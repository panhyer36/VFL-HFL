"""
VFL 訓練主腳本 - 垂直聯邦學習 + FedAvg

**訓練流程**:
1. 初始化 Server (全局 Weather Model)
2. 初始化 Clients (HFL Model + Fusion Model)
3. 載入並預處理數據 (Weather + HFL)
4. 聯邦學習訓練循環 (FedAvg)
5. 保存模型和可視化結果

**數據流**:
- Server: 管理 Weather Model，聚合梯度
- Clients:
  * Weather 數據 → Weather Model → Weather 嵌入
  * HFL 數據 → HFL Model → HFL 嵌入
  * 雙方嵌入 → Fusion Model → 預測

**三階段通訊優化**:
- Phase 0: Fusion 預熱期 - 只訓練 Fusion Model，Weather Model 凍結
- Phase 1: 聯合訓練期 - 每輪都訓練 Fusion + Weather
- Phase 2: 通訊優化期 - 4輪訓練 Fusion，1輪訓練 Weather (節省通訊)
"""

import argparse
import os
import glob
import torch
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from dotenv import load_dotenv
import requests

# 導入自定義模組
from config import load_config
from src.Server import VFLServer
from src.Client import VFLClient
from src.Personalizer import initialize_personalized_models, save_personalized_models

# 載入環境變數
load_dotenv()


def send_message(message):
    """
    發送消息到 Webhook

    Args:
        message: 要發送的消息內容
    """
    if os.getenv('HOST_LINK') is None:
        return
    url = os.getenv('HOST_LINK')
    name = os.getenv('NAME')
    payload = {
        "name": name,
        "message": message
    }
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Error sending message: {e}")

def load_weather_data(config):
    """
    載入 Weather 數據並標準化

    Args:
        config: 配置對象

    Returns:
        weather_data_scaled: 標準化後的 Weather 數據
        weather_scaler: Weather 標準化器
    """
    print(f"\n{'=' * 70}")
    print("Loading Weather Data (Cloud)")
    print(f"{'=' * 70}")

    # 讀取 Weather CSV
    weather_csv_path = os.path.join(config.data_path, f"{config.weather_csv}.csv")
    weather_df = pd.read_csv(weather_csv_path)

    print(f"  - Weather raw data shape: {weather_df.shape}")
    print(f"  - Number of Weather features: {len(config.weather_features)}")

    # 提取 Weather 特徵
    weather_data_raw = weather_df[config.weather_features].values

    # 標準化器路徑
    weather_scaler_path = os.path.join(config.data_path, "weather_scaler.pkl")

    if os.path.exists(weather_scaler_path):
        # 載入已有的標準化器
        with open(weather_scaler_path, 'rb') as f:
            weather_scaler = pickle.load(f)
        print(f"  V Weather scaler loaded")
    else:
        # 創建新的標準化器 (僅在訓練集上擬合，防止數據洩漏)
        train_end = int(len(weather_data_raw) * config.train_ratio)
        weather_scaler = StandardScaler()
        weather_scaler.fit(weather_data_raw[:train_end])
        with open(weather_scaler_path, 'wb') as f:
            pickle.dump(weather_scaler, f)
        print(f"  V Weather scaler created (fitted on training portion: {train_end}/{len(weather_data_raw)} rows)")

    # 標準化
    weather_data_scaled = weather_scaler.transform(weather_data_raw)
    print(f"  V Weather data normalized: {weather_data_scaled.shape}")

    return weather_data_scaled, weather_scaler


def create_weather_sequences(weather_data, seq_len, total_len):
    """
    創建 Weather 時序序列

    Args:
        weather_data: Weather 數據 (標準化後)
        seq_len: 序列長度
        total_len: 需要的序列總數 (與 HFL 對齊)

    Returns:
        weather_sequences: Weather 序列數組
    """
    sequences = []
    for i in range(min(len(weather_data) - seq_len + 1, total_len)):
        sequences.append(weather_data[i:i + seq_len])
    return np.array(sequences)


def load_client_data(config, weather_sequences, client_csv_files):
    """
    載入所有客戶端的 HFL 數據並創建 DataLoader

    Args:
        config: 配置對象
        weather_sequences: Weather 序列數組
        client_csv_files: 客戶端 CSV 文件路徑列表

    Returns:
        client_dataloaders: 字典 {client_name: {'train': loader, 'val': loader, 'train_size': int}}
        client_names: 客戶端名稱列表
        client_target_scalers: 字典 {client_name: target_scaler} 每個客戶端自己的目標標準化器
    """
    print(f"\n{'=' * 70}")
    print("Loading Client Data (Local)")
    print(f"{'=' * 70}")

    from src.DataLoader import SequenceCSVDataset

    client_dataloaders = {}
    client_names = []
    client_target_scalers = {}

    # 序列參數
    seq_length = config.input_length
    output_length = config.output_length
    batch_size = config.batch_size

    processed_dir = os.path.join(config.data_path, 'processed')

    for idx, csv_file in enumerate(client_csv_files):
        client_name = os.path.basename(csv_file).replace('.csv', '')
        client_names.append(client_name)

        print(f"\nClient [{idx + 1}/{len(client_csv_files)}]: {client_name}")

        # 讀取客戶端數據
        client_df = pd.read_csv(csv_file)

        # 檢查目標變量
        if config.target[0] not in client_df.columns:
            target_col = 'Consumption_Total'
        else:
            target_col = config.target[0]

        # 提取特徵和目標
        client_hfl_data = client_df[config.hfl_features].values
        client_target_data = client_df[target_col].values

        # 載入前處理階段的 per-client scaler (與 PerFedAvg 訓練時一致)
        hfl_scaler_path = os.path.join(processed_dir, f'{client_name}_feature_scaler.pkl')
        target_scaler_path = os.path.join(processed_dir, f'{client_name}_target_scaler.pkl')

        with open(hfl_scaler_path, 'rb') as f:
            hfl_scaler = pickle.load(f)
        with open(target_scaler_path, 'rb') as f:
            target_scaler = pickle.load(f)
        client_target_scalers[client_name] = target_scaler
        print(f"  V Loaded per-client scalers from {processed_dir}")

        # 標準化
        client_hfl_scaled = hfl_scaler.transform(client_hfl_data)
        client_target_scaled = target_scaler.transform(client_target_data.reshape(-1, 1)).flatten()

        # 創建序列（與 PerFedAvg DataLoader 對齊）
        def create_sequences(weather, hfl, targets, seq_len):
            X_w, X_h, y = [], [], []
            # 修正：weather 已經是 sequences，不需要再減 seq_len
            # limit 應基於 hfl/targets 的長度，與 PerFedAvg 對齊
            limit = min(len(weather), len(hfl) - seq_len, len(targets) - seq_len)
            for i in range(limit):
                X_w.append(weather[i])
                X_h.append(hfl[i:i + seq_len])
                y.append(targets[i + seq_len])
            return np.array(X_w), np.array(X_h), np.array(y)

        # 不截斷 hfl/target，讓 create_sequences 內部的 limit 正確計算
        X_w, X_h, y = create_sequences(
            weather_sequences,
            client_hfl_scaled,
            client_target_scaled,
            seq_length
        )

        # 分割數據集 (8:1:1)
        total = len(X_w)
        train_size = int(config.train_ratio * total)
        val_size = int(config.val_ratio * total)

        # 訓練集
        X_w_train = torch.FloatTensor(X_w[:train_size]).to(config.device)
        X_h_train = torch.FloatTensor(X_h[:train_size]).to(config.device)
        y_train = torch.FloatTensor(y[:train_size]).unsqueeze(1).to(config.device)

        # 驗證集
        X_w_val = torch.FloatTensor(X_w[train_size:train_size + val_size]).to(config.device)
        X_h_val = torch.FloatTensor(X_h[train_size:train_size + val_size]).to(config.device)
        y_val = torch.FloatTensor(y[train_size:train_size + val_size]).unsqueeze(1).to(config.device)

        # 創建 DataLoader
        train_dataset = TensorDataset(X_w_train, X_h_train, y_train)
        val_dataset = TensorDataset(X_w_val, X_h_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        client_dataloaders[client_name] = {
            'train': train_loader,
            'val': val_loader,
            'train_size': len(train_dataset)
        }

        print(f"  V Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    print(f"\nTotal loaded {len(client_dataloaders)} clients")
    return client_dataloaders, client_names, client_target_scalers


def train(args):
    """
    VFL 訓練主函數

    Args:
        args: 命令行參數
    """
    print("\n" + "=" * 70)
    print("VFL Vertical Federated Learning Training - FedAvg")
    print("=" * 70)

    # === 步驟 1: 載入配置 ===
    config = load_config(args.config)
    device = config.device

    print(f"\nConfiguration Summary:")
    print(f"  - Algorithm: {config.algorithm}")
    print(f"  - Total rounds: {config.K}")
    print(f"  - Number of clients: {config.num_users}")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Learning rate: {config.beta}")
    print(f"  - Device: {device}")

    # === 步驟 2: 載入 Weather 數據 ===
    weather_data_scaled, weather_scaler = load_weather_data(config)

    # === 步驟 3: 載入客戶端數據 ===
    # 獲取客戶端 CSV 文件
    csv_pattern = os.path.join(config.data_path, config.hfl_csv_pattern + ".csv")
    all_files = sorted(glob.glob(csv_pattern))

    # 過濾出真正的 CSV 檔案 (排除 .pkl.csv 等)
    client_csv_files = [f for f in all_files if f.endswith('.csv') and not '.pkl' in f][:config.num_users]

    if not client_csv_files:
        raise FileNotFoundError(f"Client data not found: {csv_pattern}")

    print(f"\nFound {len(client_csv_files)} client files")

    # 創建 Weather 序列 (與第一個客戶端對齊)
    # 這裡先估算序列數量
    first_client_df = pd.read_csv(client_csv_files[0])
    total_hfl_sequences = len(first_client_df) - config.input_length
    weather_sequences = create_weather_sequences(
        weather_data_scaled,
        config.input_length,
        total_hfl_sequences
    )

    print(f"\nWeather sequences created: {weather_sequences.shape}")

    # 載入所有客戶端數據
    client_dataloaders, client_names, client_target_scalers = load_client_data(
        config,
        weather_sequences,
        client_csv_files
    )

    # === 步驟 4: 初始化 Per-FedAvg 個性化模型 (可選) ===
    client_hfl_models = {}
    if config.use_personalized_hfl and config.hfl_model_path:
        print(f"\n{'=' * 70}")
        print("Per-FedAvg Personalized HFL Model Initialization")
        print(f"{'=' * 70}")
        try:
            import torch
            from src.Model import TransformerModel
            from src.DataLoader import SequenceCSVDataset

            # 檢查 HFL 全局模型是否存在
            if os.path.exists(config.hfl_model_path):
                print(f"  V Found HFL global model: {config.hfl_model_path}")

                # 創建 HFL 模型架構
                global_hfl_model = TransformerModel(
                    feature_dim=config.hfl_feature_dim,
                    d_model=config.hfl_d_model,
                    nhead=config.hfl_nhead,
                    num_layers=config.hfl_num_layers,
                    output_dim=config.hfl_output_dim,
                    max_seq_length=config.hfl_max_seq_length,
                    dropout=config.hfl_dropout
                ).to(device)

                # 載入全局模型權重
                global_hfl_model.load_state_dict(torch.load(config.hfl_model_path, map_location=device))
                print(f"  V Successfully loaded HFL global model weights")

                # 為每個客戶端進行個性化適應
                print(f"\n  Starting personalization adaptation for each client...")
                print(f"  Adaptation parameters: lr={config.adaptation_lr}, steps={config.personalization_steps}")
                print()

                for i, csv_file in enumerate(client_csv_files):
                    client_name = os.path.basename(csv_file).replace('.csv', '')
                    print(f"  [{i+1}/{len(client_csv_files)}] {client_name}:")

                    # 載入客戶端數據集 (用於個性化)
                    try:
                        client_dataset = SequenceCSVDataset(
                            csv_path=os.path.dirname(csv_file),
                            csv_name=client_name,
                            input_len=config.input_length,
                            output_len=config.output_length,
                            features=config.hfl_features,
                            target=config.target,
                            save_path=os.path.dirname(csv_file),
                            train_ratio=config.train_ratio,
                            val_ratio=config.val_ratio,
                            split_type='time_based',
                            fit_scalers=False  # 使用已保存的標準化器
                        )

                        # 使用 Personalizer 進行個性化適應
                        from src.Personalizer import personalize_model_for_client
                        personalized_state = personalize_model_for_client(
                            global_model=global_hfl_model,
                            dataset=client_dataset,
                            config=config,
                            client_name=client_name
                        )

                        # 保存個性化後的模型權重
                        client_hfl_models[client_name] = personalized_state

                    except Exception as e:
                        print(f"    ! Personalization failed: {e}")
                        print(f"    -> Using global model weights")
                        client_hfl_models[client_name] = global_hfl_model.state_dict()

                print(f"\n  V Completed personalization adaptation for {len(client_hfl_models)} clients")

                # 保存個性化 HFL 模型到磁碟
                save_personalized_models(client_hfl_models, config.model_save_path)
                print(f"  V Personalized HFL models saved to: {config.model_save_path}/")
            else:
                print(f"  ! HFL global model file not found: {config.hfl_model_path}")
                print(f"  -> Will use randomly initialized HFL model")
        except Exception as e:
            import traceback
            print(f"  ! Failed to load/personalize HFL model: {e}")
            print(traceback.format_exc())
            print(f"  -> Will use randomly initialized HFL model")

    # === 步驟 5: 初始化 Server 和 Clients ===
    print(f"\n{'=' * 70}")
    print("Initializing VFL Server and Clients")
    print(f"{'=' * 70}")

    # 初始化 Server
    server = VFLServer(config, device)

    # 初始化 Clients
    clients = {}
    for client_name in client_names:
        hfl_state_dict = client_hfl_models.get(client_name, None)
        client = VFLClient(
            client_id=client_name,
            config=config,
            device=device,
            hfl_model_state_dict=hfl_state_dict
        )
        clients[client_name] = client

    # === 步驟 5.5: 初始凍結 LoRA (Phase 0) ===
    if config.lora_rank > 0:
        for client in clients.values():
            client.set_lora_training(False)
        print(f"\n  V LoRA parameters frozen (Phase 0 warmup)")

    # === 步驟 6: 初始化學習率調度器 ===
    fusion_schedulers = {}
    for client_name in client_names:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            clients[client_name].fusion_optimizer, T_max=config.K
        )
        fusion_schedulers[client_name] = scheduler
    server_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        server.global_optimizer, T_max=config.K
    )
    print(f"\n  V Learning rate schedulers initialized (CosineAnnealingLR, T_max={config.K})")

    # === 步驟 7: 聯邦學習訓練循環 ===
    print(f"\n{'=' * 70}")
    print("Starting Federated Learning Training...")
    send_message("Starting Federated Learning Training...")
    print(f"{'=' * 70}")

    for round_idx in range(config.K):
        server.current_round = round_idx

        # 獲取當前階段資訊
        phase_info = server.get_current_phase_info()
        train_weather = phase_info['train_weather']

        print(f"\n{'─' * 70}")
        print(f"Federated Learning Round [{round_idx + 1}/{config.K}]")
        print(f"Phase {phase_info['phase']} - {phase_info['phase_name']} [{phase_info['phase_round']}/{phase_info['phase_total']}]")
        print(f"{'─' * 70}")

        # Phase 0 -> Phase 1 轉換時解凍 LoRA
        if config.lora_rank > 0 and round_idx == config.phase0_rounds:
            for client in clients.values():
                client.set_lora_training(True)
            print(f"  V LoRA parameters unfrozen (entering Phase 1)")

        # 根據階段顯示訓練模式
        if phase_info['phase'] == 0:
            print(f"  Training mode: Fusion Model only (Weather frozen - warmup period)")
        elif train_weather:
            print(f"  Training mode: Fusion Model + Weather Model")
        else:
            print(f"  Training mode: Fusion Model only (Save communication)")

        # 客戶端選擇
        selected_clients = server.select_clients(client_names)
        if phase_info['phase'] == 0:
            print(f"\n  Selected clients: All {len(selected_clients)} clients (Fusion warmup - full participation)")
        else:
            print(f"\n  Selected clients ({len(selected_clients)}/{len(client_names)}): {selected_clients}")

        # === Split Learning 訓練 (Batch-wise with FedAvg Aggregation) ===
        print(f"\n  Distributed Training (Split Learning) with FedAvg aggregation:")

        if train_weather:
            server.zero_weather_model_grad()

        client_losses = []
        client_grad_lists = []
        client_sample_counts = []

        # 逐個客戶端進行訓練 (Batch-wise)
        for client_name in selected_clients:
            client = clients[client_name]
            train_loader = client_dataloaders[client_name]['train']

            # --- 訓練階段 ---
            client_total_loss = 0
            client_num_batches = 0

            # 遍歷每個 Batch
            for batch_idx, (weather_batch, hfl_batch, targets) in enumerate(train_loader):
                weather_batch = weather_batch.to(device)

                # 1. Server 計算 Weather 嵌入 (Batch-wise)
                weather_embedding = server.compute_weather_embeddings(
                    weather_batch,
                    requires_grad=train_weather
                )

                # 2. Client 執行訓練步
                loss, weather_grad = client.train_batch(
                    weather_embedding,
                    hfl_batch,
                    targets,
                    train_weather=train_weather
                )

                # 3. Server 反向傳播 (累積梯度)
                if train_weather and weather_grad is not None:
                    weather_embedding.backward(weather_grad)

                client_total_loss += loss
                client_num_batches += 1

                # 釋放記憶體
                del weather_embedding, weather_grad, loss

            # 計算該客戶端的平均 Loss
            avg_loss = client_total_loss / client_num_batches if client_num_batches > 0 else 0
            client_losses.append(avg_loss)
            print(f"    V {client_name}: Train Loss = {avg_loss:.6f}")

            if train_weather and client_num_batches > 0:
                # 擷取累積梯度並除以 batch 數量 (取平均)
                # 防止梯度與 FedAvg 樣本數加權產生二次加權
                raw_grads = server.capture_weather_model_grads()
                avg_grads = [g / client_num_batches for g in raw_grads]
                client_grad_lists.append(avg_grads)
                client_sample_counts.append(len(train_loader.dataset))
                server.zero_weather_model_grad()

        # === Server FedAvg 聚合並更新 Weather Model ===
        if train_weather and client_grad_lists:
            aggregated_grads = server.aggregate_weather_gradients(
                client_grad_lists,
                client_sample_counts
            )
            if aggregated_grads:  # 可能因 NaN/Inf 過濾後為空
                server.apply_aggregated_gradients(aggregated_grads)
                print(f"    V Global Weather Model updated (FedAvg)")

        # === 驗證所有客戶端 (穩定的早停信號) ===
        all_val_losses = []
        for client_name in client_names:
            cl = clients[client_name]
            val_loader = client_dataloaders[client_name]['val']

            client_val_loss = 0
            val_batches = 0
            for weather_batch, hfl_batch, targets in val_loader:
                weather_batch = weather_batch.to(device)
                weather_embedding = server.compute_weather_embeddings(
                    weather_batch, requires_grad=False
                )
                l = cl.evaluate_batch(weather_embedding, hfl_batch, targets)
                client_val_loss += l
                val_batches += 1
                del weather_embedding

            avg_val = client_val_loss / val_batches if val_batches > 0 else 0
            all_val_losses.append(avg_val)

        # 全局評估
        avg_train_loss = sum(client_losses) / len(client_losses) if client_losses else 0
        avg_val_loss = sum(all_val_losses) / len(all_val_losses) if all_val_losses else 0

        print(f"\n  [Round Results]")
        print(f"    Average train loss ({len(selected_clients)} clients): {avg_train_loss:.6f}")
        print(f"    Average val loss (all {len(client_names)} clients): {avg_val_loss:.6f}")

        # 早停檢查 (使用所有客戶端的驗證損失，避免客戶端抽樣噪聲)
        should_stop, is_best = server.evaluate_global(avg_train_loss, avg_val_loss, selected_clients)

        # 保存最佳 Fusion Models + LoRA (與最佳 Weather Model 同步)
        if is_best:
            for cn, cl in clients.items():
                fp = os.path.join(config.model_save_path, f"best_{cn}_fusion_model.pth")
                cl.save_fusion_model(fp)
                if config.lora_rank > 0:
                    lp = os.path.join(config.model_save_path, f"best_{cn}_lora_model.pth")
                    cl.save_lora_model(lp)
            print(f"    V Best models saved (val loss: {avg_val_loss:.6f})")

        if should_stop:
            print(f"\nEarly stopping triggered, training ended at round {round_idx + 1}")
            break

        # 學習率調度
        for cn in client_names:
            fusion_schedulers[cn].step()
        server_scheduler.step()

        # 定期評估
        if (round_idx + 1) % config.eval_interval == 0:
            current_lr = fusion_schedulers[client_names[0]].get_last_lr()[0]
            print(f"\n  [Evaluation Summary - Round {round_idx + 1}]")
            print(f"    Best val loss: {server.best_val_loss:.6f}")
            print(f"    Early stopping counter: {server.patience_counter}/{config.early_stopping_patience}")
            print(f"    Current learning rate: {current_lr:.6f}")

    # === 步驟 8: 保存最終模型 ===
    print(f"\n{'=' * 70}")
    print("Saving final models...")
    print(f"{'=' * 70}")

    server.save_final_model()

    # 保存最終客戶端 Fusion Models + LoRA
    for client_name, client in clients.items():
        fusion_path = os.path.join(
            config.model_save_path,
            f"{client_name}_fusion_model.pth"
        )
        client.save_fusion_model(fusion_path)
        if config.lora_rank > 0:
            lora_path = os.path.join(
                config.model_save_path,
                f"{client_name}_lora_model.pth"
            )
            client.save_lora_model(lora_path)
        print(f"  V {client_name} Fusion Model saved (final)")

    # === 步驟 9: 訓練摘要 ===
    summary = server.get_training_summary()

    print(f"\n{'=' * 70}")
    print("Training completed!")
    send_message("Training completed!")
    print(f"{'=' * 70}")
    print(f"\nTraining Summary:")
    print(f"  - Total rounds: {summary['total_rounds']}")
    print(f"  - Training strategy: {summary['training_strategy']}")
    print(f"  - Weather Model updates: {summary['weather_updates']}")
    print(f"  - Actual communication saving: {summary['comm_saving_actual']:.1f}%")
    print(f"  - Best val loss: {summary['best_val_loss']:.6f}")
    print(f"  - Final train loss: {summary['final_train_loss']:.6f}")
    print(f"  - Final val loss: {summary['final_val_loss']:.6f}")

    print(f"\nModels saved to: {config.model_save_path}/")
    print(f"  - Best models: best_weather_model.pth + best_*_fusion_model.pth")
    print(f"  - Final models: final_weather_model.pth + *_fusion_model.pth")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VFL Model Training')
    parser.add_argument('--config', default='config.yaml',
                        help='配置文件路徑 (default: config.yaml)')
    args = parser.parse_args()

    train(args)

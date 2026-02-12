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


def format_bytes(num_bytes):
    """將 bytes 轉換為人類可讀格式 (KB/MB/GB)"""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 ** 2:
        return f"{num_bytes / 1024:.2f} KB"
    elif num_bytes < 1024 ** 3:
        return f"{num_bytes / 1024 ** 2:.2f} MB"
    else:
        return f"{num_bytes / 1024 ** 3:.2f} GB"


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
            'train_size': len(train_dataset),
            # 原始 tensor (用於快取路徑建立臨時 DataLoader)
            'train_weather_raw': X_w_train,
            'train_hfl': X_h_train,
            'train_targets': y_train,
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

    # === 步驟 5.5: 初始化通訊統計 ===
    server.init_comm_stats(client_names)

    # === 步驟 5.6: 初始凍結 LoRA (Phase 0) ===
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

        client_losses = []
        client_grad_lists = []
        client_sample_counts = []
        # B-LEC 壓縮品質指標收集 (per-client per-round)
        round_dl_metrics = []   # 下載端 SVD 指標
        round_ul_metrics = []   # 上傳端 Top-k 指標

        if train_weather:
            # === 路徑 A: train_weather=True (Phase 1, Phase 2 更新輪) ===
            # 流程: Server 一次性下發嵌入 (no-grad, 省記憶體) → Client 逐 batch 訓練並累積嵌入梯度
            #       → 累積完成後一次上傳 → Server 分塊反向傳播 (避免 OOM)

            for client_name in selected_clients:
                client = clients[client_name]

                # 1. Server 一次性前向計算所有訓練集 Weather 嵌入 (不保留計算圖，節省記憶體)
                X_w_raw = client_dataloaders[client_name]['train_weather_raw']
                if config.download_compression_enabled:
                    # B-LEC 下載端壓縮: 低秩殘差 SVD
                    client_z_old = client.get_cached_weather_embeddings()
                    if client_z_old is None:
                        client_z_old = torch.zeros(
                            X_w_raw.size(0), config.weather_d_model, device=device
                        )
                    A, B, z_new, dl_metrics = server.download_weather_embeddings_compressed(
                        X_w_raw, client_z_old, config.svd_rank
                    )
                    dl_metrics['client'] = client_name
                    round_dl_metrics.append(dl_metrics)
                    if A is None:
                        # 首輪未壓縮，直接使用完整嵌入
                        all_embeddings = z_new.detach()
                        server.record_download(client_name, all_embeddings)
                    else:
                        server.record_download(client_name, A)
                        server.record_download(client_name, B)
                        all_embeddings = VFLServer.reconstruct_from_low_rank(
                            A, B, client_z_old
                        )
                    # 更新快取 (供下輪 delta 計算使用)
                    client.cache_weather_embeddings(
                        all_embeddings, server.weather_model_version
                    )
                else:
                    with torch.no_grad():
                        all_embeddings = server.download_weather_embeddings(
                            X_w_raw, requires_grad=False
                        )
                    server.record_download(client_name, all_embeddings)

                # 2. Client 逐 batch 訓練，累積嵌入梯度
                X_h = client_dataloaders[client_name]['train_hfl']
                y = client_dataloaders[client_name]['train_targets']
                N = X_h.size(0)
                indices = torch.arange(N, device=device)
                idx_dataset = TensorDataset(indices, X_h, y)
                idx_loader = DataLoader(
                    idx_dataset, batch_size=config.batch_size, shuffle=True
                )

                accumulated_grad = torch.zeros_like(all_embeddings)
                client_total_loss = 0
                client_num_batches = 0

                for idx_batch, hfl_batch, targets in idx_loader:
                    emb_batch = all_embeddings[idx_batch].detach().requires_grad_(True)
                    loss, weather_grad = client.train_batch(
                        emb_batch, hfl_batch, targets, train_weather=True
                    )
                    if weather_grad is not None:
                        accumulated_grad[idx_batch] = weather_grad
                    client_total_loss += loss
                    client_num_batches += 1

                avg_loss = client_total_loss / client_num_batches if client_num_batches > 0 else 0
                client_losses.append(avg_loss)
                print(f"    V {client_name}: Train Loss = {avg_loss:.6f}")

                # 3. 累積嵌入梯度取平均後一次上傳，Server 分塊反向傳播
                if client_num_batches > 0:
                    avg_embedding_grad = accumulated_grad / client_num_batches

                    # B-LEC 上傳端壓縮: Top-k Error Feedback
                    if config.upload_compression_enabled:
                        compressed_vals, comp_indices, grad_shape, ul_metrics = \
                            client.compress_gradient_with_error_feedback(avg_embedding_grad)
                        ul_metrics['client'] = client_name
                        round_ul_metrics.append(ul_metrics)
                        server.record_upload(client_name, [compressed_vals, comp_indices])
                        decompressed_grad = VFLServer.decompress_gradient(
                            compressed_vals, comp_indices, grad_shape, device
                        )
                        server.backward_weather_embeddings_chunked(
                            X_w_raw, decompressed_grad
                        )
                    else:
                        server.record_upload(client_name, [avg_embedding_grad])
                        server.backward_weather_embeddings_chunked(
                            X_w_raw, avg_embedding_grad
                        )

                    raw_grads = server.capture_weather_model_grads()
                    client_grad_lists.append(raw_grads)
                    client_sample_counts.append(N)

                del all_embeddings, accumulated_grad

            # Server FedAvg 聚合並更新 Weather Model
            if client_grad_lists:
                updated = server.upload_and_aggregate_gradients(
                    client_grad_lists, client_sample_counts
                )
                if updated:
                    print(f"    V Global Weather Model updated (FedAvg, version={server.weather_model_version})")

        else:
            # === 路徑 B: train_weather=False (Phase 0, Phase 2 非更新輪) ===
            for client_name in selected_clients:
                client = clients[client_name]

                # 快取檢查: 版本不匹配時一次性計算整個訓練集嵌入
                if not client.is_cache_valid(server.weather_model_version):
                    X_w_raw = client_dataloaders[client_name]['train_weather_raw']
                    if config.download_compression_enabled:
                        # B-LEC 下載端壓縮: 低秩殘差 SVD
                        client_z_old = client.get_cached_weather_embeddings()
                        if client_z_old is None:
                            client_z_old = torch.zeros(
                                X_w_raw.size(0), config.weather_d_model, device=device
                            )
                        A, B, z_new, dl_metrics = server.download_weather_embeddings_compressed(
                            X_w_raw, client_z_old, config.svd_rank
                        )
                        dl_metrics['client'] = client_name
                        round_dl_metrics.append(dl_metrics)
                        if A is None:
                            Z_w = z_new.detach()
                            server.record_download(client_name, Z_w)
                        else:
                            server.record_download(client_name, A)
                            server.record_download(client_name, B)
                            Z_w = VFLServer.reconstruct_from_low_rank(
                                A, B, client_z_old
                            )
                    else:
                        with torch.no_grad():
                            Z_w = server.download_weather_embeddings(X_w_raw, requires_grad=False)
                        server.record_download(client_name, Z_w)
                    client.cache_weather_embeddings(Z_w, server.weather_model_version)

                # 用快取嵌入建立臨時 DataLoader
                cached_Z = client.get_cached_weather_embeddings()
                X_h = client_dataloaders[client_name]['train_hfl']
                y = client_dataloaders[client_name]['train_targets']
                cached_dataset = TensorDataset(cached_Z, X_h, y)
                cached_loader = DataLoader(
                    cached_dataset, batch_size=config.batch_size, shuffle=True
                )

                client_total_loss = 0
                client_num_batches = 0

                for emb_batch, hfl_batch, targets in cached_loader:
                    loss, _ = client.train_batch(
                        emb_batch, hfl_batch, targets, train_weather=False
                    )
                    client_total_loss += loss
                    client_num_batches += 1

                avg_loss = client_total_loss / client_num_batches if client_num_batches > 0 else 0
                client_losses.append(avg_loss)
                print(f"    V {client_name}: Train Loss = {avg_loss:.6f}")

        # === B-LEC 壓縮品質報告 (當輪) ===
        if round_dl_metrics:
            compressed_dl = [m for m in round_dl_metrics if m.get('compressed')]
            if compressed_dl:
                avg_explained = sum(m['explained_var'] for m in compressed_dl) / len(compressed_dl)
                avg_rel_err  = sum(m['relative_error'] for m in compressed_dl) / len(compressed_dl)
                avg_byte_r   = sum(m['byte_ratio'] for m in compressed_dl) / len(compressed_dl)
                print(f"\n  [B-LEC Download SVD] rank={compressed_dl[0]['svd_rank']}, "
                      f"clients={len(compressed_dl)}")
                print(f"    Explained variance: {avg_explained:.4f}  |  "
                      f"Relative recon error: {avg_rel_err:.6f}  |  "
                      f"Byte ratio: {avg_byte_r:.2%}")
            else:
                # 所有客戶端都是首輪未壓縮
                print(f"\n  [B-LEC Download] First round - full embeddings sent "
                      f"({len(round_dl_metrics)} clients)")
        if round_ul_metrics:
            avg_sparsity  = sum(m['sparsity'] for m in round_ul_metrics) / len(round_ul_metrics)
            avg_rel_err   = sum(m['relative_error'] for m in round_ul_metrics) / len(round_ul_metrics)
            avg_byte_r    = sum(m['byte_ratio'] for m in round_ul_metrics) / len(round_ul_metrics)
            avg_eb_norm   = sum(m['error_buf_norm'] for m in round_ul_metrics) / len(round_ul_metrics)
            avg_grad_norm = sum(m['grad_norm'] for m in round_ul_metrics) / len(round_ul_metrics)
            print(f"  [B-LEC Upload Top-k] γ={config.top_k_ratio}, "
                  f"clients={len(round_ul_metrics)}")
            print(f"    Sparsity: {avg_sparsity:.4f}  |  "
                  f"Relative error: {avg_rel_err:.6f}  |  "
                  f"Byte ratio: {avg_byte_r:.2%}")
            print(f"    Grad norm: {avg_grad_norm:.6f}  |  "
                  f"Error buffer norm: {avg_eb_norm:.6f}")

        # === 驗證所有客戶端 (穩定的早停信號) ===
        all_val_losses = []
        for client_name in client_names:
            cl = clients[client_name]
            val_loader = client_dataloaders[client_name]['val']

            client_val_loss = 0
            val_batches = 0
            for weather_batch, hfl_batch, targets in val_loader:
                weather_batch = weather_batch.to(device)
                weather_embedding = server.download_weather_embeddings(
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
    print(f"  - Best val loss: {summary['best_val_loss']:.6f}")
    print(f"  - Final train loss: {summary['final_train_loss']:.6f}")
    print(f"  - Final val loss: {summary['final_val_loss']:.6f}")

    # === 步驟 9.5: 通訊統計 ===
    comm_stats = server.get_comm_stats()
    compression_active = config.download_compression_enabled or config.upload_compression_enabled

    print(f"\nCommunication Statistics (Training Only):")
    if compression_active:
        print(f"  B-LEC Compression: Download={'ON (SVD rank=' + str(config.svd_rank) + ')' if config.download_compression_enabled else 'OFF'}, "
              f"Upload={'ON (Top-k γ=' + str(config.top_k_ratio) + ')' if config.upload_compression_enabled else 'OFF'}")

    # 估算未壓縮通訊量 (用於壓縮率計算)
    # 每個客戶端每輪: 下載 N×D×4B, 上傳 N×D×4B (僅 Path A 輪次)
    if compression_active:
        header = f"{'Client':<25} {'Downloaded':>15} {'Uploaded':>15} {'Total':>15} {'Ratio':>10}"
        separator = '─' * 85
    else:
        header = f"{'Client':<25} {'Downloaded':>15} {'Uploaded':>15} {'Total':>15}"
        separator = '─' * 70
    print(header)
    print(separator)

    total_down = total_up = 0
    # 估算未壓縮量: 每個 Path A 輪次每客戶端 download + upload 各 N×D×4 bytes
    total_uncompressed_estimate = 0
    for name, stats in comm_stats.items():
        down = stats['bytes_downloaded']
        up = stats['bytes_uploaded']
        total_down += down
        total_up += up

        if compression_active:
            # 以首個客戶端的訓練集大小估算 per-client 未壓縮量
            train_size = client_dataloaders[name]['train_size']
            D = config.weather_d_model
            # 未壓縮估算: 實際 download/upload 次數 × N×D×4
            uncompressed_est = down + up  # 回退估算: 若無法精確計算，用未壓縮 = 同樣次數的完整傳輸
            # 更精確: 每次 download = N×D×4, 每次 upload = N×D×4
            per_round_bytes = train_size * D * 4
            # 估算完整下載輪數 (Phase 1 全部 + Phase 2 部分)
            weather_update_count = len(server.history['weather_update_rounds'])
            uncompressed_per_client = per_round_bytes * weather_update_count * 2  # download + upload
            total_uncompressed_estimate += uncompressed_per_client
            ratio = (down + up) / uncompressed_per_client if uncompressed_per_client > 0 else 1.0
            print(f"{name:<25} {format_bytes(down):>15} {format_bytes(up):>15} {format_bytes(down + up):>15} {ratio:>9.1%}")
        else:
            print(f"{name:<25} {format_bytes(down):>15} {format_bytes(up):>15} {format_bytes(down + up):>15}")

    print(separator)
    if compression_active:
        total_ratio = (total_down + total_up) / total_uncompressed_estimate if total_uncompressed_estimate > 0 else 1.0
        print(f"{'Total':<25} {format_bytes(total_down):>15} {format_bytes(total_up):>15} {format_bytes(total_down + total_up):>15} {total_ratio:>9.1%}")
    else:
        print(f"{'Total':<25} {format_bytes(total_down):>15} {format_bytes(total_up):>15} {format_bytes(total_down + total_up):>15}")

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

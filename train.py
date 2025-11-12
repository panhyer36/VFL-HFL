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

**通訊優化**:
- 階段1: 每輪都訓練 Fusion + Weather
- 階段2: 4輪訓練 Fusion，1輪訓練 Weather (節省通訊)
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

# 導入自定義模組
from config import load_config
from Server import VFLServer
from Client import VFLClient
from Personalizer import initialize_personalized_models


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
    print("載入 Weather 數據 (雲端)")
    print(f"{'=' * 70}")

    # 讀取 Weather CSV
    weather_csv_path = os.path.join(config.data_path, f"{config.weather_csv}.csv")
    weather_df = pd.read_csv(weather_csv_path)

    print(f"  - Weather 原始數據形狀: {weather_df.shape}")
    print(f"  - Weather 特徵數: {len(config.weather_features)}")

    # 提取 Weather 特徵
    weather_data_raw = weather_df[config.weather_features].values

    # 標準化器路徑
    weather_scaler_path = os.path.join(config.data_path, "weather_scaler.pkl")

    if os.path.exists(weather_scaler_path):
        # 載入已有的標準化器
        with open(weather_scaler_path, 'rb') as f:
            weather_scaler = pickle.load(f)
        print(f"  ✓ Weather 標準化器已載入")
    else:
        # 創建新的標準化器
        weather_scaler = StandardScaler()
        weather_scaler.fit(weather_data_raw)
        with open(weather_scaler_path, 'wb') as f:
            pickle.dump(weather_scaler, f)
        print(f"  ✓ Weather 標準化器已創建並保存")

    # 標準化
    weather_data_scaled = weather_scaler.transform(weather_data_raw)
    print(f"  ✓ Weather 數據已標準化: {weather_data_scaled.shape}")

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
        target_scaler: 目標變量標準化器
    """
    print(f"\n{'=' * 70}")
    print("載入客戶端數據 (本地)")
    print(f"{'=' * 70}")

    from DataLoader import SequenceCSVDataset

    client_dataloaders = {}
    client_names = []
    target_scaler = None
    hfl_scaler = None

    # 序列參數
    seq_length = config.input_length
    output_length = config.output_length
    batch_size = config.batch_size

    for idx, csv_file in enumerate(client_csv_files):
        client_name = os.path.basename(csv_file).replace('.csv', '')
        client_names.append(client_name)

        print(f"\n客戶端 [{idx + 1}/{len(client_csv_files)}]: {client_name}")

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

        # 標準化器 (第一個客戶端創建，其餘共用)
        if idx == 0:
            hfl_scaler = StandardScaler()
            hfl_scaler.fit(client_hfl_data)
            target_scaler = StandardScaler()
            target_scaler.fit(client_target_data.reshape(-1, 1))

            # 保存標準化器
            with open(os.path.join(config.data_path, 'hfl_scaler.pkl'), 'wb') as f:
                pickle.dump(hfl_scaler, f)
            with open(os.path.join(config.data_path, 'target_scaler.pkl'), 'wb') as f:
                pickle.dump(target_scaler, f)

        # 標準化
        client_hfl_scaled = hfl_scaler.transform(client_hfl_data)
        client_target_scaled = target_scaler.transform(client_target_data.reshape(-1, 1)).flatten()

        # 對齊長度
        min_len = min(len(weather_sequences), len(client_hfl_scaled), len(client_target_scaled))

        # 創建序列
        def create_sequences(weather, hfl, targets, seq_len):
            X_w, X_h, y = [], [], []
            for i in range(len(weather) - seq_len):
                X_w.append(weather[i])
                X_h.append(hfl[i:i + seq_len])
                y.append(targets[i + seq_len])
            return np.array(X_w), np.array(X_h), np.array(y)

        X_w, X_h, y = create_sequences(
            weather_sequences[:min_len],
            client_hfl_scaled[:min_len],
            client_target_scaled[:min_len],
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

        print(f"  ✓ 訓練樣本: {len(train_dataset)}, 驗證樣本: {len(val_dataset)}")

    print(f"\n總共載入 {len(client_dataloaders)} 個客戶端")
    return client_dataloaders, client_names, target_scaler


def train(args):
    """
    VFL 訓練主函數

    Args:
        args: 命令行參數
    """
    print("\n" + "=" * 70)
    print("VFL 垂直聯邦學習訓練 - FedAvg")
    print("=" * 70)

    # === 步驟 1: 載入配置 ===
    config = load_config(args.config)
    device = config.device

    print(f"\n配置摘要:")
    print(f"  - 演算法: {config.algorithm}")
    print(f"  - 全局輪數: {config.K}")
    print(f"  - 客戶端數: {config.num_users}")
    print(f"  - 批次大小: {config.batch_size}")
    print(f"  - 學習率: {config.beta}")
    print(f"  - 設備: {device}")

    # === 步驟 2: 載入 Weather 數據 ===
    weather_data_scaled, weather_scaler = load_weather_data(config)

    # === 步驟 3: 載入客戶端數據 ===
    # 獲取客戶端 CSV 文件
    csv_pattern = os.path.join(config.data_path, config.hfl_csv_pattern + ".csv")
    client_csv_files = sorted(glob.glob(csv_pattern))[:config.num_users]

    if not client_csv_files:
        raise FileNotFoundError(f"未找到客戶端數據: {csv_pattern}")

    print(f"\n找到 {len(client_csv_files)} 個客戶端文件")

    # 創建 Weather 序列 (與第一個客戶端對齊)
    # 這裡先估算序列數量
    first_client_df = pd.read_csv(client_csv_files[0])
    total_hfl_sequences = len(first_client_df) - config.input_length
    weather_sequences = create_weather_sequences(
        weather_data_scaled,
        config.input_length,
        total_hfl_sequences
    )

    print(f"\nWeather 序列已創建: {weather_sequences.shape}")

    # 載入所有客戶端數據
    client_dataloaders, client_names, target_scaler = load_client_data(
        config,
        weather_sequences,
        client_csv_files
    )

    # === 步驟 4: 初始化 Per-FedAvg 個性化模型 (可選) ===
    client_hfl_models = {}
    if config.use_personalized_hfl and config.hfl_model_path:
        print(f"\n{'=' * 70}")
        print("Per-FedAvg 個性化 HFL 模型初始化")
        print(f"{'=' * 70}")
        try:
            import torch
            from Model import TransformerModel
            from DataLoader import SequenceCSVDataset

            # 檢查 HFL 全局模型是否存在
            if os.path.exists(config.hfl_model_path):
                print(f"  ✓ 找到 HFL 全局模型: {config.hfl_model_path}")

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
                print(f"  ✓ 成功載入 HFL 全局模型權重")

                # 為每個客戶端進行個性化適應
                print(f"\n  開始為每個客戶端進行個性化適應...")
                print(f"  適應參數: lr={config.adaptation_lr}, steps={config.personalization_steps}")
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
                        from Personalizer import personalize_model_for_client
                        personalized_state = personalize_model_for_client(
                            global_model=global_hfl_model,
                            dataset=client_dataset,
                            config=config,
                            client_name=client_name
                        )

                        # 保存個性化後的模型權重
                        client_hfl_models[client_name] = personalized_state

                    except Exception as e:
                        print(f"    ⚠ 個性化失敗: {e}")
                        print(f"    → 使用全局模型權重")
                        client_hfl_models[client_name] = global_hfl_model.state_dict()

                print(f"\n  ✓ 已為 {len(client_hfl_models)} 個客戶端完成個性化適應")
            else:
                print(f"  ⚠ HFL 全局模型文件不存在: {config.hfl_model_path}")
                print(f"  → 將使用隨機初始化的 HFL 模型")
        except Exception as e:
            import traceback
            print(f"  ⚠ 載入/個性化 HFL 模型失敗: {e}")
            print(traceback.format_exc())
            print(f"  → 將使用隨機初始化的 HFL 模型")

    # === 步驟 5: 初始化 Server 和 Clients ===
    print(f"\n{'=' * 70}")
    print("初始化 VFL Server 和 Clients")
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

    # === 步驟 6: 聯邦學習訓練循環 ===
    print(f"\n{'=' * 70}")
    print("開始聯邦學習訓練...")
    print(f"{'=' * 70}")

    for round_idx in range(config.K):
        server.current_round = round_idx

        print(f"\n{'─' * 70}")
        print(f"聯邦學習輪次 [{round_idx + 1}/{config.K}]")
        print(f"{'─' * 70}")

        # 確定訓練策略
        train_weather = server.should_update_weather()

        if train_weather:
            print(f"  訓練模式: Fusion Model + Weather Model ⚡")
        else:
            print(f"  訓練模式: Fusion Model only (節省通訊) 📡")

        # 客戶端選擇
        selected_clients = server.select_clients(client_names)
        print(f"\n  選中客戶端: {selected_clients}")

        # === Split Learning 前向傳播: Server 計算 Weather 嵌入 ===
        print(f"\n  Server 計算 Weather 嵌入向量:")

        # 收集所有選中客戶端的 Weather 數據
        client_weather_data = {}
        for client_name in selected_clients:
            train_loader = client_dataloaders[client_name]['train']
            val_loader = client_dataloaders[client_name]['val']

            # 提取 Weather 數據 (訓練集)
            train_weather_batches = []
            for weather_batch, _, _ in train_loader:
                train_weather_batches.append(weather_batch)
            train_weather_data = torch.cat(train_weather_batches, dim=0).to(device)

            # 提取 Weather 數據 (驗證集)
            val_weather_batches = []
            for weather_batch, _, _ in val_loader:
                val_weather_batches.append(weather_batch)
            val_weather_data = torch.cat(val_weather_batches, dim=0).to(device)

            client_weather_data[client_name] = {
                'train': train_weather_data,
                'val': val_weather_data
            }

        # Server 計算嵌入向量
        client_weather_embeddings = {}
        for client_name in selected_clients:
            # 訓練集嵌入 (需要梯度)
            train_embeddings = server.compute_weather_embeddings(
                client_weather_data[client_name]['train'],
                requires_grad=train_weather
            )

            # 驗證集嵌入 (不需要梯度)
            val_embeddings = server.compute_weather_embeddings(
                client_weather_data[client_name]['val'],
                requires_grad=False
            )

            client_weather_embeddings[client_name] = {
                'train': train_embeddings,
                'val': val_embeddings
            }

            print(f"    ✓ {client_name}: Train Embeddings {train_embeddings.shape}, Val Embeddings {val_embeddings.shape}")

        # === 客戶端本地訓練 (使用 Server 發送的嵌入) ===
        client_losses = []
        client_val_losses = []
        client_embedding_gradients = []
        client_sample_counts = []

        print(f"\n  本地訓練 (Client 端):")
        for client_name in selected_clients:
            client = clients[client_name]
            train_loader = client_dataloaders[client_name]['train']
            val_loader = client_dataloaders[client_name]['val']

            # 本地訓練 (接收 Server 的嵌入)
            train_loss, embedding_grad, num_samples = client.local_train(
                train_loader,
                weather_embeddings=client_weather_embeddings[client_name]['train'],
                train_weather=train_weather
            )

            # 本地驗證
            val_loss = client.local_evaluate(
                val_loader,
                weather_embeddings=client_weather_embeddings[client_name]['val']
            )

            client_losses.append(train_loss)
            client_val_losses.append(val_loss)

            if train_weather and len(embedding_grad) > 0:
                client_embedding_gradients.append(embedding_grad)
                client_sample_counts.append(num_samples)

            print(f"    ✓ {client_name}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

        # === Server 聚合 Embedding 梯度並更新 Weather Model ===
        if train_weather and client_embedding_gradients:
            print(f"\n  Server 聚合 Embedding 梯度並更新 Weather Model (Split Learning + FedAvg):")

            # 使用第一個客戶端的 Weather 數據進行反向傳播 (所有客戶端共享相同的 Weather 數據)
            representative_client = selected_clients[0]
            weather_data_for_backward = client_weather_data[representative_client]['train']

            server.update_weather_model_from_embeddings(
                weather_data_for_backward,
                client_embedding_gradients,
                client_sample_counts
            )
            print(f"    ✓ 全局 Weather Model 已更新 (Chain Rule)")
            print(f"    - 參與客戶端: {len(client_embedding_gradients)}")

        # 全局評估
        avg_train_loss = sum(client_losses) / len(client_losses)
        avg_val_loss = sum(client_val_losses) / len(client_val_losses)

        print(f"\n  【本輪結果】")
        print(f"    平均訓練損失: {avg_train_loss:.6f}")
        print(f"    平均驗證損失: {avg_val_loss:.6f}")

        # 早停檢查
        should_stop = server.evaluate_global(avg_train_loss, avg_val_loss, selected_clients)
        if should_stop:
            print(f"\n早停觸發，訓練結束於第 {round_idx + 1} 輪")
            break

        # 定期評估
        if (round_idx + 1) % config.eval_interval == 0:
            print(f"\n  【評估摘要 - 第 {round_idx + 1} 輪】")
            print(f"    最佳驗證損失: {server.best_val_loss:.6f}")
            print(f"    早停計數: {server.patience_counter}/{config.early_stopping_patience}")

    # === 步驟 7: 保存模型 ===
    print(f"\n{'=' * 70}")
    print("保存模型...")
    print(f"{'=' * 70}")

    server.save_final_model()

    # 保存客戶端 Fusion Models
    for client_name, client in clients.items():
        fusion_path = os.path.join(
            config.model_save_path,
            f"{client_name}_fusion_model.pth"
        )
        client.save_fusion_model(fusion_path)
        print(f"  ✓ {client_name} Fusion Model 已保存")

    # === 步驟 8: 訓練摘要 ===
    summary = server.get_training_summary()

    print(f"\n{'=' * 70}")
    print("訓練完成！")
    print(f"{'=' * 70}")
    print(f"\n訓練摘要:")
    print(f"  - 總輪數: {summary['total_rounds']}")
    print(f"  - Weather Model 更新次數: {summary['weather_updates']}")
    print(f"  - 實際通訊節省: {summary['comm_saving_actual']:.1f}%")
    print(f"  - 最佳驗證損失: {summary['best_val_loss']:.6f}")
    print(f"  - 最終訓練損失: {summary['final_train_loss']:.6f}")
    print(f"  - 最終驗證損失: {summary['final_val_loss']:.6f}")

    print(f"\n模型已保存到: {config.model_save_path}/")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VFL Model Training')
    parser.add_argument('--config', default='config.yaml',
                        help='配置文件路徑 (default: config.yaml)')
    args = parser.parse_args()

    train(args)

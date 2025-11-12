"""
VFL Client 端模組 - 垂直聯邦學習客戶端 (Split Learning 架構)

**VFL Client 的職責**:
1. 管理本地 HFL Model (凍結的個性化模型)
2. 管理本地 Fusion Model (可訓練的融合模型)
3. 接收雲端 Server 發送的 Weather 嵌入向量
4. 本地訓練並使用 Chain Rule 計算 ∂L/∂embedding_weather
5. 將嵌入梯度回傳給 Server (不接觸 Weather Model)

**隱私保護機制 (真正的 Split Learning)**:
- Weather Model 隔離: Weather Model 只存在於 Server，Client 完全無法訪問
- 原始數據不出域: HFL 特徵和標籤只在本地處理
- 嵌入傳輸: Server → Client 傳送 Weather 嵌入
- 梯度回傳: Client → Server 回傳 ∂L/∂embedding (Chain Rule)
- 標籤保護: Power_Demand 標籤只存在於本地

**Chain Rule 梯度傳遞**:
1. Forward: Server 計算 Weather 嵌入 → Client
2. Client: 融合 HFL 嵌入 + Weather 嵌入 → 預測 → 損失
3. Backward: Client 計算 ∂L/∂embedding_weather → Server
4. Server: 使用 ∂L/∂embedding 繼續反向傳播到 Weather Model
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict
from torch.utils.data import DataLoader
from src.Model import TransformerModel, FusionModel


class VFLClient:
    """VFL 聯邦學習客戶端"""

    def __init__(
        self,
        client_id: str,
        config,
        device,
        hfl_model_state_dict: Dict = None
    ):
        """
        初始化 VFL Client

        Args:
            client_id: 客戶端 ID (如 'Consumer_01')
            config: 配置對象
            device: 計算設備
            hfl_model_state_dict: Per-FedAvg 個性化的 HFL 模型狀態字典
        """
        self.client_id = client_id
        self.config = config
        self.device = device
        self.criterion = nn.MSELoss()

        print(f"\n{'=' * 70}")
        print(f"VFL Client Initialization: {client_id}")
        print(f"{'=' * 70}")

        # === 初始化 HFL Model (本地，凍結) ===
        self.hfl_model = TransformerModel(
            feature_dim=config.hfl_feature_dim,
            d_model=config.hfl_d_model,
            nhead=config.hfl_nhead,
            num_layers=config.hfl_num_layers,
            output_dim=config.hfl_output_dim,  # None for VFL
            max_seq_length=config.hfl_max_seq_length,
            dropout=config.hfl_dropout
        ).to(device)

        # 載入 Per-FedAvg 個性化權重
        if hfl_model_state_dict is not None:
            self.hfl_model.load_state_dict(hfl_model_state_dict)
            print(f"  V Loaded Per-FedAvg personalized HFL model")
        else:
            print(f"  ! Using randomly initialized HFL model")

        # 凍結 HFL Model
        if config.freeze_hfl:
            for param in self.hfl_model.parameters():
                param.requires_grad = False
            self.hfl_model.eval()
            print(f"  V HFL Model frozen")

        hfl_params = sum(p.numel() for p in self.hfl_model.parameters())
        print(f"\nHFL Model (Local, Frozen):")
        print(f"  - Feature dimension: {config.hfl_feature_dim}")
        print(f"  - Parameters: {hfl_params:,}")

        # === 初始化 Fusion Model (本地，可訓練) ===
        self.fusion_model = FusionModel(
            embedding_dim_party_a=config.fusion_embedding_dim_weather,
            embedding_dim_party_b=config.fusion_embedding_dim_hfl,
            hidden_dim=config.fusion_hidden_dim,
            output_dim=config.fusion_output_dim,
            dropout=config.fusion_dropout
        ).to(device)

        fusion_params = sum(p.numel() for p in self.fusion_model.parameters())
        print(f"\nFusion Model (Local, Trainable):")
        print(f"  - Weather embedding dimension: {config.fusion_embedding_dim_weather}")
        print(f"  - HFL embedding dimension: {config.fusion_embedding_dim_hfl}")
        print(f"  - Parameters: {fusion_params:,}")

        # === 優化器 (只優化 Fusion Model) ===
        self.fusion_optimizer = torch.optim.Adam(
            self.fusion_model.parameters(),
            lr=config.beta,
            weight_decay=1e-4
        )

        print(f"\nDevice: {device}")
        print(f"{'=' * 70}")

    def local_train(
        self,
        train_loader: DataLoader,
        weather_embeddings: torch.Tensor,
        train_weather: bool = True
    ) -> Tuple[float, torch.Tensor, int]:
        """
        本地訓練 - VFL 場景 (Split Learning with Chain Rule)

        訓練流程:
        1. 接收來自 Server 的 Weather 嵌入向量
        2. HFL Model 生成嵌入 (凍結，不參與梯度計算)
        3. Fusion Model 融合雙方嵌入並預測
        4. 計算損失並反向傳播
        5. 更新 Fusion Model (總是更新)
        6. 提取 ∂L/∂embedding_weather (Chain Rule) 回傳 Server

        Args:
            train_loader: 訓練數據加載器
                格式: (weather_batch, hfl_batch, targets)
            weather_embeddings: Server 發送的 Weather 嵌入向量 (需要梯度)
            train_weather: 是否計算 Weather 嵌入的梯度

        Returns:
            avg_loss: 平均訓練損失
            embedding_gradients: ∂L/∂embedding_weather 的梯度列表 (如果 train_weather=True)
            num_samples: 訓練樣本數量
        """
        self.fusion_model.train()

        epoch_loss = 0.0
        num_batches = 0
        embedding_gradients = []  # 儲存每個 batch 的 embedding 梯度

        batch_idx = 0
        for _, hfl_batch, targets in train_loader:
            hfl_batch = hfl_batch.to(self.device)
            targets = targets.to(self.device)

            # === 提取當前 batch 的 Weather 嵌入 ===
            batch_size = hfl_batch.size(0)
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size

            # 從完整的 weather_embeddings 中取出當前 batch
            weather_embedding_batch = weather_embeddings[start_idx:end_idx]

            # 如果需要訓練 Weather Model，確保嵌入需要梯度
            if train_weather:
                weather_embedding_batch = weather_embedding_batch.detach().requires_grad_(True)

            # === 前向傳播 ===
            # HFL 嵌入 (凍結)
            with torch.no_grad():
                hfl_embedding = self.hfl_model.forward_embedding(hfl_batch)

            # Fusion 預測
            predictions = self.fusion_model(weather_embedding_batch, hfl_embedding)

            # 計算損失
            loss = self.criterion(predictions, targets)

            # === 反向傳播 ===
            self.fusion_optimizer.zero_grad()
            loss.backward()

            # 更新 Fusion Model
            self.fusion_optimizer.step()

            # 提取 Weather 嵌入的梯度 (∂L/∂embedding_weather)
            if train_weather and weather_embedding_batch.grad is not None:
                embedding_gradients.append(weather_embedding_batch.grad.clone())

            epoch_loss += loss.item()
            num_batches += 1
            batch_idx += 1

        # 聚合所有 batch 的 embedding 梯度
        if train_weather and embedding_gradients:
            # 將所有 batch 的梯度拼接成一個完整的梯度張量
            aggregated_embedding_grad = torch.cat(embedding_gradients, dim=0)
        else:
            aggregated_embedding_grad = []

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        num_samples = len(train_loader.dataset)

        return avg_loss, aggregated_embedding_grad, num_samples

    def local_evaluate(self, val_loader: DataLoader, weather_embeddings: torch.Tensor) -> float:
        """
        本地驗證

        Args:
            val_loader: 驗證數據加載器
            weather_embeddings: Server 發送的 Weather 嵌入向量

        Returns:
            avg_val_loss: 平均驗證損失
        """
        self.fusion_model.eval()

        val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            batch_idx = 0
            for _, hfl_batch, targets in val_loader:
                hfl_batch = hfl_batch.to(self.device)
                targets = targets.to(self.device)

                # 提取當前 batch 的 Weather 嵌入
                batch_size = hfl_batch.size(0)
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                weather_embedding_batch = weather_embeddings[start_idx:end_idx]

                # 前向傳播
                hfl_embedding = self.hfl_model.forward_embedding(hfl_batch)
                predictions = self.fusion_model(weather_embedding_batch, hfl_embedding)

                # 計算損失
                loss = self.criterion(predictions, targets)
                val_loss += loss.item()
                num_batches += 1
                batch_idx += 1

        avg_val_loss = val_loss / num_batches if num_batches > 0 else 0.0
        return avg_val_loss

    def local_test(
        self,
        test_loader: DataLoader,
        weather_embeddings: torch.Tensor
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        本地測試

        Args:
            test_loader: 測試數據加載器
            weather_embeddings: Server 發送的 Weather 嵌入向量

        Returns:
            avg_test_loss: 平均測試損失
            all_predictions: 所有預測結果
            all_targets: 所有真實標籤
        """
        self.fusion_model.eval()

        test_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            batch_idx = 0
            for _, hfl_batch, targets in test_loader:
                hfl_batch = hfl_batch.to(self.device)
                targets = targets.to(self.device)

                # 提取當前 batch 的 Weather 嵌入
                batch_size = hfl_batch.size(0)
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                weather_embedding_batch = weather_embeddings[start_idx:end_idx]

                # 前向傳播
                hfl_embedding = self.hfl_model.forward_embedding(hfl_batch)
                predictions = self.fusion_model(weather_embedding_batch, hfl_embedding)

                # 計算損失
                loss = self.criterion(predictions, targets)
                test_loss += loss.item()
                num_batches += 1
                batch_idx += 1

                # 收集預測結果
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        avg_test_loss = test_loss / num_batches if num_batches > 0 else 0.0
        return avg_test_loss, np.array(all_predictions), np.array(all_targets)

    def save_fusion_model(self, save_path: str):
        """保存 Fusion Model"""
        torch.save(self.fusion_model.state_dict(), save_path)

    def load_fusion_model(self, load_path: str):
        """載入 Fusion Model"""
        self.fusion_model.load_state_dict(torch.load(load_path, map_location=self.device))

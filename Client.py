"""
VFL Client 端模組 - 垂直聯邦學習客戶端

**VFL Client 的職責**:
1. 管理本地 HFL Model (凍結的個性化模型)
2. 管理本地 Fusion Model (可訓練的融合模型)
3. 接收雲端 Weather Model 的嵌入向量
4. 本地訓練並計算梯度
5. 將 Weather Model 的梯度回傳給 Server

**隱私保護機制**:
- 原始數據不出域: HFL 特徵只在本地處理
- 嵌入傳輸: 只傳輸 Weather 嵌入和梯度
- 標籤保護: Power_Demand 標籤只存在於本地

**與 HFL Client 的區別**:
- HFL: 訓練完整模型並上傳參數
- VFL: 只訓練 Fusion Model，Weather 梯度回傳
"""

import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from typing import Tuple, List, Dict
from torch.utils.data import DataLoader
from Model import TransformerModel, FusionModel


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
        print(f"VFL Client 初始化: {client_id}")
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
            print(f"  ✓ 載入 Per-FedAvg 個性化 HFL 模型")
        else:
            print(f"  ⚠ 使用隨機初始化的 HFL 模型")

        # 凍結 HFL Model
        if config.freeze_hfl:
            for param in self.hfl_model.parameters():
                param.requires_grad = False
            self.hfl_model.eval()
            print(f"  ✓ HFL Model 已凍結")

        hfl_params = sum(p.numel() for p in self.hfl_model.parameters())
        print(f"\nHFL Model (本地，凍結):")
        print(f"  - 特徵維度: {config.hfl_feature_dim}")
        print(f"  - 參數量: {hfl_params:,}")

        # === 初始化 Fusion Model (本地，可訓練) ===
        self.fusion_model = FusionModel(
            embedding_dim_party_a=config.fusion_embedding_dim_weather,
            embedding_dim_party_b=config.fusion_embedding_dim_hfl,
            hidden_dim=config.fusion_hidden_dim,
            output_dim=config.fusion_output_dim,
            dropout=config.fusion_dropout
        ).to(device)

        fusion_params = sum(p.numel() for p in self.fusion_model.parameters())
        print(f"\nFusion Model (本地，可訓練):")
        print(f"  - Weather 嵌入維度: {config.fusion_embedding_dim_weather}")
        print(f"  - HFL 嵌入維度: {config.fusion_embedding_dim_hfl}")
        print(f"  - 參數量: {fusion_params:,}")

        # === 優化器 (只優化 Fusion Model) ===
        self.fusion_optimizer = torch.optim.Adam(
            self.fusion_model.parameters(),
            lr=config.beta,
            weight_decay=1e-4
        )

        # === 本地 Weather Model 副本 (用於計算梯度) ===
        self.local_weather_model = None

        print(f"\n設備: {device}")
        print(f"='* 70}")

    def set_weather_model(self, global_weather_model: TransformerModel):
        """
        接收來自 Server 的全局 Weather Model

        Args:
            global_weather_model: Server 的全局 Weather Model
        """
        self.local_weather_model = deepcopy(global_weather_model).to(self.device)
        self.local_weather_model.train()

        # 為 Weather Model 創建優化器 (用於計算梯度)
        self.weather_optimizer = torch.optim.Adam(
            self.local_weather_model.parameters(),
            lr=self.config.beta,
            weight_decay=1e-4
        )

    def local_train(
        self,
        train_loader: DataLoader,
        train_weather: bool = True
    ) -> Tuple[float, List[torch.Tensor], int]:
        """
        本地訓練 - VFL 場景

        訓練流程:
        1. Weather Model 生成嵌入 (如果 train_weather=True 則參與梯度計算)
        2. HFL Model 生成嵌入 (凍結，不參與梯度計算)
        3. Fusion Model 融合雙方嵌入並預測
        4. 計算損失並反向傳播
        5. 更新 Fusion Model (總是更新)
        6. 更新 Weather Model (根據 train_weather)
        7. 提取 Weather Model 梯度回傳 Server

        Args:
            train_loader: 訓練數據加載器
                格式: (weather_batch, hfl_batch, targets)
            train_weather: 是否訓練 Weather Model

        Returns:
            avg_loss: 平均訓練損失
            weather_gradients: Weather Model 的梯度列表 (如果 train_weather=True)
            num_samples: 訓練樣本數量
        """
        if self.local_weather_model is None:
            raise RuntimeError("Weather Model 未設置! 請先調用 set_weather_model()")

        self.fusion_model.train()
        if train_weather:
            self.local_weather_model.train()

        epoch_loss = 0.0
        num_batches = 0
        weather_gradients = []

        for weather_batch, hfl_batch, targets in train_loader:
            weather_batch = weather_batch.to(self.device)
            hfl_batch = hfl_batch.to(self.device)
            targets = targets.to(self.device)

            # === 前向傳播 ===
            # Weather 嵌入
            if train_weather:
                weather_embedding = self.local_weather_model.forward_embedding(weather_batch)
            else:
                with torch.no_grad():
                    weather_embedding = self.local_weather_model.forward_embedding(weather_batch)

            # HFL 嵌入 (凍結)
            with torch.no_grad():
                hfl_embedding = self.hfl_model.forward_embedding(hfl_batch)

            # Fusion 預測
            predictions = self.fusion_model(weather_embedding, hfl_embedding)

            # 計算損失
            loss = self.criterion(predictions, targets)

            # === 反向傳播 ===
            self.fusion_optimizer.zero_grad()
            if train_weather:
                self.weather_optimizer.zero_grad()

            loss.backward()

            # 更新 Fusion Model
            self.fusion_optimizer.step()

            # 更新 Weather Model
            if train_weather:
                self.weather_optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        # 提取 Weather Model 梯度
        if train_weather:
            weather_gradients = [
                param.grad.clone() if param.grad is not None else torch.zeros_like(param)
                for param in self.local_weather_model.parameters()
            ]

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        num_samples = len(train_loader.dataset)

        return avg_loss, weather_gradients, num_samples

    def local_evaluate(self, val_loader: DataLoader) -> float:
        """
        本地驗證

        Args:
            val_loader: 驗證數據加載器

        Returns:
            avg_val_loss: 平均驗證損失
        """
        if self.local_weather_model is None:
            raise RuntimeError("Weather Model 未設置!")

        self.fusion_model.eval()
        self.local_weather_model.eval()

        val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for weather_batch, hfl_batch, targets in val_loader:
                weather_batch = weather_batch.to(self.device)
                hfl_batch = hfl_batch.to(self.device)
                targets = targets.to(self.device)

                # 前向傳播
                weather_embedding = self.local_weather_model.forward_embedding(weather_batch)
                hfl_embedding = self.hfl_model.forward_embedding(hfl_batch)
                predictions = self.fusion_model(weather_embedding, hfl_embedding)

                # 計算損失
                loss = self.criterion(predictions, targets)
                val_loss += loss.item()
                num_batches += 1

        avg_val_loss = val_loss / num_batches if num_batches > 0 else 0.0
        return avg_val_loss

    def local_test(
        self,
        test_loader: DataLoader
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        本地測試

        Args:
            test_loader: 測試數據加載器

        Returns:
            avg_test_loss: 平均測試損失
            all_predictions: 所有預測結果
            all_targets: 所有真實標籤
        """
        if self.local_weather_model is None:
            raise RuntimeError("Weather Model 未設置!")

        self.fusion_model.eval()
        self.local_weather_model.eval()

        test_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for weather_batch, hfl_batch, targets in test_loader:
                weather_batch = weather_batch.to(self.device)
                hfl_batch = hfl_batch.to(self.device)
                targets = targets.to(self.device)

                # 前向傳播
                weather_embedding = self.local_weather_model.forward_embedding(weather_batch)
                hfl_embedding = self.hfl_model.forward_embedding(hfl_batch)
                predictions = self.fusion_model(weather_embedding, hfl_embedding)

                # 計算損失
                loss = self.criterion(predictions, targets)
                test_loss += loss.item()
                num_batches += 1

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

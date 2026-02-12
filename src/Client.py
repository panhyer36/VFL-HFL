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
from src.LoRA import (
    apply_lora_to_transformer, get_lora_state_dict, load_lora_state_dict,
    set_lora_trainable, count_lora_params
)


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
        self.lambda_decorr = getattr(config, 'lambda_decorr', 0.01)

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

        # === LoRA 注入 HFL Transformer ===
        self.lora_rank = getattr(config, 'lora_rank', 0)
        self.lora_alpha = getattr(config, 'lora_alpha', 4.0)
        self.lora_trainable = False  # Phase 0 預設凍結

        if self.lora_rank > 0:
            apply_lora_to_transformer(self.hfl_model, self.lora_rank, self.lora_alpha)
            self.hfl_model.to(device)  # LoRA 新建的參數在 CPU，需移至正確設備
            lora_total, lora_trainable = count_lora_params(self.hfl_model)
            print(f"  V LoRA applied (rank={self.lora_rank}, alpha={self.lora_alpha})")
            print(f"  V LoRA parameters: {lora_total:,} (trainable: {lora_trainable:,})")

        hfl_params = sum(p.numel() for p in self.hfl_model.parameters())
        print(f"\nHFL Model (Local, Frozen + LoRA):")
        print(f"  - Feature dimension: {config.hfl_feature_dim}")
        print(f"  - Parameters: {hfl_params:,}")

        # === 初始化 Fusion Model (本地，可訓練) ===
        self.fusion_model = FusionModel(
            embedding_dim_party_a=config.fusion_embedding_dim_weather,
            embedding_dim_party_b=config.fusion_embedding_dim_hfl,
            hidden_dim=config.fusion_hidden_dim,
            output_dim=config.fusion_output_dim,
            dropout=config.fusion_dropout,
            adapter_bottleneck_dim=getattr(config, 'fusion_adapter_bottleneck_dim', 64),
            use_cross_attention=getattr(config, 'fusion_use_cross_attention', True),
            cross_attention_dim=getattr(config, 'fusion_cross_attention_dim', 64)
        ).to(device)

        fusion_params = sum(p.numel() for p in self.fusion_model.parameters())
        print(f"\nFusion Model (Local, Trainable):")
        print(f"  - Weather embedding dimension: {config.fusion_embedding_dim_weather}")
        print(f"  - HFL embedding dimension: {config.fusion_embedding_dim_hfl}")
        print(f"  - Parameters: {fusion_params:,}")

        # === 優化器 (Fusion Model + LoRA 參數) ===
        optim_params = list(self.fusion_model.parameters())
        if self.lora_rank > 0:
            lora_params = [p for p in self.hfl_model.parameters() if p.requires_grad]
            optim_params += lora_params
        self.fusion_optimizer = torch.optim.Adam(
            optim_params,
            lr=config.beta,
            weight_decay=1e-4
        )

        # === Weather 嵌入快取 (z_old) ===
        self.cached_weather_embeddings = None   # 快取的完整訓練集 Weather 嵌入
        self.weather_cache_version = -1         # 對應的 Weather Model 版本

        # === B-LEC 下載端: Client 側 z_old 管理 ===
        self.z_old_client = None  # 壓縮基準值，與 Server 側 z_old_server 同步

        # === B-LEC 上傳端 Error Feedback 壓縮 ===
        self.error_buffer = None  # 殘差緩衝區，首次使用時初始化，跨輪次持續累積
        self.top_k_ratio = getattr(config, 'top_k_ratio', 0.05)
        self.upload_compression_enabled = getattr(config, 'upload_compression_enabled', False)

        print(f"Device: {device}")
        print(f"{'=' * 70}")

    def _compute_decorr_loss(self, embeddings):
        """計算去相關損失，防止嵌入維度坍縮 (FedDecorr)"""
        batch_size = embeddings.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device)

        centered = embeddings - embeddings.mean(dim=0, keepdim=True)
        cov = (centered.T @ centered) / (batch_size - 1)
        std = cov.diag().sqrt().clamp(min=1e-8)
        corr = cov / (std.unsqueeze(0) * std.unsqueeze(1))

        eye = torch.eye(corr.size(0), device=corr.device)
        return (corr - eye).pow(2).mean()

    def train_batch(
        self,
        weather_embedding: torch.Tensor,
        hfl_batch: torch.Tensor,
        targets: torch.Tensor,
        train_weather: bool = True
    ) -> Tuple[float, torch.Tensor]:
        """
        執行單個 Batch 的訓練步
        
        Args:
            weather_embedding: Weather 嵌入 (batch_size, dim)
            hfl_batch: HFL 數據 (batch_size, seq_len, dim)
            targets: 目標值 (batch_size, 1)
            train_weather: 是否需要計算 weather 梯度
            
        Returns:
            loss: 損失值
            weather_grad: weather_embedding 的梯度 (如果 train_weather=True)
        """
        self.fusion_model.train()
        
        # 確保數據在正確設備
        hfl_batch = hfl_batch.to(self.device)
        targets = targets.to(self.device)
        
        # 如果需要訓練 Weather Model，確保嵌入需要梯度
        # 我們總是 detach 以截斷梯度流，並手動獲取對 embedding 的梯度
        # 這樣可以在 Server 端進行梯度聚合和處理
        if train_weather:
             weather_embedding = weather_embedding.detach().requires_grad_(True)

        # 前向傳播
        # HFL 嵌入: LoRA trainable 時讓梯度流過 LoRA 參數
        if self.lora_rank > 0 and self.lora_trainable:
            hfl_embedding = self.hfl_model.forward_embedding(hfl_batch)
        else:
            with torch.no_grad():
                hfl_embedding = self.hfl_model.forward_embedding(hfl_batch)

        # Fusion 預測 (同時獲取 adapted embeddings 用於 FedDecorr)
        predictions, adapted_w, adapted_h = self.fusion_model(
            weather_embedding, hfl_embedding, return_adapted=True
        )

        # 計算損失 (MSE + FedDecorr 去相關正則化)
        mse_loss = self.criterion(predictions, targets)
        if self.lambda_decorr > 0:
            decorr_loss = (
                self._compute_decorr_loss(adapted_w) +
                self._compute_decorr_loss(adapted_h)
            ) / 2
            loss = mse_loss + self.lambda_decorr * decorr_loss
        else:
            loss = mse_loss

        # 反向傳播
        self.fusion_optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪 (防止梯度爆炸，特別是 Split Learning 跨網路梯度傳遞場景)
        if hasattr(self.config, 'gradient_clip_norm') and self.config.gradient_clip_norm > 0:
            clip_params = list(self.fusion_model.parameters())
            if self.lora_rank > 0 and self.lora_trainable:
                clip_params += [p for p in self.hfl_model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(
                clip_params, self.config.gradient_clip_norm
            )

        # 更新 Fusion Model + LoRA 參數
        self.fusion_optimizer.step()

        # 提取 Weather 嵌入的梯度
        weather_grad = None
        if train_weather and weather_embedding.grad is not None:
            weather_grad = weather_embedding.grad.clone()
            
        return loss.item(), weather_grad

    def evaluate_batch(
        self,
        weather_embedding: torch.Tensor,
        hfl_batch: torch.Tensor,
        targets: torch.Tensor
    ) -> float:
        """執行單個 Batch 的驗證"""
        self.fusion_model.eval()
        hfl_batch = hfl_batch.to(self.device)
        targets = targets.to(self.device)
        
        with torch.no_grad():
            hfl_embedding = self.hfl_model.forward_embedding(hfl_batch)
            predictions = self.fusion_model(weather_embedding, hfl_embedding)
            loss = self.criterion(predictions, targets)
            
        return loss.item()

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

        WARNING: weather_embeddings 必須與 train_loader 的迭代順序對齊。
        若 train_loader 設定 shuffle=True，嵌入將與 HFL/target 錯位！
        主訓練迴圈 (train.py) 使用 batch-wise 的 train_batch() 方法，不受此限制。

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

        ptr = 0
        for _, hfl_batch, targets in train_loader:
            hfl_batch = hfl_batch.to(self.device)
            targets = targets.to(self.device)

            # === 提取當前 batch 的 Weather 嵌入 (ptr-based 索引) ===
            batch_size = hfl_batch.size(0)
            end_idx = ptr + batch_size

            # 從完整的 weather_embeddings 中取出當前 batch
            weather_embedding_batch = weather_embeddings[ptr:end_idx].to(self.device)
            ptr = end_idx

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

            # 梯度裁剪
            if hasattr(self.config, 'gradient_clip_norm') and self.config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.fusion_model.parameters(), self.config.gradient_clip_norm
                )

            # 更新 Fusion Model
            self.fusion_optimizer.step()

            # 提取 Weather 嵌入的梯度 (∂L/∂embedding_weather)
            if train_weather and weather_embedding_batch.grad is not None:
                embedding_gradients.append(weather_embedding_batch.grad.clone())

            epoch_loss += loss.item()
            num_batches += 1

        # 聚合所有 batch 的 embedding 梯度
        if train_weather and embedding_gradients:
            # 將所有 batch 的梯度拼接成一個完整的梯度張量
            aggregated_embedding_grad = torch.cat(embedding_gradients, dim=0)
        else:
            aggregated_embedding_grad = []

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        num_samples = len(train_loader.dataset)

        return avg_loss, aggregated_embedding_grad, num_samples

    def local_evaluate(
        self,
        val_loader: DataLoader,
        weather_embeddings: torch.Tensor,
        return_outputs: bool = False
    ):
        """
        本地驗證

        Args:
            val_loader: 驗證數據加載器
            weather_embeddings: Server 發送的 Weather 嵌入向量
            return_outputs: 是否回傳預測與目標（用於測試階段）

        Returns:
            avg_val_loss: 平均驗證損失
            (predictions, targets): 當 return_outputs=True 時額外回傳
        """
        self.fusion_model.eval()

        val_loss = 0.0
        num_batches = 0
        ptr = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for _, hfl_batch, targets in val_loader:
                hfl_batch = hfl_batch.to(self.device)
                targets = targets.to(self.device)

                # 提取當前 batch 的 Weather 嵌入
                batch_size = hfl_batch.size(0)
                end_idx = ptr + batch_size
                weather_embedding_batch = weather_embeddings[ptr:end_idx].to(self.device)
                ptr = end_idx

                # 前向傳播
                hfl_embedding = self.hfl_model.forward_embedding(hfl_batch)
                predictions = self.fusion_model(weather_embedding_batch, hfl_embedding)

                # 計算損失
                loss = self.criterion(predictions, targets)
                val_loss += loss.item()
                num_batches += 1
                if return_outputs:
                    all_predictions.append(predictions.detach().cpu())
                    all_targets.append(targets.detach().cpu())

        avg_val_loss = val_loss / num_batches if num_batches > 0 else 0.0
        if return_outputs:
            preds = torch.cat(all_predictions, dim=0) if all_predictions else torch.empty(0, 1)
            trgs = torch.cat(all_targets, dim=0) if all_targets else torch.empty(0, 1)
            return avg_val_loss, preds, trgs
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
            ptr = 0
            for _, hfl_batch, targets in test_loader:
                hfl_batch = hfl_batch.to(self.device)
                targets = targets.to(self.device)

                # 提取當前 batch 的 Weather 嵌入 (ptr-based 索引，與 local_evaluate 一致)
                batch_size = hfl_batch.size(0)
                end_idx = ptr + batch_size
                weather_embedding_batch = weather_embeddings[ptr:end_idx].to(self.device)
                ptr = end_idx

                # 前向傳播
                hfl_embedding = self.hfl_model.forward_embedding(hfl_batch)
                predictions = self.fusion_model(weather_embedding_batch, hfl_embedding)

                # 計算損失
                loss = self.criterion(predictions, targets)
                test_loss += loss.item()
                num_batches += 1

                # 收集預測結果
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        avg_test_loss = test_loss / num_batches if num_batches > 0 else 0.0
        return avg_test_loss, np.array(all_predictions), np.array(all_targets)

    def compress_gradient_with_error_feedback(
        self,
        gradient: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Size, dict]:
        """
        Top-k Error Feedback 梯度壓縮 (B-LEC)

        流程:
        1. v = gradient + error_buffer (累積殘差補償)
        2. Top-k 選取絕對值最大的 k 個元素
        3. error_buffer ← v - compressed (未傳送部分存入殘差，下輪補償)

        數學保證: Error Feedback 確保長期梯度更新無偏性 (Unbiasedness)

        Args:
            gradient: 平均嵌入梯度 (N, D)

        Returns:
            compressed_values: Top-k 壓縮值 (k,)
            indices: Top-k 元素索引 (k,)
            shape: 原始梯度形狀 (用於解壓縮)
            metrics: dict 壓縮品質指標
        """
        # 首次使用時初始化殘差緩衝區為零張量
        if self.error_buffer is None:
            self.error_buffer = torch.zeros_like(gradient)

        # Error Feedback 累積: 補償上輪未傳送的殘差
        v = gradient + self.error_buffer

        # Top-k 壓縮: 取絕對值最大的前 k 個元素
        k = max(1, int(v.numel() * self.top_k_ratio))
        _, topk_indices = torch.topk(v.abs().flatten(), k)
        compressed_values = v.flatten()[topk_indices]

        # 更新殘差緩衝區: 未傳送部分存入 "存錢罐"
        self.error_buffer = v.clone()
        self.error_buffer.view(-1)[topk_indices] = 0

        # === 壓縮品質指標 ===
        grad_norm = gradient.norm().item()
        v_norm = v.norm().item()
        error_buf_norm = self.error_buffer.norm().item()
        # 壓縮相對誤差: ||v - decompress(compress(v))|| / ||v||
        approx_error = self.error_buffer.norm().item()  # = ||v - compressed||
        relative_error = approx_error / v_norm if v_norm > 0 else 0.0
        # 通訊量壓縮比
        full_bytes = gradient.nelement() * gradient.element_size()
        comp_bytes = compressed_values.nelement() * compressed_values.element_size() \
                     + topk_indices.nelement() * topk_indices.element_size()

        metrics = {
            'k': k,
            'total_elements': v.numel(),
            'sparsity': k / v.numel(),
            'grad_norm': grad_norm,
            'ef_input_norm': v_norm,
            'error_buf_norm': error_buf_norm,
            'relative_error': relative_error,
            'full_bytes': full_bytes,
            'comp_bytes': comp_bytes,
            'byte_ratio': comp_bytes / full_bytes if full_bytes > 0 else 1.0,
        }

        return compressed_values, topk_indices, v.shape, metrics

    def receive_embeddings(self, payload: dict) -> torch.Tensor:
        """
        Client ← Server: 接收 Weather 嵌入（統一接口）

        根據 payload['type'] 自動處理:
        - 'full': 直接使用完整嵌入，並重置 z_old_client = embeddings
        - 'compressed': z_hat = z_old_client + A @ B，並更新 z_old_client = z_hat

        Args:
            payload: Server 傳來的 dict
                - {'type': 'full', 'embeddings': tensor}
                - {'type': 'compressed', 'A': tensor, 'B': tensor}

        Returns:
            embeddings: 重構後的嵌入 (N, D)
        """
        if payload['type'] == 'full':
            embeddings = payload['embeddings']
            self.z_old_client = embeddings.detach().clone()
            return embeddings
        else:
            A, B = payload['A'], payload['B']
            if self.z_old_client is None:
                raise RuntimeError("Cannot decompress: z_old_client not initialized")
            z_hat = self.z_old_client + A @ B
            self.z_old_client = z_hat.detach().clone()
            return z_hat

    def send_gradients(self, gradient: torch.Tensor) -> dict:
        """
        Client → Server: 發送嵌入梯度（統一接口）

        根據 self.upload_compression_enabled 自動選擇:
        - True: Top-k Error Feedback 壓縮
        - False: 完整梯度

        Args:
            gradient: 平均嵌入梯度 (N, D)

        Returns:
            payload: dict
                - {'type': 'full', 'gradient': tensor}
                - {'type': 'compressed', 'values': tensor, 'indices': tensor,
                   'shape': Size, 'metrics': dict}
        """
        if self.upload_compression_enabled:
            values, indices, shape, metrics = \
                self.compress_gradient_with_error_feedback(gradient)
            return {
                'type': 'compressed',
                'values': values,
                'indices': indices,
                'shape': shape,
                'metrics': metrics,
            }
        else:
            return {
                'type': 'full',
                'gradient': gradient,
            }

    def cache_weather_embeddings(self, embeddings: torch.Tensor, version: int):
        """快取 Weather 嵌入與對應版本號"""
        self.cached_weather_embeddings = embeddings.detach()
        self.weather_cache_version = version

    def get_cached_weather_embeddings(self) -> torch.Tensor:
        """取得快取的 Weather 嵌入"""
        return self.cached_weather_embeddings

    def is_cache_valid(self, current_version: int) -> bool:
        """檢查快取是否與當前 Weather Model 版本匹配"""
        return (self.cached_weather_embeddings is not None
                and self.weather_cache_version == current_version)

    def clear_weather_cache(self):
        """清除快取"""
        self.cached_weather_embeddings = None
        self.weather_cache_version = -1

    def set_lora_training(self, trainable: bool):
        """設定 LoRA 訓練狀態 (Phase 0 凍結 / Phase 1+ 解凍)"""
        self.lora_trainable = trainable
        if self.lora_rank > 0:
            set_lora_trainable(self.hfl_model, trainable)

    def save_lora_model(self, save_path: str):
        """保存 LoRA 權重"""
        if self.lora_rank > 0:
            state_dict = get_lora_state_dict(self.hfl_model)
            torch.save(state_dict, save_path)

    def load_lora_model(self, load_path: str):
        """載入 LoRA 權重"""
        if self.lora_rank > 0:
            state_dict = torch.load(load_path, map_location=self.device)
            load_lora_state_dict(self.hfl_model, state_dict)

    def save_fusion_model(self, save_path: str):
        """保存 Fusion Model"""
        torch.save(self.fusion_model.state_dict(), save_path)

    def load_fusion_model(self, load_path: str):
        """載入 Fusion Model"""
        self.fusion_model.load_state_dict(torch.load(load_path, map_location=self.device))

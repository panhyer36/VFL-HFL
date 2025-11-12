"""
VFL Server 端模組 - 垂直聯邦學習協調器 (Split Learning 架構)

**VFL Server 的職責**:
1. 管理全局 Weather Model (雲端模型，Client 無法訪問)
2. 計算並分發 Weather 嵌入向量給 Clients
3. 接收 Clients 回傳的 ∂L/∂embedding_weather
4. 使用 Chain Rule 更新 Weather Model
5. 分階段訓練策略: 平衡性能與通訊效率

**Split Learning + FedAvg 架構**:
- Weather Model 隔離: 只存在於 Server，Client 無法訪問
- 前向傳播: Server 計算 Weather 嵌入 → 分發給 Clients
- 反向傳播: Clients 計算 ∂L/∂embedding → Server 聚合並更新模型
- FedAvg 聚合: 根據客戶端數據量加權平均 embedding 梯度

**Chain Rule 梯度更新流程**:
1. Server: weather_data → Weather_Model → embeddings → Clients
2. Clients: 本地訓練 → ∂L/∂embedding → Server
3. Server: FedAvg 聚合 → 反向傳播 → 更新 Weather Model
"""

import torch
import random
from typing import List, Dict
from src.Model import TransformerModel


class VFLServer:
    """VFL 聯邦學習服務器 - Weather Model 協調器"""

    def __init__(self, config, device):
        """
        初始化 VFL Server

        Args:
            config: 配置對象 (包含模型架構、訓練參數等)
            device: 計算設備 (cuda/mps/cpu)
        """
        self.config = config
        self.device = device
        self.current_round = 0

        # === 初始化全局 Weather Model (雲端) ===
        print("=" * 70)
        print("VFL Server Initialization")
        print("=" * 70)

        self.global_weather_model = TransformerModel(
            feature_dim=config.weather_feature_dim,
            d_model=config.weather_d_model,
            nhead=config.weather_nhead,
            num_layers=config.weather_num_layers,
            output_dim=config.weather_output_dim,  # None for VFL
            max_seq_length=config.weather_max_seq_length,
            dropout=config.weather_dropout
        ).to(device)

        # === 載入 SSL 預訓練權重 (可選) ===
        if hasattr(config, 'use_ssl_pretrain') and config.use_ssl_pretrain:
            if hasattr(config, 'ssl_pretrain_path') and config.ssl_pretrain_path:
                import os
                ssl_path = config.ssl_pretrain_path
                if os.path.exists(ssl_path):
                    try:
                        print(f"\nLoading SSL pretrained weights:")
                        print(f"  - Path: {ssl_path}")

                        ssl_checkpoint = torch.load(ssl_path, map_location=device)

                        # 處理 checkpoint 格式 (包含 model_state_dict key)
                        if isinstance(ssl_checkpoint, dict) and 'model_state_dict' in ssl_checkpoint:
                            ssl_state_dict = ssl_checkpoint['model_state_dict']
                            print(f"  - Detected checkpoint format, extracting model_state_dict")
                        else:
                            ssl_state_dict = ssl_checkpoint

                        # 載入權重 (允許部分匹配)
                        model_dict = self.global_weather_model.state_dict()
                        pretrained_dict = {k: v for k, v in ssl_state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}

                        if pretrained_dict:
                            model_dict.update(pretrained_dict)
                            self.global_weather_model.load_state_dict(model_dict)
                            print(f"  V Successfully loaded {len(pretrained_dict)}/{len(ssl_state_dict)} weight layers")
                        else:
                            print(f"  ! No matching weight layers, using random initialization")
                    except Exception as e:
                        print(f"  ! Failed to load SSL weights: {e}")
                        print(f"  -> Using random initialization")
                else:
                    print(f"\n  ! SSL pretrained weights not found: {ssl_path}")
                    print(f"  -> Using random initialization")

        # 統計模型參數
        total_params = sum(p.numel() for p in self.global_weather_model.parameters())
        trainable_params = sum(p.numel() for p in self.global_weather_model.parameters() if p.requires_grad)

        print(f"\nGlobal Weather Model (Cloud):")
        print(f"  - Feature dimension: {config.weather_feature_dim}")
        print(f"  - Model dimension: {config.weather_d_model}")
        print(f"  - Number of attention heads: {config.weather_nhead}")
        print(f"  - Number of Transformer layers: {config.weather_num_layers}")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        print(f"  - Device: {device}")

        # === 全局優化器 ===
        self.global_optimizer = torch.optim.Adam(
            self.global_weather_model.parameters(),
            lr=config.beta,
            weight_decay=1e-4
        )

        # === 訓練歷史記錄 ===
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'selected_clients': [],
            'weather_update_rounds': []  # 記錄哪些輪次更新了 Weather Model
        }

        # === 早停機制 ===
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        print(f"\nTraining Strategy:")
        print(f"  - Total rounds: {config.K}")
        print(f"  - Phase 1 ({config.phase1_rounds} rounds): Train Fusion + Weather every round")
        print(f"  - Phase 2 ({config.phase2_rounds} rounds): {config.phase2_fusion_freq} rounds Fusion, 1 round Weather")
        print(f"  - Communication saving: {self._estimate_comm_saving():.1f}%")
        print("=" * 70)

    def _estimate_comm_saving(self):
        """估計通訊節省比例"""
        phase1_updates = self.config.phase1_rounds
        phase2_updates = self.config.phase2_rounds // (self.config.phase2_fusion_freq + 1)
        total_updates = phase1_updates + phase2_updates
        total_rounds = self.config.K
        saving = (1 - total_updates / total_rounds) * 100
        return saving

    def select_clients(self, client_names: List[str]) -> List[str]:
        """
        選擇參與本輪訓練的客戶端 (FedAvg 策略)

        Args:
            client_names: 所有客戶端名稱列表

        Returns:
            selected_clients: 被選中的客戶端名稱列表

        Note:
            - VFL 場景通常需要所有客戶端參與 (client_fraction=1.0)
            - 如果部分參與，使用隨機選擇
        """
        num_selected = max(1, int(len(client_names) * self.config.r))
        selected = random.sample(client_names, num_selected)
        return selected

    def should_update_weather(self) -> bool:
        """
        判斷當前輪次是否需要更新 Weather Model

        分階段訓練策略:
        - 階段1 (前 phase1_rounds 輪): 每輪都更新
        - 階段2 (後續輪次): 每隔 phase2_fusion_freq 輪更新一次

        Returns:
            bool: True 表示需要更新 Weather Model
        """
        if self.current_round < self.config.phase1_rounds:
            # 階段1: 每輪都更新
            return True
        else:
            # 階段2: 週期性更新
            rounds_in_phase2 = self.current_round - self.config.phase1_rounds
            return (rounds_in_phase2 % (self.config.phase2_fusion_freq + 1)) == self.config.phase2_fusion_freq

    def aggregate_weather_gradients(
        self,
        client_gradients: List[List[torch.Tensor]],
        client_weights: List[int]
    ) -> List[torch.Tensor]:
        """
        FedAvg 梯度聚合 - Weather Model

        聚合策略:
        1. 加權平均: 根據客戶端數據量加權
        2. 正規化: 確保權重總和為 1
        3. 參數對應: 逐參數進行加權平均

        Args:
            client_gradients: 每個客戶端的梯度列表
                格式: [[param1_grad, param2_grad, ...], ...]
            client_weights: 每個客戶端的數據量

        Returns:
            aggregated_grads: 聚合後的梯度列表
        """
        if not client_gradients:
            return []

        # 正規化權重
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]

        # 逐參數聚合
        aggregated_grads = []
        num_params = len(client_gradients[0])

        for param_idx in range(num_params):
            weighted_grad = None
            for client_idx, grads in enumerate(client_gradients):
                if weighted_grad is None:
                    weighted_grad = grads[param_idx] * normalized_weights[client_idx]
                else:
                    weighted_grad += grads[param_idx] * normalized_weights[client_idx]
            aggregated_grads.append(weighted_grad)

        return aggregated_grads

    def apply_aggregated_gradients(self, aggregated_grads: List[torch.Tensor]):
        """
        將聚合後的梯度應用到全局 Weather Model

        Args:
            aggregated_grads: 聚合後的梯度列表
        """
        for param, grad in zip(self.global_weather_model.parameters(), aggregated_grads):
            if param.grad is None:
                param.grad = grad.clone()
            else:
                param.grad.copy_(grad)

        # 執行優化步驟
        self.global_optimizer.step()
        self.global_optimizer.zero_grad()

    def update_weather_model_from_embeddings(
        self,
        weather_data: torch.Tensor,
        client_embedding_gradients: List[torch.Tensor],
        client_sample_counts: List[int]
    ):
        """
        使用 Chain Rule 更新全局 Weather Model (Split Learning)

        流程:
        1. 收集所有客戶端的 ∂L/∂embedding_weather
        2. 根據數據量進行加權平均 (FedAvg)
        3. Server 重新前向傳播計算 weather_embeddings
        4. 使用聚合的 embedding 梯度進行反向傳播
        5. 提取並應用 Weather Model 的參數梯度

        Args:
            weather_data: Weather 輸入數據 (用於重新計算嵌入)
            client_embedding_gradients: 客戶端的 ∂L/∂embedding 列表
            client_sample_counts: 客戶端數據量列表

        Note:
            plit Learning 梯度傳遞機制:
            - Client: loss → ∂L/∂embedding (chain rule 第一步)
            - Server: ∂L/∂embedding → ∂L/∂weather_params (chain rule 第二步)
        """
        # === 步驟 1: FedAvg 聚合 embedding 梯度 ===
        total_weight = sum(client_sample_counts)
        normalized_weights = [w / total_weight for w in client_sample_counts]

        # 聚合所有客戶端的 embedding 梯度
        aggregated_embedding_grad = None
        for client_idx, embedding_grad in enumerate(client_embedding_gradients):
            weighted_grad = embedding_grad * normalized_weights[client_idx]
            if aggregated_embedding_grad is None:
                aggregated_embedding_grad = weighted_grad
            else:
                aggregated_embedding_grad += weighted_grad

        # === 步驟 2: Server 重新前向傳播 ===
        self.global_weather_model.train()
        self.global_optimizer.zero_grad()

        # 計算 weather embeddings (保留計算圖)
        weather_embeddings = self.global_weather_model.forward_embedding(weather_data)

        # === 步驟 3: Chain Rule 反向傳播 ===
        # 使用聚合的 embedding 梯度作為 backward 的 gradient 參數
        weather_embeddings.backward(gradient=aggregated_embedding_grad)

        # === 步驟 4: 更新 Weather Model ===
        self.global_optimizer.step()

        # 記錄更新
        self.history['weather_update_rounds'].append(self.current_round)

    def compute_weather_embeddings(
        self,
        weather_data: torch.Tensor,
        requires_grad: bool = False
    ) -> torch.Tensor:
        """
        計算 Weather 嵌入向量 (前向傳播)

        Args:
            weather_data: Weather 輸入數據 (num_samples, seq_len, feature_dim)
            requires_grad: 是否需要梯度 (訓練時為 True)

        Returns:
            weather_embeddings: Weather 嵌入向量 (num_samples, d_model)

        Note:
            - Weather Model 只存在於 Server 端
            - Client 只接收嵌入向量，不接觸原始 Weather Model
        """
        if requires_grad:
            self.global_weather_model.train()
            embeddings = self.global_weather_model.forward_embedding(weather_data)
        else:
            self.global_weather_model.eval()
            with torch.no_grad():
                embeddings = self.global_weather_model.forward_embedding(weather_data)

        return embeddings

    def evaluate_global(
        self,
        avg_train_loss: float,
        avg_val_loss: float,
        selected_clients: List[str]
    ) -> bool:
        """
        全局評估與早停檢查

        Args:
            avg_train_loss: 平均訓練損失
            avg_val_loss: 平均驗證損失
            selected_clients: 本輪參與的客戶端

        Returns:
            should_stop: 是否應該早停
        """
        # 記錄歷史
        self.history['train_loss'].append(avg_train_loss)
        self.history['val_loss'].append(avg_val_loss)
        self.history['selected_clients'].append(selected_clients)

        # 早停檢查
        should_stop = False
        if avg_val_loss < self.best_val_loss - self.config.early_stopping_min_delta:
            self.best_val_loss = avg_val_loss
            self.patience_counter = 0
            # 保存最佳模型
            self.save_best_model()
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"\nEarly stopping triggered (patience={self.config.early_stopping_patience})")
                should_stop = True

        return should_stop

    def save_best_model(self):
        """保存最佳全局 Weather Model"""
        import os
        save_path = os.path.join(
            self.config.model_save_path,
            "best_weather_model.pth"
        )
        torch.save(self.global_weather_model.state_dict(), save_path)

    def save_final_model(self):
        """保存最終全局 Weather Model"""
        import os
        save_path = os.path.join(
            self.config.model_save_path,
            "final_weather_model.pth"
        )
        torch.save(self.global_weather_model.state_dict(), save_path)
        print(f"\nFinal model saved: {save_path}")

    def get_training_summary(self) -> Dict:
        """
        獲取訓練摘要統計

        Returns:
            summary: 訓練摘要字典
        """
        weather_updates = len(self.history['weather_update_rounds'])
        total_rounds = self.current_round + 1

        summary = {
            'total_rounds': total_rounds,
            'weather_updates': weather_updates,
            'comm_saving_actual': (1 - weather_updates / total_rounds) * 100,
            'best_val_loss': self.best_val_loss,
            'final_train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else 0,
            'final_val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else 0
        }

        return summary

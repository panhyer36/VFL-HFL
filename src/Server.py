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

        print(f"\nThree-Phase Training Strategy:")
        print(f"  - Total rounds: {config.K}")
        print(f"  - Phase 0 ({config.phase0_rounds} rounds): Fusion warmup - Weather frozen")
        print(f"  - Phase 1 ({config.phase1_rounds} rounds): Joint training - Fusion + Weather every round")
        print(f"  - Phase 2 ({config.phase2_rounds} rounds): Comm optimized - {config.phase2_fusion_freq} rounds Fusion, 1 round Weather")
        print(f"  - Estimated communication saving: {self._estimate_comm_saving():.1f}%")
        print("=" * 70)

    def _estimate_comm_saving(self):
        """
        估計通訊節省比例

        三階段策略通訊分析:
        - Phase 0: 無 Weather Model 更新 (0 次通訊)
        - Phase 1: 每輪更新 Weather Model (phase1_rounds 次通訊)
        - Phase 2: 週期性更新 (actual_phase2 / (phase2_fusion_freq + 1) 次通訊)
        """
        phase0_updates = 0  # Phase 0 不更新 Weather Model
        phase1_updates = self.config.phase1_rounds
        # 使用實際 Phase 2 輪數 (已由 config.py 自動計算為 K - phase0 - phase1)
        actual_phase2 = self.config.phase2_rounds
        phase2_updates = actual_phase2 // (self.config.phase2_fusion_freq + 1)
        total_updates = phase0_updates + phase1_updates + phase2_updates
        total_rounds = self.config.K
        saving = (1 - total_updates / total_rounds) * 100
        return saving

    def select_clients(self, client_names: List[str]) -> List[str]:
        """
        選擇參與本輪訓練的客戶端

        選擇策略 (根據訓練階段):
        - Phase 0 (Fusion 預熱期): 選擇所有客戶端 (client_fraction = 1.0)
          → 確保每個 Fusion Model 都能預熱，且此階段無通訊開銷
        - Phase 1 & 2: 按比例隨機選擇 (client_fraction = config.r)
          → 模擬聯邦學習場景，部分客戶端參與

        Args:
            client_names: 所有客戶端名稱列表

        Returns:
            selected_clients: 被選中的客戶端名稱列表
        """
        phase0_rounds = self.config.phase0_rounds

        # Phase 0: 所有客戶端參與 Fusion 預熱
        if self.current_round < phase0_rounds:
            return client_names.copy()

        # Phase 1 & 2: 按比例隨機選擇
        num_selected = max(1, int(len(client_names) * self.config.r))
        selected = random.sample(client_names, num_selected)
        return selected

    def should_update_weather(self) -> bool:
        """
        判斷當前輪次是否需要更新 Weather Model

        三階段訓練策略:
        - Phase 0 (前 phase0_rounds 輪): Fusion 預熱期，Weather Model 凍結 → return False
        - Phase 1 (接下來 phase1_rounds 輪): 聯合訓練期，每輪都更新 → return True
        - Phase 2 (剩餘輪次): 通訊優化期，週期性更新 → 條件判斷

        Returns:
            bool: True 表示需要更新 Weather Model
        """
        phase0_rounds = self.config.phase0_rounds  # 預設 5
        phase1_rounds = self.config.phase1_rounds  # 10

        # Phase 0: Fusion Model 預熱期 (Weather 凍結)
        if self.current_round < phase0_rounds:
            return False

        # Phase 1: 聯合訓練期 (每輪都更新)
        adjusted_round = self.current_round - phase0_rounds
        if adjusted_round < phase1_rounds:
            return True

        # Phase 2: 通訊優化期 (週期性更新)
        phase2_round = adjusted_round - phase1_rounds
        return (phase2_round + 1) % (self.config.phase2_fusion_freq + 1) == 0

    def get_current_phase_info(self) -> Dict:
        """
        獲取當前訓練階段的詳細資訊

        Returns:
            dict: 包含階段名稱、階段內輪次、總輪次等資訊
        """
        phase0_rounds = self.config.phase0_rounds
        phase1_rounds = self.config.phase1_rounds
        phase2_rounds = self.config.phase2_rounds

        if self.current_round < phase0_rounds:
            # Phase 0: Fusion 預熱期
            return {
                'phase': 0,
                'phase_name': 'Fusion Warmup',
                'phase_round': self.current_round + 1,
                'phase_total': phase0_rounds,
                'train_weather': False
            }
        elif self.current_round < phase0_rounds + phase1_rounds:
            # Phase 1: 聯合訓練期
            return {
                'phase': 1,
                'phase_name': 'Joint Training',
                'phase_round': self.current_round - phase0_rounds + 1,
                'phase_total': phase1_rounds,
                'train_weather': True
            }
        else:
            # Phase 2: 通訊優化期
            phase2_round = self.current_round - phase0_rounds - phase1_rounds
            return {
                'phase': 2,
                'phase_name': 'Communication Optimized',
                'phase_round': phase2_round + 1,
                'phase_total': phase2_rounds,
                'train_weather': self.should_update_weather()
            }

    def aggregate_weather_gradients(
        self,
        client_gradients: List[List[torch.Tensor]],
        client_weights: List[int]
    ) -> List[torch.Tensor]:
        """
        FedAvg 梯度聚合 - Weather Model

        聚合策略:
        1. NaN/Inf 過濾: 跳過包含異常梯度的客戶端
        2. 加權平均: 根據客戶端數據量加權
        3. 正規化: 確保權重總和為 1
        4. 參數對應: 逐參數進行加權平均

        Args:
            client_gradients: 每個客戶端的梯度列表
                格式: [[param1_grad, param2_grad, ...], ...]
            client_weights: 每個客戶端的數據量

        Returns:
            aggregated_grads: 聚合後的梯度列表 (若全部異常則返回空列表)
        """
        if not client_gradients:
            return []

        # === NaN/Inf 過濾: 跳過包含異常梯度的客戶端 ===
        valid_indices = []
        for i, grads in enumerate(client_gradients):
            if all(torch.isfinite(g).all() for g in grads):
                valid_indices.append(i)
            else:
                print(f"  ! Warning: Client {i} has NaN/Inf gradients, skipping in aggregation")

        if not valid_indices:
            print("  ! Warning: All client gradients contain NaN/Inf, skipping Weather Model update")
            return []

        filtered_gradients = [client_gradients[i] for i in valid_indices]
        filtered_weights = [client_weights[i] for i in valid_indices]

        # 正規化權重
        total_weight = sum(filtered_weights)
        normalized_weights = [w / total_weight for w in filtered_weights]

        # 逐參數聚合
        aggregated_grads = []
        num_params = len(filtered_gradients[0])

        for param_idx in range(num_params):
            weighted_grad = None
            for client_idx, grads in enumerate(filtered_gradients):
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
        self.history['weather_update_rounds'].append(self.current_round)

    def zero_weather_model_grad(self):
        """清空全局 Weather Model 的梯度"""
        self.global_optimizer.zero_grad()

    def capture_weather_model_grads(self) -> List[torch.Tensor]:
        """
        擷取目前 Weather Model 參數的梯度

        Returns:
            grads: list of gradients，對應每個參數
        """
        grads = []
        for param in self.global_weather_model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param))
            else:
                grads.append(param.grad.detach().clone())
        return grads

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
    ) -> tuple:
        """
        全局評估與早停檢查

        Args:
            avg_train_loss: 平均訓練損失
            avg_val_loss: 平均驗證損失
            selected_clients: 本輪參與的客戶端

        Returns:
            (should_stop, is_best): 是否應該早停, 是否為新最佳模型
        """
        # 記錄歷史
        self.history['train_loss'].append(avg_train_loss)
        self.history['val_loss'].append(avg_val_loss)
        self.history['selected_clients'].append(selected_clients)

        # 早停檢查
        should_stop = False
        is_best = False
        if avg_val_loss < self.best_val_loss - self.config.early_stopping_min_delta:
            self.best_val_loss = avg_val_loss
            self.patience_counter = 0
            is_best = True
            # 保存最佳 Weather Model
            self.save_best_model()
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"\nEarly stopping triggered (patience={self.config.early_stopping_patience})")
                should_stop = True

        return should_stop, is_best

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
        獲取訓練摘要統計 (含三階段策略分析)

        Returns:
            summary: 訓練摘要字典
        """
        weather_updates = len(self.history['weather_update_rounds'])
        total_rounds = self.current_round + 1

        # 三階段配置
        phase0_rounds = self.config.phase0_rounds
        phase1_rounds = self.config.phase1_rounds
        phase2_rounds = self.config.phase2_rounds

        summary = {
            'total_rounds': total_rounds,
            'weather_updates': weather_updates,
            'comm_saving_actual': (1 - weather_updates / total_rounds) * 100 if total_rounds > 0 else 0,
            'best_val_loss': self.best_val_loss,
            'final_train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else 0,
            'final_val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else 0,
            # 三階段配置資訊
            'phase0_rounds': phase0_rounds,
            'phase1_rounds': phase1_rounds,
            'phase2_rounds': phase2_rounds,
            'training_strategy': f"Phase0({phase0_rounds}) + Phase1({phase1_rounds}) + Phase2({phase2_rounds})"
        }

        return summary

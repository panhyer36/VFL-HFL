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
        print("=" * 70)

        # === Weather Model 版本號 (每次成功更新 +1，用於快取判斷) ===
        self.weather_model_version = 0

        # === 通訊統計 ===
        self.comm_stats = {}  # {client_name: {'bytes_downloaded': 0, 'bytes_uploaded': 0}}

        # === B-LEC 下載端: Server 側 z_old 管理 ===
        self.z_old_server = {}  # {client_name: z_old_tensor}，每個客戶端獨立維護

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

    def upload_and_aggregate_gradients(
        self,
        client_grad_lists: List[List[torch.Tensor]],
        client_sample_counts: List[int]
    ) -> bool:
        """
        Client → Server: 接收上傳的累積梯度, FedAvg 聚合後更新 Weather Model

        Args:
            client_grad_lists: 每個客戶端的平均梯度列表
            client_sample_counts: 每個客戶端的數據量

        Returns:
            bool: 是否成功更新 Weather Model
        """
        aggregated = self._aggregate_weather_gradients(client_grad_lists, client_sample_counts)
        if aggregated:
            self._apply_aggregated_gradients(aggregated)
            self.weather_model_version += 1
            return True
        return False

    def _aggregate_weather_gradients(
        self,
        client_gradients: List[List[torch.Tensor]],
        client_weights: List[int]
    ) -> List[torch.Tensor]:
        """
        FedAvg 梯度聚合 - Weather Model (內部方法)

        Args:
            client_gradients: 每個客戶端的梯度列表
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

    def _apply_aggregated_gradients(self, aggregated_grads: List[torch.Tensor]):
        """
        將聚合後的梯度應用到全局 Weather Model (內部方法)

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

    def download_weather_embeddings(
        self,
        weather_data: torch.Tensor,
        requires_grad: bool = False
    ) -> torch.Tensor:
        """
        Server 計算 Weather 嵌入後下發給 Client

        Args:
            weather_data: Weather 輸入數據 (num_samples, seq_len, feature_dim)
            requires_grad: 是否需要梯度 (訓練時為 True)

        Returns:
            weather_embeddings: Weather 嵌入向量 (num_samples, d_model)
        """
        if requires_grad:
            self.global_weather_model.train()
            embeddings = self.global_weather_model.forward_embedding(weather_data)
        else:
            self.global_weather_model.eval()
            with torch.no_grad():
                embeddings = self.global_weather_model.forward_embedding(weather_data)

        return embeddings

    def send_embeddings(
        self,
        weather_data: torch.Tensor,
        client_name: str,
        compress: bool = False,
        svd_rank: int = 16,
        variance_threshold: float = 0.85
    ) -> tuple:
        """
        Server → Client: 發送 Weather 嵌入（統一接口）

        內部流程:
        1. 計算 z_new
        2. 若 compress=True 且非首輪:
           a. delta_z = z_new - z_old_server[client_name]
           b. SVD 截斷 → 檢查 explained_variance
           c. 若 variance >= threshold: 傳送 {A, B}
           d. 若 variance < threshold: 回退完整傳送 z_new (同時重置 z_old)
        3. 更新 z_old_server[client_name]
        4. 內部調用 record_download() 記錄帶寬

        z_old 更新策略:
        - 壓縮成功: z_old_server = z_hat = z_old + A@B (存重構值，避免誤差累積)
        - 完整傳送:  z_old_server = z_new (精確重置，清除所有累積誤差)
        - 首輪:      z_old_server = z_new (初始化)

        Args:
            weather_data: Weather 輸入數據
            client_name: 客戶端名稱 (用於查找對應的 z_old)
            compress: 是否啟用 SVD 壓縮
            svd_rank: 截斷 SVD rank
            variance_threshold: explained variance ratio 閾值

        Returns:
            payload: dict {'type': 'full'/'compressed', ...}
            z_new: 完整嵌入 (Server 內部使用，用於 backward)
            metrics: dict 壓縮品質指標
        """
        # 計算完整嵌入
        self.global_weather_model.eval()
        with torch.no_grad():
            z_new = self.global_weather_model.forward_embedding(weather_data)

        # 初始化該客戶端的 z_old (首次)
        if client_name not in self.z_old_server:
            self.z_old_server[client_name] = torch.zeros_like(z_new)

        z_old = self.z_old_server[client_name]

        # 不壓縮 或 首輪 (z_old 全零) → 完整傳送
        if not compress or torch.all(z_old == 0):
            self.z_old_server[client_name] = z_new.detach().clone()
            emb = z_new.detach()
            self.record_download(client_name, emb)
            if not compress:
                reason = 'disabled'
            else:
                reason = 'first_round'
            metrics = {
                'compressed': False,
                'reason': reason,
                'bytes_sent': emb.nelement() * emb.element_size(),
            }
            payload = {'type': 'full', 'embeddings': emb}
            return payload, z_new, metrics

        # SVD 壓縮
        delta_z = z_new - z_old

        try:
            U, S, Vh = torch.linalg.svd(delta_z.float(), full_matrices=False)
        except Exception:
            delta_cpu = delta_z.float().cpu()
            U, S, Vh = torch.linalg.svd(delta_cpu, full_matrices=False)
            U, S, Vh = U.to(delta_z.device), S.to(delta_z.device), Vh.to(delta_z.device)

        r = min(svd_rank, len(S))
        S_sq = S.float() ** 2
        total_var = S_sq.sum().item()
        explained_var = S_sq[:r].sum().item() / total_var if total_var > 0 else 1.0

        # 自適應品質控制: variance 低於閾值 → 回退完整傳送 (重置 z_old)
        if explained_var < variance_threshold:
            self.z_old_server[client_name] = z_new.detach().clone()
            emb = z_new.detach()
            self.record_download(client_name, emb)
            metrics = {
                'compressed': False,
                'reason': 'variance_fallback',
                'explained_var': explained_var,
                'threshold': variance_threshold,
                'bytes_sent': emb.nelement() * emb.element_size(),
            }
            payload = {'type': 'full', 'embeddings': emb}
            return payload, z_new, metrics

        # 壓縮成功
        A = U[:, :r] @ torch.diag(S[:r])  # (N, r)
        B = Vh[:r, :]                       # (r, D)

        # z_old 更新為 z_hat (重構值)，不用 z_new
        z_hat = z_old + A @ B
        self.z_old_server[client_name] = z_hat.detach().clone()

        A_out = A.detach()
        B_out = B.detach()

        # 帶寬記錄
        self.record_download(client_name, A_out)
        self.record_download(client_name, B_out)
        bytes_sent = A_out.nelement() * A_out.element_size() + B_out.nelement() * B_out.element_size()
        full_bytes = delta_z.nelement() * delta_z.element_size()

        recon = A @ B
        recon_error_norm = (delta_z.float() - recon).norm().item()
        delta_norm = delta_z.float().norm().item()

        metrics = {
            'compressed': True,
            'svd_rank': r,
            'explained_var': explained_var,
            'relative_error': recon_error_norm / delta_norm if delta_norm > 0 else 0.0,
            'delta_norm': delta_norm,
            'full_bytes': full_bytes,
            'bytes_sent': bytes_sent,
            'byte_ratio': bytes_sent / full_bytes if full_bytes > 0 else 1.0,
        }
        payload = {'type': 'compressed', 'A': A_out, 'B': B_out}
        return payload, z_new, metrics

    def receive_gradients(
        self,
        client_name: str,
        compressed_values: torch.Tensor = None,
        indices: torch.Tensor = None,
        grad_shape: torch.Size = None,
        full_gradient: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Server ← Client: 接收上傳梯度（統一接口）

        內部自動調用 record_upload() 記錄帶寬。
        根據傳入參數自動判斷壓縮/未壓縮:
        - 若 compressed_values 不為 None → Top-k 解壓縮
        - 若 full_gradient 不為 None → 直接使用

        Args:
            client_name: 客戶端名稱 (用於通訊統計)
            compressed_values: Top-k 壓縮值 (壓縮模式)
            indices: Top-k 索引 (壓縮模式)
            grad_shape: 原始梯度形狀 (壓縮模式)
            full_gradient: 完整梯度 (非壓縮模式)

        Returns:
            gradient: 解壓縮後 (或原始) 的密集梯度
        """
        if compressed_values is not None:
            self.record_upload(client_name, [compressed_values, indices])
            gradient = self.decompress_gradient(
                compressed_values, indices, grad_shape, compressed_values.device
            )
        else:
            self.record_upload(client_name, [full_gradient])
            gradient = full_gradient
        return gradient

    @staticmethod
    def decompress_gradient(
        compressed_values: torch.Tensor,
        indices: torch.Tensor,
        shape: torch.Size,
        device: torch.device
    ) -> torch.Tensor:
        """
        Server 端解壓縮 Top-k 稀疏梯度為密集張量

        在 VFL 架構中，Client 上傳壓縮梯度後，Server 負責解壓縮再進行
        FedAvg 聚合與反向傳播，確保解壓縮邏輯完全在 Server 端完成。

        Args:
            compressed_values: Top-k 壓縮值 (k,)
            indices: Top-k 元素索引 (k,)
            shape: 原始梯度形狀
            device: 計算設備

        Returns:
            full_grad: 解壓縮後的密集梯度張量 (可直接進入 FedAvg 流程)
        """
        full_grad = torch.zeros(shape, device=device).flatten()
        full_grad[indices] = compressed_values.to(device)
        return full_grad.reshape(shape)

    def backward_weather_embeddings_chunked(
        self,
        weather_data: torch.Tensor,
        embedding_grad: torch.Tensor,
        chunk_size: int = 256
    ):
        """
        分塊反向傳播 Weather Model，避免一次性存儲所有中間激活值導致 OOM

        數學等價: Σ_chunks backward(chunk_grad) ≡ backward(full_grad)
        因為每個樣本在 Weather Model 中是獨立計算的 (無跨樣本狀態)

        Args:
            weather_data: Weather 輸入數據 (N, seq_len, feature_dim)
            embedding_grad: 嵌入梯度 (N, d_model)
            chunk_size: 每次反向傳播的樣本數 (越小越省記憶體，越大越快)
        """
        self.global_weather_model.eval()  # eval mode: 與 no-grad forward 一致 (無 dropout)
        self.global_optimizer.zero_grad()
        N = weather_data.size(0)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk_emb = self.global_weather_model.forward_embedding(
                weather_data[start:end]
            )
            chunk_emb.backward(embedding_grad[start:end])
            del chunk_emb

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

    def init_comm_stats(self, client_names: List[str]):
        """初始化通訊統計"""
        self.comm_stats = {name: {'bytes_downloaded': 0, 'bytes_uploaded': 0} for name in client_names}

    def record_download(self, client_name: str, tensor: torch.Tensor):
        """記錄 Server → Client 下載通訊量"""
        if client_name in self.comm_stats:
            self.comm_stats[client_name]['bytes_downloaded'] += tensor.nelement() * tensor.element_size()

    def record_upload(self, client_name: str, tensors: List[torch.Tensor]):
        """記錄 Client → Server 上傳通訊量"""
        if client_name in self.comm_stats:
            for t in tensors:
                self.comm_stats[client_name]['bytes_uploaded'] += t.nelement() * t.element_size()

    def get_comm_stats(self) -> Dict:
        """獲取通訊統計"""
        return self.comm_stats

    def get_training_summary(self) -> Dict:
        """
        獲取訓練摘要統計 (含三階段策略分析)

        Returns:
            summary: 訓練摘要字典
        """
        total_rounds = self.current_round + 1

        # 三階段配置
        phase0_rounds = self.config.phase0_rounds
        phase1_rounds = self.config.phase1_rounds
        phase2_rounds = self.config.phase2_rounds

        summary = {
            'total_rounds': total_rounds,
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

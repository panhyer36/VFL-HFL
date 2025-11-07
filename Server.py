"""
VFL Server 端模組 - 垂直聯邦學習協調器

**VFL Server 的職責**:
1. 管理全局 Weather Model (雲端模型)
2. 協調多個客戶端的訓練過程
3. 聚合來自客戶端的梯度 (FedAvg)
4. 分階段訓練策略: 平衡性能與通訊效率

**與 HFL Server 的區別**:
- HFL: 聚合多個客戶端的完整模型參數
- VFL: 只訓練雲端的 Weather Model，客戶端保留 HFL + Fusion 模型
- 通訊: VFL 只傳輸嵌入向量和梯度，不傳輸完整模型

**FedAvg 聚合策略**:
- 加權平均: 根據客戶端數據量加權
- 只聚合 Weather Model 的梯度
- 支持部分客戶端參與 (client_fraction)
"""

import torch
import torch.nn as nn
import numpy as np
import random
from copy import deepcopy
from typing import List, Dict, Tuple
from Model import TransformerModel


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
        print("VFL Server 初始化")
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

        # 統計模型參數
        total_params = sum(p.numel() for p in self.global_weather_model.parameters())
        trainable_params = sum(p.numel() for p in self.global_weather_model.parameters() if p.requires_grad)

        print(f"\n全局 Weather Model (雲端):")
        print(f"  - 特徵維度: {config.weather_feature_dim}")
        print(f"  - 模型維度: {config.weather_d_model}")
        print(f"  - 注意力頭數: {config.weather_nhead}")
        print(f"  - Transformer層數: {config.weather_num_layers}")
        print(f"  - 總參數量: {total_params:,}")
        print(f"  - 可訓練參數: {trainable_params:,}")
        print(f"  - 設備: {device}")

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

        print(f"\n訓練策略:")
        print(f"  - 總輪數: {config.K}")
        print(f"  - 階段1 ({config.phase1_rounds} 輪): 每輪都訓練 Fusion + Weather")
        print(f"  - 階段2 ({config.phase2_rounds} 輪): {config.phase2_fusion_freq} 輪 Fusion，1 輪 Weather")
        print(f"  - 通訊節省: {self._estimate_comm_saving():.1f}%")
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

    def update_weather_model(
        self,
        client_gradients: List[List[torch.Tensor]],
        client_sample_counts: List[int]
    ):
        """
        更新全局 Weather Model (Server 端聚合)

        流程:
        1. 收集所有客戶端的 Weather Model 梯度
        2. 根據數據量進行加權平均 (FedAvg)
        3. 應用聚合梯度到全局模型
        4. 記錄更新歷史

        Args:
            client_gradients: 客戶端梯度列表
            client_sample_counts: 客戶端數據量列表
        """
        # FedAvg 聚合
        aggregated_grads = self.aggregate_weather_gradients(
            client_gradients,
            client_sample_counts
        )

        # 應用聚合梯度
        self.apply_aggregated_gradients(aggregated_grads)

        # 記錄更新
        self.history['weather_update_rounds'].append(self.current_round)

    def get_global_weather_model(self) -> TransformerModel:
        """
        獲取當前全局 Weather Model (分發給客戶端)

        Returns:
            global_weather_model: 全局 Weather Model 的深拷貝

        Note:
            - 返回深拷貝避免客戶端直接修改全局模型
            - 客戶端使用此模型進行本地訓練
        """
        return deepcopy(self.global_weather_model)

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
                print(f"\n早停觸發 (patience={self.config.early_stopping_patience})")
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
        print(f"\n最終模型已保存: {save_path}")

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

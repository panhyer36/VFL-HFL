"""
VFL 垂直聯邦學習核心模組

這個套件包含 VFL (垂直聯邦學習) 的核心實現:
- Server: VFL 協調器，管理全局 Weather Model
- Client: VFL 客戶端，管理本地 HFL Model 和 Fusion Model
- Model: Transformer 模型和 Fusion 模型架構
- DataLoader: 時序數據加載器
- Trainer: 聯邦學習訓練輔助工具
- Personalizer: Per-FedAvg 個性化模型初始化器
"""

__version__ = "1.0.0"
__author__ = "VFL Research Team"

# 導出主要類別
from .Server import VFLServer
from .Client import VFLClient
from .Model import TransformerModel, FusionModel
from .DataLoader import SequenceCSVDataset
from .Trainer import FederatedTrainer
from .Personalizer import initialize_personalized_models, personalize_model_for_client

__all__ = [
    'VFLServer',
    'VFLClient',
    'TransformerModel',
    'FusionModel',
    'SequenceCSVDataset',
    'FederatedTrainer',
    'initialize_personalized_models',
    'personalize_model_for_client',
]

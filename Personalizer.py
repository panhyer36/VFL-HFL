"""Per-FedAvg 個性化模型初始化器

這個模組專門用於 Per-FedAvg 演算法的個性化適應：
1. 載入訓練好的全局模型
2. 為每個客戶端使用其 validation set 進行個性化適應
3. 返回所有客戶端的個性化模型狀態字典

主要功能：
- 僅針對 Per-FedAvg 演算法
- 使用 validation set 對全局模型進行適應
- 返回每個客戶端的 state_dict，用於後續 VFL 訓練
- 支援批次處理多個客戶端

使用場景：
在 VFL（垂直聯邦學習）設置中，先使用 Per-FedAvg 對每個客戶端的本地模型進行個性化，
然後將個性化後的模型用作 VFL 的初始模型。

典型工作流程：
1. Per-FedAvg 訓練 → 獲得全局模型
2. 個性化適應（本模組） → 獲得每個客戶端的個性化模型
3. VFL 訓練 → 使用個性化模型作為初始狀態
"""

import os
import glob
import torch
from copy import deepcopy
from typing import Dict, List, Tuple
from config import load_config
from Model import TransformerModel
from DataLoader import SequenceCSVDataset
from Trainer import FederatedTrainer


def load_global_model(config, model_path: str) -> TransformerModel:
    """載入 Per-FedAvg 訓練好的全局模型

    Args:
        config: 配置對象，包含模型架構參數
        model_path: 全局模型權重文件路徑

    Returns:
        載入權重的 TransformerModel 實例

    Raises:
        FileNotFoundError: 如果模型文件不存在
    """
    # 重建模型架構
    model = TransformerModel(
        feature_dim=config.feature_dim,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        output_dim=config.output_dim,
        max_seq_length=config.max_seq_length,
        dropout=config.dropout
    ).to(config.device)

    # 載入訓練好的權重
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"全局模型文件不存在: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=config.device))
    print(f"✓ 成功載入全局模型: {model_path}")

    return model


def load_client_datasets(config) -> Tuple[List[SequenceCSVDataset], List[str]]:
    """載入所有客戶端的數據集

    Args:
        config: 配置對象，包含數據路徑和處理參數

    Returns:
        datasets: 客戶端數據集列表
        client_names: 對應的客戶端名稱列表

    Note:
        - 使用 fit_scalers=False 來載入訓練時保存的標準化器
        - 保持與訓練時相同的數據分割比例（8:1:1）
    """
    datasets = []
    client_names = []

    # 掃描數據目錄
    csv_pattern = os.path.join(config.data_path, "*.csv")
    csv_files = sorted(glob.glob(csv_pattern))

    print(f"\n正在載入客戶端數據...")
    for csv_file in csv_files:
        csv_name = os.path.splitext(os.path.basename(csv_file))[0]

        try:
            dataset = SequenceCSVDataset(
                csv_path=config.data_path,
                csv_name=csv_name,
                input_len=config.input_length,
                output_len=config.output_length,
                features=config.features,
                target=config.target,
                save_path=config.data_path,
                train_ratio=0.8,
                val_ratio=0.1,
                split_type='time_based',
                fit_scalers=False  # 使用已保存的標準化器
            )

            if len(dataset) > 0:
                datasets.append(dataset)
                client_names.append(csv_name)
                print(f"  ✓ {csv_name}: {len(dataset)} 樣本")
        except Exception as e:
            print(f"  ✗ {csv_name}: 載入失敗 - {e}")

    print(f"成功載入 {len(datasets)} 個客戶端數據集")
    return datasets, client_names


def personalize_model_for_client(
    global_model: TransformerModel,
    dataset: SequenceCSVDataset,
    config,
    client_name: str
) -> Dict[str, torch.Tensor]:
    """為單個客戶端創建個性化模型

    Per-FedAvg 個性化流程：
    1. 複製全局模型（避免修改原始模型）
    2. 使用客戶端的 validation set 進行梯度下降適應
    3. 返回適應後的模型狀態字典

    Args:
        global_model: Per-FedAvg 訓練的全局模型
        dataset: 客戶端的完整數據集
        config: 配置對象（包含 adaptation_lr、personalization_steps）
        client_name: 客戶端名稱（用於日誌）

    Returns:
        personalized_state_dict: 個性化後的模型狀態字典

    Note:
        - 只使用 validation set 進行適應，test set 保留用於最終評估
        - 適應學習率通常較高（如 0.01），以實現快速適應
        - 適應步數通常較少（如 5-10 步），避免過擬合
    """
    # 準備數據
    trainer = FederatedTrainer(global_model, config, config.device)
    _, val_dataset, _ = trainer.split_dataset(dataset)
    _, val_loader, _ = trainer.create_data_loaders(None, val_dataset, None)

    # 收集 validation set 數據
    support_inputs, support_targets = [], []
    for inputs, targets in val_loader:
        support_inputs.append(inputs)
        support_targets.append(targets)

    if not support_inputs:
        print(f"  ⚠ {client_name}: 沒有 validation 數據，返回全局模型")
        return global_model.state_dict()

    # 合併批次
    support_inputs = torch.cat(support_inputs, dim=0).to(config.device)
    support_targets = torch.cat(support_targets, dim=0).to(config.device)

    print(f"  {client_name}: 使用 {len(support_inputs)} 個樣本進行個性化適應...")

    # 創建模型副本
    personalized_model = deepcopy(global_model)
    personalized_model.train()

    # 設置優化器和損失函數
    optimizer = torch.optim.Adam(
        personalized_model.parameters(),
        lr=config.adaptation_lr,
        weight_decay=1e-4
    )
    criterion = torch.nn.MSELoss()

    # 個性化適應
    initial_loss = None
    final_loss = None

    for step in range(config.personalization_steps):
        optimizer.zero_grad()
        outputs = personalized_model(support_inputs)
        loss = criterion(outputs, support_targets)
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()
        if step == config.personalization_steps - 1:
            final_loss = loss.item()

    # 輸出適應效果
    improvement = ((initial_loss - final_loss) / initial_loss * 100) if initial_loss > 0 else 0
    print(f"    初始損失: {initial_loss:.4f} → 最終損失: {final_loss:.4f} (改善: {improvement:.2f}%)")

    # 返回個性化模型的狀態字典
    return personalized_model.state_dict()


def initialize_personalized_models(
    config,
    model_path: str = None
) -> Dict[str, Dict[str, torch.Tensor]]:
    """為所有客戶端初始化個性化模型（主函數）

    這是 Per-FedAvg 個性化的核心函數：
    1. 載入全局模型
    2. 載入所有客戶端數據
    3. 為每個客戶端進行個性化適應
    4. 返回所有客戶端的模型狀態字典

    Args:
        config: 配置對象或配置文件路徑
        model_path: 全局模型路徑（默認使用 config 中的路徑）

    Returns:
        client_models: 字典 {client_name: state_dict}

    使用範例：
    ```python
    from config import load_config
    from Personalizer import initialize_personalized_models

    # 載入配置
    config = load_config('config.yaml')

    # 獲取所有客戶端的個性化模型
    client_models = initialize_personalized_models(config)

    # 使用特定客戶端的模型
    model_a = TransformerModel(...)
    model_a.load_state_dict(client_models['client_a'])
    ```

    典型工作流程（VFL 場景）：
    ```python
    # 步驟 1: Per-FedAvg 個性化
    client_models = initialize_personalized_models(config)

    # 步驟 2: 為 VFL 的兩方初始化模型
    model_party_a = TransformerModel(...)
    model_party_a.load_state_dict(client_models['party_a'])

    model_party_b = TransformerModel(...)
    model_party_b.load_state_dict(client_models['party_b'])

    # 步驟 3: VFL 訓練
    fusion_model = FusionModel(...)
    # ... VFL 訓練流程 ...
    ```
    """
    # 如果 config 是字符串路徑，載入配置
    if isinstance(config, str):
        config = load_config(config)

    print("=" * 70)
    print("Per-FedAvg 個性化模型初始化器")
    print("=" * 70)

    # 確定模型路徑
    if model_path is None:
        model_path = os.path.join(config.model_save_path, "final_global_model.pth")

    # 載入全局模型
    print(f"\n【步驟 1】載入全局模型")
    global_model = load_global_model(config, model_path)

    # 載入客戶端數據
    print(f"\n【步驟 2】載入客戶端數據")
    datasets, client_names = load_client_datasets(config)

    if not datasets:
        raise ValueError("沒有找到任何客戶端數據集！")

    # 為每個客戶端進行個性化
    print(f"\n【步驟 3】為每個客戶端進行個性化適應")
    print(f"適應參數:")
    print(f"  - 學習率: {config.adaptation_lr}")
    print(f"  - 適應步數: {config.personalization_steps}")
    print(f"  - 設備: {config.device}")
    print()

    client_models = {}

    for dataset, client_name in zip(datasets, client_names):
        state_dict = personalize_model_for_client(
            global_model,
            dataset,
            config,
            client_name
        )
        client_models[client_name] = state_dict

    # 總結
    print(f"\n【完成】成功初始化 {len(client_models)} 個客戶端的個性化模型")
    print(f"客戶端列表: {list(client_models.keys())}")
    print("=" * 70)

    return client_models


def save_personalized_models(
    client_models: Dict[str, Dict[str, torch.Tensor]],
    save_dir: str
):
    """保存所有客戶端的個性化模型到磁盤

    Args:
        client_models: 客戶端模型狀態字典 {client_name: state_dict}
        save_dir: 保存目錄

    Note:
        每個客戶端模型保存為獨立的 .pth 文件
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n保存個性化模型到: {save_dir}")
    for client_name, state_dict in client_models.items():
        save_path = os.path.join(save_dir, f"{client_name}_personalized.pth")
        torch.save(state_dict, save_path)
        print(f"  ✓ {client_name} → {save_path}")

    print(f"成功保存 {len(client_models)} 個模型")


# ============================================================================
# 使用範例
# ============================================================================

if __name__ == "__main__":
    """
    使用範例：直接運行此腳本進行個性化並保存模型

    使用方式：
    ```bash
    python Personalizer.py
    ```
    """
    import argparse

    # 解析命令行參數
    parser = argparse.ArgumentParser(description='Per-FedAvg 個性化模型初始化')
    parser.add_argument('--config', default='config.yaml',
                       help='配置文件路徑 (default: config.yaml)')
    parser.add_argument('--model_path', default=None,
                       help='全局模型路徑 (default: 使用配置文件中的路徑)')
    parser.add_argument('--save_dir', default='personalized_models',
                       help='保存個性化模型的目錄 (default: personalized_models)')
    args = parser.parse_args()

    # 載入配置
    config = load_config(args.config)

    # 初始化個性化模型
    client_models = initialize_personalized_models(config, args.model_path)

    # 保存模型
    save_personalized_models(client_models, args.save_dir)

    print("\n✓ 個性化模型初始化完成！")
    print("\n使用方式:")
    print("```python")
    print("from Model import TransformerModel")
    print("import torch")
    print()
    print("# 載入特定客戶端的個性化模型")
    print("model = TransformerModel(...)")
    print(f"model.load_state_dict(torch.load('personalized_models/client_name_personalized.pth'))")
    print("```")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

class FederatedTrainer:
    """
    聯邦學習輔助工具類 - 數據處理和評估的統一接口
    
    **設計理念**：
    在聯邦學習中，每個客戶端需要獨立處理自己的數據。FederatedTrainer
    提供了標準化的數據分割、加載器創建和模型評估功能。
    
    **核心職責**：
    1. 數據集分割：將完整數據集分為訓練/驗證/測試三部分
    2. 數據加載器創建：為不同數據集創建適當的DataLoader
    3. 模型評估：提供統一的測試評估接口
    4. 設備管理：處理CPU/GPU之間的數據轉移
    
    **為什麼不在這裡實現訓練**：
    聯邦學習中，不同算法（FedAvg vs Per-FedAvg）有不同的訓練邏輯，
    因此訓練功能被分離到Client類中，保持職責單一性。
    
    **與集中式學習的區別**：
    - 集中式：所有數據在一個Trainer中
    - 聯邦式：每個客戶端有自己的Trainer實例
    """
    
    def __init__(self, model, config, device):
        self.model = model        # 模型實例（用於評估）
        self.config = config      # 訓練配置參數
        self.device = device      # 計算設備
        self.criterion = nn.MSELoss()  # 損失函數（用於評估）
        
        # 注意：優化器現在在Client.py中實現，因為不同的FL算法需要不同的優化策略
    
    def split_dataset(self, dataset):
        """
        將數據集按時間順序分割為訓練、驗證、測試集
        
        **時間順序分割的重要性**：
        對於時序預測任務，數據分割必須嚴格按照時間順序進行：
        - 訓練集：最早的80%數據
        - 驗證集：中間的10%數據  
        - 測試集：最新的10%數據
        
        **為什麼不能隨機分割**：
        1. 避免數據洩漏：隨機分割可能讓模型"看到未來"
        2. 模擬真實場景：實際部署時只能用歷史數據預測未來
        3. 評估可靠性：測試集代表模型在真實場景下的性能
        
        **分割比例的考慮**：
        - 8:1:1是常用比例，平衡了訓練數據量和評估可靠性
        - 對於長時序數據，可以適當增加測試集比例
        
        Args:
            dataset: SequenceCSVDataset實例，已包含時間順序分割邏輯
            
        Returns:
            tuple: (train_dataset, val_dataset, test_dataset)
        """
        # 利用dataset內建的時間順序分割功能
        # 這些方法返回SequenceSubset實例，包含對應的數據索引
        train_dataset = dataset.get_train_dataset()
        val_dataset = dataset.get_val_dataset()
        test_dataset = dataset.get_test_dataset()
        
        print(f"Time-based split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def create_data_loaders(self, train_dataset, val_dataset, test_dataset):
        """
        創建PyTorch數據加載器 - 針對聯邦學習優化
        
        **數據加載器配置原則**：
        1. 訓練集：shuffle=True，增加隨機性防止過擬合
        2. 驗證/測試集：shuffle=False，保持評估結果的一致性
        3. 聯邦學習：num_workers=0，避免多進程間的衝突
        4. 記憶體優化：根據設備類型決定是否使用pin_memory
        
        **聯邦學習特殊考慮**：
        - 多客戶端並行：避免使用多進程加載，防止資源競爭
        - 記憶體效率：pin_memory只在CUDA設備上啟用
        - 批次大小：通常比集中式學習更小，適應客戶端資源限制
        
        **為什麼不直接在__init__中創建**：
        靈活性：不同的客戶端可能需要不同的數據加載配置
        
        Args:
            train_dataset: 訓練數據集（可為None）
            val_dataset: 驗證數據集（可為None）
            test_dataset: 測試數據集（可為None）
            
        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        train_loader = None
        val_loader = None  
        test_loader = None
        
        # === 訓練數據加載器 ===
        if train_dataset is not None:
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.config.batch_size,  # 通常32-128
                shuffle=True,                       # 打亂順序，增加訓練隨機性
                num_workers=0,                      # 聯邦學習：避免多進程衝突
                pin_memory=True if self.device.type == 'cuda' else False  # CUDA加速
            )
        
        # === 驗證數據加載器 ===
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset, 
                batch_size=self.config.batch_size, 
                shuffle=False,                      # 不打亂，保持評估一致性
                num_workers=0,
                pin_memory=True if self.device.type == 'cuda' else False
            )
        
        # === 測試數據加載器 ===
        if test_dataset is not None:
            test_loader = DataLoader(
                test_dataset, 
                batch_size=self.config.batch_size, 
                shuffle=False,                      # 不打亂，便於結果分析
                num_workers=0,
                pin_memory=True if self.device.type == 'cuda' else False
            )
        
        return train_loader, val_loader, test_loader
    
    def test_model(self, test_loader):
        """測試模型性能"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                
                # 收集預測結果用於後續分析
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0.0
        
        return avg_loss, np.array(all_predictions), np.array(all_targets)
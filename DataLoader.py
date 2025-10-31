import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
import psutil
from sklearn.preprocessing import StandardScaler
import pickle

class SequenceCSVDataset(Dataset):
    """
    時序CSV數據集類 - 專為聯邦學習和時序預測設計
    
    **核心功能**：
    1. 時序序列生成：將原始CSV數據轉換為滑動窗口序列
    2. 時間順序分割：確保訓練/驗證/測試集按時間順序分割，避免數據洩漏
    3. 標準化處理：僅使用訓練集統計信息，防止未來數據洩露
    4. 聯邦學習適配：支持多個客戶端的獨立數據處理
    
    **數據洩漏防護**：
    - 時間分割：測試集永遠是最新的數據，模擬真實預測場景
    - 標準化隔離：驗證集和測試集使用訓練集的統計信息標準化
    - 跨期間隔離：沒有未來數據用於過去時間點的預測
    
    **序列生成原理**：
    對於輸入長度96，輸出長度1的設置：
    - 使用過去96個時間步預測下1個時間步
    - 滑動窗口：窗口向右移動1步，產生下一個訓練樣本
    - 適合電力需求等時序預測任務
    """
    def __init__(self, csv_path, csv_name, input_len, output_len, features, target, save_path, 
                 train_ratio=0.8, val_ratio=0.1, split_type='time_based', fit_scalers=True):
        self.csv_path = csv_path          # CSV文件路徑
        self.csv_name = csv_name          # CSV文件名（不含擴展名）
        self.input_len = input_len        # 輸入序列長度（如96個時間步）
        self.output_len = output_len      # 輸出序列長度（如1個時間步）
        self.features = features          # 輸入特徵列表（25個特徵）
        self.target = target              # 預測目標列表（通常是['Power_Demand']）
        self.save_path = save_path        # 標準化器保存路徑
        self.train_ratio = train_ratio    # 訓練集比例（如0.8）
        self.val_ratio = val_ratio        # 驗證集比例（如0.1）
        self.split_type = split_type      # 分割類型（time_based時間順序）
        self.fit_scalers = fit_scalers    # 是否重新擬合標準化器
        
        # 初始化序列列表
        self.input_seq = []
        self.output_seq = []
        self.train_indices = []
        self.val_indices = []
        self.test_indices = []

        # === 步驟1：載入原始CSV數據 ===
        abs_path = os.path.abspath(csv_path)
        data = pd.read_csv(os.path.join(abs_path, self.csv_name + '.csv'))
        
        # 分離輸入特徵和預測目標
        input_data = data[features]   # 例如：25個特徵（電器使用量、天氣等）
        output_data = data[target]    # 例如：1個目標（Power_Demand）
        
        # === 步驟2：生成時序序列（滑動窗口） ===
        total_len = input_len + output_len  # 例如：96 + 1 = 97
        raw_input_seq = []   # 存儲原始輸入序列
        raw_output_seq = []  # 存儲原始輸出序列
        
        # 滑動窗口生成序列對
        # 例如：數據長度1000，窗口97，可產生904個序列
        for i in range(len(input_data) - total_len + 1):
            # 輸入序列：時間步 i 到 i+input_len-1 (96個時間步)
            input_seq = input_data.iloc[i:i+input_len].values
            
            # 輸出序列：時間步 i+input_len 到 i+total_len-1 (1個時間步)
            output_seq = output_data.iloc[i+input_len:i+total_len].values
            
            # 單步預測時將輸出展平為1維
            if self.output_len == 1:
                output_seq = output_seq.flatten()
                
            raw_input_seq.append(input_seq)
            raw_output_seq.append(output_seq)
        
        # === 步驟3：時間順序數據分割（關鍵！） ===
        total_sequences = len(raw_input_seq)
        
        if split_type == 'time_based':
            # 時間順序分割：訓練集→驗證集→測試集（按時間先後）
            # 這模擬真實場景：用歷史數據預測未來
            train_end = int(total_sequences * train_ratio)      # 前80%為訓練集
            val_end = int(total_sequences * (train_ratio + val_ratio))  # 中間10%為驗證集
            
            self.train_indices = list(range(0, train_end))              # 最早的數據
            self.val_indices = list(range(train_end, val_end))          # 中間的數據
            self.test_indices = list(range(val_end, total_sequences))   # 最新的數據
        else:
            raise ValueError(f"Unsupported split_type: {split_type}")
        
        # === 步驟4：標準化處理（防止數據洩漏） ===
        self.feature_scaler = StandardScaler()  # 特徵標準化器
        self.target_scaler = StandardScaler()   # 目標標準化器
        
        if fit_scalers:
            # 關鍵：只使用訓練集數據來擬合標準化器
            # 將所有訓練序列展平為2D矩陣進行擬合
            train_input_flat = np.concatenate([raw_input_seq[i] for i in self.train_indices])
            train_output_flat = np.concatenate([raw_output_seq[i].reshape(-1, len(target)) for i in self.train_indices])
            
            # 擬合標準化器（只看訓練數據的統計信息）
            self.feature_scaler.fit(train_input_flat)
            self.target_scaler.fit(train_output_flat)
            
            # 保存標準化器供後續使用
            with open(os.path.join(abs_path, self.csv_name + '_feature_scaler.pkl'), 'wb') as f:
                pickle.dump(self.feature_scaler, f)
            with open(os.path.join(abs_path, self.csv_name + '_target_scaler.pkl'), 'wb') as f:
                pickle.dump(self.target_scaler, f)
        else:
            # 載入已訓練好的標準化器（用於測試或推理）
            with open(os.path.join(abs_path, self.csv_name + '_feature_scaler.pkl'), 'rb') as f:
                self.feature_scaler = pickle.load(f)
            with open(os.path.join(abs_path, self.csv_name + '_target_scaler.pkl'), 'rb') as f:
                self.target_scaler = pickle.load(f)
        
        # === 步驟5：應用標準化到所有序列 ===
        for i in range(total_sequences):
            # 使用訓練集統計信息標準化所有序列（包括驗證集和測試集）
            input_seq = self.feature_scaler.transform(raw_input_seq[i])
            output_seq = self.target_scaler.transform(raw_output_seq[i].reshape(-1, len(target)))
            
            if self.output_len == 1:
                output_seq = output_seq.flatten()
                
            self.input_seq.append(input_seq)
            self.output_seq.append(output_seq)
        
        # 輸出數據分割信息
        print(f"Dataset split: Train={len(self.train_indices)}, Val={len(self.val_indices)}, Test={len(self.test_indices)}")
        print(f"Feature scaler fitted on {len(self.train_indices)} training sequences")
        print(f"Target scaler fitted on {len(self.train_indices)} training sequences")
    
    def get_train_dataset(self):
        """獲取訓練集子集"""
        return SequenceSubset(self, self.train_indices)
    
    def get_val_dataset(self):
        """獲取驗證集子集"""
        return SequenceSubset(self, self.val_indices)
    
    def get_test_dataset(self):
        """獲取測試集子集"""
        return SequenceSubset(self, self.test_indices)
    
    def inverse_transform_input(self, input_seq):
        return self.feature_scaler.inverse_transform(input_seq)
    
    def inverse_transform_output(self, output_seq):
        # 確保輸出形狀正確用於反標準化
        if output_seq.ndim == 1:
            output_seq = output_seq.reshape(-1, 1)
        elif output_seq.ndim == 2 and output_seq.shape[1] != len(self.target):
            output_seq = output_seq.reshape(-1, len(self.target))
        return self.target_scaler.inverse_transform(output_seq)

    def __len__(self):
        return len(self.input_seq)
    
    def __getitem__(self, idx):
        input_seq = self.input_seq[idx]
        output_seq = self.output_seq[idx]
        # 保持輸出維度一致性，不壓縮單一預測值
        # 確保數據類型為 float32 以兼容 MPS
        return torch.tensor(input_seq, dtype=torch.float32), torch.tensor(output_seq, dtype=torch.float32)


class SequenceSubset(Dataset):
    """時序數據集子集類別"""
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

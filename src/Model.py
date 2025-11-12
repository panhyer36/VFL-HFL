import torch
import torch.nn as nn
import math

class PositionalEncoder(nn.Module):
    """
    位置編碼器：為序列添加位置信息 - Transformer的關鍵組件
    
    **為什麼需要位置編碼？**
    Transformer的自注意力機制是位置不變的，即打亂序列順序不會改變輸出。
    但對於時序數據（如電力需求預測），位置信息至關重要。
    位置編碼讓模型能夠理解序列中每個元素的位置。
    
    **正弦-餘弦位置編碼的優勢**：
    1. 確定性：不需要學習，直接計算得出
    2. 外推性：可以處理比訓練時更長的序列
    3. 相對位置：模型可以學習到相對位置關係
    4. 週期性：不同頻率的正弦波能捕捉不同尺度的位置模式
    
    **數學原理**：
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    其中pos是位置，i是維度索引
    """
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        
        # 創建位置編碼矩陣 [max_seq_length, d_model]
        pe = torch.zeros(max_seq_length, d_model)
        
        # 創建位置索引 [max_seq_length, 1]
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # 計算分母項：10000^(2i/d_model) for i in [0, d_model/2)
        # 這創建了不同頻率的正弦波，從高頻到低頻
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # 偶數維度使用sin，奇數維度使用cos
        # 這種交替模式讓模型能夠區分相對位置
        pe[:, 0::2] = torch.sin(position * div_term)  # sin(pos/10000^(2i/d_model))
        pe[:, 1::2] = torch.cos(position * div_term)  # cos(pos/10000^(2i/d_model))
        
        # 註冊為buffer：不是可學習參數，但會隨模型一起保存/載入
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_seq_length, d_model)
    
    def forward(self, x):
        """
        前向傳播：將位置編碼加到輸入序列上
        
        Args:
            x: 輸入序列 (batch_size, seq_length, d_model)
            
        Returns:
            添加位置編碼後的序列 (batch_size, seq_length, d_model)
            
        **位置編碼的加法操作**：
        直接相加而非拼接的原因：
        1. 保持維度不變，不增加參數量
        2. Transformer原論文的標準做法
        3. 讓模型學習如何平衡內容信息和位置信息
        """
        # 廣播加法：pe[:, :x.size(1), :] 自動擴展到 (batch_size, seq_length, d_model)
        # 只使用序列長度內的位置編碼，支持變長序列
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerModel(nn.Module):
    """
    用於時序預測的Transformer模型 - 針對電力需求預測優化
    
    **模型架構設計原理**：
    1. 編碼器專用：只使用Transformer編碼器，適合序列到標量的預測任務
    2. 注意力聚合：使用可學習的注意力權重將序列信息聚合為單一預測值
    3. 殘差連接：每層都有跳躍連接，防止梯度消失
    4. 層正規化：穩定訓練過程，加速收斂
    
    **為什麼選擇Transformer？**
    1. 長距離依賴：能捕捉電力需求的長期週期性模式
    2. 並行計算：相比RNN/LSTM，訓練效率更高
    3. 可解釋性：注意力權重可以可視化，理解模型關注的時間點
    4. 擴展性：容易調整模型大小以適應不同複雜度的任務
    
    **針對時序預測的改進**：
    - GELU激活：相比ReLU更平滑，適合時序數據
    - 注意力聚合：學習序列中哪些時間點對預測最重要
    - 輸入投影：將多維特徵映射到統一的Transformer維度
    """
    def __init__(self, feature_dim, d_model=512, nhead=8, num_layers=4, 
                 output_dim=None, max_seq_length=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model           # Transformer隱藏維度
        self.feature_dim = feature_dim   # 輸入特徵維度
        
        # === 輸入處理層 ===
        # 將多維特徵(如25維電力相關特徵)投影到Transformer的統一維度
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)  # 輸入正規化，穩定訓練
        
        # === 位置編碼 ===
        # 為序列添加位置信息，讓模型理解時間順序
        self.pos_encoder = PositionalEncoder(d_model, max_seq_length)
        
        # === 正則化 ===
        self.dropout = nn.Dropout(dropout)  # 防止過擬合
        
        # === Transformer核心 ===
        # 創建單個編碼器層的配置
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,                    # 模型維度
            nhead=nhead,                        # 注意力頭數，典型值：8-16
            dim_feedforward=d_model * 4,        # 前饋網絡維度，通常是d_model的4倍
            dropout=dropout,                    # Dropout比例
            activation='gelu',                  # GELU比ReLU更適合序列建模
            batch_first=True                    # 輸入格式：(batch, seq, feature)
        )
        
        # 堆疊多個編碼器層
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers,              # 層數，平衡性能與計算成本
            norm=nn.LayerNorm(d_model)          # 最終層正規化
        )
        
        # === 序列聚合層 ===
        # 將整個序列的信息聚合為單一向量，用於最終預測
        # 使用可學習的注意力權重，而非簡單的平均或最後一個時間步
        self.attention_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),   # 降維
            nn.Tanh(),                          # 非線性激活
            nn.Linear(d_model // 2, 1)          # 輸出標量權重
        )
        
        # === 輸出投影層 ===
        if output_dim is not None:
            self.output_proj = nn.Sequential(
                nn.Linear(d_model, d_model // 2),  # 第一層降維
                nn.ReLU(),                         # 非線性
                nn.Dropout(dropout),               # 防過擬合
                nn.Linear(d_model // 2, output_dim) # 最終輸出
            )
        else:
            self.output_proj = None
            
        # 權重初始化，使用Xavier初始化提高訓練穩定性
        self.init_weights()
    
    def init_weights(self):
        """使用Xavier初始化權重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _encode(self, x, src_mask=None):
        """執行從輸入到Transformer編碼器的過程"""
        # 投影到d_model維度
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = x * math.sqrt(self.d_model)  # 縮放因子

        # 添加位置編碼
        x = self.pos_encoder(x)
        x = self.dropout(x)

        # 通過transformer
        output = self.transformer(x, src_mask)  # (batch_size, seq_len, d_model)
        return output

    def forward(self, x, src_mask=None):
        """
        前向傳播 - 完整的時序預測流程
        
        Args:
            x: 輸入時序數據 (batch_size, seq_len, feature_dim)
               例如：(32, 96, 25) 表示32個樣本，每個96個時間步，25個特徵
            src_mask: 源序列遮罩，用於屏蔽某些位置（在此應用中通常為None）
            
        Returns:
            預測結果 (batch_size, output_dim)
            例如：(32, 1) 表示32個樣本的電力需求預測值
            
        **前向傳播流程**：
        1. 特徵投影：將原始特徵投影到Transformer空間
        2. 位置編碼：添加時間位置信息
        3. Transformer編碼：提取序列特徵和注意力模式
        4. 注意力聚合：將序列信息聚合為固定大小向量
        5. 輸出投影：生成最終預測結果
        """
        # 步驟1-3：特徵提取和編碼
        output = self._encode(x, src_mask)  # (batch_size, seq_len, d_model)

        # 步驟4：注意力聚合機制
        # 學習序列中每個時間步的重要性權重
        attention_scores = self.attention_proj(output)  # (batch_size, seq_len, 1)
        
        # 使用softmax確保權重和為1，形成概率分佈
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch_size, seq_len, 1)

        # 使用注意力權重對序列進行加權平均
        # 這比簡單平均或取最後時間步更加靈活和有效
        weighted_output = (attention_weights * output).sum(dim=1)  # (batch_size, d_model)

        # 步驟5：輸出投影
        if self.output_proj is not None:
            # 將聚合後的特徵映射到最終預測維度
            final_output = self.output_proj(weighted_output)  # (batch_size, output_dim)
        else:
            final_output = weighted_output  # 直接返回聚合特徵

        return final_output

    def forward_embedding(self, x, src_mask=None):
        """
        前向傳播（不含輸出投影）- 用於聯邦學習或特徵融合

        Args:
            x: 輸入時序數據 (batch_size, seq_len, feature_dim)
            src_mask: 源序列遮罩

        Returns:
            聚合後的嵌入向量 (batch_size, d_model)

        **用途**：
        在聯邦學習場景中，各方保留自己的原始數據，只共享嵌入向量。
        這個方法返回聚合後的特徵表示，可以傳送給融合模型進行最終預測。
        """
        # 特徵提取和編碼
        output = self._encode(x, src_mask)  # (batch_size, seq_len, d_model)

        # 注意力聚合
        attention_scores = self.attention_proj(output)  # (batch_size, seq_len, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch_size, seq_len, 1)
        weighted_output = (attention_weights * output).sum(dim=1)  # (batch_size, d_model)

        return weighted_output

    def get_attention_weights(self, x, src_mask=None):
        """獲取注意力權重用於可視化"""
        output = self._encode(x, src_mask)

        # 計算注意力權重
        attention_scores = self.attention_proj(output)
        attention_weights = torch.softmax(attention_scores, dim=1)

        return attention_weights


class FusionModel(nn.Module):
    """
    融合模型 - 用於垂直聯邦學習（Vertical Federated Learning, VFL）

    **VFL場景說明**：
    在垂直聯邦學習中，不同參與方擁有相同樣本的不同特徵：
    - Party A：擁有部分特徵（如氣象數據）
    - Party B：擁有另一部分特徵（如歷史電力數據）
    - Server：接收雙方的嵌入向量，進行融合預測

    **隱私保護機制**：
    1. 原始數據不出域：各方只在本地處理自己的數據
    2. 只共享嵌入：傳輸的是高維特徵表示，而非原始數據
    3. 梯度保護：反向傳播時只傳遞梯度，不暴露原始特徵

    **融合策略**：
    採用深度融合（Deep Fusion）而非簡單拼接：
    1. 拼接雙方嵌入：組合來自不同數據源的特徵
    2. 多層非線性轉換：學習特徵間的交互關係
    3. Dropout正則化：防止過度依賴某一方的特徵
    4. 殘差連接（可選）：保留原始特徵信息

    **架構優勢**：
    - 靈活性：支持不同維度的輸入嵌入
    - 可擴展性：易於擴展到多方（>2）聯邦學習
    - 可解釋性：可以分析各方特徵的貢獻度
    """

    def __init__(self, embedding_dim_party_a, embedding_dim_party_b,
                 hidden_dim=256, output_dim=1, dropout=0.1):
        """
        初始化融合模型

        Args:
            embedding_dim_party_a: 參與方A的嵌入維度（通常是TransformerModel的d_model）
            embedding_dim_party_b: 參與方B的嵌入維度
            hidden_dim: 融合層的隱藏層維度
            output_dim: 輸出維度（預測電力需求，通常為1）
            dropout: Dropout比例，用於正則化
        """
        super().__init__()

        self.embedding_dim_party_a = embedding_dim_party_a
        self.embedding_dim_party_b = embedding_dim_party_b
        self.hidden_dim = hidden_dim

        # === 融合網絡 ===
        # 多層感知機（MLP）用於學習雙方特徵的交互
        fusion_input_dim = embedding_dim_party_a + embedding_dim_party_b

        self.fusion_network = nn.Sequential(
            # 第一層：降維並提取交互特徵
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 層正規化，穩定訓練
            nn.ReLU(),
            nn.Dropout(dropout),

            # 第二層：進一步提取高階特徵
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            # 輸出層：映射到預測值
            nn.Linear(hidden_dim // 2, output_dim)
        )

        # === 可選：注意力機制 ===
        # 學習如何平衡雙方特徵的重要性
        self.use_attention = True
        if self.use_attention:
            self.attention_party_a = nn.Linear(embedding_dim_party_a, 1)
            self.attention_party_b = nn.Linear(embedding_dim_party_b, 1)

        # 權重初始化
        self.init_weights()

    def init_weights(self):
        """使用Xavier初始化權重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, embedding_party_a, embedding_party_b):
        """
        前向傳播 - 融合雙方嵌入並預測

        Args:
            embedding_party_a: 參與方A的嵌入向量 (batch_size, embedding_dim_party_a)
            embedding_party_b: 參與方B的嵌入向量 (batch_size, embedding_dim_party_b)

        Returns:
            預測的電力需求 (batch_size, output_dim)

        **融合流程**：
        1. （可選）注意力加權：動態調整雙方特徵的重要性
        2. 特徵拼接：組合雙方的嵌入向量
        3. 深度融合：通過多層網絡學習特徵交互
        4. 輸出預測：生成最終的電力需求預測值
        """
        # === 可選：注意力機制 ===
        if self.use_attention:
            # 計算每一方的注意力分數
            attn_score_a = self.attention_party_a(embedding_party_a)  # (batch_size, 1)
            attn_score_b = self.attention_party_b(embedding_party_b)  # (batch_size, 1)

            # 使用softmax歸一化注意力權重
            attn_scores = torch.cat([attn_score_a, attn_score_b], dim=1)  # (batch_size, 2)
            attn_weights = torch.softmax(attn_scores, dim=1)  # (batch_size, 2)

            # 應用注意力權重
            weighted_embedding_a = embedding_party_a * attn_weights[:, 0:1]  # (batch_size, dim_a)
            weighted_embedding_b = embedding_party_b * attn_weights[:, 1:2]  # (batch_size, dim_b)

            # 拼接加權後的嵌入
            fused_embedding = torch.cat([weighted_embedding_a, weighted_embedding_b], dim=1)
        else:
            # 直接拼接雙方嵌入
            fused_embedding = torch.cat([embedding_party_a, embedding_party_b], dim=1)

        # === 深度融合網絡 ===
        # 通過多層MLP學習特徵交互並生成預測
        output = self.fusion_network(fused_embedding)  # (batch_size, output_dim)

        return output

    def get_attention_weights(self, embedding_party_a, embedding_party_b):
        """
        獲取雙方特徵的注意力權重，用於可解釋性分析

        Returns:
            attention_weights: (batch_size, 2) 其中[:,0]是Party A的權重，[:,1]是Party B的權重
        """
        if not self.use_attention:
            # 如果未使用注意力機制，返回均勻權重
            batch_size = embedding_party_a.size(0)
            return torch.ones(batch_size, 2, device=embedding_party_a.device) * 0.5

        attn_score_a = self.attention_party_a(embedding_party_a)
        attn_score_b = self.attention_party_b(embedding_party_b)
        attn_scores = torch.cat([attn_score_a, attn_score_b], dim=1)
        attn_weights = torch.softmax(attn_scores, dim=1)

        return attn_weights

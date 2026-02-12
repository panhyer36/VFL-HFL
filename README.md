# VFL 垂直聯邦學習電力需求預測系統

## 項目概述

基於 **垂直聯邦學習（Vertical Federated Learning, VFL）** 的電力需求預測研究項目。使用 **Transformer 模型** 搭配 **FedAvg 聯邦平均算法**，在保護數據隱私的前提下，融合雲端天氣數據與本地用電數據進行電力需求預測。

**核心特色**：
- **隱私保護**：天氣數據與用電數據嚴格分離，僅傳輸嵌入向量與梯度
- **Per-FedAvg 個性化**：使用預訓練的個性化 HFL 模型作為基底
- **LoRA 適配**：per-client 低秩適配，深層個性化 HFL 嵌入
- **三階段訓練策略**：Fusion 預熱 → 聯合訓練 → 通訊優化
- **B-LEC 通訊壓縮**：SVD 低秩殘差壓縮 + Top-k Error Feedback，大幅降低通訊量
- **異質性應對**：LoRA + Element-wise Gate + FedDecorr + CrossAttention 四合一方案

## 系統架構

```
┌───────────────────────────┐            ┌───────────────────────────────┐
│       Server (雲端)        │            │       Client (×15, 本地)      │
│                           │            │                               │
│  Weather Model            │  Weather   │  HFL Model (凍結+LoRA)        │
│  (Transformer)            │  嵌入 (↓)  │  (Per-FedAvg 個性化)           │
│  9特徵 → 256維嵌入          │──────────→│  14特徵 → 256維嵌入             │
│                           │  B-LEC     │                              │
│  SSL 預訓練初始化           │  SVD 壓縮  │  Fusion Model (可訓練)         │
│                           │            │  ├ Weather Adapter (殘差瓶頸)  │
│                           │  嵌入梯度  │  ├ CrossAttention (4頭)        │
│  FedAvg 梯度聚合           │  (↑)       │  ├ Element-wise Gate          │
│  更新 Weather Model        │←──────────│  └ MLP → Power_Demand 預測     │
│                           │  Top-k     │                               │
│                           │  Error     │  損失 = MSE + FedDecorr       │
│                           │  Feedback  │                               │
└───────────────────────────┘            └───────────────────────────────┘
```

## 數據流

```
雲端 (Server):
  Weather 特徵 (9維) → Weather Model (Transformer) → Weather 嵌入 (256維)
                                    ↓ send_embeddings() [B-LEC SVD 壓縮]

本地 (Client):
  HFL 特徵 (14維) → HFL Model (凍結基底 + LoRA) → HFL 嵌入 (256維)

  Weather 嵌入 → [Weather Adapter (殘差瓶頸)] → 適配 Weather 嵌入
              ↓
  [CrossAttentionLayer] ← 適配 Weather 嵌入 + HFL 嵌入 (2-token 自注意力)
              ↓
  enriched_w, enriched_h
              ↓
  [Element-wise Gate] → gate ∈ (0,1)^256
  weighted_w = enriched_w × gate
  weighted_h = enriched_h × (1 - gate)
              ↓
  [weighted_w ‖ weighted_h] → Fusion MLP → Power_Demand 預測
              ↓
  損失 = MSE + λ_decorr × FedDecorr
              ↓ 逐 batch 累積嵌入梯度
  累積梯度 (∂L/∂emb) → [Top-k Error Feedback 壓縮] → 上傳 → Server FedAvg 聚合
```

## 隱私保護機制

| 數據類型 | 位置 | 是否傳輸 |
|----------|------|----------|
| Weather 原始特徵 | 僅雲端 | 不傳輸 |
| HFL 用電特徵 | 僅本地 | 不傳輸 |
| Power_Demand 標籤 | 僅本地 | 不傳輸 |
| Weather 嵌入向量 | 雲端 → 本地 | 傳輸 (256維向量) |
| 嵌入梯度 | 本地 → 雲端 | 傳輸 (256維梯度) |
| LoRA 權重 | 僅本地 | 不傳輸，不參與聯邦聚合 |

## 三階段訓練策略

| 階段 | 名稱 | 訓練目標 | Weather Model | LoRA | 客戶端選擇 | 通訊路徑 |
|------|------|----------|---------------|------|-----------|----------|
| Phase 0 | Fusion 預熱期 | 只訓練 Fusion Model | 凍結 | 凍結 | 所有客戶端 (100%) | 路徑 B (首輪下發，後續快取) |
| Phase 1 | 聯合訓練期 | Fusion + Weather + LoRA | 每輪更新 | 解凍 | 按 client_fraction 隨機選擇 | 路徑 A (每輪 download + upload) |
| Phase 2 | 通訊優化期 | 交替 Fusion-only 與聯合更新 | 週期性更新 | 解凍 | 按 client_fraction 隨機選擇 | 更新輪路徑 A，非更新輪路徑 B |

### 通訊路徑

- **路徑 A** (train_weather=True)：Server 一次性下發嵌入 (保留計算圖) → Client 逐 batch 訓練並累積嵌入梯度 → 累積完成後一次上傳 → Server 一次性反向傳播 → FedAvg 聚合更新
- **路徑 B** (train_weather=False)：快取檢查 (版本號比對) → 快取未命中時一次性下發嵌入並快取 → Client 用快取嵌入建立臨時 DataLoader 訓練

## B-LEC 通訊壓縮

### 下載端：低秩殘差壓縮 (SVD)

- **原理**：相鄰輪次 Weather 嵌入差值 `delta_z` 具有強低秩性
- **流程**：`delta_z = z_new - z_old` → 截斷 SVD → 傳送 `{A(N,r), B(r,D)}`
- **壓縮率**：`O((N+D)×r) / O(N×D)`，rank=4 時壓縮率極高
- **自適應品質控制**：`explained_var < variance_threshold` 時回退完整傳送，重置 z_old

### 上傳端：Top-k Error Feedback 壓縮

- **原理**：Error Feedback 確保長期梯度更新無偏性
- **流程**：`v = grad + E_k` → `Top-k(v, γ)` → 傳送 `{values, indices}` → `E_k ← v - compressed`
- **壓縮率**：γ=0.05 時僅傳送 5% 元素
- **生命週期**：error_buffer per-client，跨輪次持續累積不重置

### 對稱化通訊接口

| 方向 | Server 端 | Client 端 |
|------|-----------|-----------|
| 下載 (Cloud→Client) | `send_embeddings()` | `receive_embeddings()` |
| 上傳 (Client→Cloud) | `receive_gradients()` | `send_gradients()` |

Server 和 Client 各自獨立維護 `z_old`，帶寬統計封裝在接口內部。

## 異質性應對方案

| 方案 | 組件 | 目的 | 配置項 |
|------|------|------|--------|
| LoRA | HFL Transformer 內部適配 | 深層 per-client HFL 嵌入適配 | `hfl_model.lora_rank` / `lora_alpha` |
| 方案 1 | Element-wise Gate | 逐元素門控 (256維獨立權重) | 內建於 FusionModel |
| 方案 2 | FedDecorr | 去相關正則化，防止嵌入維度坍縮 | `regularization.lambda_decorr` |
| 方案 6 | CrossAttentionLayer | 輕量級交叉注意力，捕捉天氣-用電雙向交互 | `fusion_model.use_cross_attention` |

## 快速開始

### 前置需求

- Python 3.8+
- PyTorch 2.0+
- 預訓練模型：
  - `pretrain_model/ssl_pretrain.pt` (Weather Model SSL 預訓練權重)
  - `pretrain_model/HFL_global_model.pth` (Per-FedAvg HFL 全局模型)
- 數據文件：
  - `data/Weather.csv` (天氣數據)
  - `data/processed/Consumer_01~14.csv` + `Public_Building.csv` (15 個客戶端數據)
  - `data/processed/{client}_feature_scaler.pkl` / `{client}_target_scaler.pkl` (Per-FedAvg 產生的 scaler)

### 安裝步驟

```bash
# 1. 進入項目目錄
cd VFL

# 2. 安裝依賴
pip install -r requirements.txt

# 3. 檢查配置
python config.py
```

### 開始訓練

#### 方法一：後台訓練（推薦）

```bash
# 開始訓練
./run.sh

# 監控訓練過程
tail -f logs/train_*.log

# 停止訓練
./stop.sh
```

#### 方法二：直接運行

```bash
# 使用默認配置
python train.py

# 使用自定義配置
python train.py --config custom_config.yaml
```

### 測試評估

```bash
python test.py [--config config.yaml]
```

## 項目結構

```
VFL/
├── train.py               # VFL 訓練主腳本 (兩條通訊路徑 + 通訊統計)
├── test.py                # VFL 測試/評估腳本
├── config.py              # 配置載入器
├── config.yaml            # 核心配置文件
├── src/
│   ├── Server.py          # VFL 協調器：Weather Model, FedAvg 聚合, 通訊壓縮/統計
│   ├── Client.py          # VFL 客戶端：HFL(凍結+LoRA) + Fusion, 嵌入快取, 通訊壓縮
│   ├── Model.py           # TransformerModel + EmbeddingAdapter + CrossAttention + FusionModel
│   ├── LoRA.py            # LoRALinear + LoRAMultiheadAttention + 工具函數
│   ├── DataLoader.py      # 時序數據加載器
│   ├── Trainer.py         # 數據分割 + 評估
│   └── Personalizer.py    # Per-FedAvg 個性化 HFL 模型
├── data/
│   ├── Weather.csv                          # 天氣數據 (雲端)
│   └── processed/
│       ├── Consumer_01~14.csv               # 客戶端用電數據
│       ├── Public_Building.csv              # 公共建築用電數據
│       ├── {client}_feature_scaler.pkl      # Per-FedAvg 特徵 scaler
│       └── {client}_target_scaler.pkl       # Per-FedAvg 目標 scaler
├── pretrain_model/
│   ├── ssl_pretrain.pt                      # Weather Model SSL 預訓練權重
│   └── HFL_global_model.pth                # Per-FedAvg HFL 全局模型
├── checkpoints/                             # 模型檢查點
│   ├── best_weather_model.pth               # 最佳 Weather Model
│   ├── best_{client}_fusion_model.pth       # 最佳 Fusion Model (per-client)
│   ├── best_{client}_lora_model.pth         # 最佳 LoRA Model (per-client)
│   ├── final_weather_model.pth              # 最終 Weather Model
│   ├── {client}_fusion_model.pth            # 最終 Fusion Model
│   ├── {client}_lora_model.pth              # 最終 LoRA Model
│   └── {client}_hfl_personalized.pth        # 個性化 HFL 基底
├── plots/                                   # 性能圖表, 損失曲線
├── logs/                                    # 訓練日誌
├── run.sh                                   # 後台訓練腳本
├── stop.sh                                  # 停止訓練腳本
└── requirements.txt                         # Python 依賴
```

## 配置說明

### 聯邦學習配置

```yaml
federated_learning:
  algorithm: "vfl_fedavg"      # VFL 聯邦平均算法
  global_rounds: 200           # 全局訓練輪數
  client_fraction: 1           # 參與訓練的客戶端比例/數量
  num_clients: 15              # 客戶端數量 (Consumer_01~14 + Public_Building)

  training_strategy:
    phase0_rounds: 5           # Phase 0: Fusion 預熱期
    phase1_rounds: 194         # Phase 1: 聯合訓練期
    phase2_rounds: 1           # Phase 2: 通訊優化期 (自動計算: K - phase0 - phase1)
    phase2_fusion_freq: 4      # Phase 2 中每 N 輪 Fusion 後更新一次 Weather
```

### 模型配置

```yaml
# Weather Model (雲端, Transformer)
weather_model:
  feature_dim: 9               # 天氣特徵維度
  d_model: 256                 # Transformer 模型維度
  nhead: 8                     # 注意力頭數
  num_layers: 4                # Transformer 層數
  use_ssl_pretrain: true       # 使用 SSL 預訓練權重

# HFL Model (本地, 凍結 + LoRA)
hfl_model:
  feature_dim: 14              # 用電特徵維度
  d_model: 256                 # Transformer 模型維度
  freeze: true                 # 凍結 HFL 基底
  lora_rank: 8                 # LoRA rank (0 = 停用)
  lora_alpha: 8.0              # LoRA scaling (alpha / rank)

# Fusion Model (本地, 可訓練)
fusion_model:
  hidden_dim: 256              # 融合層隱藏維度
  output_dim: 1                # Power_Demand 預測
  adapter_bottleneck_dim: 64   # Weather Adapter 瓶頸維度 (0 = 不使用)
  use_cross_attention: true    # 啟用 CrossAttention
  cross_attention_dim: 64      # CrossAttention 投影維度
```

### 訓練配置

```yaml
local_training:
  learning_rate: 0.001         # 學習率 (CosineAnnealingLR)
  batch_size: 32               # 批次大小
  gradient_clip_norm: 1.0      # 梯度裁剪範數

regularization:
  lambda_decorr: 0.01          # FedDecorr 去相關損失權重

training:
  early_stopping:
    patience: 30               # 早停耐心值 (基於所有客戶端驗證損失)
```

### B-LEC 通訊壓縮配置

```yaml
communication_compression:
  download:
    enabled: true              # 下載端 SVD 壓縮
    svd_rank: 4                # 截斷 SVD rank
    variance_threshold: 0.85   # 品質閾值，低於此值回退完整傳送
  upload:
    enabled: true              # 上傳端 Top-k 壓縮
    top_k_ratio: 0.05          # 保留前 5% 最大梯度元素
```

### Per-FedAvg 個性化配置

```yaml
personalization:
  use_personalized_hfl: true
  hfl_model_path: "pretrain_model/HFL_global_model.pth"
  personalization_steps: 3     # 個性化適應步數
  adaptation_lr: 0.001         # 個性化適應學習率
```

## 模型參數量

| 組件 | 配置 | 說明 |
|------|------|------|
| Weather Model | 9 → 256維 Transformer | 全局共享，FedAvg 聚合 |
| HFL Model | 14 → 256維 Transformer | Per-client 個性化基底，凍結 |
| LoRA | rank=8, 注入 Q/V/out_proj/FFN ×4層 | Per-client，不參與聯邦聚合 |
| Weather Adapter | 256→64→256 殘差瓶頸 | Per-client，零初始化 |
| CrossAttentionLayer | 4頭, d_attn=64 | Per-client，~66K params |
| Element-wise Gate | 512→64→256 + Sigmoid | Per-client，~49K params |
| Fusion MLP | 512→256→128→1 | Per-client，~166K params |

## Per-FedAvg 整合

本項目與 **PerFedAvgHVP** 項目協同工作：

1. **PerFedAvgHVP** 負責 HFL 訓練，產出個性化 Consumer 模型和 per-client scaler
2. **VFL** 使用個性化 HFL 模型（凍結基底 + LoRA 適配）+ 雲端 Weather 數據融合預測

```
PerFedAvgHVP → HFL_global_model.pth → VFL HFL Model (凍結) → LoRA 疊加適配
PerFedAvgHVP → {client}_feature/target_scaler.pkl → VFL 數據標準化
```

### 個性化流程

1. 載入 Per-FedAvg 全局 HFL 模型（含 output_proj）
2. 使用 Personalizer 在客戶端 validation set 上快速適應（3 steps）
3. 凍結個性化後的 HFL 基底
4. 注入 LoRA 層（Phase 0 凍結，Phase 1+ 解凍訓練）
5. VFL 訓練時使用 `forward_embedding()` 跳過輸出層

## 使用指南

### 命令一覽

| 命令 | 功能 |
|------|------|
| `./run.sh` | 後台訓練（推薦） |
| `./stop.sh` | 停止訓練 |
| `python train.py` | 前台訓練 |
| `python train.py --config xxx.yaml` | 使用自定義配置 |
| `python test.py` | 測試評估 |
| `tail -f logs/train_*.log` | 監控訓練日誌 |

### 輸出說明

訓練結束後會輸出：
- **Per-client 通訊統計表格**：Downloaded / Uploaded / Total bytes（含壓縮率）
- **最佳模型**：`checkpoints/best_*.pth`（驗證損失最低時保存）
- **最終模型**：`checkpoints/final_*.pth` 和 `checkpoints/{client}_*.pth`
- **可視化圖表**：`plots/` 目錄下的性能圖表和損失曲線

## 故障排除

### 常見問題

1. **數據文件找不到**
```bash
# 檢查數據目錄結構
ls data/Weather.csv
ls data/processed/
```

2. **預訓練模型缺失**
```bash
# 確認預訓練模型存在
ls pretrain_model/ssl_pretrain.pt
ls pretrain_model/HFL_global_model.pth
```

3. **記憶體不足**
```yaml
# 修改 config.yaml，減小批次大小
local_training:
  batch_size: 16
```

4. **設備錯誤**
```yaml
# 強制使用 CPU
device:
  type: "cpu"
```

5. **LoRA 停用**
```yaml
# 設置 rank=0 關閉 LoRA
hfl_model:
  lora_rank: 0
```

6. **關閉通訊壓縮**
```yaml
# 完全回退未壓縮傳輸
communication_compression:
  download:
    enabled: false
  upload:
    enabled: false
```

## 套件版本

| 套件 | 版本要求 | 功能 |
|------|----------|------|
| **torch** | ≥2.0.0 | PyTorch 深度學習框架 |
| **torchvision** | ≥0.15.0 | PyTorch 視覺工具 |
| **numpy** | ≥1.21.0 | 高性能數值計算 |
| **pandas** | ≥1.3.0 | 數據處理和分析 |
| **scikit-learn** | ≥1.0.0 | 機器學習工具 (scaler) |
| **matplotlib** | ≥3.5.0 | 可視化繪圖 |
| **seaborn** | ≥0.11.0 | 統計可視化 |
| **pyyaml** | ≥6.0 | 配置文件解析 |
| **psutil** | ≥5.8.0 | 系統資源監控 |

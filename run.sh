#!/bin/bash
# VFL 訓練後台運行腳本

# 創建日誌目錄
mkdir -p logs

# 生成時間戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/train_${TIMESTAMP}.log"

echo "=========================================="
echo "VFL 垂直聯邦學習訓練"
echo "=========================================="
echo "啟動時間: $(date)"
echo "日誌文件: ${LOG_FILE}"
echo "=========================================="

# 後台運行訓練腳本
nohup python train.py > "${LOG_FILE}" 2>&1 &

# 獲取進程ID
PID=$!
echo "${PID}" > logs/train.pid

echo "訓練已在後台啟動 (PID: ${PID})"
echo ""
echo "監控日誌:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "停止訓練:"
echo "  ./stop.sh"
echo "=========================================="

#!/bin/bash
# VFL 訓練停止腳本

PID_FILE="logs/train.pid"

if [ -f "${PID_FILE}" ]; then
    PID=$(cat "${PID_FILE}")
    echo "正在停止訓練進程 (PID: ${PID})..."

    if ps -p ${PID} > /dev/null; then
        kill ${PID}
        echo "訓練進程已停止"
    else
        echo "進程 ${PID} 不存在或已停止"
    fi

    rm -f "${PID_FILE}"
else
    echo "未找到 PID 文件，嘗試搜索並停止所有訓練進程..."
    pkill -f "python train.py"
    echo "完成"
fi

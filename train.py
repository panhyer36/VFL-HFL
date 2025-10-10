import argparse


def train(args):
    # 初始化
    #1. 載入SSL pretrain的VFL模型(僅pretrain Encoder)
    #2. 重寫並分離成兩個forward函數，分別是Encoder和Attention層
    #3. 載入Encoder forward及參數，凍結Encoder參數
    #4. 載入Attention forward及初始化參數
    #5. 載入HfL Pretrain模型
    #6. 下發HFL Pretrain模型到客戶端，並且進行tau步訓練

    #開始交替訓練：
    #0. 載入k個客戶端
    #1. 伺服器發起批次
    #2. 客戶端準備數據
    #3. 計算幾度
    #4. 更新fusion_net參數或傳送attention_proj參數給雲端聚合計算

    # 評估

    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VFL Model Training')
    
    args = parser.parse_args()
    train(args)
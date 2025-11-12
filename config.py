import yaml
import torch
import os
from types import SimpleNamespace

def load_config(config_path="config.yaml"):
    """
    從YAML配置文件載入並解析 VFL 訓練參數

    **VFL (垂直聯邦學習) 配置系統特點**:
    1. 雙模型架構: Weather Model (雲端) + HFL Model (本地)
    2. 融合模型: Fusion Model 負責整合雙方嵌入
    3. 分階段訓練: 通訊效率優化策略
    4. Per-FedAvg 整合: 支援使用個性化的 HFL 模型

    Args:
        config_path: YAML配置文件路徑

    Returns:
        SimpleNamespace: 包含所有配置參數的對象

    Raises:
        FileNotFoundError: 配置文件不存在
        ValueError: YAML格式錯誤
        RuntimeError: 其他讀取錯誤
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration file {config_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Error reading configuration file {config_path}: {e}")

    # 創建命名空間對象
    config = SimpleNamespace()

    # === 聯邦學習核心配置 ===
    fl_config = config_dict['federated_learning']
    config.algorithm = fl_config['algorithm']
    config.K = fl_config['global_rounds']
    config.r = fl_config['client_fraction']
    config.num_users = fl_config['num_clients']
    config.eval_interval = fl_config['eval_interval']

    # VFL 訓練策略配置
    training_strategy = fl_config['training_strategy']
    config.phase1_rounds = training_strategy['phase1_rounds']
    config.phase2_rounds = training_strategy['phase2_rounds']
    config.phase2_fusion_freq = training_strategy['phase2_fusion_freq']

    # === 本地訓練配置 ===
    local_config = config_dict['local_training']
    config.tau = local_config['local_epochs']
    config.beta = local_config['learning_rate']
    config.batch_size = local_config['batch_size']

    # === Weather Model 配置 (雲端) ===
    weather_config = config_dict['weather_model']
    config.weather_feature_dim = weather_config['feature_dim']
    config.weather_d_model = weather_config['d_model']
    config.weather_nhead = weather_config['nhead']
    config.weather_num_layers = weather_config['num_layers']
    config.weather_output_dim = weather_config['output_dim']
    config.weather_max_seq_length = weather_config['max_seq_length']
    config.weather_dropout = weather_config['dropout']
    # SSL 預訓練配置
    config.use_ssl_pretrain = weather_config.get('use_ssl_pretrain', False)
    config.ssl_pretrain_path = weather_config.get('ssl_pretrain_path', None)

    # === HFL Model 配置 (本地) ===
    hfl_config = config_dict['hfl_model']
    config.hfl_feature_dim = hfl_config['feature_dim']
    config.hfl_d_model = hfl_config['d_model']
    config.hfl_nhead = hfl_config['nhead']
    config.hfl_num_layers = hfl_config['num_layers']
    config.hfl_output_dim = hfl_config['output_dim']
    config.hfl_max_seq_length = hfl_config['max_seq_length']
    config.hfl_dropout = hfl_config['dropout']
    config.freeze_hfl = hfl_config['freeze']

    # === Fusion Model 配置 (本地) ===
    fusion_config = config_dict['fusion_model']
    config.fusion_embedding_dim_weather = fusion_config['embedding_dim_weather']
    config.fusion_embedding_dim_hfl = fusion_config['embedding_dim_hfl']
    config.fusion_hidden_dim = fusion_config['hidden_dim']
    config.fusion_output_dim = fusion_config['output_dim']
    config.fusion_dropout = fusion_config['dropout']

    # === 數據配置 ===
    data_config = config_dict['data']
    config.data_path = data_config['data_path']
    config.weather_csv = data_config['weather_csv']
    config.hfl_csv_pattern = data_config['hfl_csv_pattern']
    config.input_length = data_config['input_length']
    config.output_length = data_config['output_length']
    config.weather_features = data_config['weather_features']
    config.hfl_features = data_config['hfl_features']
    config.target = data_config['target']
    config.train_ratio = data_config['train_ratio']
    config.val_ratio = data_config['val_ratio']
    config.test_ratio = data_config['test_ratio']

    # === 訓練配置 ===
    training_config = config_dict['training']
    config.early_stopping_patience = training_config['early_stopping']['patience']
    config.early_stopping_min_delta = training_config['early_stopping']['min_delta']

    # === 設備配置 ===
    device_config = config_dict['device']
    config.device = get_device(device_config['type'])

    # === 日誌配置 ===
    logging_config = config_dict['logging']
    config.save_model = logging_config['save_model']
    config.model_save_path = logging_config['model_save_path']
    config.log_level = logging_config['log_level']

    # === Per-FedAvg 個性化配置 ===
    personalization_config = config_dict.get('personalization', {})
    config.use_personalized_hfl = personalization_config.get('use_personalized_hfl', False)
    config.hfl_model_path = personalization_config.get('hfl_model_path', None)
    config.personalization_steps = personalization_config.get('personalization_steps', 10)
    config.adaptation_lr = personalization_config.get('adaptation_lr', 0.01)

    # === 可視化配置 ===
    viz_config = config_dict['visualization']
    config.save_plots = viz_config['save_plots']
    config.plot_path = viz_config['plot_path']
    config.save_only = viz_config['save_only']

    # 創建必要的目錄
    os.makedirs(config.model_save_path, exist_ok=True)
    os.makedirs(config.plot_path, exist_ok=True)

    return config

def get_device(device_type):
    """
    智能設備選擇函數 - VFL 場景優化

    **VFL 設備考慮**:
    - Server (Weather Model): 通常在雲端，可能使用 GPU
    - Clients (HFL + Fusion Model): 可能在邊緣設備，支持多種設備類型

    Args:
        device_type: 設備類型字符串 ("auto", "cuda", "mps", "cpu")

    Returns:
        torch.device: PyTorch設備對象
    """
    if device_type == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    elif device_type in ["cuda", "mps", "cpu"]:
        if device_type == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif device_type == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        print(f"Invalid device type: {device_type}, using CPU")
        return torch.device("cpu")

if __name__ == "__main__":
    # 測試配置加載
    config = load_config()
    print(f"Device: {config.device}")
    print(f"Algorithm: {config.algorithm}")
    print(f"Global rounds: {config.K}")
    print(f"Number of clients: {config.num_users}")
    print(f"Weather features: {len(config.weather_features)}")
    print(f"HFL features: {len(config.hfl_features)}")
    print(f"Phase 1 rounds: {config.phase1_rounds}")
    print(f"Phase 2 rounds: {config.phase2_rounds}")

"""聯邦學習測試腳本 - 評估訓練好的模型性能

這個腳本提供了完整的模型測試和評估功能，支持：
1. 標準 FedAvg 測試：直接在測試集上評估全局模型
2. Per-FedAvg 個性化測試：使用 validation set 進行個性化適應，在 test set 上評估
3. 多種性能指標計算：MSE、MAE、RMSE、R²、sMAPE
4. 豐富的可視化功能：預測對比圖、時序圖、注意力權重圖等

主要特點：
- 自動識別算法類型（FedAvg vs Per-FedAvg）
- 正確的數據分割：validation set 用於適應，test set 用於評估
- 支持多客戶端批量測試
- 生成詳細的性能報告和可視化結果
- 保存測試結果到 CSV 文件便於後續分析

Per-FedAvg 評估流程：
- 使用 validation set 對全局模型進行個性化適應
- 在完全未見過的 test set 上評估適應後的性能
- 確保評估結果的公正性和可靠性
- 與 FedAvg 使用完全相同的 targets，確保可比較性

數據一致性保證：
- 兩種算法的 targets 來自相同的 trainer.test_model() 方法
- 完整的時間序列可視化，不進行隨機採樣
- 確保評估結果的可靠性和公平比較
"""

import argparse
import glob
import os
import pickle
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

from config import load_config
from train import create_weather_sequences, load_weather_data
from src.Client import VFLClient
from src.DataLoader import SequenceCSVDataset
from src.Model import TransformerModel
from src.Trainer import FederatedTrainer


def load_scaler(path: str, name: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} 標準化器不存在: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def create_sequences(weather, hfl, targets, seq_len):
    X_w, X_h, y = [], [], []
    # 修正：weather 已經是 sequences，不需要再減 seq_len
    # limit 應基於 hfl/targets 的長度，與 PerFedAvg 對齊
    # PerFedAvg: total_sequences = len(data) - (input_len + output_len) + 1 = len(data) - 96
    limit = min(len(weather), len(hfl) - seq_len, len(targets) - seq_len)
    if limit <= 0:
        return np.array([]), np.array([]), np.array([])
    for i in range(limit):
        X_w.append(weather[i])
        X_h.append(hfl[i : i + seq_len])
        y.append(targets[i + seq_len])
    return np.array(X_w), np.array(X_h), np.array(y)


def build_personalized_hfl_models(
    config, client_csv_files: List[str]
) -> Dict[str, Dict[str, torch.Tensor]]:
    """載入訓練時已保存的個性化 HFL 模型

    從 config.model_save_path 目錄中載入每個 client 的
    {client_name}_hfl_personalized.pth 檔案，而非重新執行個性化適應。

    Raises:
        FileNotFoundError: 若找不到某個 client 的個性化模型檔案
    """
    client_states = {}
    if not (config.use_personalized_hfl and config.hfl_model_path):
        return client_states

    for csv_file in client_csv_files:
        client_name = os.path.basename(csv_file).replace(".csv", "")
        model_path = os.path.join(
            config.model_save_path, f"{client_name}_hfl_personalized.pth"
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"找不到 {client_name} 的個性化 HFL 模型: {model_path}\n"
                f"請先執行 train.py 以生成個性化 HFL 模型檔案。"
            )
        state = torch.load(model_path, map_location=config.device)
        client_states[client_name] = state
        print(f"  V 載入個性化 HFL 模型: {client_name}")

    print(f"  V 成功載入 {len(client_states)} 個個性化 HFL 模型")
    return client_states


def prepare_client_test_loader(
    csv_file: str,
    weather_sequences: np.ndarray,
    hfl_scaler,
    target_scaler,
    config,
) -> Tuple[DataLoader, int]:
    client_df = pd.read_csv(csv_file)
    target_col = config.target[0] if config.target[0] in client_df.columns else "Consumption_Total"
    hfl_data = client_df[config.hfl_features].values
    target_data = client_df[target_col].values

    hfl_scaled = hfl_scaler.transform(hfl_data)
    target_scaled = target_scaler.transform(target_data.reshape(-1, 1)).flatten()

    # 不截斷 hfl/target，讓 create_sequences 內部的 limit 正確計算
    # limit = min(len(weather_sequences), len(hfl)-96, len(target)-96) 與 PerFedAvg 對齊
    if len(hfl_scaled) <= config.input_length:
        return None, 0

    X_w, X_h, y = create_sequences(
        weather_sequences,
        hfl_scaled,
        target_scaled,
        config.input_length,
    )
    total = len(X_w)
    if total == 0:
        return None, 0

    train_size = int(config.train_ratio * total)
    val_size = int(config.val_ratio * total)
    test_start = train_size + val_size
    if test_start >= total:
        return None, 0

    X_w_test = torch.FloatTensor(X_w[test_start:])
    X_h_test = torch.FloatTensor(X_h[test_start:])
    y_test = torch.FloatTensor(y[test_start:]).unsqueeze(1)

    dataset = TensorDataset(X_w_test, X_h_test, y_test)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    return loader, len(dataset)


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(targets, preds)
    mae = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, preds) if np.var(targets) > 0 else 0.0
    smape = (
        np.mean(
            2.0 * np.abs(preds - targets) / (np.abs(preds) + np.abs(targets) + 1e-8)
        )
        * 100
    )
    return {
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "smape": float(smape),
    }


def evaluate_client(
    client: VFLClient,
    test_loader: DataLoader,
    weather_model: TransformerModel,
    target_scaler,
    device: torch.device,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    weather_batches = []
    for weather_batch, _, _ in test_loader:
        weather_batches.append(weather_batch.to(device))
    if not weather_batches:
        return None, None, None

    with torch.no_grad():
        weather_tensor = torch.cat(weather_batches, dim=0)
        weather_embeddings = weather_model.forward_embedding(weather_tensor)

    avg_loss, preds_tensor, targets_tensor = client.local_evaluate(
        test_loader, weather_embeddings, return_outputs=True
    )

    preds = target_scaler.inverse_transform(preds_tensor.numpy()).flatten()
    targets = target_scaler.inverse_transform(targets_tensor.numpy()).flatten()
    metrics = compute_metrics(preds, targets)
    metrics["scaled_loss"] = float(avg_loss)
    metrics["num_samples"] = len(preds)
    return metrics, preds, targets


def plot_predictions_vs_targets(predictions, targets, client_name, save_path):
    plt.figure(figsize=(10, 8))
    plt.scatter(targets, predictions, alpha=0.6, s=20)
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'Predictions vs True Values - {client_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    mse = mean_squared_error(targets, predictions)
    r2 = r2_score(targets, predictions) if np.var(targets) > 0 else 0.0
    plt.text(
        0.05,
        0.95,
        f'MSE: {mse:.6f}\nR²: {r2:.4f}',
        transform=plt.gca().transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
    )
    plt.tight_layout()
    plot_path = os.path.join(save_path, f'{client_name}_predictions_vs_targets.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved prediction plot to {plot_path}")


def plot_time_series_comparison(predictions, targets, client_name, save_path):
    n_samples = len(predictions)
    fig_width = max(15, min(30, n_samples // 100))
    plt.figure(figsize=(fig_width, 6))
    plt.plot(range(n_samples), targets, label='True Values', linewidth=1.0, alpha=0.8)
    plt.plot(range(n_samples), predictions, label='Predictions', linewidth=1.0, alpha=0.8)
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.title(f'Complete Time Series Prediction Comparison - {client_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    mse = np.mean((targets - predictions) ** 2)
    mae = np.mean(np.abs(targets - predictions))
    stats_text = f'Total samples: {n_samples}\nMSE: {mse:.6f}\nMAE: {mae:.6f}'
    plt.text(
        0.98,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
    )
    plt.tight_layout()
    plot_path = os.path.join(save_path, f'{client_name}_time_series_complete.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved complete time series plot to {plot_path} (showing all {n_samples} points)")


def plot_attention_weights(model, dataset, client_name, save_path, config, num_samples=5):
    show_attention = getattr(config, "show_attention", False)
    if not show_attention:
        return
    trainer = FederatedTrainer(model, config, config.device)
    _, _, test_dataset = trainer.split_dataset(dataset)
    _, _, test_loader = trainer.create_data_loaders(None, None, test_dataset)
    model.eval()
    sample_count = 0
    with torch.no_grad():
        for data, _ in test_loader:
            if sample_count >= num_samples:
                break
            data = data.to(config.device).float()
            attention_weights = model.get_attention_weights(data)
            for i in range(min(data.size(0), num_samples - sample_count)):
                plt.figure(figsize=(12, 4))
                attn = attention_weights[i].cpu().numpy().squeeze()
                plt.subplot(1, 2, 1)
                sns.heatmap(attn.reshape(1, -1), cmap='Blues', cbar=True)
                plt.title(f'Attention Weights - Sample {sample_count + 1}')
                plt.xlabel('Sequence Position')
                plt.subplot(1, 2, 2)
                input_seq = data[i].cpu().numpy().squeeze()
                positions = range(len(input_seq))
                plt.plot(positions, input_seq, 'b-', label='Input Sequence', alpha=0.7)
                plt.plot(positions, attn * np.max(input_seq), 'r-', label='Attention Weights (scaled)', alpha=0.8)
                plt.xlabel('Sequence Position')
                plt.ylabel('Value')
                plt.title('Input Sequence with Attention')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plot_path = os.path.join(save_path, f'{client_name}_attention_sample_{sample_count + 1}.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                sample_count += 1
                if sample_count >= num_samples:
                    break
    print(f"Saved attention plots for {client_name}")


def sMAPE(predictions, targets):
    if np.allclose(targets, 0) and np.allclose(predictions, 0):
        return 0.0
    return (
        np.mean(2.0 * np.abs(predictions - targets) / (np.abs(predictions) + np.abs(targets) + 1e-8))
        * 100
    )


def plot_smape_comparison(predictions, targets, client_name, save_path):
    smape_values = []
    for pred, target in zip(predictions, targets):
        smape_values.append(sMAPE(pred, target))
    smape_values = np.array(smape_values)
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(range(len(smape_values)), smape_values, 'b-', alpha=0.7, linewidth=1)
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('sMAPE (%)')
    ax1.set_title(f'sMAPE Over Samples - {client_name}')
    ax1.grid(True, alpha=0.3)
    mean_smape = np.mean(smape_values)
    median_smape = np.median(smape_values)
    ax1.axhline(y=mean_smape, color='r', linestyle='--', alpha=0.8, label=f'Mean: {mean_smape:.2f}%')
    ax1.axhline(y=median_smape, color='g', linestyle='--', alpha=0.8, label=f'Median: {median_smape:.2f}%')
    ax1.legend()
    ax2.hist(smape_values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('sMAPE (%)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'sMAPE Distribution - {client_name}')
    ax2.grid(True, alpha=0.3)
    stats_text = (
        f'Mean: {mean_smape:.2f}%\n'
        f'Median: {median_smape:.2f}%\n'
        f'Std: {np.std(smape_values):.2f}%\n'
        f'Min: {np.min(smape_values):.2f}%\n'
        f'Max: {np.max(smape_values):.2f}%'
    )
    ax2.text(
        0.75,
        0.95,
        stats_text,
        transform=ax2.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
    )
    plt.tight_layout()
    plot_path = os.path.join(save_path, f'{client_name}_smape_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved sMAPE comparison plot to {plot_path}")


def plot_overall_performance(results: Dict[str, Dict], save_path: str):
    client_names = list(results.keys())
    metrics = ['mse', 'mae', 'rmse', 'r2']
    _, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    for i, metric in enumerate(metrics):
        values = [results[client][metric] for client in client_names]
        axes[i].bar(range(len(client_names)), values)
        axes[i].set_title(f'{metric.upper()} per Client')
        axes[i].set_xlabel('Client')
        axes[i].set_ylabel(metric.upper())
        if len(client_names) <= 20:
            axes[i].set_xticks(range(len(client_names)))
            axes[i].set_xticklabels(client_names, rotation=45, ha='right')
        else:
            step = max(1, len(client_names) // 10)
            axes[i].set_xticks(range(0, len(client_names), step))
            axes[i].set_xticklabels(
                [client_names[j] for j in range(0, len(client_names), step)], rotation=45, ha='right'
            )
        axes[i].grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(save_path, 'overall_performance.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved overall performance plot to {plot_path}")


def save_results_to_csv(results: Dict[str, Dict], save_path: str):
    rows = []
    for client_name, metrics in results.items():
        row = {'client': client_name}
        row.update({k: v for k, v in metrics.items() if k not in ['predictions', 'targets']})
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(save_path, 'vfl_test_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n測試結果已儲存至 {csv_path}")
    print("\nTest Results Summary:")
    print("=" * 50)
    for metric in ['mse', 'mae', 'rmse', 'r2', 'smape']:
        if metric in df.columns:
            mean_val = df[metric].mean()
            std_val = df[metric].std()
            print(f"{metric.upper()}: {mean_val:.6f} ± {std_val:.6f}")


def main():
    parser = argparse.ArgumentParser(description="VFL Testing Script")
    parser.add_argument("--config", default="config.yaml", help="配置檔路徑")
    args = parser.parse_args()

    config = load_config(args.config)
    device = config.device

    csv_pattern = os.path.join(config.data_path, config.hfl_csv_pattern + ".csv")
    client_csv_files = [
        f for f in sorted(glob.glob(csv_pattern)) if f.endswith(".csv") and ".pkl" not in f
    ][: config.num_users]
    if not client_csv_files:
        raise FileNotFoundError(f"找不到客戶端資料：{csv_pattern}")

    weather_data_scaled, _ = load_weather_data(config)
    first_client_df = pd.read_csv(client_csv_files[0])
    total_sequences = len(first_client_df) - config.input_length
    weather_sequences = create_weather_sequences(
        weather_data_scaled, config.input_length, total_sequences
    )

    processed_dir = os.path.join(config.data_path, 'processed')

    print("\n初始化個性化 HFL 模型...")
    client_hfl_states = build_personalized_hfl_models(config, client_csv_files)

    weather_model = TransformerModel(
        feature_dim=config.weather_feature_dim,
        d_model=config.weather_d_model,
        nhead=config.weather_nhead,
        num_layers=config.weather_num_layers,
        output_dim=config.weather_output_dim,
        max_seq_length=config.weather_max_seq_length,
        dropout=config.weather_dropout,
    ).to(device)
    # 優先載入最佳模型，若不存在則回退至最終模型
    best_weather_ckpt = os.path.join(config.model_save_path, "best_weather_model.pth")
    final_weather_ckpt = os.path.join(config.model_save_path, "final_weather_model.pth")
    if os.path.exists(best_weather_ckpt):
        weather_ckpt = best_weather_ckpt
        print(f"  V 載入最佳 Weather Model: {weather_ckpt}")
    elif os.path.exists(final_weather_ckpt):
        weather_ckpt = final_weather_ckpt
        print(f"  ! 最佳模型不存在，載入最終 Weather Model: {weather_ckpt}")
    else:
        raise FileNotFoundError(f"找不到 Weather Model 權重：{best_weather_ckpt} 或 {final_weather_ckpt}")
    weather_model.load_state_dict(torch.load(weather_ckpt, map_location=device))
    weather_model.eval()

    results_dict: Dict[str, Dict] = {}
    for csv_file in client_csv_files:
        client_name = os.path.basename(csv_file).replace(".csv", "")
        print(f"\n=== 測試 {client_name} ===")

        # 載入 per-client scalers (與 PerFedAvg 前處理一致)
        hfl_scaler = load_scaler(
            os.path.join(processed_dir, f"{client_name}_feature_scaler.pkl"),
            f"{client_name} HFL"
        )
        target_scaler = load_scaler(
            os.path.join(processed_dir, f"{client_name}_target_scaler.pkl"),
            f"{client_name} Target"
        )

        test_loader, test_size = prepare_client_test_loader(
            csv_file, weather_sequences, hfl_scaler, target_scaler, config
        )
        if test_loader is None or test_size == 0:
            print("  ✗ 測試集樣本不足，略過")
            continue

        client_state = client_hfl_states.get(client_name)
        client = VFLClient(
            client_id=client_name,
            config=config,
            device=device,
            hfl_model_state_dict=client_state,
        )

        # 優先載入最佳 Fusion Model，若不存在則回退至最終模型
        best_fusion_path = os.path.join(config.model_save_path, f"best_{client_name}_fusion_model.pth")
        final_fusion_path = os.path.join(config.model_save_path, f"{client_name}_fusion_model.pth")
        if os.path.exists(best_fusion_path):
            fusion_path = best_fusion_path
        elif os.path.exists(final_fusion_path):
            fusion_path = final_fusion_path
            print(f"  ! {client_name}: 最佳 Fusion Model 不存在，使用最終模型")
        else:
            print(f"  ✗ 找不到 Fusion Model：{best_fusion_path} 或 {final_fusion_path}")
            continue
        client.load_fusion_model(fusion_path)
        client.fusion_model.eval()

        metrics, preds, targets = evaluate_client(
            client, test_loader, weather_model, target_scaler, device
        )
        if metrics is None:
            print("  ✗ 評估失敗")
            continue

        metrics_entry = dict(metrics)
        metrics_entry["predictions"] = preds
        metrics_entry["targets"] = targets
        results_dict[client_name] = metrics_entry

        print(
            f"  ✓ samples={metrics['num_samples']}, "
            f"MSE={metrics['mse']:.6f}, MAE={metrics['mae']:.6f}, "
            f"RMSE={metrics['rmse']:.6f}, R2={metrics['r2']:.4f}, "
            f"sMAPE={metrics['smape']:.2f}%"
        )

        if config.save_plots:
            plot_predictions_vs_targets(preds, targets, client_name, config.plot_path)
            plot_time_series_comparison(preds, targets, client_name, config.plot_path)
            plot_smape_comparison(preds, targets, client_name, config.plot_path)
            if len(results_dict) <= 5:
                dataset = SequenceCSVDataset(
                    csv_path=os.path.dirname(csv_file),
                    csv_name=client_name,
                    input_len=config.input_length,
                    output_len=config.output_length,
                    features=getattr(config, "features", config.hfl_features),
                    target=config.target,
                    save_path=os.path.dirname(csv_file),
                    train_ratio=config.train_ratio,
                    val_ratio=config.val_ratio,
                    split_type="time_based",
                    fit_scalers=False,
                )
                plot_attention_weights(client.hfl_model, dataset, client_name, config.plot_path, config)

    if not results_dict:
        print("未產生任何測試結果，請確認模型與資料。")
        return

    if config.save_plots:
        plot_overall_performance(results_dict, config.plot_path)

    save_results_to_csv(results_dict, config.model_save_path)


if __name__ == "__main__":
    main()


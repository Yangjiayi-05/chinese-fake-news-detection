import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from config import FIGURES_DIR

# 使用纯英文字体，避免 Kaggle/Linux 环境下的中文乱码报错
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False


def plot_training_curve(log_path, save_name):
    """Plot training loss and F1 curve from log file."""
    epochs, train_loss, dev_loss, train_acc, dev_acc, dev_f1 = [], [], [], [], [], []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('Epoch'):
                parts = line.split('|')
                ep = int(parts[0].split()[1])
                tl = float(parts[1].split('Train Loss:')[1].split()[0])
                # 兼容新旧日志格式
                if 'Train Acc:' in parts[1]:
                    ta = float(parts[1].split('Train Acc:')[1].split()[0])
                else:
                    ta = 0.0
                
                df = float(parts[2].split('F1:')[1].split()[0])
                
                epochs.append(ep)
                train_loss.append(tl)
                train_acc.append(ta)
                dev_f1.append(df)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss Curve
    axes[0].plot(epochs, train_loss, 'b-o', label='Train Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Curve')
    axes[0].legend()
    axes[0].grid(True)

    # F1 Curve
    axes[1].plot(epochs, dev_f1, 'r-o', label='Dev F1')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('Dev F1 Curve')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, save_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_name='confusion_matrix_en.png', title='Confusion Matrix'):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    save_path = os.path.join(FIGURES_DIR, save_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_attention_heatmap(tokens, weights, save_name='attention_heatmap.png',
                           pred_label=None, true_label=None, confidence=None):
    """Plot attention weights for a single sample."""
    plt.figure(figsize=(max(8, len(tokens) * 0.6), 2.5))
    sns.heatmap([weights], xticklabels=tokens, yticklabels=['Attention'],
                cmap='YlOrRd', cbar=True, annot=False)
    plt.xticks(rotation=45, ha='right')

    if pred_label is not None and true_label is not None:
        correct = " | Correct: Yes" if pred_label == true_label else " | Correct: No"
        title = f"Prediction: {pred_label}"
        if confidence is not None:
            title += f" (Conf: {confidence:.4f})"
        title += f" | True: {true_label}{correct}"
    else:
        title = 'Attention Heatmap'

    plt.title(title, fontsize=11)
    save_path = os.path.join(FIGURES_DIR, save_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_model_comparison(results, save_name='model_comparison_avg.png'):
    """Plot bar chart comparing different models."""
    models = list(results.keys())
    f1s = list(results.values())
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, f1s, color=colors, width=0.5)
    for bar, val in zip(bars, f1s):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{val:.4f}', ha='center', fontsize=11)
    plt.ylabel('Average F1 Score')
    plt.title('Model Comparison & Ablation Study')
    plt.ylim(0.7, 0.95)
    plt.grid(axis='y', alpha=0.3)
    
    save_path = os.path.join(FIGURES_DIR, save_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

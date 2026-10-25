import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from config import FIGURES_DIR

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_training_curve(log_path, save_name):
    """从日志解析Loss/F1，画训练曲线"""
    epochs, train_loss, dev_loss, train_acc, dev_acc, dev_f1 = [], [], [], [], [], []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('Epoch'):
                parts = line.split('|')
                ep = int(parts[0].split()[1])
                tl = float(parts[1].split('Train Loss:')[1].split()[0])
                ta = float(parts[1].split('Train Acc:')[1].split()[0])
                dl = float(parts[2].split('Dev Loss:')[1].split()[0])
                da = float(parts[2].split('Dev Acc:')[1].split()[0])
                df = float(parts[2].split('F1:')[1].split()[0])
                epochs.append(ep)
                train_loss.append(tl)
                dev_loss.append(dl)
                train_acc.append(ta)
                dev_acc.append(da)
                dev_f1.append(df)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, train_loss, 'b-o', label='Train Loss')
    axes[0].plot(epochs, dev_loss, 'r-o', label='Dev Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curve')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, train_acc, 'b-o', label='Train Acc')
    axes[1].plot(epochs, dev_acc, 'g-o', label='Dev Acc')
    axes[1].plot(epochs, dev_f1, 'r-o', label='Dev F1')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Accuracy / F1 Curve')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, save_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存：{save_path}")


def plot_model_comparison(results, save_name='model_comparison_avg.png'):
    """results = {'TextCNN': 0.8167, 'BiLSTM': 0.7978, 'BiLSTM+Attention': 0.8259}"""
    models = list(results.keys())
    f1s = list(results.values())
    colors = ['#4C72B0', '#DD8452', '#55A868']

    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, f1s, color=colors, width=0.5)
    for bar, val in zip(bars, f1s):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{val:.4f}', ha='center', fontsize=11)
    plt.ylabel('Average F1 Score')
    plt.title('Model Comparison (Average of 3 Seeds)')
    plt.ylim(0.7, 0.9)
    plt.grid(axis='y', alpha=0.3)
    save_path = os.path.join(FIGURES_DIR, save_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存：{save_path}")


def plot_stability_boxplot(data, save_name='stability_boxplot.png'):
    """data = {'TextCNN': [0.8143, 0.8378, 0.7981],
               'BiLSTM': [0.8243, 0.8280, 0.7410],
               'BiLSTM+Attention': [0.8055, 0.8256, 0.8466]}"""
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data.values(), labels=data.keys(), patch_artist=True,
                    boxprops=dict(facecolor='lightblue'))
    ax.set_ylabel('F1 Score')
    ax.set_title('Model Stability Across 3 Seeds')
    ax.grid(axis='y', alpha=0.3)
    save_path = os.path.join(FIGURES_DIR, save_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存：{save_path}")


def plot_confusion_matrix(y_true, y_pred, save_name='confusion_matrix.png',
                          title='Confusion Matrix'):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['虚假', '真实'], yticklabels=['虚假', '真实'])
    plt.xlabel('预测')
    plt.ylabel('真实')
    plt.title(title)
    save_path = os.path.join(FIGURES_DIR, save_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存：{save_path}")


def plot_attention_heatmap(tokens, weights, save_name='attention_heatmap.png',
                           pred_label=None, true_label=None, confidence=None):
    """画单条样本的注意力热力图，标题显示预测结果和真实标签"""
    plt.figure(figsize=(max(8, len(tokens) * 0.6), 2.5))
    sns.heatmap([weights], xticklabels=tokens, yticklabels=['Attention'],
                cmap='YlOrRd', cbar=True, annot=False)
    plt.xticks(rotation=45, ha='right')

    if pred_label is not None and true_label is not None:
        correct = " | 判定结果：√" if pred_label == true_label else " | 判定结果：×"
        title = f"预测：{pred_label}"
        if confidence is not None:
            title += f"（置信度 {confidence:.4f}）"
        title += f" | 真实：{true_label} {correct}"
    else:
        title = 'Attention Heatmap'

    plt.title(title, fontsize=11)
    save_path = os.path.join(FIGURES_DIR, save_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存：{save_path}")
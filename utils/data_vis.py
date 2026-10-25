import json
import matplotlib.pyplot as plt
import os
from collections import Counter
from config import FIGURES_DIR

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_label_distribution(raw_dir, save_name='label_distribution.png'):
    """画标签分布饼图"""
    with open(os.path.join(raw_dir, 'train.json'), 'r', encoding='utf-8') as f:
        data = json.load(f)

    labels = [item['label'] for item in data if item['label'] != 2]
    counter = Counter(labels)

    real = counter.get(0, 0)
    fake = counter.get(1, 0)

    plt.figure(figsize=(6, 6))
    plt.pie([fake, real], labels=['虚假', '真实'], autopct='%1.1f%%',
            colors=['#FF6B6B', '#4ECDC4'], startangle=90,
            textprops={'fontsize': 13})
    plt.title('CHEF训练集标签分布', fontsize=14)
    save_path = os.path.join(FIGURES_DIR, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存：{save_path}")


def plot_length_distribution(raw_dir, save_name='length_distribution.png'):
    """画文本长度分布直方图"""
    with open(os.path.join(raw_dir, 'train.json'), 'r', encoding='utf-8') as f:
        data = json.load(f)

    lengths = [len(item['claim']) for item in data if item['label'] != 2]
    avg_len = sum(lengths) / len(lengths)

    plt.figure(figsize=(10, 5))
    plt.hist(lengths, bins=50, color='#4C72B0', edgecolor='white')
    plt.xlabel('文本长度（字）', fontsize=12)
    plt.ylabel('样本数', fontsize=12)
    plt.title(f'CHEF训练集文本长度分布（平均{avg_len:.1f}字）', fontsize=13)
    plt.axvline(avg_len, color='red', linestyle='--',
                label=f'平均值 {avg_len:.1f}')
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    save_path = os.path.join(FIGURES_DIR, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存：{save_path}")
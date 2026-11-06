# -*- coding: utf-8 -*-
"""出图：模型对比（柱 + 误差棒 + 种子散点）与训练曲线。

两张图：
    figures/fig_model_comparison.png  四个主模型的 Macro-F1 均值 ± 标准差，叠加各种子散点
    figures/fig_training_curves.png   四个主模型的验证集 Macro-F1 随 epoch 变化（三种子）

图内用英文标签，避免 Kaggle 上缺中文字体导致方块；论文里可用 Word/PPT 二次标注。

用法（项目根目录）：
    python scripts/plot_results.py
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402

NAMES = {
    'fasttext': 'FastText',
    'bilstm': 'BiLSTM',
    'bilstm_avgpool': 'BiLSTM+AvgPool',
    'bilstm_attention': 'BiLSTM+Attention',
    'bilstm_attention_nomask': 'Attention (no mask)',
}


def seed_scores(model):
    """返回该模型每个种子的测试集 Macro-F1。"""
    scores = []
    for seed in config.SEEDS:
        tag = f'{model}_seed{seed}'
        y_path = os.path.join(config.CHECKPOINT_DIR, f'test_labels_{tag}.npy')
        p_path = os.path.join(config.CHECKPOINT_DIR, f'test_preds_{tag}.npy')
        if not (os.path.exists(y_path) and os.path.exists(p_path)):
            return None
        scores.append(f1_score(np.load(y_path), np.load(p_path), average='macro'))
    return np.array(scores)


def plot_comparison(models, out_path):
    data, labels = [], []
    for m in models:
        s = seed_scores(m)
        if s is not None:
            data.append(s)
            labels.append(NAMES.get(m, m))
    if not data:
        print('没有可用的预测文件，跳过对比图')
        return

    means = [d.mean() for d in data]
    stds = [d.std(ddof=1) if len(d) > 1 else 0.0 for d in data]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(data))
    ax.bar(x, means, yerr=stds, capsize=6, width=0.55,
           color=['#9aa5b1', '#7fb069', '#f2c14e', '#e4572e', '#8c6bb1'][:len(data)],
           edgecolor='black', alpha=0.9, label='Mean ± Std')

    rng = np.random.default_rng(0)
    for i, d in enumerate(data):
        jitter = rng.uniform(-0.08, 0.08, len(d))
        ax.scatter(np.full(len(d), x[i]) + jitter, d, s=36, color='black',
                   zorder=3, alpha=0.75, label='Individual seeds' if i == 0 else None)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Macro-F1 (test set)', fontsize=12)
    ax.set_title(f'Model comparison over {len(config.SEEDS)} seeds', fontsize=13)
    lo = min(min(d) for d in data)
    hi = max(max(d) for d in data)
    ax.set_ylim(lo - 0.03, hi + 0.03)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f'已保存 {out_path}')


def plot_curves(models, out_path):
    available = [m for m in models
                 if os.path.exists(os.path.join(config.LOGS_DIR,
                                                f'curve_{m}_seed{config.SEEDS[0]}.csv'))]
    if not available:
        print('没有可用的训练曲线文件，跳过曲线图')
        return

    import pandas as pd
    n = len(available)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)

    for idx, model in enumerate(available):
        ax = axes[idx // cols][idx % cols]
        curves = []
        for seed in config.SEEDS:
            path = os.path.join(config.LOGS_DIR, f'curve_{model}_seed{seed}.csv')
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            ax.plot(df['epoch'], df['dev_macro_f1'], linewidth=1.0,
                    alpha=0.55, label=f'seed {seed}')
            curves.append(df['dev_macro_f1'].to_numpy())
        if curves:
            min_len = min(len(c) for c in curves)
            mean_curve = np.mean([c[:min_len] for c in curves], axis=0)
            ax.plot(np.arange(1, min_len + 1), mean_curve, linewidth=2.4,
                    color='black', label='mean')
        ax.set_title(NAMES.get(model, model), fontsize=12)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Dev Macro-F1')
        ax.grid(linestyle='--', alpha=0.4)
        ax.legend(fontsize=8)

    for k in range(len(available), rows * cols):
        axes[k // cols][k % cols].axis('off')

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f'已保存 {out_path}')


def main():
    config.ensure_dir(config.FIGURES_DIR)
    models = config.MAIN_MODEL_LIST
    plot_comparison(models, os.path.join(config.FIGURES_DIR, 'fig_model_comparison.png'))
    plot_curves(models, os.path.join(config.FIGURES_DIR, 'fig_training_curves.png'))


if __name__ == '__main__':
    main()

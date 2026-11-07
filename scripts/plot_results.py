# -*- coding: utf-8 -*-
"""出图：三版箱线图 + 训练曲线 + 混淆矩阵 + 消融曲线。

产物（figures/）：
    fig1_box_main.png        主对比箱线图（2 基线 + 核心）
    fig2_box_all.png         全部配置箱线图
    fig3_box_ablation.png    消融箱线图（核心 + 3 变体）
    fig4_curves_main.png     主实验训练曲线（3 模型 × 3 种子）
    fig5_confusion.png       混淆矩阵（3 模型，3 种子累加）
    fig7_curves_ablation.png 消融训练曲线（4 条均值线）

图内用英文标签，避免 Kaggle 缺中文字体。
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from utils.naming import variant_suffix  # noqa: E402

MAIN = [('fasttext', 'FastText'),
        ('bilstm', 'BiLSTM'),
        ('bilstm_attention', 'BiLSTM+Attention')]

ABL = [('bilstm_attention', 'Full model', ''),
       ('bilstm_attention', 'w/o Mask', variant_suffix('mask')),
       ('bilstm_attention', 'With Class Weight', variant_suffix('cw')),
       ('bilstm_attention', 'w/o Word2Vec', variant_suffix('w2v'))]

# 主实验后缀：默认空（带类权重那版）；--suffix _nocw 可画无类权重那版
MAIN_SUFFIX = ''


def tag_of(model, seed, suffix=''):
    return f'{model}_seed{seed}{suffix}'


def seed_scores(model, suffix=''):
    """返回该配置每个种子的测试集 Macro-F1。"""
    scores = []
    for seed in config.SEEDS:
        tag = tag_of(model, seed, suffix)
        y = os.path.join(config.CHECKPOINT_DIR, f'test_labels_{tag}.npy')
        p = os.path.join(config.CHECKPOINT_DIR, f'test_preds_{tag}.npy')
        if not (os.path.exists(y) and os.path.exists(p)):
            return None
        scores.append(f1_score(np.load(y), np.load(p), average='macro'))
    return np.array(scores)


def boxplot(groups, title, path, ylabel='Macro-F1 (test set)'):
    data, labels = [], []
    for model, label, suffix in groups:
        s = seed_scores(model, suffix)
        if s is not None:
            data.append(s)
            labels.append(label)
    if not data:
        print(f'跳过 {os.path.basename(path)}：没有可用的预测文件')
        return

    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(data)), 5))
    ax.boxplot(data, widths=0.5, showfliers=False,
               medianprops=dict(color='black', linewidth=2))
    rng = np.random.default_rng(0)
    for i, d in enumerate(data, start=1):
        ax.scatter(i + rng.uniform(-0.08, 0.08, len(d)), d,
                   s=42, color='black', zorder=3, alpha=0.85)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'{title} ({len(config.SEEDS)} seeds)', fontsize=13)
    lo = min(min(d) for d in data)
    hi = max(max(d) for d in data)
    ax.set_ylim(lo - 0.02, hi + 0.02)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print('已保存', os.path.basename(path))


def curves(models, path, title):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), squeeze=False)
    for idx, (model, label) in enumerate(models):
        ax = axes[idx // 2][idx % 2]
        series = []
        for seed in config.SEEDS:
            f = os.path.join(config.LOGS_DIR,
                             f'curve_{tag_of(model, seed, MAIN_SUFFIX)}.csv')
            if not os.path.exists(f):
                continue
            df = pd.read_csv(f)
            ax.plot(df['epoch'], df['dev_macro_f1'], linewidth=1.0,
                    alpha=0.55, label=f'seed {seed}')
            series.append(df['dev_macro_f1'].to_numpy())
        if series:
            m = min(len(s) for s in series)
            ax.plot(np.arange(1, m + 1), np.mean([s[:m] for s in series], axis=0),
                    linewidth=2.4, color='black', label='mean')
        ax.set_title(label, fontsize=11)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Dev Macro-F1')
        ax.grid(linestyle='--', alpha=0.4)
        ax.legend(fontsize=7)
    for k in range(len(models), 4):
        axes[k // 2][k % 2].axis('off')
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print('已保存', os.path.basename(path))


def curves_ablation(path):
    """消融：每个配置一条 3 种子均值曲线。"""
    specs = [('bilstm_attention', '', 'Full model'),
             ('bilstm_attention', variant_suffix('mask'), 'w/o Mask'),
             ('bilstm_attention', variant_suffix('cw'), 'With Class Weight'),
             ('bilstm_attention', variant_suffix('w2v'), 'w/o Word2Vec')]
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, suffix, label in specs:
        series = []
        for seed in config.SEEDS:
            f = os.path.join(config.LOGS_DIR,
                             f'curve_{tag_of(model, seed, suffix)}.csv')
            if not os.path.exists(f):
                continue
            series.append(pd.read_csv(f)['dev_macro_f1'].to_numpy())
        if not series:
            continue
        m = min(len(s) for s in series)
        mean = np.mean([s[:m] for s in series], axis=0)
        ax.plot(np.arange(1, m + 1), mean, linewidth=2, label=label)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Dev Macro-F1')
    ax.set_title('Ablation: mean validation curves over seeds')
    ax.grid(linestyle='--', alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print('已保存', os.path.basename(path))


def confusion(path):
    fig, axes = plt.subplots(1, len(MAIN), figsize=(4.2 * len(MAIN), 4))
    if len(MAIN) == 1:
        axes = [axes]
    for ax, (model, label) in zip(axes, MAIN):
        total = np.zeros((2, 2), dtype=int)
        for seed in config.SEEDS:
            tag = tag_of(model, seed, MAIN_SUFFIX)
            y = os.path.join(config.CHECKPOINT_DIR, f'test_labels_{tag}.npy')
            p = os.path.join(config.CHECKPOINT_DIR, f'test_preds_{tag}.npy')
            if os.path.exists(y) and os.path.exists(p):
                total += confusion_matrix(np.load(y), np.load(p), labels=[0, 1])
        im = ax.imshow(total, cmap='Blues')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(total[i, j]), ha='center', va='center',
                        fontsize=13, color='black')
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['Fake', 'Real']); ax.set_yticklabels(['Fake', 'Real'])
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.set_title(label, fontsize=11)
    fig.suptitle('Confusion matrices (summed over seeds)', fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print('已保存', os.path.basename(path))


def main():
    global MAIN_SUFFIX
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--suffix', default='',
                    help='主实验后缀，如 _nocw 表示画无类权重那版')
    MAIN_SUFFIX = ap.parse_args().suffix

    config.ensure_dir(config.FIGURES_DIR)

    groups_main = [(m, lb, MAIN_SUFFIX) for m, lb in MAIN]
    groups_abl = [(m, lb, sf) for m, lb, sf in ABL]
    groups_all = groups_main + groups_abl[1:]

    boxplot(groups_main, 'Model comparison', 
            os.path.join(config.FIGURES_DIR, 'fig1_box_main.png'))
    boxplot(groups_all, 'All configurations',
            os.path.join(config.FIGURES_DIR, 'fig2_box_all.png'))
    boxplot([(m, lb, sf) for m, lb, sf in groups_abl], 'Ablation study',
            os.path.join(config.FIGURES_DIR, 'fig3_box_ablation.png'))

    curves(MAIN, os.path.join(config.FIGURES_DIR, 'fig4_curves_main.png'),
           'Training curves (3 models x 3 seeds)')
    confusion(os.path.join(config.FIGURES_DIR, 'fig5_confusion.png'))
    curves_ablation(os.path.join(config.FIGURES_DIR, 'fig7_curves_ablation.png'))


if __name__ == '__main__':
    main()

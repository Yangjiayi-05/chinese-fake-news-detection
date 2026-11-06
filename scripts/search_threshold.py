# -*- coding: utf-8 -*-
"""决策阈值搜索与置信区间。

协议：
    在验证集上网格搜索使 Macro-F1 最大的分界线，写回 config.DECISION_THRESHOLD；
    在测试集上只用这个阈值报一次结果，不做二次挑选；
    用 bootstrap 给出测试集 Macro-F1 的 95% 置信区间。

用法（在项目根目录执行）：
    python scripts/search_threshold.py --model bilstm_attention --seed 42
    python scripts/search_threshold.py --all
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import f1_score

# 允许从 scripts/ 子目录执行：把项目根加入模块搜索路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402


def macro_f1_at(y, p_real, threshold):
    """给定分界线时，把"事实概率"切成 0/1 后算 Macro-F1。"""
    return float(f1_score(y, (p_real >= threshold).astype(int), average='macro'))


def search(y, p_real, low=0.05, high=0.95, step=0.005):
    thresholds = np.arange(low, high + 1e-9, step)
    scores = [macro_f1_at(y, p_real, t) for t in thresholds]
    best = int(np.argmax(scores))
    return float(round(thresholds[best], 4)), float(scores[best])


def bootstrap_ci(y, p_real, threshold, n=2000, seed=0):
    """有放回重采样，给出 Macro-F1 的 2.5% 与 97.5% 分位点。"""
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        values.append(macro_f1_at(y[idx], p_real[idx], threshold))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def evaluate_one(tag):
    def load(name):
        path = os.path.join(config.CHECKPOINT_DIR, f'{name}_{tag}.npy')
        if not os.path.exists(path):
            raise FileNotFoundError(f'缺少 {path}，请先跑 main.py')
        return np.load(path)

    dev_y, dev_p = load('dev_labels'), load('dev_probs')
    test_y, test_p = load('test_labels'), load('test_probs')

    threshold, dev_best = search(dev_y, dev_p)
    low, high = bootstrap_ci(test_y, test_p, threshold)

    print(f'\n--- {tag} ---')
    print(f'验证集最优分界线 {threshold:.3f}（验证集 Macro-F1 {dev_best:.4f}）')
    print(f'测试集 @0.50  Macro-F1 {macro_f1_at(test_y, test_p, 0.5):.4f}')
    print(f'测试集 @{threshold:.3f}  Macro-F1 {macro_f1_at(test_y, test_p, threshold):.4f}'
          f'  95% CI [{low:.4f}, {high:.4f}]')

    result = {
        'tag': tag,
        'threshold': threshold,
        'dev_macro_f1': round(dev_best, 4),
        'test_macro_f1_at_0.5': round(macro_f1_at(test_y, test_p, 0.5), 4),
        'test_macro_f1_at_threshold': round(macro_f1_at(test_y, test_p, threshold), 4),
        'ci95': [round(low, 4), round(high, 4)],
    }
    config.ensure_dir(config.RESULTS_DIR)
    with open(os.path.join(config.RESULTS_DIR, f'threshold_{tag}.json'), 'w',
              encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=config.DEFAULT_MODEL)
    parser.add_argument('--seed', type=int, default=config.DEPLOY_SEED)
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    if args.all:
        tags = [f'{m}_seed{s}' for m in config.MODEL_LIST for s in config.SEEDS]
    else:
        tags = [f'{args.model}_seed{args.seed}']

    results = []
    for tag in tags:
        try:
            results.append(evaluate_one(tag))
        except FileNotFoundError as exc:
            print(f'跳过 {tag}：{exc}')

    core = [r for r in results if r['tag'].startswith('bilstm_attention')]
    if core:
        mean_th = float(np.mean([r['threshold'] for r in core]))
        print(f'\nAttention 三个种子的分界线均值 {mean_th:.3f}')
        print(f'建议把 config.DECISION_THRESHOLD 设为 {mean_th:.2f}（取整到 0.05 刻度）')


if __name__ == '__main__':
    main()

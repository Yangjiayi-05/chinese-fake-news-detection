# -*- coding: utf-8 -*-
"""一键跑完 4 个模型 × 3 个种子，并汇总成"均值 ± 标准差"。

用法：
    python run_all.py
    set CFND_PROFILE=tuned && python run_all.py        (Windows 切到调优档位)

产物：
    results/results.csv   逐次实验明细
    results/summary.csv   按模型聚合，论文表格直接用这个
"""
import csv
import os

import numpy as np

import config
from main import run


def summarize(rows):
    """按模型聚合三个指标，std 用样本标准差（ddof=1）。"""
    stats = {}
    for model in config.MODEL_LIST:
        sub = [r for r in rows if r['model'] == model]
        if not sub:
            continue
        stats[model] = {}
        for metric in ('test_acc', 'test_f1', 'test_macro_f1'):
            vals = np.array([r[metric] for r in sub], dtype=float)
            stats[model][metric] = (float(vals.mean()),
                                    float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)
    return stats


def main():
    rows = []
    for model in config.MODEL_LIST:
        for seed in config.SEEDS:
            print(f'\n>>>>>>>>>> {model} | seed={seed} <<<<<<<<<<')
            rows.append(run(model, seed))

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    detail_path = os.path.join(config.RESULTS_DIR, 'results.csv')
    with open(detail_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    stats = summarize(rows)
    summary_path = os.path.join(config.RESULTS_DIR, 'summary.csv')
    with open(summary_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'macro_f1_mean', 'macro_f1_std', 'f1_mean',
                         'f1_std', 'acc_mean', 'acc_std', 'n_runs'])
        for model, m in stats.items():
            writer.writerow([
                model,
                f'{m["test_macro_f1"][0]:.4f}', f'{m["test_macro_f1"][1]:.4f}',
                f'{m["test_f1"][0]:.4f}', f'{m["test_f1"][1]:.4f}',
                f'{m["test_acc"][0]:.4f}', f'{m["test_acc"][1]:.4f}',
                len([r for r in rows if r['model'] == model]),
            ])

    print(f'\n逐次结果：{detail_path}')
    print(f'汇总结果：{summary_path}\n')
    print('=' * 72)
    print(f'{"模型":<20}{"Macro-F1 (mean ± std)":<30}{"正类 F1":<12}')
    print('-' * 72)
    for model, m in stats.items():
        print(f'{model:<20}{m["test_macro_f1"][0]:.4f} ± {m["test_macro_f1"][1]:.4f}'
              f'{"":<12}{m["test_f1"][0]:.4f}')
    print('=' * 72)


if __name__ == '__main__':
    main()

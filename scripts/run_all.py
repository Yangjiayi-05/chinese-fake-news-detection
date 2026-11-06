# -*- coding: utf-8 -*-
"""批量跑实验，按阶段分组。

用法（项目根目录）：
    python scripts/run_all.py --group all        # 主实验 + 消融（18 次）
    python scripts/run_all.py --group main       # 只跑主实验（9 次）
    python scripts/run_all.py --group ablation   # 只跑消融（9 次）
    python scripts/run_all.py --group reg        # 正则档位选择（9 次，只看验证集）

产物：
    results/results.csv    逐次明细
    results/summary.csv    按配置聚合（mean ± std）
    results/reg_summary.csv 仅 --group reg 时生成
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from main import run  # noqa: E402

# 主实验：2 个基线 + 核心
MAIN_RUNS = [(m, dict()) for m in config.MAIN_MODEL_LIST]

# 消融：一条一个变体，都只改一个变量
ABLATION_RUNS = [
    ('bilstm_attention_nomask', dict()),                      # 去掉 Padding Mask
    ('bilstm_attention', dict(use_cw=False, suffix='_nocw')),  # 去掉类别权重
    ('bilstm_attention', dict(use_w2v=False, suffix='_now2v')),  # 去掉词向量初始化
]

# 正则档位选择：只跑核心模型
REG_RUNS = [('bilstm_attention', dict(reg=r, suffix=f'_reg{r}'))
            for r in config.REG_PROFILES]

GROUPS = {
    'main': MAIN_RUNS,
    'ablation': ABLATION_RUNS,
    'reg': REG_RUNS,
    'all': MAIN_RUNS + ABLATION_RUNS,
}

LABELS = {
    'fasttext': 'FastText',
    'bilstm': 'BiLSTM',
    'bilstm_attention': 'BiLSTM+Attention',
    'bilstm_attention_nomask': 'Attention 无 Mask',
}


def label_of(tag):
    if tag.endswith('_nocw'):
        base = tag[:-5].rsplit('_seed', 1)[0]
        return LABELS.get(base, base) + ' 无类权重'
    if tag.endswith('_now2v'):
        return 'Attention 无词向量'
    if '_reg' in tag:
        return 'Attention 正则' + tag.split('_reg')[1]
    base = tag.rsplit('_seed', 1)[0]
    return LABELS.get(base, base)


def summarize(rows, key='label'):
    stats = {}
    for lab in dict.fromkeys(r[key] for r in rows):
        sub = [r for r in rows if r[key] == lab]
        vals = np.array([r['test_macro_f1'] for r in sub], dtype=float)
        devs = np.array([r['dev_macro_f1'] for r in sub], dtype=float)
        stats[lab] = dict(
            n=len(sub),
            macro_mean=float(vals.mean()),
            macro_std=float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            dev_mean=float(devs.mean()),
            dev_std=float(devs.std(ddof=1)) if len(devs) > 1 else 0.0,
            binary_mean=float(np.mean([r['test_f1'] for r in sub])),
            acc_mean=float(np.mean([r['test_acc'] for r in sub])),
        )
    return stats


def write_summary(path, stats, by_dev=False):
    """by_dev=True 时按验证集排序（阶段 0 选超参只能看验证集）。"""
    sort_key = 'dev_mean' if by_dev else 'macro_mean'
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['label', 'n', 'dev_macro_f1_mean', 'dev_macro_f1_std',
                    'macro_f1_mean', 'macro_f1_std', 'binary_f1_mean', 'acc_mean'])
        for lab, s in sorted(stats.items(), key=lambda kv: -kv[1][sort_key]):
            w.writerow([lab, s['n'],
                        f"{s['dev_mean']:.4f}", f"{s['dev_std']:.4f}",
                        f"{s['macro_mean']:.4f}", f"{s['macro_std']:.4f}",
                        f"{s['binary_mean']:.4f}", f"{s['acc_mean']:.4f}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', default='all', choices=list(GROUPS))
    ap.add_argument('--reg', default=config.REG, choices=list(config.REG_PROFILES),
                    help='主实验/消融使用的正则档位')
    ap.add_argument('--models', default='',
                    help='逗号分隔，只跑这些模型（默认全部；对照组只跑核心时用）')
    ap.add_argument('--no-cw', action='store_true',
                    help='本次全部关闭类别权重，结果加 _nocw 后缀，不覆盖带权重的结果')
    args = ap.parse_args()

    runs = GROUPS[args.group]
    if args.models:
        wanted = {m.strip() for m in args.models.split(',') if m.strip()}
        runs = [(m, kw) for m, kw in runs if m in wanted]
    if not runs:
        print('没有匹配的配置，检查 --models')
        return
    if args.no_cw:
        runs = [(m, dict(kw, use_cw=False,
                         suffix=kw.get('suffix', '') + '_nocw'))
                for m, kw in runs if 'use_cw' not in kw]
    print(f'分组：{args.group} | 配置数：{len(runs)} | 次数：{len(runs) * len(config.SEEDS)}')
    print(f'正则档位（主实验/消融）：{args.reg}')

    rows = []
    for model, kw in runs:
        for seed in config.SEEDS:
            if args.group != 'reg':
                kw = dict(kw, reg=args.reg)
            print(f'\n>>>>>>>>>> {model} | seed={seed} | {kw} <<<<<<<<<<')
            rows.append(run(model, seed, **kw))

    config.ensure_dir(config.RESULTS_DIR)
    for r in rows:
        r['label'] = label_of(r['tag'])

    detail = os.path.join(config.RESULTS_DIR, 'results.csv')
    with open(detail, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    stats = summarize(rows)
    is_reg = args.group == 'reg'
    name = 'reg_summary.csv' if is_reg else 'summary.csv'
    out = os.path.join(config.RESULTS_DIR, name)
    write_summary(out, stats, by_dev=is_reg)

    print(f'\n逐次结果：{detail}')
    print(f'汇总结果：{out}\n')
    print('=' * 80)
    if is_reg:
        print('阶段 0：按【验证集】Macro-F1 排序，选最高档位（测试集不参与选择）')
        print(f'{"档位":<18}{"dev Macro-F1":<22}{"test Macro-F1（仅参考）"}')
        print('-' * 80)
        for lab, s in sorted(stats.items(), key=lambda kv: -kv[1]['dev_mean']):
            print(f'{lab:<18}{s["dev_mean"]:.4f} ± {s["dev_std"]:.4f}'
                  f'{"":<6}{s["macro_mean"]:.4f}')
    else:
        print(f'{"配置":<26}{"Macro-F1 (mean ± std)":<26}{"binary F1":<12}')
        print('-' * 80)
        for lab, s in sorted(stats.items(), key=lambda kv: -kv[1]['macro_mean']):
            print(f'{lab:<26}{s["macro_mean"]:.4f} ± {s["macro_std"]:.4f}'
                  f'{"":<8}{s["binary_mean"]:.4f}')
    print('=' * 80)


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""批量实验：全因子探索 + 主实验 + 消融 + 分组对照。

设计说明
--------
核心模型采用 3（正则）× 2（类别权重）× 2（掩码）× 2（词向量）全因子设计，
共 24 个配置；每个配置跑 3 个种子，因此 72 次。另外 2 个基线模型与 2 种划分
对照各 3 次，合计 84 次训练。

同一个配置可能被多个批次引用（例如"基准配置"既是主表的核心行，也是消融与
正则探索的对照），run() 会比对配置签名并自动跳过，因此重复执行不会重复训练。

用法（项目根目录）
------------------
    python scripts/run_all.py --group factorial   # 24 配置 × 3 种子 = 72 次
    python scripts/run_all.py --group main        # 3 模型（基准配置）= 9 次
    python scripts/run_all.py --group ablation    # 3 个消融变体 = 9 次
    python scripts/run_all.py --group reg         # 正则 A/B/C 三档 = 9 次（B 即基准）

分组对照（只看核心模型）：
    set CFND_SPLIT=random && python scripts/run_all.py --group main --models bilstm_attention
    set CFND_SPLIT=time   && python scripts/run_all.py --group main --models bilstm_attention
"""
import argparse
import csv
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from main import run  # noqa: E402
from utils.naming import final_cfg, suffix_of, variant_cfg, variant_suffix  # noqa: E402

CORE = 'bilstm_attention'

# 逐次结果 csv 的固定列：与单次结果 json 的字段对应
DETAIL_FIELDS = ['tag', 'model', 'seed', 'reg', 'use_cw', 'use_w2v', 'use_mask',
                 'best_epoch', 'dev_macro_f1', 'test_acc', 'test_precision',
                 'test_recall', 'test_f1', 'test_macro_f1', 'label']


def base_kwargs():
    """基准配置（= config.FINAL_*）对应的 run() 参数。"""
    c = final_cfg()
    return dict(reg=c['reg'], use_cw=c['cw'], use_mask=c['mask'], use_w2v=c['w2v'])


def factorial_runs():
    """3 × 2 × 2 × 2 = 24 个核心配置，基准配置用空后缀。"""
    runs = []
    for reg in config.REG_PROFILES:
        for cw in (False, True):
            for mask in (True, False):
                for w2v in (True, False):
                    runs.append((CORE, dict(
                        reg=reg, use_cw=cw, use_mask=mask, use_w2v=w2v,
                        suffix=suffix_of(reg, cw, mask, w2v))))
    return runs


# 主实验：两个基线 + 核心，都用基准配置
MAIN_RUNS = [(m, dict(**base_kwargs())) for m in config.MAIN_MODEL_LIST]

# 消融：相对基准各翻转一个因素
ABLATION_RUNS = [
    (CORE, dict(reg=c['reg'], use_cw=c['cw'], use_mask=c['mask'], use_w2v=c['w2v'],
                suffix=variant_suffix(flip)))
    for flip in ('cw', 'mask', 'w2v')
    for c in [variant_cfg(flip)]
]

# 正则探索：三档 A/B/C 完整跑一遍；最终档用空后缀，
# 后续主实验跑到核心模型时会按签名跳过并复用这一组
REG_RUNS = [
    (CORE, dict(reg=r, use_cw=c['cw'], use_mask=c['mask'], use_w2v=c['w2v'],
                suffix=suffix_of(r, c['cw'], c['mask'], c['w2v'])))
    for r in config.REG_PROFILES
    for c in [final_cfg()]
]

GROUPS = {
    'factorial': factorial_runs(),
    'main': MAIN_RUNS,
    'ablation': ABLATION_RUNS,
    'reg': REG_RUNS,
    'all': factorial_runs(),
}

LABELS = {
    'fasttext': 'FastText',
    'bilstm': 'BiLSTM',
    'bilstm_attention': 'BiLSTM+Attention',
    'bilstm_attention_nomask': 'BiLSTM+Attention(no-mask)',
}

FAC_RE = re.compile(r'_r([ABC])_c([01])_m([01])_v([01])$')


def label_of(tag):
    m = FAC_RE.search(tag)
    base = FAC_RE.sub('', tag).rsplit('_seed', 1)[0]
    name = LABELS.get(base, base)
    if m:
        reg, cw, mask, w2v = m.groups()
        return f'{name} {reg}/c{cw}m{mask}v{w2v}'
    return name


def summarize(rows):
    stats = {}
    for lab in dict.fromkeys(r['label'] for r in rows):
        sub = [r for r in rows if r['label'] == lab]
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


def write_summary(path, stats):
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['label', 'n', 'dev_macro_f1_mean', 'dev_macro_f1_std',
                    'macro_f1_mean', 'macro_f1_std', 'binary_f1_mean', 'acc_mean'])
        for lab, s in sorted(stats.items(), key=lambda kv: -kv[1]['macro_mean']):
            w.writerow([lab, s['n'],
                        f"{s['dev_mean']:.4f}", f"{s['dev_std']:.4f}",
                        f"{s['macro_mean']:.4f}", f"{s['macro_std']:.4f}",
                        f"{s['binary_mean']:.4f}", f"{s['acc_mean']:.4f}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', default='factorial', choices=list(GROUPS))
    ap.add_argument('--models', default='',
                    help='逗号分隔，只跑这些模型（默认全部）')
    args = ap.parse_args()

    runs = GROUPS[args.group]
    if args.models:
        wanted = {m.strip() for m in args.models.split(',') if m.strip()}
        runs = [(m, kw) for m, kw in runs if m in wanted]
    if not runs:
        print('没有匹配的配置，检查 --models')
        return

    print(f'分组：{args.group} | 配置数：{len(runs)} | 次数：{len(runs) * len(config.SEEDS)}')

    rows = []
    for model, kw in runs:
        for seed in config.SEEDS:
            rows.append(run(model, seed, **kw))

    config.ensure_dir(config.RESULTS_DIR)
    for r in rows:
        r['label'] = label_of(r['tag'])

    detail = os.path.join(config.RESULTS_DIR, 'results.csv')
    with open(detail, 'w', newline='', encoding='utf-8-sig') as f:
        # 字段固定：复用来的结果会多带 reused_as，这里统一忽略，避免列不一致
        w = csv.DictWriter(f, fieldnames=DETAIL_FIELDS, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

    stats = summarize(rows)
    out = os.path.join(config.RESULTS_DIR, 'summary.csv')
    write_summary(out, stats)

    # 跳过记录：写明哪个配置复用了哪次已有结果，便于核对"实验没有重复训练"
    skipped = [r for r in rows if r.get('reused_as')]
    if skipped:
        skip_path = os.path.join(config.RESULTS_DIR, 'skipped.csv')
        with open(skip_path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['本次需要的 tag', '复用的已有结果', '原因'])
            for r in skipped:
                w.writerow([r['reused_as'], r['tag'],
                            '模型/种子/正则/类权重/掩码/词向量/超参完全一致'])
    print(f'\n本次 {len(runs)} 个配置 × {len(config.SEEDS)} 种子 = {len(rows)} 个槽位：'
          f'跳过 {len(skipped)} 个（已有相同配置），实际训练 {len(rows) - len(skipped)} 次')
    if skipped:
        print(f'跳过明细已写入 {os.path.join(config.RESULTS_DIR, "skipped.csv")}')

    print(f'\n逐次结果：{detail}')
    print(f'汇总结果：{out}\n')
    print(f'\n{"=" * 92}')
    print(f'分组 {args.group} 结果（按测试集 Macro-F1 排序）')
    print('-' * 92)
    print(f'{"配置":<30}{"n":>3}   {"dev":<8}{"test Macro-F1":<20}'
          f'{"binary":<9}{"acc":<8}')
    for lab, s in sorted(stats.items(), key=lambda kv: -kv[1]['macro_mean']):
        print(f'{lab:<30}{s["n"]:>3}   {s["dev_mean"]:.4f}  '
              f'{s["macro_mean"]:.4f} ± {s["macro_std"]:.4f}   '
              f'{s["binary_mean"]:.4f}   {s["acc_mean"]:.4f}')
    print('=' * 92)


if __name__ == '__main__':
    main()

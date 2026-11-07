# -*- coding: utf-8 -*-
"""全因子实验的主效应与交互效应分析。

读取 results/result_*.json，抽出核心模型的 24 个配置（3 正则 × 2 权重 ×
2 掩码 × 2 词向量）× 3 种子，计算：

    results/factorial_table.csv     24 行明细（dev / test 均值±标准差）
    results/factorial_effects.csv   每个因素各水平的主效应
    results/factorial_interaction.csv  正则 × 类别权重的 3×2 交互表

用法（项目根目录）：python scripts/analyze_factorial.py
"""
import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402


def load_rows():
    rows = []
    for path in sorted(glob.glob(os.path.join(config.RESULTS_DIR, 'result_*.json'))):
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
        sig = d.get('signature')
        if not sig or sig.get('model') != 'bilstm_attention':
            continue
        if sig.get('split') != 'group':
            continue
        rows.append(dict(
            reg=sig['reg'], cw=int(sig['use_cw']),
            mask=int(sig['use_mask']), w2v=int(sig['use_w2v']),
            seed=d['seed'],
            dev=d['dev_macro_f1'], test=d['test_macro_f1'],
        ))
    return pd.DataFrame(rows)


def main():
    df = load_rows()
    if df.empty:
        print('没有找到全因子结果，先跑 scripts/run_all.py --group factorial')
        return

    print(f'读到 {len(df)} 次核心模型实验，覆盖 {df.groupby(["reg", "cw", "mask", "w2v"]).ngroups} 个配置')

    table = (df.groupby(['reg', 'cw', 'mask', 'w2v'])
               .agg(n=('test', 'size'),
                    dev_mean=('dev', 'mean'), dev_std=('dev', 'std'),
                    test_mean=('test', 'mean'), test_std=('test', 'std'))
               .reset_index()
               .sort_values('test_mean', ascending=False))
    table.to_csv(os.path.join(config.RESULTS_DIR, 'factorial_table.csv'),
                 index=False, encoding='utf-8-sig')
    print('\n=== 24 个配置（按测试集均值排序，前 10）===')
    print(table.head(10).round(4).to_string(index=False))

    effects = []
    for factor in ['reg', 'cw', 'mask', 'w2v']:
        for level, g in df.groupby(factor):
            effects.append({'factor': factor, 'level': level,
                            'n': len(g), 'dev_mean': g['dev'].mean(),
                            'test_mean': g['test'].mean()})
    eff = pd.DataFrame(effects)
    eff.to_csv(os.path.join(config.RESULTS_DIR, 'factorial_effects.csv'),
               index=False, encoding='utf-8-sig')
    print('\n=== 各因素主效应（测试集均值）===')
    for factor in ['reg', 'cw', 'mask', 'w2v']:
        sub = eff[eff['factor'] == factor]
        span = sub['test_mean'].max() - sub['test_mean'].min()
        detail = '  '.join(f'{r.level}={r.test_mean:.4f}' for r in sub.itertuples())
        print(f'  {factor:5s} 极差 {span:.4f}   {detail}')

    inter = (df.pivot_table(index='reg', columns='cw', values='test',
                            aggfunc='mean').round(4))
    inter.to_csv(os.path.join(config.RESULTS_DIR, 'factorial_interaction.csv'),
                 encoding='utf-8-sig')
    print('\n=== 正则 × 类别权重 交互（测试集均值）===')
    print(inter.to_string())


if __name__ == '__main__':
    main()

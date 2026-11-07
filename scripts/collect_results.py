# -*- coding: utf-8 -*-
"""把 results/ 下所有单次结果汇总成总表。

run_all.py 每跑一个分组都会重写 summary.csv（只含该组），
本脚本扫全部 result_*.json 重新聚合，供论文表格与核查使用。

产物：
    results/all_results.csv    逐次明细（含配置签名）
    results/summary_all.csv    按配置聚合的均值 ± 标准差
"""
import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402


def load():
    rows = []
    for path in sorted(glob.glob(os.path.join(config.RESULTS_DIR, 'result_*.json'))):
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
        sig = d.get('signature') or {}
        rows.append(dict(
            tag=d['tag'], model=d['model'], seed=d['seed'],
            reg=sig.get('reg', d.get('reg')),
            use_cw=int(sig.get('use_cw', d.get('use_cw', 1))),
            use_mask=int(sig.get('use_mask', d.get('use_mask', 1))),
            use_w2v=int(sig.get('use_w2v', d.get('use_w2v', 1))),
            split=sig.get('split', config.SPLIT_MODE),
            best_epoch=d.get('best_epoch'),
            dev=d['dev_macro_f1'], test=d['test_macro_f1'],
            binary=d['test_f1'], acc=d['test_acc'],
        ))
    return pd.DataFrame(rows)


def main():
    df = load()
    if df.empty:
        print('results/ 下没有单次结果')
        return

    detail = os.path.join(config.RESULTS_DIR, 'all_results.csv')
    df.to_csv(detail, index=False, encoding='utf-8-sig')

    key = ['model', 'reg', 'use_cw', 'use_mask', 'use_w2v', 'split']
    summary = (df.groupby(key)
                 .agg(n=('test', 'size'),
                      dev_mean=('dev', 'mean'), dev_std=('dev', 'std'),
                      test_mean=('test', 'mean'), test_std=('test', 'std'),
                      binary_mean=('binary', 'mean'), acc_mean=('acc', 'mean'))
                 .reset_index()
                 .sort_values('test_mean', ascending=False))
    out = os.path.join(config.RESULTS_DIR, 'summary_all.csv')
    summary.to_csv(out, index=False, encoding='utf-8-sig')

    print(f'共 {len(df)} 次实验，{len(summary)} 个配置')
    print(f'明细：{detail}\n汇总：{out}\n')
    print(summary.round(4).to_string(index=False))


if __name__ == '__main__':
    main()

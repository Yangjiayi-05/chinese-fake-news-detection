# -*- coding: utf-8 -*-
"""按论文表格的形式输出统计结果。

把 results/、results_random/、results_time/ 下的单次结果按配置聚合，
直接打印论文第四节的四张表（主表 / 消融 / 正则 / 划分对照），
每行给出 均值 ± 标准差 与样本数，方便直接抄进论文，也方便跑完当场核对。

用法（项目根目录）：python scripts/report_paper.py
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from utils.naming import final_cfg, variant_cfg  # noqa: E402

MODEL_CN = {
    'fasttext': 'FastText',
    'bilstm': 'BiLSTM',
    'bilstm_attention': 'BiLSTM+Attention',
}
SPLIT_CN = {'group': '分组划分（本文）', 'random': '随机分层', 'time': '时间切分'}


def load_all():
    """读三个划分目录下的全部单次结果。"""
    rows = []
    dirs = [config.RESULTS_DIR,
            os.path.join(config.WORK_DIR, 'results_random'),
            os.path.join(config.WORK_DIR, 'results_time')]
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, 'result_*.json'))):
            with open(path, encoding='utf-8') as f:
                r = json.load(f)
            sig = r.get('signature') or {}
            rows.append(dict(
                model=r['model'], seed=r['seed'],
                reg=sig.get('reg'), cw=int(sig.get('use_cw', 1)),
                mask=int(sig.get('use_mask', 1)), w2v=int(sig.get('use_w2v', 1)),
                split=sig.get('split', 'group'),
                dev=r['dev_macro_f1'], test=r['test_macro_f1'],
                binary=r['test_f1'], acc=r['test_acc'],
            ))
    return rows


def agg(rows):
    if not rows:
        return None
    a = lambda k: np.array([r[k] for r in rows], dtype=float)  # noqa: E731
    return dict(n=len(rows),
                dev=a('dev').mean(), test=a('test').mean(),
                test_std=a('test').std(ddof=1) if len(rows) > 1 else 0.0,
                binary=a('binary').mean(), acc=a('acc').mean())


def line(label, s, ref=None):
    if s is None:
        return f'{label:<24}{"未跑":>10}'
    delta = ''
    if ref is not None:
        delta = f'{s["test"] - ref:+.4f}'
    return (f'{label:<24}{s["n"]:>3}{"":<2}{s["dev"]:.4f}{"":<4}'
            f'{s["test"]:.4f} ± {s["test_std"]:.4f}{"":<3}{s["binary"]:.4f}'
            f'{"":<3}{s["acc"]:.4f}{"":<3}{delta}')


HEAD = (f'{"配置":<24}{"n":>3}{"":<2}{"dev":<8}{"test Macro-F1":<20}'
        f'{"binary":<9}{"acc":<9}相对基准')


def pick(rows, **cond):
    out = rows
    for k, v in cond.items():
        if v is not None:
            out = [r for r in out if r[k] == v]
    return out


def main():
    rows = load_all()
    if not rows:
        print('没有结果，先跑 scripts/run_all.py')
        return

    c = final_cfg()
    base = dict(model='bilstm_attention', split='group',
                reg=c['reg'], cw=int(c['cw']), mask=int(c['mask']), w2v=int(c['w2v']))
    base_stat = agg(pick(rows, **base))

    print(f'\n读取 {len(rows)} 次实验\n')

    print('=' * 96)
    print(f'表 1 · 主表（分组划分，基准配置 {c["reg"]}）')
    print('-' * 96)
    print(HEAD)
    for m in config.MAIN_MODEL_LIST:
        print(line(MODEL_CN.get(m, m), agg(pick(rows, model=m, split='group',
                                                reg=c['reg'], cw=int(c['cw']),
                                                mask=int(c['mask']), w2v=int(c['w2v'])))))
    print('=' * 96)

    print('\n表 2 · 消融（相对完整模型）')
    print('-' * 96)
    print(HEAD)
    ref = base_stat['test'] if base_stat else 0.0
    variants = [('完整模型', base)]
    for flip, name in (('mask', '去掉 Padding Mask'),
                       ('w2v', '去掉词向量初始化'),
                       ('cw', '加回类别权重')):
        v = variant_cfg(flip)
        variants.append((name, dict(base, cw=int(v['cw']),
                                    mask=int(v['mask']), w2v=int(v['w2v']))))
    for name, cond in variants:
        print(line(name, agg(pick(rows, **cond)), ref))

    print('\n表 3 · 正则档位')
    print('-' * 96)
    print(HEAD)
    for r in ('A', 'B', 'C'):
        print(line(f'正则 {r}', agg(pick(rows, **dict(base, reg=r))), ref))

    print('\n表 4 · 数据划分对照（仅核心模型）')
    print('-' * 96)
    print(HEAD)
    for sp in ('group', 'random', 'time'):
        print(line(SPLIT_CN.get(sp, sp), agg(pick(rows, **dict(base, split=sp))), ref))
    print('=' * 96)


if __name__ == '__main__':
    main()

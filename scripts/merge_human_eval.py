# -*- coding: utf-8 -*-
"""汇总多评分者的盲评结果：平均切题词数、切题率、Fleiss' kappa。

评分是 0~5 的"切题词数"，属于序数数据：
- 平均切题词数 / 5 即为关键词切题率；
- Fleiss' kappa 按 0~5 六个类别计算，衡量评分者之间的一致性水平。

用法
----
    python scripts/merge_human_eval.py
"""
import glob
import os
import sys

import numpy as np
from openpyxl import load_workbook

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402

MAXK = 5


def read_scores(path):
    wb = load_workbook(path, data_only=True)
    ws = wb['评估表']
    out = {}
    for r in range(2, ws.max_row + 1):
        sid = ws.cell(row=r, column=1).value
        score = ws.cell(row=r, column=4).value
        if sid is None:
            continue
        out[sid] = int(score) if isinstance(score, (int, float)) and 0 <= score <= MAXK else None
    return out


def fleiss_kappa(table):
    """table: 行=样本，列=类别计数（0~5 共 6 类）。"""
    n_rat = table.sum(axis=1)
    if n_rat.min() == 0:
        return float('nan')
    p_i = ((table ** 2).sum(axis=1) - n_rat) / (n_rat * (n_rat - 1))
    p_bar = p_i.mean()
    p_j = table.sum(axis=0) / table.sum()
    p_e = (p_j ** 2).sum()
    return (p_bar - p_e) / (1 - p_e) if p_e < 1 else float('nan')


def main():
    files = sorted(glob.glob(os.path.join(config.RESULTS_DIR, 'human_eval_rater*.xlsx')))
    if not files:
        print('未找到 human_eval_rater*.xlsx，请先运行 build_eval_xlsx.py')
        return

    per_rater = {}
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0].replace('human_eval_rater', '评分者 ')
        per_rater[name] = read_scores(f)

    ids = sorted(set().union(*[set(v) for v in per_rater.values()]))
    raters = list(per_rater)

    print(f'读取 {len(files)} 份评分表：{", ".join(raters)}')
    print(f'样本 {len(ids)} 条\n')

    table = np.zeros((len(ids), MAXK + 1), dtype=int)
    detail = []
    for i, sid in enumerate(ids):
        vals = [per_rater[r].get(sid) for r in raters]
        vals = [v for v in vals if v is not None]
        for v in vals:
            table[i, v] += 1
        detail.append((sid, vals))

    all_scores = [s for _, vals in detail for s in vals]
    filled = sum(1 for _, vals in detail if vals)
    if not all_scores:
        print('还没有任何评分')
        return

    mean = float(np.mean(all_scores))
    print(f'已填样本：{filled} / {len(ids)}')
    print(f'总判断次数：{len(all_scores)}')
    print(f'\n平均切题词数：{mean:.2f} / {MAXK}（切题率 {mean / MAXK:.1%}）')

    print('\n切题词数分布：')
    for k in range(MAXK, -1, -1):
        c = all_scores.count(k)
        if c:
            print(f'  {k} 个：{c:3d} 次（{c / len(all_scores):.1%}）')

    print('\n各评分者：')
    for r in raters:
        v = [s for s in per_rater[r].values() if s is not None]
        if v:
            print(f'  {r}：n={len(v)} 平均 {np.mean(v):.2f} / {MAXK}')

    k = fleiss_kappa(table[table.sum(axis=1) > 0])
    if k == k:
        level = ('一致性较差' if k < 0.2 else '一致性一般' if k < 0.4
                 else '中等一致' if k < 0.6 else '一致性较好')
        print(f"\nFleiss' kappa = {k:.3f}（{level}）")
    else:
        print("\nFleiss' kappa 无法计算（评分不全）")

    print('\n逐条明细（各评分者填写值）：')
    for sid, vals in detail:
        print(f'  {sid}: {vals}')


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""汇总多评分者的盲评结果：平均分、切题率、Fleiss' kappa。

Fleiss' kappa 用于 3 人及以上、类别为 1/2/3 的名义一致性；
评分是序数，kappa 偏低时不必强行解释为"标注质量差"，
也可能是评分者对"切题"的尺度不同，论文里照实报告即可。

用法
----
    python scripts/merge_human_eval.py
"""
import glob
import os
import sys

from openpyxl import load_workbook

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402


def read_scores(path):
    wb = load_workbook(path, data_only=True)
    ws = wb['评估表']
    out = {}
    for r in range(2, ws.max_row + 1):
        sid = ws.cell(row=r, column=1).value
        score = ws.cell(row=r, column=4).value
        if sid is None:
            continue
        out[sid] = int(score) if score in (1, 2, 3) else None
    return out


def fleiss_kappa(table):
    """table: 行=样本，列=类别计数（1/2/3）。"""
    n_subj, n_cat = table.shape
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

    import numpy as np
    table = np.zeros((len(ids), 3), dtype=int)
    report = []
    for i, sid in enumerate(ids):
        for k, r in enumerate(raters):
            s = per_rater[r].get(sid)
            if s in (1, 2, 3):
                table[i, s - 1] += 1
        vals = [per_rater[r].get(sid) for r in raters if per_rater[r].get(sid) in (1, 2, 3)]
        report.append((sid, vals))

    valid = [v for v in table.sum(axis=1) if v > 0]
    print(f'已填样本 {len(valid)} / {len(ids)}')

    all_scores = [s for _, vals in report for s in vals]
    if not all_scores:
        print('还没有任何评分')
        return

    mean = sum(all_scores) / len(all_scores)
    c3 = sum(1 for s in all_scores if s == 3) / len(all_scores)
    c12 = sum(1 for s in all_scores if s >= 2) / len(all_scores)

    print(f'\n总判断次数：{len(all_scores)}')
    print(f'平均切题度：{mean:.2f} / 3')
    print(f'3 分占比  ：{c3:.1%}')
    print(f'切题率(≥2)：{c12:.1%}')

    print('\n各评分者：')
    for r in raters:
        v = [s for s in per_rater[r].values() if s in (1, 2, 3)]
        if v:
            print(f'  {r}：n={len(v)} 平均 {sum(v) / len(v):.2f} '
                  f'3分 {sum(1 for s in v if s == 3) / len(v):.0%}')

    k = fleiss_kappa(table[table.sum(axis=1) > 0])
    if k == k:
        level = ('一致性较差' if k < 0.2 else '一致性一般' if k < 0.4
                 else '中等一致' if k < 0.6 else '一致性较好')
        print(f"\nFleiss' kappa = {k:.3f}（{level}）")
    else:
        print("\nFleiss' kappa 无法计算（评分不全）")

    print('\n逐条明细：')
    for sid, vals in report:
        print(f'  {sid}: {vals}')


if __name__ == '__main__':
    main()

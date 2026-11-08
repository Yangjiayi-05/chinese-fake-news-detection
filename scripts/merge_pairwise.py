# -*- coding: utf-8 -*-
"""汇总二选一评估：注意力组被选比例、两组都差/都不错的比例、评分者一致性。

支持四选一（组1 / 组2 / 两组都差 / 两组都不错），
把"差不多"拆成两个方向后，可以直接量化"两组都不够好"的比例。

用法
----
    python scripts/merge_pairwise.py            # 读 k=5 版本
    python scripts/merge_pairwise.py --tag _k8  # 读 k=8 版本
"""
import argparse
import csv
import glob
import os
import sys
from math import comb

import numpy as np
from openpyxl import load_workbook

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402

CATS = ['注意力组', '随机组', '两组都差', '两组都不错']
ALIAS = {'差不多': '两组都差'}  # 兼容旧表的"差不多"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='')
    args = ap.parse_args()

    src = os.path.join(config.RESULTS_DIR, f'pairwise_sheet{args.tag}.csv')
    rows = list(csv.DictReader(open(src, encoding='utf-8-sig')))
    att_is = {r['序号']: int(r['注意力组']) for r in rows}
    ids = [r['序号'] for r in rows]

    files = sorted(glob.glob(os.path.join(config.RESULTS_DIR,
                                          f'pairwise{args.tag}_rater*.xlsx')))
    if not files:
        print(f'未找到 pairwise{args.tag}_rater*.xlsx')
        return

    rater_votes = {}
    for f in files:
        tag = os.path.basename(f).replace(f'pairwise{args.tag}_rater', '').replace('.xlsx', '')
        ws = load_workbook(f, data_only=True)['评估表']
        votes = {}
        for r in range(2, ws.max_row + 1):
            sid = ws.cell(row=r, column=1).value
            ans = ws.cell(row=r, column=5).value
            if sid is None or ans is None:
                continue
            ans = ALIAS.get(ans, ans)
            if ans in ('组1', '组2'):
                votes[sid] = '注意力组' if int(ans[-1]) == att_is.get(sid) else '随机组'
            elif ans in ('两组都差', '两组都不错'):
                votes[sid] = ans
        rater_votes[tag] = votes

    tags = list(rater_votes)
    all_votes = [v for votes in rater_votes.values() for v in votes.values()]
    if not all_votes:
        print('还没有任何评分')
        return

    n = len(all_votes)
    print(f'评分者：{", ".join(tags)}')
    print(f'有效判断：{n} 次\n')
    for c in CATS:
        k = all_votes.count(c)
        print(f'  {c}：{k} 次（{k / n:.1%}）')

    dec = all_votes.count('注意力组') + all_votes.count('随机组')
    if dec:
        w = all_votes.count('注意力组')
        p = 2 * sum(comb(dec, i) for i in range(w, dec + 1)) / 2 ** dec
        print(f'\n明确二选一：{dec} 次，其中注意力组 {w} 次'
              f'（{w / dec:.1%}，二项检验 p={min(p, 1):.4f}）')
    bad = all_votes.count('两组都差')
    print(f'两组都差：{bad} 次（{bad / n:.1%}）')

    table = np.zeros((len(ids), len(CATS)), dtype=int)
    for i, sid in enumerate(ids):
        for t in tags:
            v = rater_votes[t].get(sid)
            if v in CATS:
                table[i, CATS.index(v)] += 1
    valid = table[table.sum(axis=1) > 0]
    if len(valid) and valid.sum(axis=1).min() > 1:
        n_rat = valid.sum(axis=1)
        p_i = ((valid ** 2).sum(axis=1) - n_rat) / (n_rat * (n_rat - 1))
        p_bar = p_i.mean()
        p_j = valid.sum(axis=0) / valid.sum()
        p_e = (p_j ** 2).sum()
        kappa = (p_bar - p_e) / (1 - p_e) if p_e < 1 else float('nan')
        level = ('一致性较差' if kappa < 0.2 else '一致性一般' if kappa < 0.4
                 else '中等一致' if kappa < 0.6 else '一致性较好')
        print(f"\nFleiss' kappa = {kappa:.3f}（{level}）")
    else:
        print('\n一致性需要至少 2 位评分者对同一题打分')

    print('\n各评分者：')
    for t in tags:
        v = list(rater_votes[t].values())
        if v:
            print(f'  {t}：n={len(v)} ' + ' / '.join(f'{c} {v.count(c)}' for c in CATS))


if __name__ == '__main__':
    main()

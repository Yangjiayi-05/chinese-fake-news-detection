# -*- coding: utf-8 -*-
"""汇总二选一人工评估：注意力组被选比例、评分者一致性。"""
import csv
import glob
import os
import sys

import numpy as np
from openpyxl import load_workbook

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402

CATS = ['注意力组', '随机组', '差不多']


def main():
    rows = list(csv.DictReader(open(os.path.join(config.RESULTS_DIR, 'pairwise_sheet.csv'),
                                    encoding='utf-8-sig')))
    att_is = {r['序号']: int(r['注意力组']) for r in rows}
    ids = [r['序号'] for r in rows]

    files = sorted(glob.glob(os.path.join(config.RESULTS_DIR, 'pairwise_rater*.xlsx')))
    if not files:
        print('未找到 pairwise_rater*.xlsx')
        return

    rater_votes = {}
    for f in files:
        tag = os.path.basename(f).replace('pairwise_rater', '').replace('.xlsx', '')
        ws = load_workbook(f, data_only=True)['评估表']
        votes = {}
        for r in range(2, ws.max_row + 1):
            sid = ws.cell(row=r, column=1).value
            ans = ws.cell(row=r, column=5).value
            if sid is None or ans not in ('组1', '组2', '差不多'):
                continue
            if ans == '差不多':
                votes[sid] = '差不多'
            else:
                votes[sid] = '注意力组' if int(ans[-1]) == att_is.get(sid) else '随机组'
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
        print(f'\n排除"差不多"后，注意力组被选中率：'
              f'{all_votes.count("注意力组") / dec:.1%}（n={dec}）')

    table = np.zeros((len(ids), len(CATS)), dtype=int)
    for i, sid in enumerate(ids):
        for t in tags:
            v = rater_votes[t].get(sid)
            if v:
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
        print('\n一致性需要至少 2 位评分者填完同一题才能计算')

    print('\n各评分者：')
    for t in tags:
        v = list(rater_votes[t].values())
        if v:
            print(f'  {t}：n={len(v)} 注意力 {v.count("注意力组")} / '
                  f'随机 {v.count("随机组")} / 差不多 {v.count("差不多")}')


if __name__ == '__main__':
    main()

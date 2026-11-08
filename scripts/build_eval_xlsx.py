# -*- coding: utf-8 -*-
"""生成多评分者盲评表：每位评分者一份，互不影响。

评分方式
--------
不设阈值：评分者直接数「5 个关键词里有几个能说明这句话在讲什么」，填 0~5。
这样不会把 4 个和 5 个切题压成同一档，也避免"几个算切题"的人为定义。

设计要点
--------
1. 盲评：不显示模型判断、置信度、任何预判；
2. 独立：每人一个文件，文件名带编号；
3. 统一：三份内容完全相同，只有文件名不同；
4. 可算一致性：同一批 60 条 × 3 人 = 180 次判断，可算 Fleiss' kappa。

用法
----
    python scripts/build_eval_xlsx.py --raters 3
"""
import argparse
import csv
import os
import sys

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402

SRC = os.path.join(config.RESULTS_DIR, 'human_eval_sheet.csv')
NAMES = 'ABCDE'

RULES = [
    ('怎么打分', '数一数：这 5 个关键词里，有几个能说明这句话在讲什么'),
    ('', ''),
    ('填 5', '5 个都能说明主题'),
    ('填 3', '3 个能说明，另外 2 个是泛词或无关词'),
    ('填 1', '只有 1 个能说明'),
    ('填 0', '一个都不能说明（完全看不出主题）'),
    ('', ''),
    ('注意 1', '不确定的词不算切题，宁可少算，不要多算'),
    ('注意 2', '不要判断新闻真假，也不要猜模型判得对不对'),
    ('注意 3', '独立完成，不要和其他评分者讨论'),
]


def build_one(rows, rater, out_dir):
    wb = Workbook()
    ws = wb.active
    ws.title = '评估表'

    head = ['序号', '原文', '关键词', '切题词数(0-5)', '备注']
    ws.append(head)

    thin = Side(style='thin', color='BFBFBF')
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    for c in range(1, len(head) + 1):
        cell = ws.cell(row=1, column=c)
        cell.font = Font(bold=True, color='FFFFFF')
        cell.fill = PatternFill('solid', fgColor='4A7BA7')
        cell.alignment = Alignment(horizontal='center', vertical='center')
        cell.border = border

    for i, r in enumerate(rows, start=2):
        ws.cell(row=i, column=1, value=r['序号'])
        ws.cell(row=i, column=2, value=r['原文'])
        ws.cell(row=i, column=3, value=r['关键词'].replace(' / ', '、'))
        for c in range(1, 6):
            cell = ws.cell(row=i, column=c)
            cell.border = border
            cell.alignment = Alignment(vertical='top', wrap_text=(c == 2))
        ws.row_dimensions[i].height = 58

    for col, w in {'A': 9, 'B': 62, 'C': 34, 'D': 13, 'E': 20}.items():
        ws.column_dimensions[col].width = w
    ws.freeze_panes = 'A2'

    dv = DataValidation(type='list', formula1='"0,1,2,3,4,5"', allow_blank=True,
                        showDropDown=False,
                        error='请填写 0~5 的整数',
                        promptTitle='切题词数',
                        prompt='数一数 5 个关键词里有几个能说明主题')
    ws.add_data_validation(dv)
    dv.add(f'D2:D{len(rows) + 1}')

    ws2 = wb.create_sheet('评分规则')
    ws2.cell(row=1, column=1, value=f'评分者 {rater}').font = Font(bold=True, size=13)
    for i, (a, b) in enumerate(RULES, start=3):
        ws2.cell(row=i, column=1, value=a).font = Font(bold=True)
        ws2.cell(row=i, column=2, value=b)
    ws2.column_dimensions['A'].width = 12
    ws2.column_dimensions['B'].width = 66

    path = os.path.join(out_dir, f'human_eval_rater{rater}.xlsx')
    wb.save(path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raters', type=int, default=3, help='评分者人数')
    args = ap.parse_args()

    rows = list(csv.DictReader(open(SRC, encoding='utf-8-sig')))
    made = [build_one(rows, NAMES[i], config.RESULTS_DIR) for i in range(args.raters)]

    old = os.path.join(config.RESULTS_DIR, 'human_eval_v2.xlsx')
    if os.path.exists(old):
        os.remove(old)
        print(f'已删除旧的单人评分表：{old}')

    print(f'已生成 {len(made)} 份盲评表（每份 {len(rows)} 条）：')
    for p in made:
        print('  ', p)
    print(f'\n合计判断次数：{len(rows)} 条 × {len(made)} 人 = {len(rows) * len(made)} 次')


if __name__ == '__main__':
    main()

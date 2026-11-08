# -*- coding: utf-8 -*-
"""把人工评估表转成盲评 Excel：隐藏模型判断与置信度，只留原文与关键词。

设计原则
--------
标注者不应看到模型结论，否则会产生锚定效应；也不应看到任何预判。
只保留「序号 / 原文 / 关键词 / 评分 / 备注」，评分用 1~3 的下拉框。

用法
----
    python scripts/build_eval_xlsx.py
"""
import csv
import os
import sys

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402

SRC = os.path.join(config.RESULTS_DIR, 'human_eval_sheet.csv')
DST = os.path.join(config.RESULTS_DIR, 'human_eval_v2.xlsx')

RULES = [
    ('评分', '标准'),
    ('3 分', '5 个关键词里至少 3 个直接说明这句话在讲什么'),
    ('2 分', '只有 1~2 个切题，其余是数字、泛词或无意义词'),
    ('1 分', '基本不切题：全是数字、标点、<UNK> 或与主题无关的词'),
    ('', ''),
    ('注意', '只判断「关键词是否切中这句话的主题」'),
    ('', '不要判断新闻真假，也不要猜模型判得对不对'),
    ('', '每条独立判断，不要回看前面已经打过的分'),
]


def main():
    rows = list(csv.DictReader(open(SRC, encoding='utf-8-sig')))
    wb = Workbook()

    ws = wb.active
    ws.title = '评估表'
    head = ['序号', '原文', '关键词', '评分(1-3)', '备注']
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
        ws.cell(row=i, column=4, value=None)
        ws.cell(row=i, column=5, value=None)
        for c in range(1, 6):
            cell = ws.cell(row=i, column=c)
            cell.border = border
            cell.alignment = Alignment(vertical='top', wrap_text=(c == 2))
        ws.row_dimensions[i].height = 58

    widths = {'A': 9, 'B': 62, 'C': 34, 'D': 11, 'E': 20}
    for col, w in widths.items():
        ws.column_dimensions[col].width = w
    ws.freeze_panes = 'A2'

    dv = DataValidation(type='list', formula1='"1,2,3"', allow_blank=True,
                        showDropDown=False, error='请填写 1、2 或 3',
                        promptTitle='评分', prompt='3=切题 2=部分 1=不切题')
    ws.add_data_validation(dv)
    dv.add(f'D2:D{len(rows) + 1}')

    ws2 = wb.create_sheet('评分规则')
    for i, (a, b) in enumerate(RULES, start=1):
        ws2.cell(row=i, column=1, value=a).font = Font(bold=True)
        ws2.cell(row=i, column=2, value=b)
    ws2.column_dimensions['A'].width = 10
    ws2.column_dimensions['B'].width = 62
    ws2.freeze_panes = 'A2'

    wb.save(DST)
    print(f'已生成盲评表 → {DST}')
    print(f'共 {len(rows)} 条；「评分」列为下拉框，选 1/2/3 即可')
    print('未包含：模型判断、置信度、任何预判')


if __name__ == '__main__':
    main()

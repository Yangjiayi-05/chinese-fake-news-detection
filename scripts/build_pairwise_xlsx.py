# -*- coding: utf-8 -*-
"""把二选一评估数据做成评分表（Excel），并附一页说明。

删除「注意力组」列，评分者看不到哪组来自模型，避免锚定。
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

SRC = os.path.join(config.RESULTS_DIR, 'pairwise_sheet.csv')
NAMES = 'AB'

GUIDE = [
    ('要做什么',
     '每题给出一句话和两组关键词，请判断哪一组更能说明这句话在讲什么'),
    ('', ''),
    ('怎么选',
     '· 选「组1」或「组2」：觉得这组词更像在概括这句话'),
     ('', '· 选「差不多」：两组都行，或两组都不行'),
    ('', ''),
    ('注意 1', '只看关键词，不要管这句话是真是假'),
    ('', ''),
    ('注意 2', '不需要逐词分析，读完两遍凭整体感觉选即可，10 秒一题'),
    ('', ''),
    ('校准示例（先看这两个，再开始）', ''),
    ('示例 1', '原文：上海警方通报：家政人员自述老人饿死视频系编造'),
    ('', '组1：上海、警方、编造、通报、家政'),
    ('', '组2：自述、视频、人员、饿死、系'),
    ('', '标准答案：选组1（组1 是主体+核心动作，组2 只是零散词）'),
    ('', ''),
    ('示例 2', '原文：特朗普警告称，如果克林顿当选，将爆发第三次世界大战'),
    ('', '组1：特朗普、警告、称、当选、克林顿'),
    ('', '组2：如果、将、爆发、第、次'),
    ('', '标准答案：选组1（组2 全是虚词和数字碎片）'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raters', type=int, default=2)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(SRC, encoding='utf-8-sig')))

    thin = Side(style='thin', color='BFBFBF')
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    for ri in range(args.raters):
        rater = NAMES[ri]
        wb = Workbook()
        ws = wb.active
        ws.title = '评估表'

        head = ['序号', '原文', '组1', '组2', '哪组更好', '备注']
        ws.append(head)
        for c in range(1, len(head) + 1):
            cell = ws.cell(row=1, column=c)
            cell.font = Font(bold=True, color='FFFFFF')
            cell.fill = PatternFill('solid', fgColor='4A7BA7')
            cell.alignment = Alignment(horizontal='center', vertical='center')
            cell.border = border

        for i, r in enumerate(rows, start=2):
            ws.cell(row=i, column=1, value=r['序号'])
            ws.cell(row=i, column=2, value=r['原文'])
            ws.cell(row=i, column=3, value=r['组1'])
            ws.cell(row=i, column=4, value=r['组2'])
            for c in range(1, 7):
                cell = ws.cell(row=i, column=c)
                cell.border = border
                cell.alignment = Alignment(vertical='top', wrap_text=(c in (2, 3, 4)))
            ws.row_dimensions[i].height = 62

        for col, w in {'A': 9, 'B': 52, 'C': 26, 'D': 26, 'E': 13, 'F': 16}.items():
            ws.column_dimensions[col].width = w
        ws.freeze_panes = 'A2'

        dv = DataValidation(type='list', formula1='"组1,组2,差不多"', allow_blank=True,
                            showDropDown=False,
                            error='请选择 组1 / 组2 / 差不多',
                            promptTitle='哪组更好',
                            prompt='哪组关键词更能说明这句话在讲什么')
        ws.add_data_validation(dv)
        dv.add(f'E2:E{len(rows) + 1}')

        ws2 = wb.create_sheet('怎么评')
        ws2.cell(row=1, column=1, value=f'评分者 {rater}').font = Font(bold=True, size=13)
        for i, (a, b) in enumerate(GUIDE, start=3):
            ca = ws2.cell(row=i, column=1, value=a)
            if a:
                ca.font = Font(bold=True)
            ws2.cell(row=i, column=2, value=b)
        ws2.column_dimensions['A'].width = 30
        ws2.column_dimensions['B'].width = 76

        path = os.path.join(config.RESULTS_DIR, f'pairwise_rater{rater}.xlsx')
        wb.save(path)
        print('已生成', path)

    print(f'\n共 {len(rows)} 题 × {args.raters} 人')


if __name__ == '__main__':
    main()

import csv
import matplotlib.pyplot as plt
import os
from config import FIGURES_DIR

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_external_summary(csv_path, save_name='external_summary.png'):
    """画外部验证结果汇总柱状图"""
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    correct = sum(1 for r in rows if r['correct'] == '✅')
    wrong = sum(1 for r in rows if r['correct'] == '❌')
    unlabeled = sum(1 for r in rows if r['correct'] == '—')

    plt.figure(figsize=(8, 5))
    bars = plt.bar(['判断正确', '判断错误', '未标注'],
                    [correct, wrong, unlabeled],
                    color=['#55A868', '#C44E52', '#8172B2'], width=0.5)
    for bar, val in zip(bars, [correct, wrong, unlabeled]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 str(val), ha='center', fontsize=12)
    plt.ylabel('样本数', fontsize=12)
    plt.title('外部验证结果汇总', fontsize=13)
    plt.grid(axis='y', alpha=0.3)
    save_path = os.path.join(FIGURES_DIR, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存：{save_path}")
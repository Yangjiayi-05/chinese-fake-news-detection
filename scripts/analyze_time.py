# -*- coding: utf-8 -*-
"""时间切分的证据分析：证明"为什么不用时间划分"。

产出：
    figures/fig9_time_drift.png   按年份的样本量与谣言占比（漂移证据）
    figures/fig8_split_compare.png  三种划分的 Macro-F1 对比
    results/time_stats.csv        按年份的统计明细
    results/split_compare.csv     三种划分的对比明细

用法（项目根目录）：python scripts/analyze_time.py
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402


def time_drift():
    """统计每个年份的样本量与谣言占比，画漂移图。"""
    df = pd.read_csv(config.RAW_CSV, encoding='utf-8')
    cn = ['微博', '新浪', '腾讯', '界面', '网易', '搜狐',
          '百家号', '凤凰', '腾讯新闻较真平台', '中国互联网联合辟谣平台']
    df = df[(df['label'] != '尚无定论') & (df['platform'].isin(cn))].copy()
    df['year'] = pd.to_datetime(df['publish_time'], errors='coerce').dt.year
    df = df[df['year'].notna()]

    g = df.groupby('year').agg(
        n=('label', 'size'),
        rumor=('label', lambda s: (s == '谣言').sum()),
    )
    g['rumor_rate'] = g['rumor'] / g['n']
    g = g.sort_index()
    g.to_csv(os.path.join(config.RESULTS_DIR, 'time_stats.csv'),
             encoding='utf-8-sig')
    print(g.to_string())
    print(f'\n时间范围：{int(g.index.min())} ～ {int(g.index.max())}')
    print(f'谣言占比区间：{g["rumor_rate"].min():.1%} ～ {g["rumor_rate"].max():.1%}')

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(g.index, g['n'], color='#9aa5b1', alpha=0.7, label='Samples')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Number of samples')

    ax2 = ax1.twinx()
    ax2.plot(g.index, g['rumor_rate'], color='#e4572e', marker='o',
             linewidth=2.2, label='Rumor ratio')
    ax2.set_ylabel('Rumor ratio')
    ax2.set_ylim(0, 1)

    ax1.set_title('Temporal distribution: sample count vs rumor ratio')
    fig.tight_layout()
    path = os.path.join(config.FIGURES_DIR, 'fig9_time_drift.png')
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print('已保存', os.path.basename(path))


def split_compare():
    """对比三种划分下核心模型的 Macro-F1（读各自的 summary.csv）。"""
    rows = []
    for mode, sub in [('group', config.RESULTS_DIR),
                      ('random', os.path.join(config.WORK_DIR, 'results_random')),
                      ('time', os.path.join(config.WORK_DIR, 'results_time'))]:
        f = os.path.join(sub, 'summary.csv')
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f)
        core = df[df['label'].astype(str).str.contains('Attention')]
        if core.empty:
            continue
        r = core.iloc[0]
        rows.append(dict(split=mode, n=int(r['n']),
                         macro_mean=float(r['macro_f1_mean']),
                         macro_std=float(r['macro_f1_std'])))
    if not rows:
        print('没有可对比的 summary.csv')
        return

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(config.RESULTS_DIR, 'split_compare.csv'),
               index=False, encoding='utf-8-sig')
    print(out.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(out['split'], out['macro_mean'], yerr=out['macro_std'],
           capsize=6, color=['#7fb069', '#f2c14e', '#e4572e'][:len(out)],
           edgecolor='black', alpha=0.9)
    for i, (m, s) in enumerate(zip(out['macro_mean'], out['macro_std'])):
        ax.text(i, m + s + 0.002, f'{m:.4f}', ha='center', fontsize=11)
    ax.set_ylabel('Macro-F1 (test set)')
    ax.set_title('Effect of data split on the core model')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()
    path = os.path.join(config.FIGURES_DIR, 'fig8_split_compare.png')
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print('已保存', os.path.basename(path))


def main():
    config.ensure_dir(config.RESULTS_DIR)
    config.ensure_dir(config.FIGURES_DIR)
    print('=== 时间分布漂移 ===')
    time_drift()
    print('\n=== 三种划分对比 ===')
    split_compare()


if __name__ == '__main__':
    main()

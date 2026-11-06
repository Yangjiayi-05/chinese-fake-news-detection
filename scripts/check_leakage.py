# -*- coding: utf-8 -*-
"""训练集 / 测试集近似重复自检（只统计，不改数据）。

背景：本项目放弃了 SimHash 去重，改用全量 17,524 条训练。
评审自然会问「那训练集和测试集之间有没有内容重叠、指标会不会虚高」。
这个脚本用字符 8-gram 指纹统计重叠比例，为「泄漏风险可忽略」提供量化依据。

口径说明（都用「测试样本的内容有多大比例出现在同一条训练样本里」来衡量）：
    - 完全重复：文本完全相同；
    - 重度重叠：包含度 ≥ 0.9；高度重叠：≥ 0.8；部分重叠：≥ 0.5。
    - 只有「高度重叠」及以上才算泄漏风险；部分重叠多由同题报道的公共短语造成，属正常。

用法（项目根目录）：python scripts/check_leakage.py
     切换划分方式：set CFND_SPLIT=random / export CFND_SPLIT=random
"""
import os
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from utils.preprocess import load_data  # noqa: E402
from utils.split import assign_clusters, build_clusters  # noqa: E402

K = 8            # 字符 n-gram 长度
CAP = 20         # 单个指纹最多保留多少条训练样本，控制内存与耗时


def shingles(text):
    """字符级 n-gram 指纹集合；短文本退化为整串本身。"""
    if len(text) <= K:
        return {hash(text)}
    return {hash(text[i:i + K]) for i in range(len(text) - K + 1)}


def split_texts():
    """按 config.SPLIT_MODE 复现同一套划分，返回 (训练文本, 测试文本, 三份条数)。"""
    df = load_data(config.RAW_CSV)
    texts = df['text'].tolist()
    labels = df['label_binary'].astype(int).tolist()

    if config.SPLIT_MODE == 'group':
        clusters = build_clusters(texts, cont_ratio=config.GROUP_CONT_RATIO,
                                  template_df=config.GROUP_TEMPLATE_DF)
        tr, dv, te = assign_clusters(clusters, labels)
        pick = lambda idx: [texts[i] for i in idx]  # noqa: E731
        return pick(tr), pick(te), (len(tr), len(dv), len(te))

    if config.SPLIT_MODE == 'time':
        ts = pd.to_datetime(df['publish_time'], errors='coerce')
        order = df.assign(_ts=ts).sort_values('_ts', na_position='last').index
        df = df.loc[order]
        n = len(df)
        i1, i2 = int(n * 0.7), int(n * 0.85)
        return (df['text'].iloc[:i1].tolist(), df['text'].iloc[i2:].tolist(),
                (i1, i2 - i1, n - i2))

    from sklearn.model_selection import train_test_split
    train_df, tmp_df = train_test_split(
        df, test_size=0.3, random_state=config.SPLIT_SEED,
        stratify=df['label_binary'])
    dev_df, test_df = train_test_split(
        tmp_df, test_size=0.5, random_state=config.SPLIT_SEED,
        stratify=tmp_df['label_binary'])
    return (train_df['text'].tolist(), test_df['text'].tolist(),
            (len(train_df), len(dev_df), len(test_df)))


def main():
    print(f'划分方式：{config.SPLIT_MODE}')
    print('加载数据并复现同一套划分 ...')
    train_texts, test_texts, sizes = split_texts()
    print(f'训练集 {sizes[0]} / 验证集 {sizes[1]} / 测试集 {sizes[2]}')

    print('构建训练集指纹索引 ...')
    index = {}
    for tid, text in enumerate(train_texts):
        for h in shingles(text):
            bucket = index.setdefault(h, [])
            if len(bucket) < CAP:
                bucket.append(tid)
    print(f'  指纹条目 {len(index)} 个')

    exact_set = {}
    for tid, text in enumerate(train_texts):
        exact_set.setdefault(hash(text), tid)

    print('比对测试集 ...')
    n_exact = 0
    cont_09 = cont_08 = cont_05 = 0
    examples = []

    for i, text in enumerate(test_texts):
        if hash(text) in exact_set:
            n_exact += 1

        ts = shingles(text)
        counter = Counter()
        for h in ts:
            for tid in index.get(h, ()):
                counter[tid] += 1

        if counter:
            tid, hits = counter.most_common(1)[0]
            cont = hits / max(len(ts), 1)   # 测试样本有多大比例的内容出现在同一条训练样本里
            if cont >= 0.9:
                cont_09 += 1
            if cont >= 0.8:
                cont_08 += 1
                if len(examples) < 3:
                    examples.append((cont, train_texts[tid][:40], text[:40]))
            if cont >= 0.5:
                cont_05 += 1

        if (i + 1) % 500 == 0:
            print(f'  已比对 {i + 1}/{len(test_texts)}')

    n = len(test_texts)
    print('\n================ 结果 ================')
    print(f'测试集样本数                    : {n}')
    print(f'完全重复（文本完全相同）        : {n_exact}  ({n_exact / n:.2%})')
    print(f'重度重叠（包含度 ≥ 0.9）        : {cont_09}  ({cont_09 / n:.2%})')
    print(f'高度重叠（包含度 ≥ 0.8）        : {cont_08}  ({cont_08 / n:.2%})')
    print(f'部分重叠（包含度 ≥ 0.5）        : {cont_05}  ({cont_05 / n:.2%})')

    if examples:
        print('\n高度重叠示例（包含度 | 训练片段 | 测试片段）：')
        for cont, a, b in examples:
            print(f'  {cont:.2f} | {a} ... | {b} ...')

    print('\n判读建议：')
    if cont_08 / n < 0.01:
        print('  高度重叠低于 1%，可写「训练/测试集内容重叠可忽略，保留全量数据不构成泄漏风险」。')
    elif cont_08 / n < 0.05:
        print('  高度重叠在 1%~5%，建议在论文中如实报告，并补一组按重复簇划分的敏感性实验。')
    else:
        print('  高度重叠超过 5%，建议改用「按重复簇分组划分」：不删任何样本，把近似重复')
        print('  的样本整体放进同一个集合，既保住训练量又消除泄漏。')


if __name__ == '__main__':
    main()

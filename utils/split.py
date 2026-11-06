# -*- coding: utf-8 -*-
"""按重复簇分组划分。

为什么要这一步：
    MCFEND 是多平台数据集，同一条新闻会被微博、百家号、腾讯新闻等反复转载，
    文本高度相似。随机划分会把这些"孪生样本"拆到训练集和测试集两侧，
    实测测试集里有 7.49% 与训练集完全相同、12.29% 内容包含度超过 0.8，
    导致测试指标虚高（俗称数据泄漏）。

本模块的做法：
    一条样本都不删，只把互相近似重复的样本绑成一个簇，整簇分配到同一侧。
    这样测试集里不会再出现训练时见过的内容，样本总量保持不变。
"""
from collections import Counter

import numpy as np

K = 8        # 字符 n-gram 长度
CAP = 50     # 单个指纹最多保留多少条样本，控制内存占用


def _shingles(text):
    """字符级 n-gram 指纹集合；短文本退化为整串本身。"""
    if len(text) <= K:
        return {hash(text)}
    return {hash(text[i:i + K]) for i in range(len(text) - K + 1)}


def build_clusters(texts, cont_ratio=0.8, template_df=0.02):
    """把互相近似重复的样本聚成簇，返回下标列表的列表。

    先剔除高频模板 n-gram（如平台免责声明），否则不相干的新闻会被套话连成一簇。
    """
    raw = [_shingles(t) for t in texts]
    n = len(texts)

    freq = Counter()
    for s in raw:
        freq.update(s)
    cut = template_df * n
    template = {h for h, c in freq.items() if c > cut}
    sh = [s - template for s in raw]

    index = {}
    for i, s in enumerate(sh):
        for h in s:
            bucket = index.setdefault(h, [])
            if len(bucket) < CAP:
                bucket.append(i)

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # 文本完全相同的先直接合并：即使它的 n-gram 全被当成模板剔除，也不能漏掉
    exact = {}
    for i, t in enumerate(texts):
        if t in exact:
            ra, rb = find(i), find(exact[t])
            if ra != rb:
                parent[rb] = ra
        else:
            exact[t] = i

    for i, s in enumerate(sh):
        counter = Counter()
        for h in s:
            for j in index.get(h, ()):
                if j != i:
                    counter[j] += 1
        for j, hits in counter.items():
            if hits / max(len(s), 1) >= cont_ratio:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[rb] = ra

    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    return list(clusters.values())


def assign_clusters(clusters, labels, ratios=(0.7, 0.15, 0.15)):
    """把整簇分配到训练/验证/测试，同时约束"样本量"和"正负类配额"。

    只按样本量分配会把转载更多的谣言堆进训练集、事实全剩到测试集
    （实测会偏成 23.8% 与 52.8%），所以必须同时盯着正负类的配额余量。
    """
    n = len(labels)
    targets = np.array(ratios, dtype=float) * n
    g = float(np.mean(labels))
    t_pos = targets * g
    t_neg = targets * (1 - g)

    order = sorted(clusters, key=len, reverse=True)
    cur_n = np.zeros(3)
    cur_p = np.zeros(3)
    out = [[], [], []]

    for cl in order:
        size = len(cl)
        pos = float(sum(labels[i] for i in cl))
        rem_pos = np.maximum(t_pos - cur_p, 0) / t_pos
        rem_neg = np.maximum(t_neg - (cur_n - cur_p), 0) / t_neg
        score = pos * rem_pos + (size - pos) * rem_neg
        if score.max() <= 0:
            score = np.maximum(targets - cur_n, 0)
        k = int(np.argmax(score))
        out[k].extend(cl)
        cur_n[k] += size
        cur_p[k] += pos

    return [sorted(x) for x in out]

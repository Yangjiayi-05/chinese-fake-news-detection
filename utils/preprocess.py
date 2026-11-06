# -*- coding: utf-8 -*-
"""数据预处理：MCFEND 中文平台合并版 -> 全量清洗 -> 分词 -> 划分 -> 建词表。

两种划分方式（由 config.SPLIT_MODE 决定，环境变量 CFND_SPLIT 可切换）：
    random：随机分层划分，实现简单，但同源转载的新闻会跨训练/测试，存在泄漏；
    group ：按重复簇分组划分，近似重复的样本绑在同一侧，泄漏为 0，样本一条不删。

刻意不做 SimHash 删除：去重会把 17,524 条砍到约 4,400 条，训练集只剩约 3,100，
对参数量 660 万的轻量模型属于严重欠采样，因此只做基础清洗。
"""
import json
import os
import pickle
import re
import sys
from collections import Counter

import jieba
import pandas as pd

# 允许直接执行 python utils/preprocess.py：把项目根目录加入模块搜索路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from utils.split import assign_clusters, build_clusters  # noqa: E402

CN_PLATFORMS = ['微博', '新浪', '腾讯', '界面', '网易', '搜狐',
                '百家号', '凤凰', '腾讯新闻较真平台', '中国互联网联合辟谣平台']

# 控制字符与零宽字符，属于"乱码"清洗；只清理字符，不借此删样本
_CTRL_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200f\ufeff]')


def clean_text(s):
    """去控制字符、压缩连续空白，不做除截断以外的任何删样本操作。"""
    s = _CTRL_RE.sub('', str(s))
    s = re.sub(r'[ \t\r\n\u3000]+', ' ', s)
    return s.strip()


def load_data(csv_path):
    """加载并仅做基础清洗，返回条数应当正好是 17,524。"""
    df = pd.read_csv(csv_path, encoding='utf-8')

    # 1. 剔除"尚无定论"，它不属于二分类任务
    df = df[df['label'] != '尚无定论'].copy()

    # 2. 仅保留中文平台，避免多语言混入
    df = df[df['platform'].isin(CN_PLATFORMS)].copy()

    # 3. 标题与正文拼接，标题往往携带强判别特征
    def merge_text(row):
        t = clean_text(row['title']) if pd.notna(row['title']) else ''
        c = clean_text(row['content']) if pd.notna(row['content']) else ''
        if t and c:
            return t + '。' + c
        return t or c

    df['text'] = df.apply(merge_text, axis=1)

    # 4. 只剔除空文本，这是唯一的删样本操作
    df = df[df['text'].str.len() > 0].copy()

    # 5. 标签二值化：谣言为 0，事实为 1
    df['label_binary'] = df['label'].map(config.LABEL_MAP)
    df = df[df['label_binary'].notna()].copy()

    return df.reset_index(drop=True)


def build_vocab(tokenized_texts, min_freq=2):
    """只用训练集建词表，词频低于 min_freq 的丢弃。"""
    word_counts = Counter()
    for tokens in tokenized_texts:
        word_counts.update(tokens)
    kept = [w for w, f in word_counts.items() if f >= min_freq]
    # 按词频降序排序，保证多次运行词表顺序一致、索引可复现
    kept.sort(key=lambda w: (-word_counts[w], w))
    idx2word = ['<PAD>', '<UNK>'] + kept
    word2idx = {w: i for i, w in enumerate(idx2word)}
    return word2idx, idx2word


def text_to_indices(tokens, word2idx, max_len):
    """转索引：超长截断，不足补 0（<PAD>）。"""
    indices = [word2idx.get(w, word2idx['<UNK>']) for w in tokens]
    if len(indices) < max_len:
        indices += [0] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return indices


def main():
    # Kaggle 的 /kaggle/input 是只读挂载区，预处理只应在本地做一次
    if config.IS_KAGGLE:
        raise SystemExit(
            'Kaggle 上不需要跑预处理：请把本地生成的目录作为 Dataset 挂载，'
            'config 会自动找到它。')

    csv_path = config.RAW_CSV
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'找不到原始数据：{csv_path}')
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)

    print(f'[1/5] 读取原始数据：{csv_path}')
    df = load_data(csv_path)
    print(f'      样本总数：{len(df)}')
    print('      标签分布：')
    print(df['label_binary'].value_counts().to_string())

    print('[2/5] Jieba 分词')
    df['tokens'] = df['text'].apply(lambda x: jieba.lcut(x))
    tok_len = df['tokens'].apply(len)
    trunc_rate = float((tok_len > config.MAX_LEN).mean())
    print(f'      token 长度：中位数 {int(tok_len.median())}，'
          f'p99 {int(tok_len.quantile(0.99))}，最大 {int(tok_len.max())}')
    print(f'      MAX_LEN={config.MAX_LEN} 截断率：{trunc_rate:.2%}')

    labels = df['label_binary'].astype(int).tolist()
    n_clusters = None

    if config.SPLIT_MODE == 'group':
        print('[3/5] 划分：按重复簇分组')
        clusters = build_clusters(df['text'].tolist(),
                                  cont_ratio=config.GROUP_CONT_RATIO,
                                  template_df=config.GROUP_TEMPLATE_DF)
        n_clusters = len(clusters)
        print(f'      {n_clusters} 个重复簇，最大簇 {max(len(c) for c in clusters)} 条')
        tr, dv, te = assign_clusters(clusters, labels)
        train_df, dev_df, test_df = df.iloc[tr], df.iloc[dv], df.iloc[te]
    elif config.SPLIT_MODE == 'time':
        print('[3/5] 划分：按发布时间切分 7 : 1.5 : 1.5')
        ts = pd.to_datetime(df['publish_time'], errors='coerce')
        n_bad = int(ts.isna().sum())
        print(f'      无法解析的时间 {n_bad} 条（排到最后）')
        df = df.assign(_ts=ts).sort_values('_ts', na_position='last').drop(columns='_ts')
        n_all = len(df)
        i1, i2 = int(n_all * 0.7), int(n_all * 0.85)
        train_df, dev_df, test_df = df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]
        print(f'      时间范围：{ts.min()} ～ {ts.max()}')
    else:
        print('[3/5] 划分：随机分层 7 : 1.5 : 1.5')
        from sklearn.model_selection import train_test_split
        train_df, tmp_df = train_test_split(
            df, test_size=0.3, random_state=config.SPLIT_SEED,
            stratify=df['label_binary'])
        dev_df, test_df = train_test_split(
            tmp_df, test_size=0.5, random_state=config.SPLIT_SEED,
            stratify=tmp_df['label_binary'])

    print(f'      训练集 {len(train_df)} / 验证集 {len(dev_df)} / 测试集 {len(test_df)}')
    for name, sub in [('train', train_df), ('dev', dev_df), ('test', test_df)]:
        print(f'      {name} 事实占比 {sub["label_binary"].mean():.2%}')

    print('[4/5] 构建词表')
    word2idx, idx2word = build_vocab(train_df['tokens'].tolist(), min_freq=2)
    print(f'      词表大小：{len(word2idx)}')

    # 类别权重由训练集真实分布算出，供交叉熵损失使用
    counts = train_df['label_binary'].value_counts().to_dict()
    n0, n1 = counts.get(0, 0), counts.get(1, 0)
    total = n0 + n1
    class_weight = {0: round(total / (2.0 * n0), 4), 1: round(total / (2.0 * n1), 4)}
    print(f'      类别权重：{class_weight}')

    print('[5/5] 转索引并保存')
    with open(os.path.join(config.PROCESSED_DIR, 'vocab.pkl'), 'wb') as f:
        pickle.dump({'word2idx': word2idx, 'idx2word': idx2word}, f)

    def process(sub_df):
        X = [text_to_indices(t, word2idx, config.MAX_LEN) for t in sub_df['tokens']]
        y = sub_df['label_binary'].astype(int).tolist()
        return {'X': X, 'y': y}

    for name, sub in [('train', train_df), ('dev', dev_df), ('test', test_df)]:
        with open(os.path.join(config.PROCESSED_DIR, f'{name}.pkl'), 'wb') as f:
            pickle.dump(process(sub), f)
        print(f'      已保存 {name}.pkl')

    # 口径存档：写论文时直接引用，不用回头猜当时的参数
    meta = {
        'source_csv': csv_path,
        'split_mode': config.SPLIT_MODE,
        'total': len(df),        'train': len(train_df),
        'dev': len(dev_df),
        'test': len(test_df),
        'pos_rate': {
            'train': round(float(train_df['label_binary'].mean()), 4),
            'dev': round(float(dev_df['label_binary'].mean()), 4),
            'test': round(float(test_df['label_binary'].mean()), 4),
        },
        'vocab_size': len(word2idx),
        'max_len': config.MAX_LEN,
        'truncation_rate': round(trunc_rate, 6),
        'class_weight': class_weight,
        'label_counts': {str(k): int(v) for k, v in counts.items()},
        'n_clusters': n_clusters,
        'group_cont_ratio': config.GROUP_CONT_RATIO if config.SPLIT_MODE == 'group' else None,
        'split_seed': config.SPLIT_SEED,
        'dedup': 'none',
    }
    with open(os.path.join(config.PROCESSED_DIR, 'meta.json'), 'w',
              encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print('      已保存 meta.json')
    print('预处理完成。')


if __name__ == '__main__':
    main()

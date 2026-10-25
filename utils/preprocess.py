import pandas as pd
import pickle
import os
from collections import Counter
import jieba

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_data(csv_path):
    """加载MCFEND数据，合并中文平台"""
    df = pd.read_csv(csv_path, encoding='utf-8')

    # 过滤"尚无定论"
    df = df[df['label'] != '尚无定论']

    # 中文平台列表
    cn_platforms = ['微博', '新浪', '腾讯', '界面', '网易', '搜狐',
                    '百家号', '凤凰', '腾讯新闻较真平台', '中国互联网联合辟谣平台']
    df = df[df['platform'].isin(cn_platforms)]

    # 拼接文本：标题+正文
    def merge_text(row):
        t = str(row['title']) if pd.notna(row['title']) else ''
        c = str(row['content']) if pd.notna(row['content']) else ''
        if t and c:
            return t + '。' + c
        return t or c

    df['text'] = df.apply(merge_text, axis=1)

    # 过滤空文本
    df = df[df['text'].str.len() > 0]

    # 标签映射：谣言→0，事实→1
    df['label_binary'] = df['label'].map({'谣言': 0, '事实': 1})

    return df


def build_vocab(tokenized_texts, min_freq=2):
    word_counts = Counter()
    for tokens in tokenized_texts:
        word_counts.update(tokens)
    kept = [w for w, f in word_counts.items() if f >= min_freq]
    idx2word = ['<PAD>', '<UNK>'] + kept
    word2idx = {w: i for i, w in enumerate(idx2word)}
    return word2idx, idx2word


def text_to_indices(tokens, word2idx, max_len):
    indices = [word2idx.get(w, word2idx['<UNK>']) for w in tokens]
    if len(indices) < max_len:
        indices += [word2idx['<PAD>']] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return indices


def main():
    csv_path = os.path.join(BASE_DIR, 'MCFEND_original_dataset', 'mcfend', 'news.csv')
    processed_dir = os.path.join(BASE_DIR, 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    MAX_LEN = 200
    MIN_FREQ = 2

    print("加载数据...")
    df = load_data(csv_path)
    print(f"样本总数：{len(df)}")
    print(f"标签分布：\n{df['label_binary'].value_counts()}")

    print("分词...")
    df['tokens'] = df['text'].apply(lambda x: jieba.lcut(x))

    # 划分：7:1.5:1.5
    from sklearn.model_selection import train_test_split
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42,
                                         stratify=df['label_binary'])
    dev_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42,
                                       stratify=temp_df['label_binary'])

    print(f"训练集：{len(train_df)}，验证集：{len(dev_df)}，测试集：{len(test_df)}")

    # 构建词表（仅用训练集）
    word2idx, idx2word = build_vocab(train_df['tokens'].tolist(), MIN_FREQ)
    print(f"词表大小：{len(word2idx)}")

    # 转索引
    def process(df):
        X = [text_to_indices(t, word2idx, MAX_LEN) for t in df['tokens']]
        y = df['label_binary'].tolist()
        return {'X': X, 'y': y}

    # 保存
    with open(os.path.join(processed_dir, 'vocab.pkl'), 'wb') as f:
        pickle.dump({'word2idx': word2idx, 'idx2word': idx2word}, f)

    for name, data in [('train', train_df), ('dev', dev_df), ('test', test_df)]:
        with open(os.path.join(processed_dir, f'{name}.pkl'), 'wb') as f:
            pickle.dump(process(data), f)
        print(f"已保存 {name}.pkl")

    print("预处理完成！")


if __name__ == '__main__':
    main()
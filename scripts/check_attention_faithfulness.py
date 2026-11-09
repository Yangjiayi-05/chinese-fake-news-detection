# -*- coding: utf-8 -*-
"""可解释性验证：注意力删除检验（faithfulness）。

思路
----
对每条测试样本，把注意力权重最高的 k 个真实词替换成 <UNK>，重新预测，
记录"预测类概率"的下降幅度；再随机替换同样数量的词做对照。
若前者显著大于后者，说明注意力高权重词确实是模型判断的依据，
而不只是"看起来相关"。

产出
----
    results/faithfulness.csv      逐条的 Δ(top-k) 与 Δ(random-k)
    终端汇总：平均值、倍数、按类别的表现

用法（项目根目录）
------------------
    python scripts/check_attention_faithfulness.py
    python scripts/check_attention_faithfulness.py --n 500 --k 5
"""
import argparse
import csv
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from main import DEVICE, build_model  # noqa: E402
from utils.textfilter import is_content_token  # noqa: E402

UNK = 1  # 词表中 <UNK> 的编号


def load_vocab():
    with open(os.path.join(config.PROCESSED_DIR, 'vocab.pkl'), 'rb') as f:
        return pickle.load(f)


def load_test(n):
    with open(os.path.join(config.PROCESSED_DIR, 'test.pkl'), 'rb') as f:
        d = pickle.load(f)
    return d['X'][:n], d['y'][:n]


def load_model(vocab_size):
    model = build_model('bilstm_attention', vocab_size, use_mask=True).to(DEVICE)
    ckpt = os.path.join(config.CHECKPOINT_DIR,
                        f'best_bilstm_attention_seed{config.DEPLOY_SEED}.pth')
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f'缺少模型权重：{ckpt}\n请从 Kaggle Output 下载后放到该目录')
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    return model


def predict_probs(model, x):
    """返回 (P(事实), P(预测类))。

    删词检验要看的是"模型对自己判断的把握"，所以用预测类的概率：
    预测谣言时用 1-p，预测事实时用 p。否则多数类占优时符号会反。
    """
    with torch.no_grad():
        logits = model(x)
        p1 = torch.softmax(logits, dim=1)[:, 1]
        p_pred = torch.where(p1 >= 0.5, p1, 1.0 - p1)
        return p1, p_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=300, help='评估样本数')
    ap.add_argument('--k', type=int, default=5, help='删除的词数')
    ap.add_argument('--out', default=None, help='逐条结果 csv 路径')
    args = ap.parse_args()

    vocab = load_vocab()
    idx2word = vocab['idx2word']
    vocab_size = len(vocab['word2idx'])
    model = load_model(vocab_size)
    X, Y = load_test(args.n)

    rows = []
    rng = np.random.default_rng(0)

    for x_list, y in zip(X, Y):
        x = torch.tensor([x_list], dtype=torch.long).to(DEVICE)
        real = [i for i, t in enumerate(x_list) if t != 0]
        # 只在内容词中挑选与删除，避免把"删标点"当成无关操作
        cand = [i for i in real if is_content_token(idx2word[x_list[i]])]
        if len(cand) <= args.k:
            continue

        p1_0, pp_0 = (float(t) for t in predict_probs(model, x))
        pred0 = int(p1_0 >= 0.5)

        # 注意力权重 → 取权重最高的 k 个真实词
        with torch.no_grad():
            w = model.get_attention(x)[0].cpu().numpy()
        top = sorted(cand, key=lambda i: -w[i])[:args.k]

        # 随机选 k 个真实词
        rand = rng.choice(cand, size=args.k, replace=False).tolist()

        x_top = x.clone()
        x_top[0, top] = UNK
        p1_top, pp_top = (float(t) for t in predict_probs(model, x_top))

        x_rand = x.clone()
        x_rand[0, rand] = UNK
        p1_rand, pp_rand = (float(t) for t in predict_probs(model, x_rand))

        rows.append(dict(label=y, pred=pred0,
                         p_pred0=pp_0, p_pred_top=pp_top, p_pred_rand=pp_rand,
                         drop_top=pp_0 - pp_top, drop_rand=pp_0 - pp_rand,
                         flip_top=int((p1_top >= 0.5) != pred0)))

    if not rows:
        print('没有可用样本')
        return

    d_top = np.array([r['drop_top'] for r in rows])
    d_rand = np.array([r['drop_rand'] for r in rows])
    ratio = d_top.mean() / d_rand.mean() if abs(d_rand.mean()) > 0.005 else float('nan')
    flips = sum(r['flip_top'] for r in rows)

    print(f'\n样本数 {len(rows)} | 每条删除 {args.k} 个词'
          f'（替换为 <UNK>，其余不变）\n')
    print(f'  删除注意力 Top-{args.k} 词：预测类置信度平均下降 {d_top.mean():.4f}')
    print(f'  随机删除 {args.k} 个词      ：预测类置信度平均下降 {d_rand.mean():.4f}')
    print(f'  比值：{ratio:.1f} 倍')
    print(f'  判定翻转（删 Top-{args.k} 后类别改变）：{flips} / {len(rows)}'
          f'（{flips / len(rows):.1%}）')

    for name, lab in (('谣言（label=0）', 0), ('事实（label=1）', 1)):
        sub = [r for r in rows if r['label'] == lab]
        if sub:
            a = np.mean([r['drop_top'] for r in sub])
            b = np.mean([r['drop_rand'] for r in sub])
            ratio_s = f'{a / b:.1f} 倍' if abs(b) > 0.005 else '—（分母接近 0）'
            print(f'  {name}：Δtop {a:.4f} vs Δrandom {b:.4f}'
                  f'（{ratio_s}，n={len(sub)}）')

    out = args.out or os.path.join(config.RESULTS_DIR, 'faithfulness.csv')
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'\n逐条结果已写入 {out}')


if __name__ == '__main__':
    main()

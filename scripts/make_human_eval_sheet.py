# -*- coding: utf-8 -*-
"""生成人工评估表：随机抽取测试样本，附模型判断与注意力关键词。

供论文第五章"可解释性验证"使用。产出 results/human_eval_sheet.csv，
标注者只需在「是否合理」列填写 是/否，其余列不要改动。

用法（项目根目录）
------------------
    python scripts/make_human_eval_sheet.py
    python scripts/make_human_eval_sheet.py --per-class 10 --k 5
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-class', type=int, default=10, help='每类抽取条数')
    ap.add_argument('--k', type=int, default=5, help='展示的关键词个数')
    ap.add_argument('--seed', type=int, default=7, help='抽样随机种子')
    ap.add_argument('--min-len', type=int, default=12,
                    help='只抽内容词不少于该长度的样本，避免过短句子')
    args = ap.parse_args()

    with open(os.path.join(config.PROCESSED_DIR, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)
    idx2word = vocab['idx2word']
    with open(os.path.join(config.PROCESSED_DIR, 'test.pkl'), 'rb') as f:
        d = pickle.load(f)

    model = build_model('bilstm_attention', len(vocab['word2idx']), use_mask=True).to(DEVICE)
    ckpt = os.path.join(config.CHECKPOINT_DIR,
                        f'best_bilstm_attention_seed{config.DEPLOY_SEED}.pth')
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    rng = np.random.default_rng(args.seed)
    rows = []
    for label in (0, 1):
        idx = [i for i, y in enumerate(d['y']) if y == label
               and sum(1 for t in d['X'][i]
                       if t != 0 and is_content_token(idx2word[t])) >= args.min_len]
        pick = rng.choice(idx, size=min(args.per_class, len(idx)), replace=False)
        for n, i in enumerate(sorted(pick), 1):
            x_list = d['X'][i]
            real = [j for j, t in enumerate(x_list) if t != 0]
            # 关键词只在内容词中挑选，避免把标点/虚词当作"关键词"展示
            cand = [j for j in real if is_content_token(idx2word[x_list[j]])]
            x = torch.tensor([x_list], dtype=torch.long).to(DEVICE)
            with torch.no_grad():
                p1 = float(torch.softmax(model(x), dim=1)[0, 1])
                w = model.get_attention(x)[0].cpu().numpy()
            top = sorted(cand or real, key=lambda j: -w[j])[:args.k]
            rows.append(dict(
                序号=f'{"谣" if label == 0 else "实"}{n:02d}',
                真实标签='谣言' if label == 0 else '事实',
                模型判断='谣言' if p1 < 0.5 else '事实',
                置信度=f'{max(p1, 1 - p1):.3f}',
                原文=''.join(idx2word[x_list[j]] for j in real),
                关键词=' / '.join(idx2word[x_list[j]] for j in top),
                关键词权重=' / '.join(f'{w[j]:.3f}' for j in top),
                是否合理='',
                备注='',
            ))

    out = os.path.join(config.RESULTS_DIR, 'human_eval_sheet.csv')
    config.ensure_dir(config.RESULTS_DIR)
    with open(out, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f'已生成 {len(rows)} 条样本 → {out}')
    print('请在「是否合理」列填写 是/否，其余列保持不动。')
    for r in rows[:2]:
        print(f"\n[{r['序号']}] 真实={r['真实标签']} 判断={r['模型判断']}"
              f" 置信度={r['置信度']}\n  原文：{r['原文'][:60]}\n  关键词：{r['关键词']}")


if __name__ == '__main__':
    main()

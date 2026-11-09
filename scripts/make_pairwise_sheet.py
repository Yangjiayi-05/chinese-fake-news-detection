# -*- coding: utf-8 -*-
"""生成"二选一"人工评估数据：注意力关键词 vs 随机关键词。

设计
----
每题给出同一句话的两组关键词：
  组1 / 组2（顺序随机，评分者不知道哪组来自注意力）
评分者只需回答："哪组词更能说明这句话在讲什么？"

相比"数几个切题"，二选一不需要理解任何模糊标准，认知负担最低，
且直接对应删除检验里的"注意力 vs 随机"对照。

产出
----
    results/pairwise_sheet.csv
        序号,原文,组1,组2,注意力组(1或2),随机组(1或2)
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
    ap.add_argument('--per-class', type=int, default=15)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--min-len', type=int, default=12)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--tag', default='', help='输出文件名后缀，如 _k8')
    args = ap.parse_args()

    with open(os.path.join(config.PROCESSED_DIR, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)
    idx2word = vocab['idx2word']
    with open(os.path.join(config.PROCESSED_DIR, 'test.pkl'), 'rb') as f:
        d = pickle.load(f)

    model = build_model('bilstm_attention', len(vocab['word2idx']), use_mask=True).to(DEVICE)
    model.load_state_dict(torch.load(
        os.path.join(config.CHECKPOINT_DIR,
                     f'best_bilstm_attention_seed{config.DEPLOY_SEED}.pth'),
        map_location=DEVICE))
    model.eval()

    rng = np.random.default_rng(args.seed)
    rows = []

    for label in (0, 1):
        pool = [i for i, y in enumerate(d['y']) if y == label
                and sum(1 for t in d['X'][i]
                        if t != 0 and is_content_token(idx2word[t])) >= args.min_len + args.k]
        for n, i in enumerate(sorted(rng.choice(pool, size=min(args.per_class, len(pool)),
                                                replace=False)), 1):
            x_list = d['X'][i]
            real = [j for j, t in enumerate(x_list) if t != 0]
            cand = [j for j in real if is_content_token(idx2word[x_list[j]])]
            x = torch.tensor([x_list], dtype=torch.long).to(DEVICE)
            with torch.no_grad():
                w = model.get_attention(x)[0].cpu().numpy()

            seen, att = set(), []
            for j in sorted(cand, key=lambda j: -w[j]):
                word = idx2word[x_list[j]]
                if word in seen:
                    continue
                seen.add(word)
                att.append(j)
                if len(att) >= args.k:
                    break

            att_words = {idx2word[x_list[j]] for j in att}
            rest = [j for j in cand if idx2word[x_list[j]] not in att_words]
            rnd = rng.choice(rest, size=min(args.k, len(rest)), replace=False).tolist()

            def words(js):
                out = []
                for j in js:
                    wo = idx2word[x_list[j]]
                    if wo not in out:
                        out.append(wo)
                return '、'.join(out)

            g_att, g_rnd = words(att), words(rnd)
            if rng.random() < 0.5:
                g1, g2, att_is = g_att, g_rnd, 1
            else:
                g1, g2, att_is = g_rnd, g_att, 2

            rows.append(dict(
                序号=f'{"谣" if label == 0 else "实"}{n:02d}',
                原文=''.join(idx2word[x_list[j]] for j in real),
                组1=g1, 组2=g2,
                注意力组=att_is,
                随机组=3 - att_is,
            ))

    rng.shuffle(rows)
    out = os.path.join(config.RESULTS_DIR, f'pairwise_sheet{args.tag}.csv')
    config.ensure_dir(config.RESULTS_DIR)
    with open(out, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f'已生成 {len(rows)} 题 → {out}')
    for r in rows[:3]:
        print(f"\n[{r['序号']}] {r['原文'][:44]}")
        print(f"  组1：{r['组1']}")
        print(f"  组2：{r['组2']}")
        print(f"  （注意力组={r['注意力组']}，此列不要给评分者看）")


if __name__ == '__main__':
    main()

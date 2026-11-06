# -*- coding: utf-8 -*-
"""模型间差异的配对 bootstrap 显著性检验。

回答的问题：「核心模型比基线高出的那几个点，是真实差异还是随机波动？」

做法：同一测试集上，有放回重采样样本，同时重算两个模型的 Macro-F1，
得到「差值」的分布，取 2.5% / 97.5% 分位点作为 95% 置信区间。
区间不跨 0，才判定为显著优于。

用法（项目根目录）：
    python scripts/compare_models.py                       # 每个基线 vs 核心
    python scripts/compare_models.py --a bilstm --b bilstm_attention
"""
import argparse
import os
import sys

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402


def load_preds(model, seed):
    """读取某个模型某个种子的测试集标签与预测。"""
    base = config.CHECKPOINT_DIR
    tag = f'{model}_seed{seed}'
    y_path = os.path.join(base, f'test_labels_{tag}.npy')
    p_path = os.path.join(base, f'test_preds_{tag}.npy')
    if not (os.path.exists(y_path) and os.path.exists(p_path)):
        return None, None
    return np.load(y_path), np.load(p_path)


def macro(y, pred):
    return float(f1_score(y, pred, average='macro'))


def paired_bootstrap(y, preds_a, preds_b, n=2000, seed=0):
    """返回 (观测差值, 下限, 上限, p 值)。"""
    rng = np.random.default_rng(seed)
    size = len(y)
    obs = np.mean([macro(y, p) for p in preds_a]) - np.mean([macro(y, p) for p in preds_b])

    diffs = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, size, size)
        sub_y = y[idx]
        a = np.mean([macro(sub_y, p[idx]) for p in preds_a])
        b = np.mean([macro(sub_y, p[idx]) for p in preds_b])
        diffs[i] = a - b

    low, high = np.percentile(diffs, [2.5, 97.5])
    p_value = 2 * min(float((diffs <= 0).mean()), float((diffs >= 0).mean()))
    return obs, float(low), float(high), float(p_value)


def compare(model_a, model_b):
    """收集两个模型在所有种子上的预测，长度不一致就跳过。"""
    ya, pa, yb, pb = None, [], None, []
    for seed in config.SEEDS:
        y1, p1 = load_preds(model_a, seed)
        y2, p2 = load_preds(model_b, seed)
        if y1 is None or y2 is None:
            return None
        if not np.array_equal(y1, y2):
            print(f'  ⚠️ {model_a} 与 {model_b} 的测试集不一致，跳过')
            return None
        ya, yb = y1, y2
        pa.append(p1)
        pb.append(p2)
    if not pa:
        return None
    return paired_bootstrap(ya, pa, pb)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', default=None, help='对比另一方，默认逐个主模型')
    parser.add_argument('--b', default='bilstm_attention', help='基准模型')
    args = parser.parse_args()

    pairs = ([(args.a, args.b)] if args.a
             else [(m, args.b) for m in config.MAIN_MODEL_LIST if m != args.b])

    print(f'{"对比":<34}{"差值":>9}{"95% CI":>22}{"p":>8}   结论')
    print('-' * 88)
    for a, b in pairs:
        res = compare(a, b)
        if res is None:
            print(f'{a + " vs " + b:<34}{"数据缺失":>9}')
            continue
        obs, low, high, p = res
        sig = '显著' if low > 0 or high < 0 else '不显著'
        print(f'{a + " vs " + b:<34}{obs:>+9.4f}'
              f'{f"[{low:+.4f}, {high:+.4f}]":>22}{p:>8.3f}   {sig}')

    print('\n说明：差值为正值表示前者更好；区间跨 0 视为差异不显著。')


if __name__ == '__main__':
    main()

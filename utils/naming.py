# -*- coding: utf-8 -*-
"""配置与结果文件名之间的统一规则。

全因子实验里同一套配置可能被多个批次引用，靠"后缀"区分结果文件：
基准配置用空后缀，其余配置用 `_rC_c0_m1_v1` 形式。
这套规则只在本文件定义一次，训练与出图脚本都从这里取，避免各写一份。

因素的取值来源见 `config.FINAL_*`：改最终配置只需改 config 那一处。
"""
import config

# 因素顺序固定为 (正则, 类权重, 掩码, 词向量)
FLIPS = ('cw', 'mask', 'w2v')


def final_cfg():
    """最终配置（基准配置）。"""
    return dict(reg=config.FINAL_REG, cw=config.FINAL_USE_CW,
                mask=config.FINAL_USE_MASK, w2v=config.FINAL_USE_W2V)


def suffix_of(reg, cw, mask, w2v, plain_if_final=True):
    """基准配置返回空后缀，其余返回 _r{reg}_c{cw}_m{mask}_v{w2v}。"""
    c = final_cfg()
    if plain_if_final and (reg, cw, mask, w2v) == (c['reg'], c['cw'], c['mask'], c['w2v']):
        return ''
    return f'_r{reg}_c{int(cw)}_m{int(mask)}_v{int(w2v)}'


def variant_cfg(flip):
    """把最终配置里的某一个因素取反，用于消融实验。"""
    if flip not in FLIPS:
        raise ValueError(f'未知因素：{flip}，可选 {FLIPS}')
    c = final_cfg()
    c[flip] = not c[flip]
    return c


def variant_suffix(flip):
    """消融变体的结果文件后缀。"""
    c = variant_cfg(flip)
    return suffix_of(c['reg'], c['cw'], c['mask'], c['w2v'], plain_if_final=False)

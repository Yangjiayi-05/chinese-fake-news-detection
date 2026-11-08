# -*- coding: utf-8 -*-
"""词级过滤：挑出可用于展示/删除检验的"内容词"。

注意力权重会落在标点、空格、高频虚词上，这些词对人工阅读没有信息量，
也不适合作为"关键词"展示。这里用一个轻量规则把它们筛掉，不做词性标注，
避免引入额外依赖。
"""
import unicodedata

STOPWORDS = {
    '的', '了', '是', '在', '和', '也', '就', '都', '而', '及', '与', '着',
    '或', '一个', '我们', '他们', '你们', '这个', '那个', '这些', '那些',
    '有', '被', '把', '对', '从', '到', '为', '以', '于', '其', '这', '那',
    '将', '已', '并', '等', '中', '上', '下', '个', '我', '你', '他', '她',
    '它', '很', '还', '又', '才', '只', '但', '却', '则', '所', '之', '者',
}


def is_content_token(tok):
    """判断是否是可用于展示的内容词（非标点、非空白、非虚词）。"""
    if not tok:
        return False
    t = tok.strip()
    if not t:
        return False
    if all(unicodedata.category(ch)[0] in ('P', 'Z', 'S') for ch in t):
        return False
    if t in STOPWORDS:
        return False
    return True

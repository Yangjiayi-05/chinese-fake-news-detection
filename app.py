# -*- coding: utf-8 -*-
"""人机协同筛查系统（Streamlit）。

三档由 config 的两个参数驱动：决策分界线 DECISION_THRESHOLD，
以及分界线两侧的模糊带半宽 BAND_HALF_WIDTH。
高置信区间直接给出结论，模糊区间交给人，并高亮模型关注的原文片段。
"""
import csv
import html
import os
from datetime import datetime

import streamlit as st

import config
from inference import load_model_and_vocab, predict

st.set_page_config(page_title='中文虚假新闻辅助筛查系统', layout='centered')
st.title('中文虚假新闻辅助筛查系统')
st.caption(f'模型 BiLSTM + Self-Attention（含 Padding Mask）｜ '
           f'决策阈值 {config.DECISION_THRESHOLD} ｜ 模糊带 ±{config.BAND_HALF_WIDTH}')

LOG_FILE = os.path.join(config.LOGS_DIR, 'external_verification_log.csv')
LOW = round(config.DECISION_THRESHOLD - config.BAND_HALF_WIDTH, 2)
HIGH = round(config.DECISION_THRESHOLD + config.BAND_HALF_WIDTH, 2)


def save_log(text, status, p_real, top_words, true_label, correct):
    """把每次检测追加到 CSV，供后续统计人机协同效果。"""
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['time', 'text', 'status', 'p_real',
                             'top_words', 'true_label', 'correct'])
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            text, status, f'{p_real:.4f}',
            '; '.join(f'{w}:{s}' for w, s in top_words),
            true_label, correct,
        ])


def render_highlight(tokens, top_words, k=8):
    """在原文上高亮高权重词；用分词结果拼接，避免字符串替换破坏 HTML。"""
    marked = {w for w, _ in top_words[:k]}
    parts = []
    for token in tokens:
        escaped = html.escape(token)
        if token in marked:
            parts.append('<mark style="background:#ffe58f;padding:0 2px;'
                         f'border-radius:3px;">{escaped}</mark>')
        else:
            parts.append(escaped)
    return ''.join(parts)


@st.cache_resource
def get_model():
    return load_model_and_vocab()


try:
    model, vocab = get_model()
except FileNotFoundError as exc:
    st.error(f'模型加载失败：{exc}')
    st.stop()

text = st.text_area('请输入新闻文本：', height=150)
st.markdown('**您是否知道这条新闻的真实情况？（可选，用于统计模型准确率）**')
feedback = st.radio('选择后系统会记录，用于统计模型判断对错：',
                    ['我不知道 / 跳过', '我知道，它是真实的', '我知道，它是虚假的'],
                    index=0)

if st.button('开始检测'):
    if not text.strip():
        st.warning('请输入文本。')
        st.stop()

    p_real, top_words, tokens = predict(text, model, vocab)

    if p_real >= HIGH:
        status = '高置信事实'
        st.success(f'✅ **【高置信事实 · 绿灯放行】**  事实概率 {p_real:.4f}')
    elif p_real <= LOW:
        status = '高置信谣言'
        st.error(f'🚨 **【高置信谣言 · 红灯拦截】**  事实概率 {p_real:.4f}')
    else:
        status = '建议人工复核'
        st.warning(f'⚠️ **【模糊区间 · 黄灯】**  事实概率 {p_real:.4f}，'
                   f'落在 [{LOW:.2f}, {HIGH:.2f}]，建议人工复核')

    st.markdown('**模型关注的原文片段（按注意力权重高亮）：**')
    st.markdown(f'<div style="line-height:1.9;font-size:15px;">'
                f'{render_highlight(tokens, top_words)}</div>',
                unsafe_allow_html=True)
    st.markdown('**Top-8 关键词及权重：**')
    for word, score in top_words:
        st.write(f'- {word}（权重 {score}）')

    true_label, correct = '未标注', '—'
    if feedback == '我知道，它是真实的':
        true_label = '真实'
        correct = '✅' if p_real >= HIGH else '❌'
    elif feedback == '我知道，它是虚假的':
        true_label = '虚假'
        correct = '✅' if p_real <= LOW else '❌'

    save_log(text, status, p_real, top_words, true_label, correct)
    if correct != '—':
        st.info(f'已记录：真实标签 {true_label}，判定{"正确" if correct == "✅" else "错误"}')
    st.caption(f'检测记录已追加到 {LOG_FILE}')

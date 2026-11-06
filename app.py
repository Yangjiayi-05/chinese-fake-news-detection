# -*- coding: utf-8 -*-
"""人机协同筛查系统（Streamlit）。

三档由 config 的两个参数驱动：决策分界线 DECISION_THRESHOLD，
以及分界线两侧的模糊带半宽 BAND_HALF_WIDTH。
高置信区间直接给出结论，模糊区间交给人，并高亮模型关注的原文片段。
所有检测记录追加写入 logs/external_verification_log.csv，便于统计人机协同效果。
"""
import csv
import html
import os
from datetime import datetime

import pandas as pd
import streamlit as st

import config
from inference import load_model_and_vocab, predict

st.set_page_config(page_title='中文虚假新闻辅助筛查系统', layout='centered')
st.title('中文虚假新闻辅助筛查系统')
st.caption(f'模型 BiLSTM + Self-Attention（含 Padding Mask）｜ '
           f'决策阈值 {config.DECISION_THRESHOLD} ｜ 模糊带 ±{config.BAND_HALF_WIDTH}')

LOG_FILE = os.path.join(config.LOGS_DIR, 'external_verification_log.csv')
MODEL_TAG = f'{config.DEFAULT_MODEL}_seed{config.DEPLOY_SEED}'
TH = config.DECISION_THRESHOLD
LOW = round(TH - config.BAND_HALF_WIDTH, 2)
HIGH = round(TH + config.BAND_HALF_WIDTH, 2)

FIELDS = ['time', 'text', 'p_real', 'status', 'model_label', 'threshold',
          'band', 'model_tag', 'top_words', 'true_label', 'correct']


def save_log(text, p_real, status, model_label, top_words, true_label, correct):
    """追加一条检测记录。

    记录里带上当时的阈值和模型标识，将来换阈值重跑时才能回溯
    「哪条结果是由哪套参数产生的」。
    """
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(FIELDS)
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            text, f'{p_real:.4f}', status, model_label,
            f'{TH:.2f}', f'±{config.BAND_HALF_WIDTH}', MODEL_TAG,
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


def load_log():
    """读历史记录；文件不存在时返回空表，避免界面上报错。"""
    if not os.path.isfile(LOG_FILE):
        return pd.DataFrame(columns=FIELDS)
    return pd.read_csv(LOG_FILE, encoding='utf-8-sig', dtype=str)


tab_detect, tab_history = st.tabs(['🔍 检测', '📋 历史记录'])

with tab_detect:
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

        # 三档预警：只由 p_real 与阈值带的关系决定
        if p_real >= HIGH:
            status, model_label = '高置信事实', '真实'
            st.success(f'✅ **【高置信事实 · 绿灯放行】**  事实倾向得分 {p_real:.4f}')
        elif p_real <= LOW:
            status, model_label = '高置信谣言', '虚假'
            st.error(f'🚨 **【高置信谣言 · 红灯拦截】**  事实倾向得分 {p_real:.4f}')
        else:
            status, model_label = '建议人工复核', '未判定'
            st.warning(f'⚠️ **【模糊区间 · 黄灯】**  事实倾向得分 {p_real:.4f}，'
                       f'落在 [{LOW:.2f}, {HIGH:.2f}]，模型不下结论，建议人工复核')

        st.markdown('**模型关注的原文片段（按注意力权重高亮）：**')
        st.markdown(f'<div style="line-height:1.9;font-size:15px;">'
                    f'{render_highlight(tokens, top_words)}</div>',
                    unsafe_allow_html=True)
        st.markdown('**Top-8 关键词及权重：**')
        for word, score in top_words:
            st.write(f'- {word}（权重 {score}）')

        # 人工反馈：模糊区间里模型没有下结论，因此记为「未判定」而不是「错误」
        true_label, correct = '未标注', '—'
        if feedback == '我知道，它是真实的':
            true_label = '真实'
        elif feedback == '我知道，它是虚假的':
            true_label = '虚假'

        if true_label != '未标注':
            if model_label == '未判定':
                correct = '未判定'
            elif model_label == true_label:
                correct = '正确'
            else:
                correct = '错误'

        save_log(text, p_real, status, model_label, top_words, true_label, correct)

        if correct == '正确':
            st.info(f'已记录：人工标签「{true_label}」，模型判定一致')
        elif correct == '错误':
            st.info(f'已记录：人工标签「{true_label}」，模型判定不一致')
        elif correct == '未判定':
            st.info('已记录：该样本落在模糊区间，模型未下结论，不计入模型正确率')
        st.caption(f'记录已追加到 {LOG_FILE}')

with tab_history:
    st.subheader('既往检测记录')
    log_df = load_log()

    if log_df.empty:
        st.info('还没有任何检测记录。完成一次检测后这里会自动出现。')
    else:
        total = len(log_df)
        labeled = log_df[log_df['true_label'] != '未标注']
        decided = labeled[labeled['correct'].isin(['正确', '错误'])]
        review = log_df[log_df['status'] == '建议人工复核']

        col1, col2, col3, col4 = st.columns(4)
        col1.metric('总记录', total)
        col2.metric('模糊区间占比', f'{len(review) / total:.1%}')
        col3.metric('已人工标注', len(labeled))
        if len(decided):
            acc = (decided['correct'] == '正确').mean()
            col4.metric('模型判定准确率', f'{acc:.1%}', help='仅统计模型给出明确结论的样本')
        else:
            col4.metric('模型判定准确率', '—')

        st.markdown('**三档分布**')
        st.dataframe(
            log_df['status'].value_counts().rename_axis('档位').reset_index(name='条数'),
            width='stretch')

        st.markdown('**明细**')
        st.dataframe(log_df.iloc[::-1], width='stretch', height=360)

        st.download_button(
            '下载 CSV',
            data=log_df.to_csv(index=False).encode('utf-8-sig'),
            file_name='external_verification_log.csv',
            mime='text/csv')

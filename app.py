import streamlit as st
import csv
import os
from datetime import datetime
from inference import load_model_and_vocab, predict
from config import LOGS_DIR

st.set_page_config(page_title="中文虚假新闻检测", layout="centered")
st.title("中文虚假新闻检测系统")
st.markdown("输入一段中文新闻文本，系统将判断其真实性并展示关键注意力词。")

LOG_FILE = os.path.join(LOGS_DIR, 'external_verification_log.csv')


def save_log(text, label, conf, top_words, true_label, correct):
    os.makedirs(LOGS_DIR, exist_ok=True)
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['time', 'text', 'model_pred', 'confidence',
                             'top_words', 'true_label', 'correct'])
        top_str = '; '.join([f"{w}:{s}" for w, s in top_words])
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            text, label, f"{conf:.4f}", top_str,
            true_label, correct
        ])


@st.cache_resource
def get_model():
    return load_model_and_vocab()


model, vocab = get_model()

text = st.text_area("请输入新闻文本：", height=150)

# ===== 反馈区域放在按钮之前 =====
st.markdown("**您是否知道这条新闻的真实情况？（可选，用于评估模型准确率）**")
feedback = st.radio(
    "选择后系统会记录，用于统计模型判断对错：",
    ["我不知道 / 跳过", "我知道，它是真实的", "我知道，它是虚假的"],
    index=0
)

if st.button("检测"):
    if text.strip() == "":
        st.warning("请输入文本。")
    else:
        label, conf, words = predict(text, model, vocab)
        st.success(f"模型预测：{label}")
        st.info(f"置信度：{conf:.4f}")
        st.write("**模型关注的关键词：**")
        for w, score in words:
            st.write(f"- {w}（权重 {score}）")

        true_label = "未标注"
        correct = "—"

        if feedback == "我知道，它是真实的":
            true_label = "真实"
            correct = "✅" if label == "真实" else "❌"
            if correct == "✅":
                st.success("已记录：模型判断正确")
            else:
                st.error("已记录：模型判断错误")

        elif feedback == "我知道，它是虚假的":
            true_label = "虚假"
            correct = "✅" if label == "虚假" else "❌"
            if correct == "✅":
                st.success("已记录：模型判断正确")
            else:
                st.error("已记录：模型判断错误")

        save_log(text, label, conf, words, true_label, correct)
        st.caption("已记录到日志文件：external_verification_log.csv")
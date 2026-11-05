import streamlit as st
import csv
import os
from datetime import datetime
from inference import load_model_and_vocab, predict
from config import LOGS_DIR

st.set_page_config(page_title="中文虚假新闻检测", layout="centered")
st.title("中文虚假新闻辅助筛查系统")
st.markdown("系统采用三档预警机制，对于模糊样本建议人工复核。")

LOG_FILE = os.path.join(LOGS_DIR, 'external_verification_log.csv')

def save_log(text, label, p_real, top_words, true_label, correct, status):
    os.makedirs(LOGS_DIR, exist_ok=True)
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['time', 'text', 'model_pred', 'p_real',
                             'top_words', 'true_label', 'correct', 'status'])
        top_str = '; '.join([f"{w}:{s}" for w, s in top_words])
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            text, label, f"{p_real:.4f}", top_str,
            true_label, correct, status
        ])

@st.cache_resource
def get_model():
    return load_model_and_vocab()

model, vocab = get_model()

text = st.text_area("请输入新闻文本：", height=150)

st.markdown("**您是否知道这条新闻的真实情况？（可选，用于评估模型）**")
feedback = st.radio(
    "选择后系统会记录，用于统计模型判断对错：",
    ["我不知道 / 跳过", "我知道，它是真实的", "我知道，它是虚假的"],
    index=0
)

if st.button("开始检测"):
    if text.strip() == "":
        st.warning("请输入文本。")
    else:
        label, p_real, words = predict(text, model, vocab)
        
        # 阈值设定为 0.43，用于三档预警
        THRESHOLD = 0.43
        
        # p_real 是模型判定为“事实”的概率
        if p_real >= (1 - THRESHOLD):  # 即 >= 0.57
            status = "高置信事实"
            st.success("✅ 模型判定：**【高置信事实】**")
            st.write(f"系统认为它是真实新闻（事实概率：{p_real:.4f}），直接放行。")
        elif p_real <= THRESHOLD:      # 即 <= 0.43
            status = "高置信谣言"
            st.error("🚨 模型判定：**【高置信谣言】**")
            st.write(f"系统认为它是虚假新闻（事实概率仅：{p_real:.4f}），红色预警。")
        else:
            status = "建议人工复核"
            st.warning("⚠️ 模型判定：**【模糊区间，建议人工复核】**")
            st.write(f"系统置信度（事实概率：{p_real:.4f}）落入 [{THRESHOLD}, {1-THRESHOLD}] 的模糊带，请人工确认。")
        
        st.info("**模型关注的关键词（注意力机制提取的线索）：**")
        for w, score in words:
            st.write(f"- {w}（权重 {score}）")

        true_label = "未标注"
        correct = "—"

        if feedback == "我知道，它是真实的":
            true_label = "真实"
            correct = "✅" if label == "真实" else "❌"
        elif feedback == "我知道，它是虚假的":
            true_label = "虚假"
            correct = "✅" if label == "虚假" else "❌"

        save_log(text, label, p_real, words, true_label, correct, status)
        st.caption("检测记录已保存。")

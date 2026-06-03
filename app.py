import os
import torch
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================
# 頁面設定
# =========================
st.set_page_config(
    page_title="BERT / RoBERTa 情緒分析系統",
    page_icon="💬",
    layout="wide"
)

# =========================
# 模型路徑與裝置設定
# =========================
BERT_MODEL_PATH = "./my_bert_model"
ROBERTA_MODEL_PATH = "./my_roberta_model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensemble 權重
# class 0：RoBERTa 0.85 + BERT 0.15
# class 1：RoBERTa 0.35 + BERT 0.65
weight_roberta = torch.tensor([0.85, 0.35], dtype=torch.float32).to(device)
weight_bert = torch.tensor([0.15, 0.65], dtype=torch.float32).to(device)

# =========================
# CSS 美化
# =========================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f8fbff 0%, #eef4ff 45%, #fdf7ff 100%);
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #26324B;
    text-align: center;
    margin-bottom: 8px;
}

.sub-title {
    font-size: 18px;
    color: #5D6B82;
    text-align: center;
    margin-bottom: 35px;
}

.card {
    background-color: white;
    padding: 28px;
    border-radius: 22px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.08);
    margin-bottom: 24px;
}

.result-positive {
    background: linear-gradient(135deg, #E8FFF3, #F7FFFB);
    border-left: 8px solid #2ECC71;
    padding: 24px;
    border-radius: 18px;
    font-size: 24px;
    font-weight: 700;
    color: #1E8449;
}

.result-negative {
    background: linear-gradient(135deg, #FFF0F0, #FFF8F8);
    border-left: 8px solid #E74C3C;
    padding: 24px;
    border-radius: 18px;
    font-size: 24px;
    font-weight: 700;
    color: #C0392B;
}

.small-text {
    color: #6B7280;
    font-size: 15px;
}

.metric-box {
    background-color: #ffffff;
    border-radius: 18px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 6px 18px rgba(0,0,0,0.06);
}

.metric-number {
    font-size: 30px;
    font-weight: 800;
    color: #34495E;
}

.metric-label {
    font-size: 15px;
    color: #7F8C8D;
}

div[data-testid="stTabs"] button {
    font-size: 17px;
    font-weight: 600;
}

div.stButton > button {
    border-radius: 12px;
    height: 48px;
    font-size: 17px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# =========================
# 載入模型
# =========================
@st.cache_resource
def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        local_files_only=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True
    )

    model.to(device)
    model.eval()
    return tokenizer, model

# =========================
# 工具函式
# =========================
def label_info(predicted_class):
    if predicted_class == 1:
        return "Positive 正向", "😊"
    else:
        return "Negative 負向", "😟"


def get_probabilities(text, tokenizer, model):
    inputs = tokenizer(
        str(text),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)

    return probabilities


def predict_single_model(text, tokenizer, model):
    probabilities = get_probabilities(text, tokenizer, model)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_class].item()
    label, emoji = label_info(predicted_class)

    return label, confidence, predicted_class, emoji


def predict_ensemble(text, bert_tokenizer, bert_model, roberta_tokenizer, roberta_model):
    bert_probs = get_probabilities(text, bert_tokenizer, bert_model)
    roberta_probs = get_probabilities(text, roberta_tokenizer, roberta_model)

    final_probs = (roberta_probs * weight_roberta) + (bert_probs * weight_bert)

    predicted_class = torch.argmax(final_probs, dim=1).item()
    confidence = final_probs[0][predicted_class].item()
    label, emoji = label_info(predicted_class)

    return label, confidence, predicted_class, emoji


def predict_sentiment(text):
    if model_choice == "BERT":
        return predict_single_model(text, tokenizer, model)
    elif model_choice == "RoBERTa":
        return predict_single_model(text, tokenizer, model)
    else:
        return predict_ensemble(
            text,
            bert_tokenizer,
            bert_model,
            roberta_tokenizer,
            roberta_model
        )

# =========================
# 主標題
# =========================
st.markdown('<div class="main-title">💬 BERT / RoBERTa 情緒分析系統</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">使用深度學習模型分析文字情緒，支援單句預測、CSV 批次分析與加權融合</div>',
    unsafe_allow_html=True
)

# =========================
# 檢查模型資料夾
# =========================
bert_exists = os.path.exists(BERT_MODEL_PATH)
roberta_exists = os.path.exists(ROBERTA_MODEL_PATH)

if not bert_exists and not roberta_exists:
    st.error("找不到模型資料夾，請確認 my_bert_model 或 my_roberta_model 是否放在 app.py 同一層。")
    st.stop()

# =========================
# 選擇模型
# =========================
available_models = []

if bert_exists:
    available_models.append("BERT")

if roberta_exists:
    available_models.append("RoBERTa")

if bert_exists and roberta_exists:
    available_models.append("BERT + RoBERTa 加權融合")

st.markdown('<div class="card">', unsafe_allow_html=True)

model_choice = st.selectbox(
    "請選擇要使用的模型",
    available_models
)

if model_choice == "BERT":
    tokenizer, model = load_model(BERT_MODEL_PATH)
    current_model_name = "BERT"
elif model_choice == "RoBERTa":
    tokenizer, model = load_model(ROBERTA_MODEL_PATH)
    current_model_name = "RoBERTa"
else:
    bert_tokenizer, bert_model = load_model(BERT_MODEL_PATH)
    roberta_tokenizer, roberta_model = load_model(ROBERTA_MODEL_PATH)
    current_model_name = "BERT + RoBERTa 加權融合"

st.success(f"目前使用模型：{current_model_name}")
st.caption(f"目前運算裝置：{device}")

if model_choice == "BERT + RoBERTa 加權融合":
    st.info("加權設定：class 0 = RoBERTa 0.85 + BERT 0.15；class 1 = RoBERTa 0.35 + BERT 0.65")

st.markdown('</div>', unsafe_allow_html=True)

# =========================
# 上方功能欄位
# =========================
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 首頁",
    "🔍 單句情緒分析",
    "📁 CSV 批次分析",
    "📌 專案說明"
])

# =========================
# 首頁
# =========================
with tab1:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="card">
            <h3>🔍 單句分析</h3>
            <p class="small-text">輸入任意英文句子，系統會即時判斷情緒為正向或負向。</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <h3>📁 CSV 批次分析</h3>
            <p class="small-text">上傳包含 text 欄位的 CSV 檔案，可一次分析多筆資料。</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card">
            <h3>📊 加權融合</h3>
            <p class="small-text">可將 BERT 與 RoBERTa 的預測機率依照指定權重融合。</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card">
        <h3>系統簡介</h3>
        <p>
        本系統可使用 BERT、RoBERTa 或 BERT + RoBERTa 加權融合模型進行情緒分析。
        使用者可以輸入單一句子，或上傳 CSV 檔案進行批次分析。
        分析結果會顯示預測情緒、信心分數，並透過圖表呈現正負向比例。
        </p>
        <p><b>目前使用模型：</b>{current_model_name}</p>
    </div>
    """, unsafe_allow_html=True)

# =========================
# 單句情緒分析
# =========================
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.header("🔍 單句情緒分析")
    st.write(f"請輸入一段英文文字，系統會使用 **{current_model_name}** 進行情緒判斷。")

    user_input = st.text_area(
        "輸入文字",
        placeholder="例如：I love this product. It is amazing!",
        height=160
    )

    analyze_btn = st.button("開始分析", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    if analyze_btn:
        if user_input.strip() == "":
            st.warning("請先輸入文字。")
        else:
            label, confidence, predicted_class, emoji = predict_sentiment(user_input)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-number">{emoji}</div>
                    <div class="metric-label">情緒圖示</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-number">{predicted_class}</div>
                    <div class="metric-label">預測類別</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-number">{confidence:.2%}</div>
                    <div class="metric-label">信心分數</div>
                </div>
                """, unsafe_allow_html=True)

            st.write("")

            if predicted_class == 1:
                st.markdown(
                    f'<div class="result-positive">{emoji} 預測結果：{label}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="result-negative">{emoji} 預測結果：{label}</div>',
                    unsafe_allow_html=True
                )

            st.write("信心分數")
            st.progress(confidence)

# =========================
# CSV 批次分析
# =========================
with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.header("📁 CSV 批次情緒分析")
    st.write(f"請上傳 CSV 檔案，系統會使用 **{current_model_name}** 分析。")
    st.write("CSV 檔案中必須包含 `text` 欄位。")

    example_df = pd.DataFrame({
        "text": [
            "I love this movie.",
            "This product is terrible.",
            "The service is very good."
        ]
    })

    st.write("範例格式：")
    st.dataframe(example_df, use_container_width=True)

    uploaded_file = st.file_uploader("上傳 CSV 檔案", type=["csv"])

    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if "text" not in df.columns:
            st.error("CSV 檔案中找不到 `text` 欄位，請確認欄位名稱。")
        else:
            st.subheader("原始資料")
            st.dataframe(df, use_container_width=True)

            if st.button("開始批次分析", use_container_width=True):
                results = []
                confidences = []
                classes = []

                for text in df["text"]:
                    label, confidence, predicted_class, emoji = predict_sentiment(text)
                    results.append(label)
                    confidences.append(confidence)
                    classes.append(predicted_class)

                df["prediction"] = results
                df["class"] = classes
                df["confidence"] = confidences
                df["model"] = current_model_name

                st.subheader("分析結果")
                st.dataframe(df, use_container_width=True)

                positive_count = (df["class"] == 1).sum()
                negative_count = (df["class"] == 0).sum()
                total_count = len(df)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("總資料筆數", total_count)

                with col2:
                    st.metric("正向數量", positive_count)

                with col3:
                    st.metric("負向數量", negative_count)

                st.subheader("情緒分佈圖")

                sentiment_counts = df["prediction"].value_counts()

                fig, ax = plt.subplots(figsize=(7, 4))
                ax.bar(sentiment_counts.index, sentiment_counts.values)
                ax.set_xlabel("Sentiment")
                ax.set_ylabel("Count")
                ax.set_title(f"Sentiment Distribution - {current_model_name}")

                st.pyplot(fig)
                plt.close(fig)

                csv_result = df.to_csv(index=False).encode("utf-8-sig")

                st.download_button(
                    label="下載分析結果 CSV",
                    data=csv_result,
                    file_name="sentiment_result.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# =========================
# 專案說明
# =========================
with tab4:
    st.markdown("""
    <div class="card">
        <h2>📌 專案名稱</h2>
        <p>結合深度學習與視覺化之多平台情緒分析系統</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h2>🎯 專案目的</h2>
        <p>
        本專案希望透過 BERT、RoBERTa 與加權融合方法，分析使用者輸入文字或 CSV 資料中的情緒傾向。
        同時利用 Streamlit 建立互動式前端介面，讓使用者能更直觀地查看分析結果。
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h2>⚙️ 系統功能</h2>
        <ol>
            <li>選擇 BERT、RoBERTa 或 BERT + RoBERTa 加權融合</li>
            <li>單句文字情緒分析</li>
            <li>CSV 批次情緒分析</li>
            <li>顯示模型預測結果與信心分數</li>
            <li>產生情緒分佈視覺化圖表</li>
            <li>下載分析完成的 CSV 結果</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h2>🛠 使用技術</h2>
        <p>
        Python、Streamlit、PyTorch、Hugging Face Transformers、BERT、RoBERTa、Pandas、Matplotlib
        </p>
    </div>
    """, unsafe_allow_html=True)
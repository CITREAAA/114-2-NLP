import os
import torch
import pandas as pd
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="BERT / RoBERTa Sentiment Analysis System",
    page_icon="💬",
    layout="wide"
)

# =========================
# Model Paths and Device Settings
# =========================
BERT_MODEL_PATH = "./my_bert_model"
ROBERTA_MODEL_PATH = "./my_roberta_model"
FINAL_MODEL_PATH = "./final_model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensemble weights
# class 0: RoBERTa 0.85 + BERT 0.15
# class 1: RoBERTa 0.35 + BERT 0.65
weight_roberta = torch.tensor([0.85, 0.35], dtype=torch.float32).to(device)
weight_bert = torch.tensor([0.15, 0.65], dtype=torch.float32).to(device)

# =========================
# CSS Styling
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
# Load Model
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


@st.cache_resource
def load_final_model(final_model_path, tokenizer_path):
    """
    Load Final Model weights from ./final_model,
    but use RoBERTa tokenizer and config from ./my_roberta_model.
    This avoids the missing model_type error in final_model/config.json.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=True,
        local_files_only=True
    )

    config = AutoConfig.from_pretrained(
        tokenizer_path,
        local_files_only=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        final_model_path,
        config=config,
        local_files_only=True
    )

    model.to(device)
    model.eval()

    return tokenizer, model

# =========================
# Utility Functions
# =========================
def label_info(predicted_class):
    if predicted_class == 1:
        return "Positive", "😊"
    else:
        return "Negative", "😟"


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

    elif model_choice == "Final Model":
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
# Main Title
# =========================
st.markdown(
    '<div class="main-title">💬 BERT / RoBERTa Sentiment Analysis System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Analyze text sentiment with deep learning models, supporting single sentence prediction, CSV batch analysis, final model prediction, and weighted ensemble</div>',
    unsafe_allow_html=True
)

# =========================
# Check Model Folders
# =========================
bert_exists = os.path.exists(BERT_MODEL_PATH)
roberta_exists = os.path.exists(ROBERTA_MODEL_PATH)
final_exists = os.path.exists(FINAL_MODEL_PATH)

if not bert_exists and not roberta_exists and not final_exists:
    st.error(
        "Model folder not found. Please make sure my_bert_model, my_roberta_model, or final_model is in the same directory as app.py."
    )
    st.stop()

# =========================
# Model Selection
# =========================
available_models = []

if bert_exists:
    available_models.append("BERT")

if roberta_exists:
    available_models.append("RoBERTa")

if final_exists:
    available_models.append("Final Model")

if bert_exists and roberta_exists:
    available_models.append("BERT + RoBERTa Weighted Ensemble")

model_choice = st.selectbox(
    "Select a model",
    available_models
)

if model_choice == "BERT":
    tokenizer, model = load_model(BERT_MODEL_PATH)
    current_model_name = "BERT"

elif model_choice == "RoBERTa":
    tokenizer, model = load_model(ROBERTA_MODEL_PATH)
    current_model_name = "RoBERTa"

elif model_choice == "Final Model":
    if not roberta_exists:
        st.error(
            "Final Model needs the RoBERTa tokenizer and config. Please keep my_roberta_model in the same directory as app.py."
        )
        st.stop()

    tokenizer, model = load_final_model(FINAL_MODEL_PATH, ROBERTA_MODEL_PATH)
    current_model_name = "Final Model"

else:
    bert_tokenizer, bert_model = load_model(BERT_MODEL_PATH)
    roberta_tokenizer, roberta_model = load_model(ROBERTA_MODEL_PATH)
    current_model_name = "BERT + RoBERTa Weighted Ensemble"

st.success(f"Current model: {current_model_name}")
st.caption(f"Current device: {device}")

if model_choice == "BERT + RoBERTa Weighted Ensemble":
    st.info(
        "Weight settings: class 0 = RoBERTa 0.85 + BERT 0.15; class 1 = RoBERTa 0.35 + BERT 0.65"
    )

if model_choice == "Final Model":
    st.info(
        "Final Model is loaded from ./final_model. The tokenizer and config are loaded from ./my_roberta_model."
    )

# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Home",
    "🔍 Single Sentence Analysis",
    "📁 CSV Batch Analysis",
    "📌 Project Description"
])

# =========================
# Home
# =========================
with tab1:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="card">
            <h3>🔍 Single Sentence Analysis</h3>
            <p class="small-text">
            Enter an English sentence and the system will predict whether the sentiment is positive or negative.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <h3>📁 CSV Batch Analysis</h3>
            <p class="small-text">
            Upload a CSV file containing a text column to analyze multiple records at once.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card">
            <h3>📊 Model Selection</h3>
            <p class="small-text">
            Choose BERT, RoBERTa, Final Model, or a weighted ensemble model for prediction.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card">
        <h3>System Overview</h3>
        <p>
        This system can perform sentiment analysis using BERT, RoBERTa, Final Model,
        or the BERT + RoBERTa weighted ensemble model.
        Users can enter a single sentence or upload a CSV file for batch analysis.
        The analysis results display the predicted sentiment, class label, confidence score, and the model used.
        </p>
        <p><b>Current model:</b> {current_model_name}</p>
    </div>
    """, unsafe_allow_html=True)

# =========================
# Single Sentence Analysis
# =========================
with tab2:
    st.header("🔍 Single Sentence Sentiment Analysis")
    st.write(
        f"Enter an English text, and the system will use **{current_model_name}** to predict its sentiment."
    )

    user_input = st.text_area(
        "Input Text",
        placeholder="Example: I love this product. It is amazing!",
        height=160
    )

    analyze_btn = st.button("Analyze", use_container_width=True)

    if analyze_btn:
        if user_input.strip() == "":
            st.warning("Please enter some text first.")
        else:
            label, confidence, predicted_class, emoji = predict_sentiment(user_input)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-number">{emoji}</div>
                    <div class="metric-label">Sentiment Icon</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-number">{predicted_class}</div>
                    <div class="metric-label">Predicted Class</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-number">{confidence:.2%}</div>
                    <div class="metric-label">Confidence Score</div>
                </div>
                """, unsafe_allow_html=True)

            st.write("")

            if predicted_class == 1:
                st.markdown(
                    f'<div class="result-positive">{emoji} Prediction Result: {label}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="result-negative">{emoji} Prediction Result: {label}</div>',
                    unsafe_allow_html=True
                )

            st.write("Confidence Score")
            st.progress(confidence)

# =========================
# CSV Batch Analysis
# =========================
with tab3:
    st.header("📁 CSV Batch Sentiment Analysis")
    st.write(
        f"Upload a CSV file, and the system will analyze it using **{current_model_name}**."
    )
    st.write("The CSV file must contain a `text` column.")

    example_df = pd.DataFrame({
        "text": [
            "I love this movie.",
            "This product is terrible.",
            "Great job Mercedes, another masterclass in disappointment."
        ]
    })

    st.write("Example format:")
    st.dataframe(example_df, use_container_width=True)

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if "text" not in df.columns:
            st.error(
                "The `text` column was not found in the CSV file. Please check the column name."
            )
        else:
            st.subheader("Original Data")
            st.dataframe(df, use_container_width=True)

            if st.button("Start Batch Analysis", use_container_width=True):
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

                st.subheader("Analysis Results")
                st.dataframe(df, use_container_width=True)

                positive_count = (df["class"] == 1).sum()
                negative_count = (df["class"] == 0).sum()
                total_count = len(df)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Total Records", total_count)

                with col2:
                    st.metric("Positive Count", positive_count)

                with col3:
                    st.metric("Negative Count", negative_count)

                csv_result = df.to_csv(index=False).encode("utf-8-sig")

                st.download_button(
                    label="Download Result CSV",
                    data=csv_result,
                    file_name="sentiment_result.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# =========================
# Project Description
# =========================
with tab4:
    st.markdown("""
    <div class="card">
        <h2>📌 Project Title</h2>
        <p>A Multi-platform Sentiment Analysis System Combining Deep Learning and Visualization</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h2>🎯 Project Objective</h2>
        <p>
        This project aims to analyze the sentiment tendency of user-input text or CSV data
        using BERT, RoBERTa, Final Model, and a weighted ensemble method.
        Streamlit is used to build an interactive frontend interface, allowing users to view
        analysis results more intuitively.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h2>⚙️ System Features</h2>
        <ol>
            <li>Select BERT, RoBERTa, Final Model, or BERT + RoBERTa weighted ensemble</li>
            <li>Single sentence sentiment analysis</li>
            <li>CSV batch sentiment analysis</li>
            <li>Display prediction results and confidence scores</li>
            <li>Download completed CSV analysis results</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h2>🛠 Technologies Used</h2>
        <p>
        Python, Streamlit, PyTorch, Hugging Face Transformers, BERT, RoBERTa, Pandas
        </p>
    </div>
    """, unsafe_allow_html=True)
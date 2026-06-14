import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
df = pd.read_csv("dataset/processed_data.csv", encoding='utf-8')
df.columns = df.columns.str.strip()
print("實際欄位：", df.columns.tolist())
print("Sentiment in columns:", 'Sentiment' in df.columns)

# ================= 1. 配置參數 =================
MODEL_PATH  = "./final_model"           # train.py 儲存的模型路徑
DATA_PATH   = "dataset/processed_data.csv"
SAMPLE_SIZE = 5000                      # 隨機抽樣筆數
MAX_LEN     = 64
BATCH_SIZE  = 32
RANDOM_SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用裝置：{device}")

# ================= 2. Dataset =================
class TwitterDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts  = df['clean_text'].values
        self.labels = df['Sentiment'].values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids':      enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ================= 3. 讀取資料並抽樣 =================
print(f"\n正在讀取資料：{DATA_PATH}")
df = pd.read_csv(DATA_PATH, encoding='utf-8')
df.columns = df.columns.str.strip()

# 確認欄位
assert 'clean_text' in df.columns and 'Sentiment' in df.columns, \
    f"找不到必要欄位！現有欄位：{df.columns.tolist()}"

# 分層抽樣：確保兩個類別比例均衡
sample_per_class = SAMPLE_SIZE // 2
sampled = (
    df.groupby('Sentiment', group_keys=False)
      .apply(lambda x: x.sample(min(len(x), sample_per_class), random_state=RANDOM_SEED))
      .reset_index(drop=True)
)
print(f"抽樣完成：共 {len(sampled)} 筆")
print(sampled['Sentiment'].value_counts().to_string())

# ================= 4. 載入模型與 Tokenizer =================
print(f"\n正在載入模型：{MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()

# ================= 5. 推論 =================
loader = DataLoader(
    TwitterDataset(sampled, tokenizer, MAX_LEN),
    batch_size=BATCH_SIZE,
    shuffle=False
)

all_preds, all_labels, all_probs = [], [], []

print("\n正在推論中...")
with torch.no_grad():
    for batch in loader:
        ids    = batch['input_ids'].to(device)
        mask   = batch['attention_mask'].to(device)
        labels = batch['labels']

        outputs = model(ids, attention_mask=mask)
        probs   = torch.softmax(outputs.logits, dim=1).cpu().numpy()  # shape: (B, 2)
        preds   = np.argmax(probs, axis=1)

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_probs.extend(probs[:, 1])  # Positive 類別的機率

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)

# ================= 6. 指標計算 =================
print("\n" + "="*50)
print("        模型評估報告（抽樣 5000 筆）")
print("="*50)

accuracy = accuracy_score(all_labels, all_preds)
auc      = roc_auc_score(all_labels, all_probs)
report   = classification_report(
    all_labels, all_preds,
    target_names=['Negative', 'Positive'],
    digits=4
)

print(f"\n整體準確率 (Accuracy) : {accuracy:.4f}")
print(f"AUC-ROC              : {auc:.4f}")
print(f"\n分類報告：\n{report}")

# ================= 7. 視覺化 =================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Model Evaluation (5,000 Samples)", fontsize=15, fontweight='bold')

# --- 7a. Confusion Matrix ---
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
    xticklabels=['Negative', 'Positive'],
    yticklabels=['Negative', 'Positive'],
    linewidths=0.5
)
axes[0].set_title("Confusion Matrix")
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("True Label")

# 在格子裡補上百分比
total = cm.sum()
for i in range(2):
    for j in range(2):
        axes[0].text(
            j + 0.5, i + 0.72,
            f"({cm[i,j]/total*100:.1f}%)",
            ha='center', va='center',
            fontsize=9, color='gray'
        )

# --- 7b. ROC Curve ---
fpr, tpr, _ = roc_curve(all_labels, all_probs)
axes[1].plot(fpr, tpr, color='steelblue', lw=2, label=f"AUC = {auc:.4f}")
axes[1].plot([0, 1], [0, 1], 'k--', lw=1, label="Random Classifier")
axes[1].fill_between(fpr, tpr, alpha=0.1, color='steelblue')
axes[1].set_title("ROC Curve")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].legend(loc='lower right')
axes[1].set_xlim([0, 1])
axes[1].set_ylim([0, 1.02])

plt.tight_layout()
output_path = "evaluation_results.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✅ 視覺化圖表已儲存：{output_path}")
plt.show()

# ================= 8. 錯誤分析（選用） =================
sampled['pred']       = all_preds
sampled['prob_pos']   = all_probs
sampled['is_correct'] = (all_preds == all_labels)

wrong = sampled[~sampled['is_correct']]
print(f"\n❌ 預測錯誤共 {len(wrong)} 筆（錯誤率 {len(wrong)/len(sampled)*100:.2f}%）")
print("\n最具信心但預測錯誤的 10 筆（high-confidence mistakes）：")

# 找出模型最確定但還是錯的樣本
wrong = wrong.copy()
wrong['confidence'] = wrong.apply(
    lambda r: r['prob_pos'] if r['pred'] == 1 else 1 - r['prob_pos'], axis=1
)
print(
    wrong.nlargest(10, 'confidence')[['clean_text', 'Sentiment', 'pred', 'confidence']]
      .rename(columns={'Sentiment':'true', 'pred':'predicted'})
      .to_string(index=False)
)

print("\n✅ 評估完成！")
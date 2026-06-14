import pandas as pd
import numpy as np
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm # 用於顯示進度條

# ================= 配置設定 =================
DATA_PATH = 'dataset/Sentiment-Analysis-Dataset.csv'
MODEL_NAME = 'google-bert/bert-base-uncased'
MAX_LEN = 64
BATCH_SIZE = 16
EPOCHS = 2 # 建議跑 2~3 個 Epoch 即可
LEARNING_RATE = 2e-5
SAMPLE_SIZE = 100000

# 確認計算裝置 (GPU 或 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用裝置: {device}")

# ================= 1. 讀取與清理資料 =================
print(f"正在讀取並抽樣 {SAMPLE_SIZE} 筆資料...")
df = pd.read_csv(DATA_PATH, on_bad_lines='skip', low_memory=False)
df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

def clean_for_bert(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text) # 去除網址
    text = re.sub(r'@\w+', '', text)                   # 去除使用者標記
    # 備註：我們保留標點符號與 Emoji，因為 BERT 能理解其語氣
    return text.strip()

df['CleanedText'] = df['SentimentText'].apply(clean_for_bert)

# ================= 2. 建立資料載入器 =================
class TwitterDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text, add_special_tokens=True, max_length=self.max_len,
            return_token_type_ids=False, padding='max_length',
            truncation=True, return_attention_mask=True, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

train_data = TwitterDataset(train_df.CleanedText.values, train_df.Sentiment.values, tokenizer, MAX_LEN)
val_data = TwitterDataset(val_df.CleanedText.values, val_df.Sentiment.values, tokenizer, MAX_LEN)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

# ================= 3. 模型與優化器設定 =================
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# ================= 4. 訓練循環 =================
for epoch in range(EPOCHS):
    print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
    model.train()
    total_train_loss = 0
    
    # 訓練階段
    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        model.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_train_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 梯度裁剪防止梯度爆炸
        optimizer.step()
        scheduler.step()
        
    avg_train_loss = total_train_loss / len(train_loader)
    print(f"平均訓練 Loss: {avg_train_loss:.4f}")

    # 驗證階段
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).flatten().cpu().numpy()
            
            val_preds.extend(preds)
            val_labels.extend(labels.cpu().numpy())
            
    print(f"Validation Accuracy: {accuracy_score(val_labels, val_preds):.4f}")

# ================= 5. 儲存模型 =================
save_path = "./my_bert_model"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"\n訓練完成！模型已儲存至 {save_path}")
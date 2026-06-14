import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
from tqdm import tqdm

# ================= 1. 配置參數 =================
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
DATA_PATH  = "dataset/processed_data.csv"
MAX_LEN    = 64
BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 4
EPOCHS     = 1
TRAIN_SIZE = None  # 每個類別抽 5 萬，總計 10 萬筆（設 None 則用全部資料）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用裝置：{device}")

# ================= 2. 融合模型架構 =================
class RoBERTaWithFeatures(nn.Module):
    """
    架構圖：
    
    輸入文字 → RoBERTa → [CLS] 向量 (768維) ──┐
                                               concat → Linear(771→256) → ReLU → Dropout → Linear(256→2)
    手工特徵 (3維) ─────────────────────────────┘
    """
    def __init__(self, roberta_model, num_features=3, num_labels=2, dropout=0.3):
        super().__init__()
        self.roberta = roberta_model
        hidden_size  = self.roberta.config.hidden_size  # 768

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask, features):
        # 取 [CLS] token 的輸出（第 0 個 token）
        outputs    = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch, 768)

        # 拼接手工特徵
        combined = torch.cat([cls_output, features], dim=1)  # (batch, 771)
        return self.classifier(combined)

# ================= 3. Dataset =================
class TwitterFusionDataset(Dataset):
    def __init__(self, df, tokenizer, scaler, max_len, fit_scaler=False):
        self.texts  = df['clean_text'].values
        self.labels = df['Sentiment'].values
        self.tokenizer = tokenizer
        self.max_len   = max_len

        # 手工特徵標準化
        raw_features = df[['cap_ratio', 'punct_density', 'negation_count']].values.astype(np.float32)
        if fit_scaler:
            self.features = scaler.fit_transform(raw_features)
        else:
            self.features = scaler.transform(raw_features)

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
            'features':       torch.tensor(self.features[idx], dtype=torch.float),
            'labels':         torch.tensor(self.labels[idx],   dtype=torch.long)
        }

# ================= 4. 讀取資料 =================
print("\n正在讀取資料...")
df = pd.read_csv(DATA_PATH, encoding='utf-8')
df.columns = df.columns.str.strip()

required = ['clean_text', 'Sentiment', 'cap_ratio', 'punct_density', 'negation_count']
missing  = [c for c in required if c not in df.columns]
assert not missing, f"缺少欄位：{missing}"

# 分層抽樣（各類別各取 TRAIN_SIZE 筆）
if TRAIN_SIZE:
    parts = [grp.sample(min(len(grp), TRAIN_SIZE), random_state=42)
             for _, grp in df.groupby('Sentiment', group_keys=False)]
    df = pd.concat(parts).reset_index(drop=True)
    print(f"抽樣後資料量：{len(df)} 筆（每類別最多 {TRAIN_SIZE} 筆）")

train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
print(f"訓練集：{len(train_df)} 筆　驗證集：{len(val_df)} 筆")

# ================= 5. Tokenizer、Scaler、DataLoader =================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
scaler    = StandardScaler()

train_dataset = TwitterFusionDataset(train_df, tokenizer, scaler, MAX_LEN, fit_scaler=True)
val_dataset   = TwitterFusionDataset(val_df,   tokenizer, scaler, MAX_LEN, fit_scaler=False)

train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# 儲存 scaler 供推論時使用
joblib.dump(scaler, "fusion_scaler.pkl")
print("✅ Scaler 已儲存：fusion_scaler.pkl")

# ================= 6. 模型、損失函數、優化器 =================
base_model = AutoModel.from_pretrained(MODEL_NAME)

# 凍結前 6 層，減少運算量
for param in base_model.encoder.layer[:6].parameters():
    param.requires_grad = False

model = RoBERTaWithFeatures(base_model).to(device)

class_counts = df['Sentiment'].value_counts()
weights  = torch.tensor(
    [len(df) / class_counts[0], len(df) / class_counts[1]],
    dtype=torch.float
).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
total_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * EPOCHS
scheduler   = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

# ================= 7. 訓練迴圈 =================
for epoch in range(EPOCHS):
    model.train()
    model.zero_grad()
    print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")

    for i, batch in enumerate(tqdm(train_loader)):
        ids      = batch['input_ids'].to(device)
        mask     = batch['attention_mask'].to(device)
        features = batch['features'].to(device)
        labels   = batch['labels'].to(device)

        logits = model(ids, mask, features)
        loss   = criterion(logits, labels) / GRAD_ACCUM_STEPS
        loss.backward()

        if (i + 1) % GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            scheduler.step()
            model.zero_grad()

    # --- 驗證 ---
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            ids      = batch['input_ids'].to(device)
            mask     = batch['attention_mask'].to(device)
            features = batch['features'].to(device)
            labels   = batch['labels']

            logits = model(ids, mask, features)
            preds  = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print(classification_report(all_labels, all_preds, target_names=['Neg', 'Pos'], digits=4))

# ================= 8. 儲存 =================
# 分開儲存 RoBERTa 主體與分類頭
base_model.save_pretrained("./fusion_model")
torch.save(model.classifier.state_dict(), "./fusion_model/classifier_head.pt")
tokenizer.save_pretrained("./fusion_model")
print("\n✅ 融合模型訓練完成並儲存至 ./fusion_model")
import torch
import io
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report
from tqdm import tqdm
import sys
import os

# 1. 基本設定
save_path = "./my_bert_model"
data_path = 'dataset/Sentiment-Analysis-Dataset.csv'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 計算裝置: {device}")

# 2. 載入模型
print("正在載入模型...")
tokenizer = BertTokenizer.from_pretrained(save_path)
model = BertForSequenceClassification.from_pretrained(save_path).to(device)
model.eval()

# 3. 核彈級讀取方案：逐行掃描 (不使用 Pandas)
print("正在讀取資料集 (原生逐行掃描模式)...")
y_true = []
texts = []

try:
    with open(data_path, 'r', encoding='latin-1') as f:
        # 跳過第一行 (Header)
        header = f.readline()
        
        # 讀取前 5000 筆有效資料
        skip_rows = 500000 
        count = 0
        skipped = 0
        for line in f:
            if skipped < skip_rows:
                skipped += 1
                continue
            if count >= 5000: break
            
            # Twitter CSV 格式通常是: ItemID, Sentiment, SentimentSource, SentimentText
            # 我們用逗號切割，但只切前 3 個逗號，剩下的全是文字內容
            parts = line.split(',', 3)
            if len(parts) >= 4:
                sentiment = parts[1].strip().replace('"', '')
                text = parts[3].strip().replace('"', '')
                
                if sentiment in ['0', '1']:
                    y_true.append(int(sentiment))
                    texts.append(text)
                    count += 1
                    
    print(f"✅ 成功讀取 {len(texts)} 筆資料！")
except Exception as e:
    print(f"❌ 讀取失敗：{str(e)}")
    sys.exit()

# 4. 批次預測
y_pred = []
batch_size = 32
print(f"開始進行效能評估...")

for i in tqdm(range(0, len(texts), batch_size), desc="Evaluating"):
    batch_texts = texts[i:i+batch_size]
    inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        y_pred.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())

# 5. 產出報告
if len(y_pred) > 0:
    report = classification_report(y_true, y_pred, target_names=['Negative', 'Positive'], digits=4)
    print("\n" + "="*40 + "\n" + report + "\n" + "="*40)
    with open("final_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print("✅ 指標報表已儲存至 final_report.txt")
else:
    print("❌ 錯誤：沒有預測結果產生。")
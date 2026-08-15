import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix

test_df = pd.read_csv('data/raw/test.csv')

tokenizer = AutoTokenizer.from_pretrained("models/router")
model = AutoModelForSequenceClassification.from_pretrained("models/router")
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
model.eval()

preds = []
batch_size = 16
texts = test_df['text'].tolist()

for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    inputs = tokenizer(batch, truncation=True, max_length=128, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    batch_preds = outputs.logits.argmax(dim=1).cpu().tolist()
    preds.extend(batch_preds)
    if i % 1600 == 0:
        print(f"{i}/{len(texts)}")

id2label = model.config.id2label
pred_labels = [id2label[p] for p in preds]

print()
print(classification_report(test_df['category'], pred_labels))
print()
print("confusion matrix (rows=true, cols=predicted):")
labels_sorted = sorted(test_df['category'].unique())
cm = confusion_matrix(test_df['category'], pred_labels, labels=labels_sorted)
print("labels order:", labels_sorted)
print(cm)

import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from sklearn.metrics import f1_score

# ---- 1. load our three piles ----
train_df = pd.read_csv('data/raw/train_boosted2.csv')
val_df   = pd.read_csv('data/raw/val.csv')

# ---- 2. turn category names into numbers (model needs numbers, not words) ----
labels = sorted(train_df['category'].unique())
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
print("label map:", label2id)

train_df['label'] = train_df['category'].map(label2id)
val_df['label']   = val_df['category'].map(label2id)

# ---- 3. convert to the format the model library expects ----
train_ds = Dataset.from_pandas(train_df[['text', 'label']])
val_ds   = Dataset.from_pandas(val_df[['text', 'label']])

# ---- 4. load the pre-trained model's tokenizer ----
# a tokenizer chops text into pieces the model understands
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, max_length=128, padding='max_length')

train_ds = train_ds.map(tokenize, batched=True)
val_ds   = val_ds.map(tokenize, batched=True)

# ---- 5. load the pre-trained model, tell it we want 4 output categories ----
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
)

# ---- 6. detect the fastest available device ----
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("training on:", device)

# ---- 7. how we judge the model while it trains ----
def compute_metrics(eval_pred):
    preds = np.argmax(eval_pred.predictions, axis=1)
    return {"macro_f1": f1_score(eval_pred.label_ids, preds, average="macro")}

# ---- 8. training settings, tuned for 8GB RAM ----
args = TrainingArguments(
    output_dir="models/router_checkpoints",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    
    fp16=False,   # MPS does not support fp16 - must stay False
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

# ---- 9. train ----
trainer.train()

# ---- 10. save the finished model ----
trainer.save_model("models/router")
tokenizer.save_pretrained("models/router")
print("saved model to models/router")

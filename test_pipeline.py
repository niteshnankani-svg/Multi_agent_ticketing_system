import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from ingest import ingest_ticket

# ---- load the trained classifier, once ----
tokenizer = AutoTokenizer.from_pretrained("models/router")
model = AutoModelForSequenceClassification.from_pretrained("models/router")
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
model.eval()

def classify(clean_text: str):
    """Runs the trained model on cleaned text, returns (category, confidence)."""
    inputs = tokenizer(clean_text, truncation=True, max_length=128,
                        padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0]
    top_id = probs.argmax().item()
    category = model.config.id2label[top_id]
    confidence = probs[top_id].item()
    return category, confidence

# ---- the full pipeline: raw ticket in, category out ----
test_tickets = [
    "My invoice for last month shows the wrong amount, can you check my billing?",
    "The server keeps crashing every time I try to upload a file, getting a 500 error.",
    "I need to update my emergency contact information in the employee system.",
    "What are your business hours and where are you located?",
]

for raw in test_tickets:
    state = ingest_ticket(ticket_id="demo", raw_text=raw)
    category, confidence = classify(state["clean_text"])
    state["category"] = category
    state["category_confidence"] = confidence

    print(f"RAW:        {raw}")
    print(f"CATEGORY:   {category}  (confidence: {confidence:.2%})")
    print("-" * 60)


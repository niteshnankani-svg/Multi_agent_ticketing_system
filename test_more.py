from test_pipeline import classify, ingest_ticket

tickets = [
    ("I was denied access to the shared drive, can IT fix my permissions?", "internal"),
    ("The app crashes on login, error code 500 every time.", "backend"),
    ("Can I get a refund for my last order?", "finance"),
    ("Do you offer weekend support?", "general"),
    ("My paycheck was short this month, who do I contact?", "internal"),
    ("The database connection keeps timing out.", "backend"),
]

for text, expected in tickets:
    state = ingest_ticket("t", text)
    cat, conf = classify(state["clean_text"])
    mark = "OK" if cat == expected else "WRONG"
    print(f"[{mark}] expected={expected:9} got={cat:9} ({conf:.1%})  {text}")

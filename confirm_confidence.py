from graph import app

texts = [
    "asdkjf aslkdjf gibberish",
    "asdkjf aslkdjf random gibberish text",
    "asdkjf aslkdjf gibberish",
]

for i, text in enumerate(texts):
    config = {"configurable": {"thread_id": f"confirm-{i}"}}
    result = app.invoke({
        "ticket_id": "test", "raw_text": text, "clean_text": "",
        "category": None, "category_confidence": None,
        "answer": None, "needs_human": False, "escalation_reason": None,
    }, config=config)
    print(f"{result['category_confidence']:.4f}  {text!r}")

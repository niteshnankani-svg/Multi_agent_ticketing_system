from graph import app

config = {"configurable": {"thread_id": "sqlite-final-test"}}
result = app.invoke({
    "ticket_id": "test", "raw_text": "asdkjf aslkdjf random gibberish text", "clean_text": "",
    "category": None, "category_confidence": None,
    "answer": None, "needs_human": False, "escalation_reason": None,
}, config=config)

print("confidence:", result["category_confidence"])
print("paused:", "__interrupt__" in result)

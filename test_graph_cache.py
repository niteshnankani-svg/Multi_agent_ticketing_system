from graph import app
from semantic_cache import store_in_cache

# ---- 1. first time asking - should MISS cache, go through real pipeline ----
result1 = app.invoke({
    "ticket_id": "t1", "raw_text": "How long does a refund take?", "clean_text": "",
    "category": None, "category_confidence": None,
    "answer": None, "needs_human": False, "escalation_reason": None,
}, config={"configurable": {"thread_id": "graphcache-1"}})

print("FIRST ASK (should be a real category, not 'cached'):")
print("category:", result1["category"])
print("answer:", result1["answer"][:80])
print()

# ---- 2. manually store that answer, simulating what auto-save will do later ----
store_in_cache("How long does a refund take?", result1["answer"])

# ---- 3. ask a paraphrase - should HIT cache this time ----
result2 = app.invoke({
    "ticket_id": "t2", "raw_text": "How many days until my refund shows up?", "clean_text": "",
    "category": None, "category_confidence": None,
    "answer": None, "needs_human": False, "escalation_reason": None,
}, config={"configurable": {"thread_id": "graphcache-2"}})

print("PARAPHRASED ASK (should say 'cached'):")
print("category:", result2["category"])
print("answer:", result2["answer"][:80])

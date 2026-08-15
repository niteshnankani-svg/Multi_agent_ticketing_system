from graph import app
from langgraph.types import Command

config = {'configurable': {'thread_id': 'test-gibberish-3'}}

# ---- step 1: pause ----
result1 = app.invoke({
    'ticket_id': 'test', 'raw_text': 'asdkjf aslkdjf random gibberish text', 'clean_text': '',
    'category': None, 'category_confidence': None,
    'answer': None, 'needs_human': False, 'escalation_reason': None,
}, config=config)

print("AFTER FIRST INVOKE (should be paused):")
print(result1)
print()

# ---- step 2: resume, same process, same thread_id ----
result2 = app.invoke(
    Command(resume={'answer': "Thanks for reaching out - could you clarify what you need help with?"}),
    config=config
)

print("AFTER RESUME (should have a real answer):")
print(result2)

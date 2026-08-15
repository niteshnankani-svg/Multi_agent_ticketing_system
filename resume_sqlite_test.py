from graph import app
from langgraph.types import Command

config = {"configurable": {"thread_id": "sqlite-final-test"}}
result = app.invoke(
    Command(resume={"answer": "A human replied after a full process restart."}),
    config=config
)

print("RESUMED:", result["answer"])
print("needs_human:", result["needs_human"])
print("escalation_reason:", result["escalation_reason"])

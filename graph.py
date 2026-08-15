import torch
import sqlite3
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt
from langgraph.checkpoint.sqlite import SqliteSaver
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ingest import ingest_ticket
from generate_answer import answer_finance_ticket
from generate_answer_backend import answer_backend_ticket
from internal_agent import answer_internal_ticket
from generate_answer_general import answer_general_ticket
from semantic_cache import check_cache, store_in_cache

# ---- 1. the shared state ----
class TicketState(TypedDict):
    ticket_id: str
    raw_text: str
    clean_text: str
    category: Optional[str]
    category_confidence: Optional[float]
    answer: Optional[str]
    needs_human: bool
    escalation_reason: Optional[str]

# ---- 2. load the trained classifier once ----
tokenizer = AutoTokenizer.from_pretrained("models/router")
model = AutoModelForSequenceClassification.from_pretrained("models/router")
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
model.eval()

# ---- 3. NODE: clean the ticket ----
def ingest_node(state: TicketState) -> TicketState:
    result = ingest_ticket(state["ticket_id"], state["raw_text"])
    state["clean_text"] = result["clean_text"]
    return state

# ---- 4. NODE: check cache before doing any real work ----
def cache_check_node(state: TicketState) -> TicketState:
    cached_answer = check_cache(state["clean_text"])
    if cached_answer:
        state["answer"] = cached_answer
        state["category"] = "cached"
        state["category_confidence"] = 1.0
    return state

def cache_hit_or_miss(state: TicketState) -> str:
    if state["category"] == "cached":
        return "hit"
    return "miss"

# ---- 5. NODE: classify ----
def classify_node(state: TicketState) -> TicketState:
    inputs = tokenizer(state["clean_text"], truncation=True, max_length=128,
                        padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0]
    top_id = probs.argmax().item()
    state["category"] = model.config.id2label[top_id]
    state["category_confidence"] = probs[top_id].item()
    return state

# ---- 6. confidence gate - decides answer vs escalate ----
def confidence_gate(state: TicketState) -> str:
    if state["category_confidence"] < 0.70:
        return "escalate"
    return state["category"]

# ---- 7. NODE: the four specialist agents ----
def finance_node(state: TicketState) -> TicketState:
    state["answer"] = answer_finance_ticket(state["clean_text"])
    return state

def backend_node(state: TicketState) -> TicketState:
    state["answer"] = answer_backend_ticket(state["clean_text"])
    return state

def internal_node(state: TicketState) -> TicketState:
    state["answer"] = answer_internal_ticket(state["clean_text"])
    return state

def general_node(state: TicketState) -> TicketState:
    state["answer"] = answer_general_ticket(state["clean_text"])
    return state

# ---- 8. NODE: escalation (pauses via interrupt) ----
def escalation_node(state: TicketState) -> TicketState:
    human_response = interrupt({
        "reason": "low_confidence",
        "ticket_text": state["clean_text"],
        "predicted_category": state["category"],
        "confidence": state["category_confidence"],
    })
    state["answer"] = human_response["answer"]
    state["needs_human"] = True
    state["escalation_reason"] = "low_confidence"
    return state

# ---- 9. NODE: save a fresh answer into the cache ----
def cache_store_node(state: TicketState) -> TicketState:
    if state["category"] != "cached" and state["answer"]:
        store_in_cache(state["clean_text"], state["answer"])
    return state

# ---- 10. build the graph ----
graph = StateGraph(TicketState)

graph.add_node("ingest", ingest_node)
graph.add_node("cache_check", cache_check_node)
graph.add_node("classify", classify_node)
graph.add_node("finance", finance_node)
graph.add_node("backend", backend_node)
graph.add_node("internal", internal_node)
graph.add_node("general", general_node)
graph.add_node("escalation", escalation_node)
graph.add_node("cache_store", cache_store_node)

graph.set_entry_point("ingest")
graph.add_edge("ingest", "cache_check")

graph.add_conditional_edges("cache_check", cache_hit_or_miss, {
    "hit": END,
    "miss": "classify",
})

graph.add_conditional_edges("classify", confidence_gate, {
    "finance": "finance",
    "backend": "backend",
    "internal": "internal",
    "general": "general",
    "escalate": "escalation",
})

graph.add_edge("finance", "cache_store")
graph.add_edge("backend", "cache_store")
graph.add_edge("internal", "cache_store")
graph.add_edge("general", "cache_store")
graph.add_edge("cache_store", END)

graph.add_edge("escalation", END)

# ---- 11. checkpointer - survives process restarts ----
conn = sqlite3.connect("runtime/checkpoints.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)
app = graph.compile(checkpointer=checkpointer)

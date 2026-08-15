import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from graph import app as ticket_graph

app = FastAPI(title="Multi-Agent Ticketing System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TicketRequest(BaseModel):
    raw_text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ticket")
def create_ticket(request: TicketRequest):
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    state = {
        "ticket_id": thread_id,
        "raw_text": request.raw_text,
        "clean_text": "",
        "category": None,
        "category_confidence": None,
        "answer": None,
        "needs_human": False,
        "escalation_reason": None,
    }

    result = ticket_graph.invoke(state, config=config)

    if "__interrupt__" in result:
        interrupt_info = result["__interrupt__"][0].value
        return {
            "status": "needs_human_review",
            "thread_id": thread_id,
            "reason": interrupt_info.get("reason"),
            "predicted_category": interrupt_info.get("predicted_category"),
            "confidence": interrupt_info.get("confidence"),
        }

    return {
        "status": "answered",
        "thread_id": thread_id,
        "category": result.get("category"),
        "answer": result.get("answer"),
        "needs_human": result.get("needs_human"),
        "escalation_reason": result.get("escalation_reason"),
    }

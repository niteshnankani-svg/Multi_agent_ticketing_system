import uuid
from typing import Any, Optional

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


# ---------------------------------------------------------------------------
# Agent Company console adapter
#
# The Agent Company trace console (agent-company-console.vercel.app) proxies
# every request to POST {AGENT_COMPANY_API_URL}/request as
# {"request_id", "raw_input"} and expects its own AgentResponse shape back
# (frontend/lib/types.ts in the agent-company repo). This endpoint runs the
# same LangGraph pipeline as /ticket and translates the result into that
# shape, so the console drives the real ticket router instead of the TF-IDF
# model it falls back to when no backend is reachable.
#
# /ticket above is left untouched for the existing ticket-router-frontend.
# ---------------------------------------------------------------------------

# Which team owns each category once Jev has classified the ticket.
CATEGORY_TEAM = {
    "finance": "Finance Support",
    "backend": "Backend / Infrastructure",
    "internal": "Internal IT",
    "general": "General Support",
}

# The Chroma collection each specialist agent actually retrieves from.
# "internal" is absent on purpose: internal_agent.py answers via the
# internal_systems tools, not a knowledge base.
CATEGORY_KB = {
    "finance": "kb_finance",
    "backend": "kb_backend",
    "general": "kb_general",
}

# How each graph node is labelled in the console's trace graph. The console
# only colours the four SourceType values, so stay inside them.
NODE_SOURCE = {
    "ingest": "local_model",
    "cache_check": "cache",
    "classify": "local_model",
    "finance": "llm",
    "backend": "llm",
    "internal": "llm",
    "general": "llm",
    "escalation": "human_escalation",
    "cache_store": "cache",
}

CONFIDENCE_GATE = 0.70


class AgentRequest(BaseModel):
    raw_input: str
    request_id: Optional[str] = None


def _node_decision(node: str, state: dict[str, Any]) -> str:
    """What the console shows on the trace-graph step for this node."""
    if node == "ingest":
        return "cleaned"
    if node == "cache_check":
        return "hit" if state.get("category") == "cached" else "miss"
    if node == "classify":
        return state.get("category") or "unclassified"
    if node == "escalation":
        return "human_escalation"
    if node == "cache_store":
        return "stored"
    return "answer_drafted"


@app.post("/request")
def agent_request(request: AgentRequest):
    thread_id = str(uuid.uuid4())
    request_id = request.request_id or thread_id
    config = {"configurable": {"thread_id": thread_id}}

    state: dict[str, Any] = {
        "ticket_id": thread_id,
        "raw_text": request.raw_input,
        "clean_text": "",
        "category": None,
        "category_confidence": None,
        "answer": None,
        "needs_human": False,
        "escalation_reason": None,
    }

    # Stream rather than invoke so the trace graph shows the nodes the graph
    # really walked, instead of a path reconstructed from the final state.
    visited: list[str] = []
    final: dict[str, Any] = dict(state)
    interrupt_value: dict[str, Any] = {}

    for chunk in ticket_graph.stream(state, config=config, stream_mode="updates"):
        for node, delta in chunk.items():
            if node == "__interrupt__":
                first = delta[0] if isinstance(delta, (list, tuple)) and delta else delta
                value = getattr(first, "value", first)
                if isinstance(value, dict):
                    interrupt_value = value
                continue
            visited.append(node)
            if isinstance(delta, dict):
                final.update(delta)

    # A node that calls interrupt() never completes, so stream() reports
    # __interrupt__ instead of an update for it and the node is missing from
    # `visited`. escalation_node is the graph's only interrupt site, so an
    # interrupt means the graph did enter it - put it back on the trail, or the
    # console would draw a ticket that stops at `classify` while the decision
    # next to it reads human_escalation.
    if interrupt_value and "escalation" not in visited:
        visited.append("escalation")

    category = final.get("category")
    confidence = final.get("category_confidence")
    answer = final.get("answer")

    # An interrupt means the escalation node paused for a human, so no agent
    # ever drafted an answer.
    escalated = bool(interrupt_value) or bool(final.get("needs_human"))
    cached = category == "cached"

    if escalated:
        category = interrupt_value.get("predicted_category", category)
        if interrupt_value.get("confidence") is not None:
            confidence = interrupt_value["confidence"]

    team = CATEGORY_TEAM.get(category or "", "Unassigned")
    kb = CATEGORY_KB.get(category or "")
    pct = f"{confidence:.0%}" if isinstance(confidence, (int, float)) else "n/a"

    if cached:
        decision = "answered_from_cache"
        source = "cache"
        rationale = (
            "Semantic-cache hit (cosine distance below 0.25) — a near-identical "
            "ticket was answered before, so classification and answer generation "
            "were both skipped.\n\n---\n\n"
            f"{answer}"
        )
        evidence = ["semantic_cache:hit", "collection:ticket_cache", f"thread:{thread_id}"]
    elif escalated:
        decision = "human_escalation"
        source = "human_escalation"
        reason = final.get("escalation_reason") or interrupt_value.get("reason") or "low_confidence"
        if reason == "low_confidence":
            rationale = (
                f"Jev classified this ticket as `{category}` at {pct} confidence, "
                f"below the {CONFIDENCE_GATE:.2f} auto-answer gate. The graph paused "
                "at the `escalation` node for human review — no answer was drafted."
            )
        else:
            rationale = (
                f"Jev classified this ticket as `{category}` at {pct} confidence. "
                f"The graph paused at the `escalation` node for human review "
                f"(`{reason}`) — no answer was drafted."
            )
        evidence = [f"jev_category:{category}", f"thread:{thread_id}"]
    else:
        decision = f"answered:{category}"
        source = "llm"
        retrieved = (
            f"retrieved context from `{kb}`"
            if kb
            else "live lookups through the `internal_systems` tools"
        )
        rationale = (
            f"**{team}** — Jev classified this ticket as `{category}` at {pct} "
            f"confidence, clearing the {CONFIDENCE_GATE:.2f} gate. The {category} "
            f"agent then drafted the answer below from {retrieved}.\n\n---\n\n"
            f"{answer}"
        )
        evidence = [f"jev_category:{category}"]
        evidence.append(f"kb:{kb}" if kb else "tools:internal_systems")
        evidence.append(f"thread:{thread_id}")

    audit_trail = [
        {
            "node": node,
            "decision": _node_decision(node, final),
            "source": NODE_SOURCE.get(node, "local_model"),
            "request_id": request_id,
        }
        for node in visited
    ]

    return {
        "decision": decision,
        "confidence": confidence if isinstance(confidence, (int, float)) else 0.0,
        "rationale": rationale,
        "evidence_ids": evidence,
        "source": source,
        "audit_trail": audit_trail,
        "department_history": None,
        # Deliberately null: the console's BackendTriage panel is hard-coded to
        # the five IT classes of the agent-company DistilBERT head
        # (Access/Hardware/Network/Security/Software). This router's four
        # categories would render as an all-zero chart, so the panel is left out
        # rather than fed mismatched scores.
        "backend_triage": None,
        "escalation_reason": final.get("escalation_reason") or interrupt_value.get("reason"),
        "needs_human": escalated,
        # Not part of AgentResponse; handy for resuming the interrupt later.
        "thread_id": thread_id,
    }

from typing import TypedDict, Optional

class TicketState(TypedDict):
    # ---- filled in when a ticket first arrives ----
    ticket_id: str
    raw_text: str          # the original, untouched ticket text
    clean_text: str        # cleaned version (built in Phase 3)

    # ---- filled in by the classifier (the model we just trained) ----
    category: Optional[str]        # 'finance' | 'backend' | 'general' | 'internal'
    category_confidence: Optional[float]   # how sure the model was, 0.0 to 1.0

    # ---- filled in by whichever agent handles the ticket ----
    answer: Optional[str]

    # ---- filled in if the AI wasn't confident enough ----
    needs_human: bool
    escalation_reason: Optional[str]


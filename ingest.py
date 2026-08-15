import re
import hashlib
from state import TicketState

def normalize_whitespace(text: str) -> str:
    """Collapses multiple spaces/newlines into single spaces."""
    return re.sub(r'\s+', ' ', text).strip()

def scrub_pii(text: str) -> str:
    """Replaces emails, phone numbers, and card-like digit runs with placeholders."""
    text = re.sub(r'[\w.+-]+@[\w-]+\.[\w.-]+', '<EMAIL>', text)
    text = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '<PHONE>', text)
    text = re.sub(r'\b(?:\d[ -]*?){13,16}\b', '<CARD>', text)
    return text

def truncate(text: str, limit: int = 2000) -> str:
    """Keeps text under a length limit, keeping start and end if too long."""
    if len(text) <= limit:
        return text
    head, tail = text[:1200], text[-800:]
    return head + " [...truncated...] " + tail

def make_dedupe_hash(text: str) -> str:
    """A fingerprint of the text - identical tickets get identical hashes."""
    normalized = text.lower().strip()
    return hashlib.sha256(normalized.encode()).hexdigest()

def clean_ticket(raw_text: str) -> str:
    """Runs all cleaning steps in order."""
    text = normalize_whitespace(raw_text)
    text = scrub_pii(text)
    text = truncate(text)
    return text

def ingest_ticket(ticket_id: str, raw_text: str) -> TicketState:
    """Takes a raw ticket, returns a filled-in TicketState ready for the classifier."""
    clean = clean_ticket(raw_text)
    return {
        'ticket_id': ticket_id,
        'raw_text': raw_text,
        'clean_text': clean,
        'category': None,
        'category_confidence': None,
        'answer': None,
        'needs_human': False,
        'escalation_reason': None,
    }


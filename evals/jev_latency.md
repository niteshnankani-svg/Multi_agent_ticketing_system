# Jev classification latency

- **Calls**: 25 (5 tickets x 5 runs each)

- **Overall min**: 0.312s
- **Overall mean**: 0.372s
- **Overall median**: 0.354s
- **Overall stdev**: 0.071s
- **Overall p95**: 0.449s
- **Overall max**: 0.670s

This measures exactly what `graph.py`'s `classify_node` calls - `typesafe_client.system_one()` with the `category` Choice question - not the full ticket pipeline (ingest/cache/agent nodes excluded).

---

## finance

**Ticket**: I was charged twice for my subscription this month, can I get a refund for the duplicate charge?
**Routed to**: `finance` (confidence 1.00)

- min: 0.359s
- mean: 0.446s
- median: 0.400s
- max: 0.670s
- all runs: 0.670s, 0.359s, 0.400s, 0.406s, 0.396s

---

## backend

**Ticket**: The app keeps crashing every time I try to log in, I've tried resetting my password already.
**Routed to**: `backend` (confidence 1.00)

- min: 0.316s
- mean: 0.372s
- median: 0.367s
- max: 0.449s
- all runs: 0.316s, 0.449s, 0.369s, 0.360s, 0.367s

---

## internal

**Ticket**: I need access to the shared marketing drive, my manager approved it last week.
**Routed to**: `internal` (confidence 1.00)

- min: 0.337s
- mean: 0.342s
- median: 0.340s
- max: 0.349s
- all runs: 0.349s, 0.346s, 0.340s, 0.338s, 0.337s

---

## general

**Ticket**: What are your business hours and where are you located?
**Routed to**: `general` (confidence 1.00)

- min: 0.312s
- mean: 0.345s
- median: 0.338s
- max: 0.417s
- all runs: 0.338s, 0.312s, 0.316s, 0.417s, 0.340s

---

## ambiguous

**Ticket**: I can't log into my account to check my last invoice and I think I was overcharged.
**Routed to**: `backend` (confidence 0.46)

- min: 0.334s
- mean: 0.356s
- median: 0.354s
- max: 0.387s
- all runs: 0.354s, 0.387s, 0.359s, 0.348s, 0.334s

---

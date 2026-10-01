# DistilBERT classification latency

- **Run at**: 2026-10-01T16:03:15.442675+00:00
- **Device**: cpu (torch 2.12.0, 4 threads)
- **Calls**: 25 (5 tickets x 5 runs each), 3 warm-up calls excluded

- **Overall min**: 13.8 ms
- **Overall mean**: 15.2 ms
- **Overall median**: 14.8 ms
- **Overall stdev**: 1.2 ms
- **Overall p95**: 17.9 ms
- **Overall max**: 18.3 ms

Measures tokenise + forward pass + softmax for one ticket, the same scope as jev_latency.md.

---

## finance

**Ticket**: I was charged twice for my subscription this month, can I get a refund for the duplicate charge?
**Routed to**: `finance` (confidence 1.00)

- mean: 14.5 ms
- median: 14.4 ms
- all runs: 14.7 ms, 14.7 ms, 14.4 ms, 14.4 ms, 14.3 ms

---

## backend

**Ticket**: The app keeps crashing every time I try to log in, I've tried resetting my password already.
**Routed to**: `finance` (confidence 1.00)

- mean: 15.3 ms
- median: 14.8 ms
- all runs: 18.3 ms, 15.1 ms, 14.7 ms, 13.8 ms, 14.8 ms

---

## internal

**Ticket**: I need access to the shared marketing drive, my manager approved it last week.
**Routed to**: `internal` (confidence 1.00)

- mean: 15.9 ms
- median: 15.3 ms
- all runs: 18.2 ms, 15.7 ms, 15.1 ms, 15.1 ms, 15.3 ms

---

## general

**Ticket**: What are your business hours and where are you located?
**Routed to**: `general` (confidence 1.00)

- mean: 15.5 ms
- median: 15.1 ms
- all runs: 16.9 ms, 15.0 ms, 15.1 ms, 15.6 ms, 14.8 ms

---

## ambiguous

**Ticket**: I can't log into my account to check my last invoice and I think I was overcharged.
**Routed to**: `finance` (confidence 1.00)

- mean: 14.8 ms
- median: 14.4 ms
- all runs: 16.7 ms, 14.6 ms, 14.0 ms, 14.4 ms, 14.4 ms

---

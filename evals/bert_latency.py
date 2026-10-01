#!/usr/bin/env python3
"""DistilBERT classification latency, measured the same way as jev_latency.md.

Same 5 tickets x 5 runs (25 calls). Each call is timed end to end:
tokenise -> forward pass -> softmax -> label. That matches what Jev's number
covers (one classification call, nothing else from the pipeline).

Usage:
    python3 bert_latency.py --model-dir /path/to/your/distilbert_folder

Optional macro-F1 on your held-out set (CSV with columns: text,label):
    python3 bert_latency.py --model-dir /path/to/model --testset test.csv

Run it on the SAME kind of machine Jev was timed from, or say which machine
you used when you quote the numbers (a Mac and an EC2 box will differ).
"""
import argparse
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Same tickets as jev_latency.md, in the same order.
TICKETS = [
    ("finance", "I was charged twice for my subscription this month, can I get a refund for the duplicate charge?"),
    ("backend", "The app keeps crashing every time I try to log in, I've tried resetting my password already."),
    ("internal", "I need access to the shared marketing drive, my manager approved it last week."),
    ("general", "What are your business hours and where are you located?"),
    ("ambiguous", "I can't log into my account to check my last invoice and I think I was overcharged."),
]
RUNS_PER_TICKET = 5
MAX_LENGTH = 128  # same as training


def classify(model, tokenizer, text):
    enc = tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)[0]
    idx = int(probs.argmax())
    return model.config.id2label[idx], float(probs[idx])


def percentile(values, pct):
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def macro_f1(model, tokenizer, csv_path):
    import csv
    from sklearn.metrics import f1_score

    y_true, y_pred = [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            label, _ = classify(model, tokenizer, row["text"])
            y_true.append(row["label"].strip().lower())
            y_pred.append(label.strip().lower())
    return f1_score(y_true, y_pred, average="macro"), len(y_true)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--testset", default=None, help="CSV with text,label columns")
    parser.add_argument("--out", default="bert_latency.md")
    args = parser.parse_args()

    device = "cpu"  # deployed inference is CPU; keep the comparison honest
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device).eval()

    # Warm-up: first calls are slow (lazy init). Not counted.
    for _ in range(3):
        classify(model, tokenizer, TICKETS[0][1])

    all_times, per_ticket = [], {}
    for name, text in TICKETS:
        times, label, conf = [], None, None
        for _ in range(RUNS_PER_TICKET):
            start = time.perf_counter()
            label, conf = classify(model, tokenizer, text)
            times.append(time.perf_counter() - start)
        per_ticket[name] = (text, label, conf, times)
        all_times.extend(times)

    lines = [
        "# DistilBERT classification latency",
        "",
        f"- **Run at**: {datetime.now(timezone.utc).isoformat()}",
        f"- **Device**: {device} (torch {torch.__version__}, {torch.get_num_threads()} threads)",
        f"- **Calls**: {len(all_times)} ({len(TICKETS)} tickets x {RUNS_PER_TICKET} runs each), 3 warm-up calls excluded",
        "",
        f"- **Overall min**: {min(all_times) * 1000:.1f} ms",
        f"- **Overall mean**: {statistics.mean(all_times) * 1000:.1f} ms",
        f"- **Overall median**: {statistics.median(all_times) * 1000:.1f} ms",
        f"- **Overall stdev**: {statistics.stdev(all_times) * 1000:.1f} ms",
        f"- **Overall p95**: {percentile(all_times, 95) * 1000:.1f} ms",
        f"- **Overall max**: {max(all_times) * 1000:.1f} ms",
        "",
        "Measures tokenise + forward pass + softmax for one ticket, the same scope as jev_latency.md.",
        "",
        "---",
        "",
    ]
    for name, (text, label, conf, times) in per_ticket.items():
        lines += [
            f"## {name}",
            "",
            f"**Ticket**: {text}",
            f"**Routed to**: `{label}` (confidence {conf:.2f})",
            "",
            f"- mean: {statistics.mean(times) * 1000:.1f} ms",
            f"- median: {statistics.median(times) * 1000:.1f} ms",
            f"- all runs: {', '.join(f'{t * 1000:.1f} ms' for t in times)}",
            "",
            "---",
            "",
        ]

    if args.testset:
        f1, n = macro_f1(model, tokenizer, args.testset)
        lines += [f"## Macro-F1 on {args.testset}", "", f"- **Macro-F1**: {f1:.3f} (n={n})", ""]

    Path(args.out).write_text("\n".join(lines))
    print("\n".join(lines[:14]))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

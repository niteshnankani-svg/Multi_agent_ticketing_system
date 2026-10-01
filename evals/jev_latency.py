#!/usr/bin/env python3
"""Latency benchmark for Jev (TypeSafe) ticket classification.

Measures wall-clock time for exactly the call graph.py's classify_node makes
(typesafe_client.system_one with the Choice question over CATEGORY_CRITERIA)
across a spread of real-looking tickets, one per department plus an
ambiguous one. Does not run ingest/cache/agent nodes - isolates the
classification step specifically, since that's what routes a ticket to a
department.

Each ticket is run multiple times (default 5) to see variance, not just a
single cold sample.

Usage: python3 evals/jev_latency.py [--runs 5]
"""
import argparse
import statistics
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typesafe_sdk import Choice, TypeSafeClient
from graph import CATEGORY_CRITERIA

TICKETS = [
    {"id": "finance", "text": "I was charged twice for my subscription this month, can I get a refund for the duplicate charge?"},
    {"id": "backend", "text": "The app keeps crashing every time I try to log in, I've tried resetting my password already."},
    {"id": "internal", "text": "I need access to the shared marketing drive, my manager approved it last week."},
    {"id": "general", "text": "What are your business hours and where are you located?"},
    {"id": "ambiguous", "text": "I can't log into my account to check my last invoice and I think I was overcharged."},
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5, help="repetitions per ticket")
    args = parser.parse_args()

    client = TypeSafeClient()
    all_latencies = []
    rows = []

    for ticket in TICKETS:
        latencies = []
        last_category = None
        last_confidence = None
        for i in range(args.runs):
            start = time.perf_counter()
            response = client.system_one(
                state=ticket["text"],
                questions={
                    "category": Choice(
                        instructions="Which team should handle this support ticket?",
                        criteria=CATEGORY_CRITERIA,
                    ),
                },
            )
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)
            answer = response.choices["category"]
            last_category = answer.choice
            last_confidence = answer.confidence
            print(f"[{ticket['id']}] run {i+1}/{args.runs}: {elapsed:.3f}s -> {answer.choice} (confidence {answer.confidence:.2f})", flush=True)

        rows.append({
            "id": ticket["id"],
            "text": ticket["text"],
            "latencies": latencies,
            "category": last_category,
            "confidence": last_confidence,
        })
        all_latencies.extend(latencies)

    print("\n" + "=" * 60)
    print("PER-TICKET STATS")
    print("=" * 60)
    for r in rows:
        lat = r["latencies"]
        print(f"{r['id']:12s} min={min(lat):.3f}s  mean={statistics.mean(lat):.3f}s  "
              f"median={statistics.median(lat):.3f}s  max={max(lat):.3f}s  -> routed to '{r['category']}'")

    print("\n" + "=" * 60)
    print(f"OVERALL ({len(all_latencies)} calls)")
    print("=" * 60)
    print(f"min:    {min(all_latencies):.3f}s")
    print(f"mean:   {statistics.mean(all_latencies):.3f}s")
    print(f"median: {statistics.median(all_latencies):.3f}s")
    if len(all_latencies) >= 2:
        print(f"stdev:  {statistics.stdev(all_latencies):.3f}s")
    sorted_lat = sorted(all_latencies)
    p95_idx = min(len(sorted_lat) - 1, int(len(sorted_lat) * 0.95))
    print(f"p95:    {sorted_lat[p95_idx]:.3f}s")
    print(f"max:    {max(all_latencies):.3f}s")

    write_report(rows, all_latencies, Path(__file__).resolve().parent / "jev_latency.md")


def write_report(rows, all_latencies, out_path: Path):
    lines = ["# Jev classification latency", "", f"- **Calls**: {len(all_latencies)} "
             f"({len(rows)} tickets x {len(rows[0]['latencies'])} runs each)", ""]
    lines.append(f"- **Overall min**: {min(all_latencies):.3f}s")
    lines.append(f"- **Overall mean**: {statistics.mean(all_latencies):.3f}s")
    lines.append(f"- **Overall median**: {statistics.median(all_latencies):.3f}s")
    if len(all_latencies) >= 2:
        lines.append(f"- **Overall stdev**: {statistics.stdev(all_latencies):.3f}s")
    sorted_lat = sorted(all_latencies)
    p95_idx = min(len(sorted_lat) - 1, int(len(sorted_lat) * 0.95))
    lines.append(f"- **Overall p95**: {sorted_lat[p95_idx]:.3f}s")
    lines.append(f"- **Overall max**: {max(all_latencies):.3f}s")
    lines.append("")
    lines.append("This measures exactly what `graph.py`'s `classify_node` calls - "
                  "`typesafe_client.system_one()` with the `category` Choice question - "
                  "not the full ticket pipeline (ingest/cache/agent nodes excluded).")
    lines.append("")
    lines.append("---")
    lines.append("")

    for r in rows:
        lat = r["latencies"]
        lines.append(f"## {r['id']}")
        lines.append("")
        lines.append(f"**Ticket**: {r['text']}")
        lines.append(f"**Routed to**: `{r['category']}` (confidence {r['confidence']:.2f})")
        lines.append("")
        lines.append(f"- min: {min(lat):.3f}s")
        lines.append(f"- mean: {statistics.mean(lat):.3f}s")
        lines.append(f"- median: {statistics.median(lat):.3f}s")
        lines.append(f"- max: {max(lat):.3f}s")
        lines.append(f"- all runs: {', '.join(f'{x:.3f}s' for x in lat)}")
        lines.append("")
        lines.append("---")
        lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

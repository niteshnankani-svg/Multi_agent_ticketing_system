#!/usr/bin/env python3
"""Accuracy + faithfulness comparison: DistilBERT router vs Jev, on top of
the speed numbers already in evals/jev_latency.md and evals/bert_latency.md.

Three distinct measurements, each with an honest, stated scope - not
blended together into one misleading number:

  1. BERT macro-F1 on the FULL held-out test set (data/raw/test.csv,
     10,464 tickets). Cheap and local (~15ms/ticket per bert_latency.md),
     so there's no reason to sample it.

  2. BERT and Jev macro-F1 on the SAME stratified sample (25 tickets per
     department = 100 tickets, data/raw/test.csv, random_state=42). Jev is
     a real network call (~0.35s/ticket per jev_latency.md) - the full
     10,464-ticket set would take roughly an hour, so this is a matched
     subset run through both classifiers for an apples-to-apples F1
     comparison, not a claim that 100 tickets is the full picture.

  3. LLM-as-judge faithfulness on that same 100-ticket sample: Gemini
     (via its REST API - a genuine separate call, not this agent's own
     reasoning) rates whether EACH classifier's predicted department is
     actually a defensible read of the ticket's content against the
     department criteria, independent of whatever label the dataset
     itself assigned. Dataset labels can be noisy; this is meant to catch
     cases where a "wrong" prediction vs. the label is actually the more
     reasonable call, and vice versa - not to replace the F1 accuracy
     number, which is reported separately.

Needs (all already in this repo):
  - GEMINI_API_KEY in .env (judge)
  - TYPESAFE_API_KEY in .env (Jev)
  - models/router/ (BERT)
  - data/raw/test.csv (held-out set)

Usage: python3 evals/accuracy_comparison.py [--sample-per-category 25]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from typesafe_sdk import Choice, TypeSafeClient
from graph import CATEGORY_CRITERIA

MAX_LENGTH = 128
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
BERT_LABEL_MAP = {"backend": "backend", "finance": "finance", "general": "general", "internal": "internal"}

JUDGE_PROMPT_TEMPLATE = """You are an impartial judge rating ticket-routing classifications for a \
4-department support system. Judge each prediction on whether it is a FAITHFUL, defensible match to the \
department criteria below - NOT whether it matches the dataset's own label, which may itself be noisy or wrong.

DEPARTMENT CRITERIA:
{criteria}

TICKET TEXT:
{text}

DATASET LABEL (reference only, may be wrong - call it out in dataset_label_looks_correct if so): {dataset_label}

PREDICTION A (BERT router): {bert_label} (model confidence {bert_conf:.2f})
PREDICTION B (Jev): {jev_label} (model confidence {jev_conf:.2f})

Respond with ONLY a JSON object, no other text:
{{
  "bert_faithful": true/false,
  "bert_reasoning": "one sentence",
  "jev_faithful": true/false,
  "jev_reasoning": "one sentence",
  "dataset_label_looks_correct": true/false
}}"""


def load_bert():
    tokenizer = AutoTokenizer.from_pretrained("models/router")
    model = AutoModelForSequenceClassification.from_pretrained("models/router").to("cpu").eval()
    return tokenizer, model


def bert_classify_batch(tokenizer, model, texts: list[str], batch_size: int = 32) -> list[str]:
    labels = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits
        preds = torch.argmax(logits, dim=-1).tolist()
        labels.extend(model.config.id2label[p] for p in preds)
        print(f"  BERT: {min(i + batch_size, len(texts))}/{len(texts)}", end="\r", flush=True)
    print()
    return labels


def bert_classify_one(tokenizer, model, text: str) -> tuple[str, float]:
    enc = tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)[0]
    idx = int(probs.argmax())
    return model.config.id2label[idx], float(probs[idx])


def jev_classify_one(client: TypeSafeClient, text: str) -> tuple[str, float, float]:
    start = time.perf_counter()
    response = client.system_one(
        state=text,
        questions={"category": Choice(instructions="Which team should handle this support ticket?", criteria=CATEGORY_CRITERIA)},
    )
    elapsed = time.perf_counter() - start
    answer = response.choices["category"]
    return answer.choice, answer.confidence, elapsed


def call_judge(client: httpx.Client, prompt: str) -> dict:
    api_key = os.environ["GEMINI_API_KEY"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
    response = client.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=60.0)
    response.raise_for_status()
    data = response.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-per-category", type=int, default=25)
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "accuracy_comparison.md"))
    args = parser.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv("data/raw/test.csv")
    print(f"Loaded {len(df)} held-out tickets, categories: {dict(df['category'].value_counts())}")

    print("\nLoading BERT model...")
    tokenizer, bert_model = load_bert()

    # --- 1. BERT macro-F1 on the FULL test set ---
    print(f"\nRunning BERT on the full {len(df)}-ticket test set...")
    full_preds = bert_classify_batch(tokenizer, bert_model, df["text"].tolist())
    full_f1 = f1_score(df["category"].tolist(), full_preds, average="macro")
    print(f"BERT full-set macro-F1: {full_f1:.3f}")

    # --- 2 & 3. matched stratified sample through both classifiers + judge ---
    sample = (
        df.groupby("category", group_keys=False)[df.columns.tolist()]
        .apply(lambda g: g.sample(min(len(g), args.sample_per_category), random_state=42))
        .reset_index(drop=True)
    )
    print(f"\nStratified sample: {len(sample)} tickets, {dict(sample['category'].value_counts())}")

    jev_client = TypeSafeClient()

    rows = []
    with httpx.Client() as http_client:
        for i, row in sample.iterrows():
            text, dataset_label = row["text"], row["category"]
            bert_label, bert_conf = bert_classify_one(tokenizer, bert_model, text)
            jev_label, jev_conf, jev_latency = jev_classify_one(jev_client, text)

            prompt = JUDGE_PROMPT_TEMPLATE.format(
                criteria="\n".join(f"- {k}: {v}" for k, v in CATEGORY_CRITERIA.items()),
                text=text[:1500],
                dataset_label=dataset_label,
                bert_label=bert_label, bert_conf=bert_conf,
                jev_label=jev_label, jev_conf=jev_conf,
            )
            try:
                verdict = call_judge(http_client, prompt)
            except Exception as exc:
                verdict = {"error": str(exc)}

            rows.append({
                "text": text, "dataset_label": dataset_label,
                "bert_label": bert_label, "bert_conf": bert_conf,
                "jev_label": jev_label, "jev_conf": jev_conf, "jev_latency": jev_latency,
                "verdict": verdict,
            })
            print(f"[{i+1}/{len(sample)}] dataset={dataset_label} bert={bert_label} jev={jev_label} "
                  f"judge_bert_ok={verdict.get('bert_faithful')} judge_jev_ok={verdict.get('jev_faithful')}", flush=True)

    sample_f1_bert = f1_score(sample["category"].tolist(), [r["bert_label"] for r in rows], average="macro")
    sample_f1_jev = f1_score(sample["category"].tolist(), [r["jev_label"] for r in rows], average="macro")

    judged = [r for r in rows if "error" not in r["verdict"]]
    bert_faithful_rate = sum(1 for r in judged if r["verdict"].get("bert_faithful")) / len(judged) if judged else None
    jev_faithful_rate = sum(1 for r in judged if r["verdict"].get("jev_faithful")) / len(judged) if judged else None
    label_quality = sum(1 for r in judged if r["verdict"].get("dataset_label_looks_correct")) / len(judged) if judged else None

    write_report(args, full_f1, sample, sample_f1_bert, sample_f1_jev, rows, judged,
                 bert_faithful_rate, jev_faithful_rate, label_quality, Path(args.out))


def write_report(args, full_f1, sample, sample_f1_bert, sample_f1_jev, rows, judged,
                  bert_faithful_rate, jev_faithful_rate, label_quality, out_path: Path):
    lines = ["# Accuracy + faithfulness: DistilBERT vs Jev", ""]
    lines.append(f"- **Held-out test set**: data/raw/test.csv, {10464} tickets total")
    lines.append(f"- **Stratified sample for matched comparison**: {len(sample)} tickets "
                 f"({args.sample_per_category}/category, random_state=42)")
    lines.append(f"- **Judge**: gemini/{GEMINI_MODEL} via the Gemini REST API")
    lines.append("")
    lines.append("## Speed (from evals/jev_latency.md and evals/bert_latency.md)")
    lines.append("")
    lines.append("| Classifier | Median | p95 |")
    lines.append("|---|---|---|")
    lines.append("| BERT | 14.8 ms | 17.9 ms |")
    lines.append("| Jev | 354 ms | 449 ms |")
    lines.append("")
    lines.append("## Accuracy (macro-F1 against dataset labels)")
    lines.append("")
    lines.append(f"- **BERT, full {10464}-ticket test set**: {full_f1:.3f}")
    lines.append(f"- **BERT, matched {len(sample)}-ticket sample**: {sample_f1_bert:.3f}")
    lines.append(f"- **Jev, matched {len(sample)}-ticket sample**: {sample_f1_jev:.3f}")
    lines.append("")
    lines.append("## Faithfulness (LLM-judge, same matched sample)")
    lines.append("")
    lines.append("Judges whether the predicted department is a defensible read of the ticket's "
                  "actual content against the department criteria - independent of the dataset's "
                  "own (possibly noisy) label.")
    lines.append("")
    if bert_faithful_rate is not None:
        lines.append(f"- **BERT faithful rate**: {bert_faithful_rate:.1%} ({len(judged)} judged)")
        lines.append(f"- **Jev faithful rate**: {jev_faithful_rate:.1%} ({len(judged)} judged)")
        lines.append(f"- **Dataset label judged correct**: {label_quality:.1%} of sampled tickets "
                      "(a check on the ground truth itself, not either classifier)")
    else:
        lines.append("*No judge verdicts recorded - all judge calls failed.*")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Per-ticket detail (matched sample)")
    lines.append("")

    disagreements = [r for r in rows if r["bert_label"] != r["jev_label"]]
    lines.append(f"### Disagreements between BERT and Jev ({len(disagreements)}/{len(rows)})")
    lines.append("")
    for r in disagreements:
        v = r["verdict"]
        lines.append(f"- **Dataset label**: `{r['dataset_label']}` | **BERT**: `{r['bert_label']}` "
                      f"(faithful={v.get('bert_faithful')}) | **Jev**: `{r['jev_label']}` "
                      f"(faithful={v.get('jev_faithful')})")
        lines.append(f"  - Ticket: {r['text'][:200]}")
        if "error" not in v:
            lines.append(f"  - Judge on BERT: {v.get('bert_reasoning')}")
            lines.append(f"  - Judge on Jev: {v.get('jev_reasoning')}")
        lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate TypeSafe AI's Jev model on a labeled classification dataset.

Endpoint:  POST https://api.typesafe.ai/v1/systemone
Model:     jev-latest
API key:   TYPESAFE_API_KEY env var, or hardcoded in TYPESAFE_API_KEY below.

Reports:
  * Accuracy, per-class precision / recall / F1, confusion matrix.
  * Calibration: Expected Calibration Error (ECE) and Brier score, computed
    from the probability Jev assigned to the ground-truth label.
  * Latency stats: mean / median / p90 / p99 per-request wall clock.
  * Optional per-row dump with --verbose.

Dataset format: JSONL. Each line must be a JSON object with:
    { "input": "<text or object>", "label": "<expected label>" }
Optional keys ("id", "notes", etc.) are ignored. The label space is inferred
from the unique labels seen in the file.

Usage:
  python app.py                                         # use built-in phishing sample
  python app.py --dataset path/to/mydata.jsonl
  python app.py --instructions "Classify the sentiment as positive/negative/neutral"
  python app.py --limit 50                              # first N rows only
  python app.py --dry-run                               # print request payload, no API call
  python app.py --verbose                               # per-row prediction + probabilities
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Configuration - edit these or export the env var
# ---------------------------------------------------------------------------
TYPESAFE_API_KEY = "sk-REPLACE-ME"     # <-- paste your TypeSafe key or use env var
JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"

DEFAULT_DATASET = Path(__file__).parent / "datasets" / "phishing_sample.jsonl"
DEFAULT_INSTRUCTIONS = (
    "Classify this email subject line. 'phishing' = a scam / fraudulent / "
    "credential-harvesting message; 'legitimate' = a real, benign message."
)
QUESTION_NAME = "label"                # key inside the request `questions` map
REQUEST_TIMEOUT_S = 30                 # HTTP timeout for a single call
CALIBRATION_BINS = 10                  # for ECE reliability bins


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Example:
    input: str
    label: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    example: Example
    predicted: str | None
    probabilities: dict[str, float]      # {label: p}
    confidence: float | None             # Jev's stated confidence (0..1) if present
    latency_ms: float
    correct: bool
    p_true_label: float                  # probability Jev assigned to the true label
    raw_answer: dict[str, Any] | None    # full answer object, useful for --verbose
    error: str | None = None


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_dataset(path: Path, limit: int | None) -> list[Example]:
    examples: list[Example] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{i}: invalid JSON ({e})")
            if "input" not in row or "label" not in row:
                raise SystemExit(f"{path}:{i}: row missing 'input' or 'label' key")
            examples.append(Example(input=str(row["input"]), label=str(row["label"]), raw=row))
            if limit and len(examples) >= limit:
                break
    if not examples:
        raise SystemExit(f"{path}: no examples loaded")
    return examples


def resolve_api_key() -> str:
    key = os.environ.get("TYPESAFE_API_KEY") or TYPESAFE_API_KEY
    if not key or "REPLACE-ME" in key:
        raise SystemExit(
            "TYPESAFE_API_KEY not set. Either export TYPESAFE_API_KEY in your "
            "shell or paste it into TYPESAFE_API_KEY at the top of app.py."
        )
    return key


# ---------------------------------------------------------------------------
# Jev call
# ---------------------------------------------------------------------------
def build_choice_question(labels: list[str], instructions: str) -> dict[str, Any]:
    """Build a Choice question over the given labels. Criteria values are null
    (no per-label description). Labels in a stable, sorted order."""
    return {
        "type": "choice",
        "instructions": instructions,
        "criteria": {lbl: None for lbl in labels},
    }


def build_request(state: str, question: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": JEV_MODEL,
        "state": state,
        "questions": {QUESTION_NAME: question},
    }


def extract_answer(payload: dict[str, Any]) -> dict[str, Any] | None:
    """The response should carry an answer keyed by our question name. Be
    generous about where it lives so we don't die on a slightly different
    wrapper shape."""
    if not isinstance(payload, dict):
        return None
    # Direct: {"label": {...}}
    if QUESTION_NAME in payload and isinstance(payload[QUESTION_NAME], dict):
        return payload[QUESTION_NAME]
    # Wrapped: {"answers": {"label": {...}}}
    answers = payload.get("answers")
    if isinstance(answers, dict) and QUESTION_NAME in answers:
        return answers[QUESTION_NAME]
    # Wrapped list: [{"name": "label", ...}]
    if isinstance(answers, list):
        for a in answers:
            if isinstance(a, dict) and a.get("name") == QUESTION_NAME:
                return a
    return None


def call_jev(session: requests.Session, api_key: str, body: dict[str, Any]
             ) -> tuple[dict[str, Any] | None, float, str | None]:
    """Returns (full_response_json, latency_ms, error_or_None)."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    t0 = time.perf_counter()
    try:
        r = session.post(JEV_ENDPOINT, headers=headers, json=body,
                         timeout=REQUEST_TIMEOUT_S)
    except requests.RequestException as e:
        return None, (time.perf_counter() - t0) * 1000, f"transport: {e}"
    latency_ms = (time.perf_counter() - t0) * 1000
    if r.status_code >= 400:
        return None, latency_ms, f"http {r.status_code}: {r.text[:200]}"
    try:
        return r.json(), latency_ms, None
    except ValueError as e:
        return None, latency_ms, f"non-json response: {e}"


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------
def evaluate(examples: list[Example], instructions: str, verbose: bool,
             dry_run: bool) -> list[RunResult]:
    labels = sorted({e.label for e in examples})
    question = build_choice_question(labels, instructions)

    if dry_run:
        print("=== DRY RUN — sample request (row 0) ===")
        print(json.dumps(build_request(examples[0].input, question), indent=2))
        print(f"\nWould send {len(examples)} requests to {JEV_ENDPOINT}")
        print(f"Label space: {labels}")
        return []

    api_key = resolve_api_key()
    session = requests.Session()
    results: list[RunResult] = []

    for i, ex in enumerate(examples, start=1):
        body = build_request(ex.input, question)
        payload, latency_ms, err = call_jev(session, api_key, body)

        predicted: str | None = None
        probabilities: dict[str, float] = {}
        confidence: float | None = None
        raw_answer = None

        if err is None and payload is not None:
            raw_answer = extract_answer(payload)
            if isinstance(raw_answer, dict):
                predicted = raw_answer.get("choice")
                probabilities = raw_answer.get("probabilities") or {}
                confidence = raw_answer.get("confidence")
            else:
                err = f"could not find answer for question {QUESTION_NAME!r} in response"

        correct = (predicted == ex.label) if predicted is not None else False
        p_true = float(probabilities.get(ex.label, 0.0)) if probabilities else 0.0

        result = RunResult(
            example=ex,
            predicted=predicted,
            probabilities={k: float(v) for k, v in probabilities.items()},
            confidence=float(confidence) if confidence is not None else None,
            latency_ms=latency_ms,
            correct=correct,
            p_true_label=p_true,
            raw_answer=raw_answer,
            error=err,
        )
        results.append(result)

        if verbose:
            _print_verbose_row(i, len(examples), result)
        else:
            mark = "." if err is None else "!"
            sys.stdout.write(mark)
            sys.stdout.flush()

    if not verbose:
        sys.stdout.write("\n")

    return results


def _print_verbose_row(i: int, total: int, r: RunResult) -> None:
    if r.error:
        print(f"[{i:>4}/{total}] ERROR   {r.error}    input={r.example.input[:60]!r}")
        return
    tick = "✓" if r.correct else "✗"
    probs = ", ".join(f"{k}={v:.3f}" for k, v in sorted(r.probabilities.items()))
    conf = f"conf={r.confidence:.3f}  " if r.confidence is not None else ""
    print(f"[{i:>4}/{total}] {tick} pred={r.predicted!r:>15} truth={r.example.label!r:>15} "
          f"lat={r.latency_ms:6.0f}ms  {conf}[{probs}]  input={r.example.input[:60]!r}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def confusion_matrix(results: list[RunResult], labels: list[str]) -> dict[tuple[str, str], int]:
    cm: dict[tuple[str, str], int] = defaultdict(int)
    for r in results:
        if r.predicted is None:
            continue
        cm[(r.example.label, r.predicted)] += 1
    return cm


def per_class_metrics(cm: dict[tuple[str, str], int], labels: list[str]
                      ) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for lbl in labels:
        tp = cm.get((lbl, lbl), 0)
        fp = sum(cm.get((other, lbl), 0) for other in labels if other != lbl)
        fn = sum(cm.get((lbl, other), 0) for other in labels if other != lbl)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[lbl] = {"precision": precision, "recall": recall, "f1": f1,
                    "support": tp + fn}
    return out


def expected_calibration_error(results: list[RunResult], bins: int
                                ) -> tuple[float, list[dict[str, float]]]:
    """ECE using p_true_label as the confidence in "the answer I actually gave"
    only when Jev's chosen label == the true label. Otherwise we use the
    probability of the predicted label and treat the example as incorrect —
    this is the standard multi-class ECE formulation."""
    usable = [r for r in results if r.predicted is not None and r.probabilities]
    if not usable:
        return 0.0, []

    entries: list[tuple[float, int]] = []
    for r in usable:
        p_pred = r.probabilities.get(r.predicted, 0.0)
        entries.append((p_pred, 1 if r.correct else 0))

    bin_edges = [i / bins for i in range(bins + 1)]
    bin_stats: list[dict[str, float]] = []
    n = len(entries)
    ece = 0.0
    for i in range(bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # last bin includes right edge
        in_bin = [(p, c) for (p, c) in entries
                  if (p >= lo and p < hi) or (i == bins - 1 and p == hi)]
        if not in_bin:
            bin_stats.append({"lo": lo, "hi": hi, "n": 0,
                              "avg_conf": 0.0, "accuracy": 0.0})
            continue
        avg_conf = sum(p for p, _ in in_bin) / len(in_bin)
        accuracy = sum(c for _, c in in_bin) / len(in_bin)
        weight = len(in_bin) / n
        ece += weight * abs(avg_conf - accuracy)
        bin_stats.append({"lo": lo, "hi": hi, "n": len(in_bin),
                          "avg_conf": avg_conf, "accuracy": accuracy})
    return ece, bin_stats


def brier_score(results: list[RunResult]) -> float:
    """Multi-class Brier: mean over rows of sum_k (p_k - y_k)^2 where y is
    one-hot on the true label. Rows without probabilities are skipped."""
    usable = [r for r in results if r.probabilities]
    if not usable:
        return 0.0
    total = 0.0
    for r in usable:
        labels = set(r.probabilities.keys()) | {r.example.label}
        s = 0.0
        for lbl in labels:
            p = r.probabilities.get(lbl, 0.0)
            y = 1.0 if lbl == r.example.label else 0.0
            s += (p - y) ** 2
        total += s
    return total / len(usable)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _pct(x: float) -> str:
    return f"{x * 100:6.2f}%"


def _latency_stats(vals: list[float]) -> str:
    if not vals:
        return "n/a"
    mean = statistics.mean(vals)
    med = statistics.median(vals)
    lo, hi = min(vals), max(vals)
    if len(vals) >= 10:
        p90 = statistics.quantiles(vals, n=10)[8]
        p99 = statistics.quantiles(vals, n=100)[98] if len(vals) >= 100 else max(vals)
        return (f"mean={mean:6.0f}  med={med:6.0f}  p90={p90:6.0f}  "
                f"p99={p99:6.0f}  min={lo:6.0f}  max={hi:6.0f}  (ms)")
    return f"mean={mean:6.0f}  med={med:6.0f}  min={lo:6.0f}  max={hi:6.0f}  (ms)"


def report(results: list[RunResult]) -> None:
    total = len(results)
    errored = [r for r in results if r.error is not None]
    successful = [r for r in results if r.error is None and r.predicted is not None]
    labels = sorted({r.example.label for r in results})

    print("\n" + "=" * 70)
    print("  Jev evaluation report")
    print("=" * 70)
    print(f"Rows total       : {total}")
    print(f"Rows evaluated   : {len(successful)}")
    print(f"Rows errored     : {len(errored)}")

    if errored:
        print("\nSample errors (first 3):")
        for r in errored[:3]:
            print(f"  - {r.error}")

    if not successful:
        print("\nNo successful rows to score.")
        return

    # Accuracy
    correct = sum(1 for r in successful if r.correct)
    accuracy = correct / len(successful)
    print(f"\nAccuracy         : {_pct(accuracy)}   ({correct}/{len(successful)})")

    # Per-class metrics
    cm = confusion_matrix(successful, labels)
    pcm = per_class_metrics(cm, labels)
    print("\nPer-class metrics:")
    print(f"  {'label':<20} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>8}")
    for lbl in labels:
        m = pcm[lbl]
        print(f"  {lbl:<20} {_pct(m['precision'])}  {_pct(m['recall'])}  "
              f"{_pct(m['f1'])}  {int(m['support']):>8}")

    # Confusion matrix
    print("\nConfusion matrix  (rows = truth, cols = predicted):")
    header = "  " + " " * 20 + "".join(f"{c[:14]:>16}" for c in labels)
    print(header)
    for row_lbl in labels:
        cells = "".join(f"{cm.get((row_lbl, col_lbl), 0):>16}" for col_lbl in labels)
        print(f"  {row_lbl:<20}{cells}")

    # Calibration
    ece, bins = expected_calibration_error(successful, CALIBRATION_BINS)
    brier = brier_score(successful)
    print(f"\nCalibration:")
    print(f"  Expected Calibration Error (ECE, {CALIBRATION_BINS} bins): {ece:.4f}")
    print(f"  Brier score (multi-class)                     : {brier:.4f}")
    print("\n  Reliability by bin (bin range | n | avg confidence | accuracy):")
    for b in bins:
        if b["n"] == 0:
            continue
        print(f"    [{b['lo']:.1f}, {b['hi']:.1f})  n={int(b['n']):>4}  "
              f"avg_conf={b['avg_conf']:.3f}  accuracy={b['accuracy']:.3f}")

    # Latency
    latencies = [r.latency_ms for r in successful]
    print(f"\nLatency:")
    print(f"  {_latency_stats(latencies)}")

    # Confidence distribution (Jev's own confidence field, if provided)
    confs = [r.confidence for r in successful if r.confidence is not None]
    if confs:
        print(f"\nJev-reported confidence distribution (n={len(confs)}):")
        buckets = Counter(round(c, 1) for c in confs)
        for k in sorted(buckets):
            print(f"  ~{k:.1f}: {buckets[k]}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Evaluate Jev on a labeled JSONL classification dataset."
    )
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                    help="Path to JSONL file (default: bundled phishing sample)")
    ap.add_argument("--instructions", default=DEFAULT_INSTRUCTIONS,
                    help="Choice question instructions passed to Jev")
    ap.add_argument("--limit", type=int, default=None,
                    help="Evaluate only the first N rows")
    ap.add_argument("--verbose", action="store_true",
                    help="Print each row's prediction, probabilities, and latency")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the request that would be sent, then exit")
    args = ap.parse_args(argv)

    dataset = load_dataset(args.dataset, args.limit)

    print(f"Dataset          : {args.dataset}")
    print(f"Rows to evaluate : {len(dataset)}")
    print(f"Label space      : {sorted({e.label for e in dataset})}")
    print(f"Model            : {JEV_MODEL}")
    print(f"Endpoint         : {JEV_ENDPOINT}")
    print(f"Instructions     : {args.instructions!r}\n")

    results = evaluate(dataset, args.instructions, args.verbose, args.dry_run)
    if not args.dry_run:
        report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())

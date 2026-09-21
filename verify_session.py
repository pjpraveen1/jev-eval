#!/usr/bin/env python3
"""Verify factual consistency of a session trace using Jev.

Reads `session.json` (a session/trace payload — text, dict, or list), sends
it as `state` to the Jev API, and asks the single Noul question:

    factually_consistent — Everything the agent told the customer is
    consistent with the data returned by its tools.

Jev's Noul answer is a probability in [0, 1] that the answer is "true".
We print the raw probability plus a PASS / FAIL decision against a
configurable threshold (default 0.5).

Usage
-----
    export TYPESAFE_API_KEY=sk-...           # or paste into the slot below
    python3 verify_session.py                # uses ./session.json
    python3 verify_session.py --session other.json --threshold 0.7 --raw
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# API key slot — paste your TypeSafe key here, or export TYPESAFE_API_KEY.
# TypeSafe keys look like `apikey_<hex>_<hex>` (not the OpenAI `sk-...` format).
# The env var takes precedence over this hardcoded value.
# ---------------------------------------------------------------------------
API_KEY_PLACEHOLDER = "PASTE_YOUR_TYPESAFE_API_KEY_HERE"
TYPESAFE_API_KEY = API_KEY_PLACEHOLDER

JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"
REQUEST_TIMEOUT_S = 30
DEFAULT_SESSION_FILE = Path(__file__).parent / "session.json"

# ---------------------------------------------------------------------------
# Questions — sent to Jev in a SINGLE request. This is the whole Jev pitch:
# multiple typed questions answered in one parallel pass, one network round-trip.
# `verify_session_bedrock.py` imports QUESTIONS from here so the two evaluators
# are guaranteed to be asking exactly the same things.
# ---------------------------------------------------------------------------
FACTUAL_QUESTION_NAME = "factually_consistent"
FACTUAL_QUESTION = {
    "type": "noul",
    "instructions": (
        "Everything the agent told the customer is consistent with the "
        "data returned by its tools."
    ),
    "criteria": {
        "true": "Every statement made to the customer matches the tool "
                "results in the trace",
        "false": "The agent told the customer something its own tool "
                 "results do not support or contradict",
    },
}

CSAT_QUESTION_NAME = "predicted_csat"
CSAT_QUESTION = {
    "type": "score",
    "instructions": (
        "Predict the satisfaction rating this customer will give in a "
        "follow-up survey one month from now."
    ),
    "criteria": [
        "Very dissatisfied: likely complaint, chargeback, or churn",
        "Dissatisfied: the outcome will fall short of what was promised or expected",
        "Neutral: acceptable outcome with friction",
        "Satisfied: issue handled competently",
        "Very satisfied: fast, complete resolution that exceeds expectations",
    ],
}

QUESTIONS: dict[str, dict[str, Any]] = {
    FACTUAL_QUESTION_NAME: FACTUAL_QUESTION,
    CSAT_QUESTION_NAME: CSAT_QUESTION,
}

# Legacy names — kept so anything else importing them still works.
QUESTION_NAME = FACTUAL_QUESTION_NAME
QUESTION = FACTUAL_QUESTION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def resolve_api_key() -> str:
    key = os.environ.get("TYPESAFE_API_KEY") or TYPESAFE_API_KEY
    if not key or key == API_KEY_PLACEHOLDER:
        raise SystemExit(
            "TYPESAFE_API_KEY is not set. Either export TYPESAFE_API_KEY in "
            "your shell (recommended) or replace the TYPESAFE_API_KEY value "
            "near the top of verify_session.py."
        )
    return key


def load_session(path: Path) -> Any:
    """Load the session file. Accepts JSON (object / array) or falls back to
    treating the file's content as a plain string state."""
    if not path.exists():
        raise SystemExit(f"session file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise SystemExit(
            f"{path} is empty. Paste the session/trace content into it first."
        )
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Not valid JSON — fine, Jev accepts a plain-text state too.
        return text


def extract_answers(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Locate all answers inside Jev's response envelope. Tolerant of a few
    plausible wrapper shapes:
      {"factually_consistent": {...}, "predicted_csat": {...}}
      {"answers": {"factually_consistent": {...}, ...}}
      {"answers": [{"name": "factually_consistent", ...}, ...]}
    Returns a name -> answer dict; missing questions simply won't be keys.
    """
    if not isinstance(payload, dict):
        return {}

    # Case 1: answers keyed at top level
    top_hits = {n: payload[n] for n in QUESTIONS
                if n in payload and isinstance(payload[n], dict)}
    if top_hits:
        return top_hits

    answers = payload.get("answers")
    # Case 2: {"answers": {name: {...}}}
    if isinstance(answers, dict):
        return {n: a for n, a in answers.items()
                if n in QUESTIONS and isinstance(a, dict)}
    # Case 3: {"answers": [{"name": name, ...}, ...]}
    if isinstance(answers, list):
        return {a["name"]: a for a in answers
                if isinstance(a, dict) and a.get("name") in QUESTIONS}
    return {}


def call_jev(state: Any, api_key: str) -> tuple[dict[str, Any] | None, float, str | None]:
    body = {
        "model": JEV_MODEL,
        "state": state,
        "questions": QUESTIONS,          # both questions in ONE request
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    t0 = time.perf_counter()
    try:
        r = requests.post(JEV_ENDPOINT, headers=headers, json=body,
                          timeout=REQUEST_TIMEOUT_S)
    except requests.RequestException as e:
        return None, (time.perf_counter() - t0) * 1000, f"transport: {e}"
    latency_ms = (time.perf_counter() - t0) * 1000
    if r.status_code >= 400:
        return None, latency_ms, f"http {r.status_code}: {r.text[:400]}"
    try:
        return r.json(), latency_ms, None
    except ValueError as e:
        return None, latency_ms, f"non-json response: {e}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--session", type=Path, default=DEFAULT_SESSION_FILE,
                    help=f"Session JSON file (default: {DEFAULT_SESSION_FILE.name})")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Probability threshold for PASS (default: 0.5)")
    ap.add_argument("--raw", action="store_true",
                    help="Also print the full raw Jev response")
    args = ap.parse_args(argv)

    api_key = resolve_api_key()
    state = load_session(args.session)

    questions_summary = ", ".join(f"{n} ({q['type']})" for n, q in QUESTIONS.items())
    print(f"Session file : {args.session}")
    print(f"State type   : {type(state).__name__}")
    print(f"Questions    : {questions_summary}")
    print(f"Model        : {JEV_MODEL}")
    print(f"Endpoint     : {JEV_ENDPOINT}")
    print(f"Threshold    : {args.threshold} (applies to noul questions)\n")

    # Explicit before/after timer around the (single) API call.
    t_send = time.perf_counter()
    send_wallclock = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    print(f"→ sending  at {send_wallclock}  ({len(QUESTIONS)} questions in one request) ...",
          flush=True)

    payload, latency_ms, err = call_jev(state, api_key)

    recv_wallclock = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    elapsed_ms = (time.perf_counter() - t_send) * 1000
    print(f"← received at {recv_wallclock}  "
          f"(wall={elapsed_ms:.0f} ms, http={latency_ms:.0f} ms)\n", flush=True)

    if err is not None:
        print(f"ERROR: {err}")
        return 1

    if args.raw:
        print("--- raw response ---")
        print(json.dumps(payload, indent=2))
        print("--- end raw ---\n")

    answers = extract_answers(payload) if payload else {}
    missing = [n for n in QUESTIONS if n not in answers]
    if missing:
        print(f"WARNING: missing answers for: {missing}")
        if not args.raw:
            print("Re-run with --raw to inspect the full payload.")

    factual_passed: bool | None = None

    for qname, question in QUESTIONS.items():
        answer = answers.get(qname)
        if not isinstance(answer, dict):
            print(f"[{qname}] no answer in response")
            continue
        qtype = question.get("type")
        print(f"[{qname}]  type={qtype}")
        if qtype == "noul":
            noul = float(answer.get("noul", 0.0))
            passed = noul >= args.threshold
            verdict = "PASS ✓" if passed else "FAIL ✗"
            print(f"    noul (P[true]) : {noul:.4f}")
            print(f"    verdict        : {verdict}   (threshold >= {args.threshold})")
            factual_passed = passed
        elif qtype == "score":
            score = answer.get("score")
            confidence = answer.get("confidence")
            probabilities = answer.get("probabilities") or {}
            levels = question.get("criteria") or []
            # Best-effort resolve chosen level from Jev's score index
            chosen_level: str | None = None
            if isinstance(score, (int, float)) and levels:
                nearest_idx = max(0, min(len(levels) - 1, int(round(float(score)))))
                chosen_level = levels[nearest_idx]
            if score is not None:
                print(f"    score          : {float(score):.3f}")
            if chosen_level:
                print(f"    nearest level  : {chosen_level!r}")
            if confidence is not None:
                print(f"    confidence     : {float(confidence):.3f}")
            if probabilities:
                # Pretty-print by ordered levels when possible, otherwise by insertion order
                keys = levels if levels and set(probabilities) == set(levels) else list(probabilities)
                print("    probabilities  :")
                for k in keys:
                    p = float(probabilities.get(k, 0.0))
                    print(f"        {p:6.3f}  {k}")
        else:
            print(f"    (unhandled question type: {qtype})")
            print(f"    raw: {answer}")
        print()

    print(f"Total latency    : {latency_ms:.0f} ms  (one request, {len(QUESTIONS)} answers)")
    if factual_passed is False:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())

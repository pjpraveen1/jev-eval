#!/usr/bin/env python3
"""Verify factual consistency of a session trace using Bedrock Claude Haiku 4.5.

Mirror of verify_session.py, but runs the same instruction and criteria through
Amazon Bedrock's Claude Haiku 4.5 instead of TypeSafe Jev. The question, the
criteria descriptions, and the session-loading logic are imported directly from
verify_session.py so the two evaluations are guaranteed to be asking the exact
same thing.

Claude does not return calibrated probabilities natively, so we ask Haiku for a
JSON-only response of the form:

    {"noul": <float 0..1>, "rationale": "<one sentence>"}

where `noul` is Claude's estimate of P(the statement is true), matching Jev's
Noul answer shape one-to-one. We then apply the same threshold-based PASS/FAIL
verdict as verify_session.py.

Wall-clock latency of the Bedrock Converse call is printed explicitly, so the
Bedrock <-> Jev comparison is apples-to-apples on timing.

Auth: standard AWS SDK credential chain (env vars, ~/.aws/credentials, or an
attached role). No key slot needed here.

Usage
-----
    python3 verify_session_bedrock.py                 # ./session.json
    python3 verify_session_bedrock.py --raw           # also dump Bedrock response
    python3 verify_session_bedrock.py --region us-west-2
    python3 verify_session_bedrock.py --threshold 0.9
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

# Reuse the exact same question definitions and session-loading logic used by
# the Jev script — single source of truth for both evaluators.
from verify_session import (  # noqa: E402
    CSAT_QUESTION_NAME,
    DEFAULT_SESSION_FILE,
    FACTUAL_QUESTION_NAME,
    QUESTIONS,
    load_session,
)

# ---------------------------------------------------------------------------
# Bedrock configuration
# ---------------------------------------------------------------------------
# Haiku 4.5 requires a cross-region inference profile for on-demand throughput
# rather than the bare model ID. Use the US CRIS profile by default.
BEDROCK_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
DEFAULT_REGION = "us-east-1"
MAX_OUTPUT_TOKENS = 800          # fits both answers + rationales in one shot
TEMPERATURE = 0.0                # deterministic-ish for an evaluator


# ---------------------------------------------------------------------------
# Prompt construction — asks Claude for BOTH questions in a single call so
# the comparison against Jev's one-shot parallel evaluation is apples-to-apples
# on request count (each side does exactly one round-trip).
# ---------------------------------------------------------------------------
def _levels(q: dict[str, Any]) -> list[str]:
    c = q.get("criteria")
    return list(c) if isinstance(c, list) else []


def _build_schema_for_question(name: str, q: dict[str, Any]) -> str:
    qtype = q.get("type")
    if qtype == "noul":
        return (f'  "{name}": {{"noul": <float 0..1>, '
                f'"rationale": "<one short sentence>"}}')
    if qtype == "score":
        levels = _levels(q)
        level_list = ", ".join(f'"{lvl}"' for lvl in levels)
        return (f'  "{name}": {{"level_index": <int 0..{len(levels) - 1}>, '
                f'"level": <one of {level_list}>, '
                f'"rationale": "<one short sentence>"}}')
    return f'  "{name}": <answer for the question>'


def _build_question_block(name: str, q: dict[str, Any]) -> str:
    parts = [f"Question id: {name}",
             f"Type: {q.get('type')}",
             f"Instructions: {q.get('instructions', '').strip()}"]
    criteria = q.get("criteria")
    if isinstance(criteria, dict):
        for k, v in criteria.items():
            parts.append(f"  {k} -> {v}")
    elif isinstance(criteria, list):
        parts.append("Ordered levels (lowest -> highest):")
        for i, lvl in enumerate(criteria):
            parts.append(f"  {i}: {lvl}")
    return "\n".join(parts)


SYSTEM_PROMPT_TEMPLATE = (
    "You are a strict evaluator that produces a SINGLE JSON object answering "
    "all the questions given by the user. Never write anything other than that "
    "JSON object — no prose, no code fences.\n\n"
    "The JSON object MUST match this schema exactly:\n"
    "{{\n{schema_lines}\n}}\n"
    "For noul questions: `noul` is the probability the condition is TRUE "
    "(1.0 = definitely true, 0.0 = definitely false). "
    "For score questions: pick the single most likely level by index (0-based) "
    "and copy its human-readable label into `level`."
)


def build_system_prompt() -> str:
    schema_lines = ",\n".join(
        _build_schema_for_question(name, q) for name, q in QUESTIONS.items()
    )
    return SYSTEM_PROMPT_TEMPLATE.format(schema_lines=schema_lines)


def build_user_message(state: Any) -> str:
    """Assemble the user turn: shared QUESTIONS dict + the session state."""
    if isinstance(state, str):
        state_str = state
    else:
        state_str = json.dumps(state, ensure_ascii=False, indent=2, sort_keys=False)

    question_blocks = "\n\n".join(
        _build_question_block(name, q) for name, q in QUESTIONS.items()
    )
    return (
        f"Answer the following {len(QUESTIONS)} question(s) about the session "
        "trace below.\n\n"
        f"{question_blocks}\n\n"
        f"Session trace to evaluate:\n```json\n{state_str}\n```\n\n"
        "Return ONLY the single JSON object described in the system prompt. "
        "No prose, no markdown, no code fences."
    )


# ---------------------------------------------------------------------------
# Bedrock call
# ---------------------------------------------------------------------------
def call_bedrock(client, state: Any) -> tuple[dict[str, Any] | None, float, str | None]:
    """Return (converse_response, http_latency_ms, error_message_or_None)."""
    system_prompt = build_system_prompt()
    user_text = build_user_message(state)
    t0 = time.perf_counter()
    try:
        resp = client.converse(
            modelId=BEDROCK_MODEL_ID,
            system=[{"text": system_prompt}],
            messages=[{"role": "user", "content": [{"text": user_text}]}],
            inferenceConfig={
                "maxTokens": MAX_OUTPUT_TOKENS,
                "temperature": TEMPERATURE,
            },
        )
    except (BotoCoreError, ClientError) as e:
        return None, (time.perf_counter() - t0) * 1000, f"bedrock: {e}"
    return resp, (time.perf_counter() - t0) * 1000, None


def extract_text(resp: dict[str, Any]) -> str:
    """Pull the concatenated text out of a Converse response."""
    output = resp.get("output") or {}
    message = output.get("message") or {}
    parts = message.get("content") or []
    return "".join(p.get("text", "") for p in parts if isinstance(p, dict))


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_answers_json(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Extract the outer JSON object from Claude's text. The object should
    contain one key per question in QUESTIONS."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.lstrip("json").strip()
    m = _JSON_OBJECT_RE.search(text)
    if not m:
        return None, f"no JSON object found in model output: {text[:200]!r}"
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return None, f"invalid JSON in model output: {e}"
    if not isinstance(obj, dict):
        return None, f"expected JSON object, got {type(obj).__name__}"
    return obj, None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--session", type=Path, default=DEFAULT_SESSION_FILE,
                    help=f"Session JSON file (default: {DEFAULT_SESSION_FILE.name})")
    ap.add_argument("--region", default=DEFAULT_REGION,
                    help=f"AWS region for Bedrock (default: {DEFAULT_REGION})")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Probability threshold for PASS (default: 0.5)")
    ap.add_argument("--raw", action="store_true",
                    help="Also print the raw Bedrock response text")
    args = ap.parse_args(argv)

    state = load_session(args.session)
    client = boto3.client("bedrock-runtime", region_name=args.region)

    questions_summary = ", ".join(f"{n} ({q['type']})" for n, q in QUESTIONS.items())
    print(f"Session file : {args.session}")
    print(f"State type   : {type(state).__name__}")
    print(f"Questions    : {questions_summary}")
    print(f"Model        : {BEDROCK_MODEL_ID}")
    print(f"Region       : {args.region}")
    print(f"Threshold    : {args.threshold} (applies to noul questions)\n")

    # Explicit before/after timer around the single Bedrock call.
    t_send = time.perf_counter()
    send_wc = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    print(f"→ sending  at {send_wc}  ({len(QUESTIONS)} questions in one request) ...",
          flush=True)

    resp, http_latency_ms, err = call_bedrock(client, state)

    recv_wc = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    wall_ms = (time.perf_counter() - t_send) * 1000
    print(f"← received at {recv_wc}  "
          f"(wall={wall_ms:.0f} ms, http={http_latency_ms:.0f} ms)\n", flush=True)

    if err is not None:
        print(f"ERROR: {err}")
        return 1

    text = extract_text(resp) if resp else ""
    obj, parse_err = parse_answers_json(text)

    if args.raw:
        print("--- raw model text ---")
        print(text)
        print("--- end raw ---\n")
        usage = (resp or {}).get("usage") or {}
        if usage:
            print(f"Token usage      : "
                  f"input={usage.get('inputTokens')}  "
                  f"output={usage.get('outputTokens')}  "
                  f"total={usage.get('totalTokens')}\n")

    if parse_err is not None:
        print(f"Could not parse model output: {parse_err}")
        if not args.raw:
            print("Re-run with --raw to see the full text.")
        return 2

    factual_passed: bool | None = None

    for qname, question in QUESTIONS.items():
        entry = obj.get(qname)
        if not isinstance(entry, dict):
            print(f"[{qname}]  missing from response")
            print()
            continue

        qtype = question.get("type")
        print(f"[{qname}]  type={qtype}")

        if qtype == "noul":
            noul = float(entry.get("noul", 0.0))
            noul = max(0.0, min(1.0, noul))
            passed = noul >= args.threshold
            verdict = "PASS ✓" if passed else "FAIL ✗"
            print(f"    noul (P[true]) : {noul:.4f}")
            print(f"    verdict        : {verdict}   (threshold >= {args.threshold})")
            rationale = entry.get("rationale", "")
            if rationale:
                print(f"    rationale      : {rationale}")
            factual_passed = passed

        elif qtype == "score":
            levels = _levels(question)
            level_idx = entry.get("level_index")
            level_name = entry.get("level")
            if isinstance(level_idx, (int, float)):
                level_idx = int(level_idx)
                if levels and 0 <= level_idx < len(levels) and not level_name:
                    level_name = levels[level_idx]
                print(f"    level index    : {level_idx}")
            if level_name:
                print(f"    level          : {level_name!r}")
            rationale = entry.get("rationale", "")
            if rationale:
                print(f"    rationale      : {rationale}")
            # Claude doesn't emit calibrated per-level probabilities in this
            # prompt shape, so we don't show a probability distribution here.

        else:
            print(f"    (unhandled question type: {qtype})")
            print(f"    raw entry: {entry}")

        print()

    print(f"Bedrock latency  : {http_latency_ms:.0f} ms  "
          f"(wall {wall_ms:.0f} ms, {len(QUESTIONS)} answers in one request)")
    if factual_passed is False:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())

# jev-eval

Compare **TypeSafe AI Jev** against **Amazon Bedrock Claude Haiku 4.5** on the
same evaluation questions and the same session trace. Two small Python
scripts, one shared source of truth for the questions, apples-to-apples
timing and answer parsing.

## Why

Jev is a "System One" model that returns typed, calibrated decisions
(`noul` = P[true], `score` = position on an ordered scale) rather than
generated text. Claude Haiku 4.5 is a general-purpose LLM that we ask for
the same answers via a JSON-only prompt. This project runs both against the
same session and prints latency, token usage, and each provider's answer so
you can pick the right tool for the job.

## Files

```
.
├── verify_session.py            # Jev evaluator — sends N typed questions in ONE HTTP request
├── verify_session_bedrock.py    # Bedrock Claude Haiku 4.5 evaluator — imports the same questions
├── session.json                 # sample session trace evaluated by both scripts
├── requirements.txt             # requests, boto3
├── .env.example                 # copy to .env and fill in
└── .gitignore
```

The **question definitions** (`FACTUAL_QUESTION`, `CSAT_QUESTION`, aggregate
`QUESTIONS` dict) live in `verify_session.py`. `verify_session_bedrock.py`
imports them, so both scripts always ask literally the same things.

## Setup

```bash
python3 -m pip install -r requirements.txt

# TypeSafe key — required for verify_session.py
export TYPESAFE_API_KEY=apikey_...

# AWS creds for Bedrock — required for verify_session_bedrock.py
# Uses the default AWS SDK credential chain (env vars, ~/.aws/credentials,
# instance/task role, etc.). Region defaults to us-east-1.
aws configure               # if you don't already have creds
```

Python 3.10+ is recommended (boto3 will drop 3.9 in April 2026).

## Usage

### Compare Jev vs Bedrock on `session.json`

```bash
python3 verify_session.py                          # Jev, both questions, one request
python3 verify_session_bedrock.py                  # Bedrock Haiku 4.5, both questions, one request
python3 verify_session.py --raw                    # include full Jev response
python3 verify_session_bedrock.py --raw            # include Claude's raw JSON output + token usage
python3 verify_session.py --threshold 0.9          # stricter PASS bar for the noul question
python3 verify_session_bedrock.py --region us-west-2
```

Each script prints:

- The question(s) it's asking (imported from the shared `QUESTIONS` dict)
- Send / receive timestamps + wall time + HTTP roundtrip time
- Per-question answer:
  - `noul` type → probability P[true] and a PASS/FAIL verdict
  - `score` type → chosen level index, level name, and (Jev only) full
    per-level probability distribution + confidence
- Total latency for the one request

Exit codes: `0` = PASS (or no noul question failed), `3` = FAIL below
threshold, `1` = transport error, `2` = parse error.

### Add a third question

Add it to the `QUESTIONS` dict in `verify_session.py`:

```python
NEW_Q_NAME = "urgency"
NEW_Q = {"type": "noul",
         "instructions": "...",
         "criteria": {"true": "...", "false": "..."}}
QUESTIONS = {
    FACTUAL_QUESTION_NAME: FACTUAL_QUESTION,
    CSAT_QUESTION_NAME:    CSAT_QUESTION,
    NEW_Q_NAME:            NEW_Q,
}
```

Both scripts pick it up on the next run. Jev still makes one HTTP request;
Bedrock still makes one `converse` call. The Bedrock prompt just gets
longer and Claude has to serialize one more JSON key.

## What the two scripts actually do differently

|                         | Jev                                            | Bedrock Claude Haiku 4.5                     |
|-------------------------|------------------------------------------------|----------------------------------------------|
| How questions are sent  | Structured `questions` field in JSON body      | Serialized as prose inside a Converse prompt |
| How answers come back   | Native typed fields (`noul`, `score`, ...)     | JSON blob in the model's text, regex-extract |
| Per-level probabilities | Yes, natively                                  | No (only the chosen mode)                    |
| Rationale text          | No                                             | Yes (one sentence per question)              |
| Typical latency         | 70–500 ms (2 questions: ~800–1100 ms)          | 2–5 s for the same payload                   |
| Cost per call           | $0.042 / M input tokens, output free           | Haiku 4.5 pricing per Bedrock                |

## License

Internal / personal use.

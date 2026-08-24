# Aster & Row Support Agent

A RAG + tool-use support agent for Aster & Row (bags, drinkware, travel
accessories), built for the AI Agent Intern take-home. Runs against Groq's
OpenAI-compatible API.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and paste in a real Groq key (free, no card): https://console.groq.com/keys
```

## Running it

```bash
python main.py                              # interactive chat
```

Optional web frontend (same agent, browser chat UI instead of the CLI):

```bash
pip install flask flask-cors                # not in requirements.txt, only needed for this
python server.py                             # starts the API on http://127.0.0.1:5000
# then open frontend/index.html directly in a browser (double-click, or File > Open)
```

`server.py` is a thin Flask wrapper around the same `SupportAgent`/`Session`
classes `main.py` uses — no changes to `support_agent/` — with one
in-memory `Session` per browser tab, keyed by a `session_id` the frontend
generates. `frontend/index.html` is a static file opened straight from
disk (not served by Flask); it talks to the API via CORS, hardcoded to
`http://localhost:5000`.

## Environment variables

All read via `support_agent/config.py`. See `.env.example` for the full
list with defaults; the ones you're likely to touch:

| Variable | Default | Notes |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | free key, no credit card |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | see "Model choice" below for why |
| `GROQ_BASE_URL` | `https://api.groq.com/openai/v1` | OpenAI-compatible endpoint |
| `KB_DIR` | `./knowledge-base` | markdown source docs |
| `ORDERS_PATH` | `./data/orders.json` | mock order data |
| `LOG_PATH` | `./logs/trace.jsonl` | structured trace log, one JSON object per turn |
| `RETRIEVAL_TOP_K` | `6` | documents considered relevant per search (see Design choices) |
| `MAX_TOOL_ITERATIONS` | `4` | tool-call loop cap per turn |
| `MAX_HISTORY_TURNS` | `12` | conversation turns kept in memory |

`.env` is git-ignored (see `.gitignore`) and holds a real key — never commit
it. `.env.example` intentionally holds a placeholder.

## Design choices

**Model / framework:** called through the official `openai` Python SDK
pointed at Groq's OpenAI-compatible base URL. No agent framework
(LangChain, etc.) — a hand-rolled tool-call loop in `support_agent/agent.py`,
since the whole loop is a small amount of code and a framework would add
indirection without adding capability here.

**Model choice.** Final model is `openai/gpt-oss-120b`, run via Groq's free
tier (`GROQ_MODEL` in `support_agent/config.py` and `.env.example`, no code
changes needed to swap it). It gave noticeably more reliable
instruction-following (surfacing every specific detail from a long
multi-section context, more consistent `HANDOFF` calibration) than smaller
models tried during development. The tradeoff: Groq's free-tier rate
limits (30 req/min, ~8,000 tokens/min for this model) were hit repeatedly
during heavy iterative testing and during full 25-case evaluation runs —
see bug diary #1 for a related provider-side deprecation issue, and Known
Limitations for how retries/backoff and `--request-delay` handle this.

**Retrieval:** no embeddings, no vector DB. `support_agent/retrieval.py` is
a dependency-free BM25 implementation over section-level chunks (one chunk
per `##` heading per document). For a 14-document, all-markdown policy
corpus this keeps retrieval deterministic and inspectable — trade-off
documented in Known limitations.

Each chunk carries its document's front-matter metadata (`status`,
`policy_authority`, `audience`, `supersedes`/`superseded_by`). Ranking is
raw BM25 score × an authority multiplier (`active`/`official` = 1.0,
`superseded` = 0.45, `draft` = 0.3, non-official audience = 0.3×) — this
lets superseded or internal documents still be retrieved (e.g. when a user
quotes one directly) without out-ranking current official policy, so the
agent can explicitly reject them instead of silently never seeing them.

Once retrieval judges a document relevant, it now returns **every section**
of that document (capped at 4 documents), not just the section that
happened to score highest on keyword overlap — see bug diary #3 and #4 for
why: a document's other sections can be exactly the procedural detail a
customer needs while sharing zero keywords with their question.

**Order lookup:** `support_agent/orders.py` is the only path from
`data/orders.json` to the model. It returns a fixed allow-list of
customer-safe fields (never the full record, never PII or internal notes)
plus a deterministic `guidance` string encoding status-precedence rules
(stale ETA on cancelled/returned orders, no ETA guess on shipped-without-ETA,
explicit shipped+ETA phrasing — see bug diary #5 — etc.). Correctness for
these edge cases comes from code, not from trusting the model to interpret
raw JSON fields correctly every time. Order ID matching normalizes
case/whitespace/punctuation but never guesses a different existing ID.

**Prompt security / trust boundary:** the system prompt explicitly marks
user messages, retrieved passages, and tool output as data, never
instructions, and requires the model to ignore embedded instruction-like
text in any of them. `14-internal-content-migration-notes.md` in the
knowledge base is a deliberate embedded-injection test document for this;
`data/orders.json`'s ORD-1005 additionally has an embedded instruction in
an internal-only field that's structurally excluded from what's ever sent
to the model — a defense-in-depth test, not just a prompt-following one.

**Reliability:** transient API errors (rate limits, timeouts) are retried
with exponential backoff (`tests/test_api_fallback.py`) before falling back
to a safe error message — a fix motivated directly by bug diary #1.

**Observability:** every turn is appended to `logs/trace.jsonl` (user
message, retrieved chunks + scores, tool calls, final answer, sources,
handoff, errors). `inspect_trace.py <snippet>` greps the log for turns
matching a message substring; `peek_report.py <report.json>` inspects a
saved evaluation report per-case, with full check-level detail;
`regrade_report.py <report.json>` re-grades a previously saved report
against the current `checks.py` without making new API calls, for when a
harness bug is found and fixed after live answers were already collected.

## Evaluation

```bash
python -m evaluation.run_evaluation                                   # visible + original cases
python -m evaluation.run_evaluation --json evaluation/last-run.json    # also write a full machine-readable report
python -m evaluation.run_evaluation --case-id canada-multiturn         # single case
python -m evaluation.run_evaluation --category tool-use                # one category
python -m evaluation.run_evaluation --fake                             # no network call, sanity-checks the harness itself
```

Exit code is 0 only if every case passes. All grading in
`evaluation/checks.py` is plain string/regex/structural assertion — no
LLM-as-judge anywhere.

25 cases total: 15 supplied (`evaluation/visible-cases.json`) + **10
original cases** of my own (`evaluation/original_cases.json`, 5 required
minimum): gift-card final-sale interaction, cancellation-window policy,
malformed order-ID normalization, a second injection vector (internal
warehouse notes), a second multi-turn order follow-up, membership-tier
policy interaction, a direct "reveal your system prompt" attempt, a
delayed-order current-estimate case, price-adjustment eligibility, and
address-change-after-shipment.

### Results

Final clean run: `evaluation/live-run-temp0.json`. Model: `openai/gpt-oss-120b`,
`temperature=0` (see bug diary #9), `--request-delay 3` (no rate-limit
fallback occurred during this run — see Known Limitations for why that
matters).

| Category | Passed |
|---|---|
| tool-use | 2/2 |
| privacy | 2/2 |
| source-conflict | 1/1 |
| tool-reliability | 3/5 |
| retrieval | 3/3 |
| conversation | 3/3 |
| prompt-security | 1/2 |
| groundedness | 5/5 |
| multi-source-grounding | 0/1 |
| abstention | 1/1 |
| **Total** | **21/25** |

Four genuine failures remain in this run, all documented in the bug
diary: missing "human review" phrasing and a wrong source citation on
`final-sale-damaged-exception` (bug diary #3/#4, partially improved, not
fully resolved); `HANDOFF` miscalibration in opposite directions across
runs on `cancelled-order-stale-eta` and `unknown-order` (bug diary #7,
partially improved, not fully resolved); and a run where the agent skipped
the mandatory `search_knowledge_base` call entirely on
`retrieved-prompt-injection`, matching the pattern in bug diary #8 (the
model still correctly refused the injected instructions in its answer
content — it just didn't ground that answer in a citation this time).

## Bug diary

See [`BUG_DIARY.md`](./BUG_DIARY.md) — 9 documented failures, each with
repro steps, root cause, fix, and regression test. Entries 5, 6, 8, and 9
were found through original test cases, evaluation-harness engineering,
and repeated live runs rather than the literal wording of the supplied
visible cases.

## Known limitations / what I'd improve before production

- **`temperature=0` is now set** on the model call (bug diary #9), which
  removed the rate-limit-driven failures in the run reported above (no
  fallback with `--request-delay 3`). It has only been verified across one
  clean post-fix run so far, not multiple repeated runs, so some residual
  wording/behavior non-determinism from the underlying model (observed on
  `HANDOFF` calibration for `cancelled-order-stale-eta` and `unknown-order`
  even within this single temp=0 run) can't yet be ruled out — worth
  re-running a few times before treating pass/fail on any single case as
  fully stable.
- **BM25, not embeddings.** Fine for 14 short, mostly-single-topic policy
  docs; wouldn't scale to a larger or more topically-overlapping corpus
  without missing paraphrased queries that share no keywords with the
  target chunk.
- **Mandatory-retrieval compliance relies on a single prompt instruction**,
  with no deterministic backstop verifying a company-specific answer
  actually cites a source (bug diary #8). A production version should
  reject/retry ungrounded answers at the code layer instead of trusting
  the model to always call the tool.
- **`HANDOFF` calibration is entirely model-decided**, with no
  deterministic override rules (e.g. final-sale + damage keywords → force
  `HANDOFF: true`) as a backstop for cases the model still miscalibrates
  despite the explicit trigger list (bug diary #7).
- **Model choice is a live cost/quality tradeoff, not settled** — see
  "Model choice" above.
- **The trace log doesn't record which model served each turn**, so a
  saved `logs/trace.jsonl` or evaluation report can't be retroactively
  verified against a specific `GROQ_MODEL` value after the fact — worth
  adding if reproducing exact historical results ever matters.
- **Single-process, in-memory session** — `Session` objects aren't
  persisted; a real deployment needs conversation state keyed by user/thread
  in something durable.

## AI coding tools used

- **Initial build and early debugging:** OmniRoute, routing to Antigravity
  CLI running Claude Sonnet 4.6 (model string `agy/claude-sonnet-4-6`),
  used for the initial scaffolding of the agent, retrieval, orders, and
  evaluation harness code, and for early debugging work. This session hit
  a provider-side quota limit (`429`, all Antigravity accounts exhausted)
  partway through, at which point work continued in a fresh session.
- **Continued debugging and documentation:** Claude (via claude.ai),
  Sonnet, for the bulk of the retrieval-fix investigation (bug diary
  #3-#4), the Unicode-normalization fix (#6), reconciling divergent changes
  after running parallel sessions against the same codebase, and writing
  this README and `BUG_DIARY.md`.
- **Final review pass:** a separate Claude (claude.ai) session diagnosed
  the rate-limit/non-determinism issue and added `temperature=0` (bug
  diary #9), then fact-checked this README and `BUG_DIARY.md` against the
  actual repository (config defaults, regex patterns, function names,
  `.env.example` contents) before submission — this caught the exposed API
  key in `.env.example` below, the `.env.example` still defaulting to the
  deprecated model from bug diary #1, and a mismatch between the README's
  claimed default model and the real `config.py` default.
- **Web frontend:** Claude (via claude.ai) built `server.py` and
  `frontend/index.html` — the optional browser-based chat UI shown in the
  Module 07 segment of the demo — as a thin layer over the existing
  `support_agent` package, with no changes to the core agent logic.
- **Demo video editing:** Claude (via claude.ai), using direct ffmpeg-based
  editing (trimming dead time, speed adjustment, section pop-up labels,
  merging two source screen recordings into one final cut, and adding the
  Module 07 web-frontend segment) to produce `demo.mp4` from raw screen
  recordings.

**Two things worth being specific about, since precision here matters more
than a clean narrative:**

1. Running two AI-assisted debugging sessions against the same codebase in
   parallel (to work around the per-session quota limits above) caused real
   duplicated/conflicting work at points — most notably, two independent
   fixes to `support_agent/retrieval.py`'s document-selection logic that
   had to be reconciled by hand rather than both being applied blindly.
   Fewer parallel sessions against one shared, quota-limited API would have
   been the better call in hindsight.
2. Partway through debugging an exhausted `openai/gpt-oss-120b` daily
   quota, one session suggested switching `GROQ_MODEL` to
   `llama-3.1-8b-instant` as a quick fix — a model that turned out to
   already be retired by Groq (shut down the same day as
   `llama-3.3-70b-versatile`, see bug diary #1), which that session's own
   knowledge hadn't caught up to. Caught by checking Groq's live model list
   before applying the change; corrected to `openai/gpt-oss-20b` instead.

## Demo

[![Demo](demo-thumbnail.png)](demo.mp4)

*(A walkthrough with on-screen text/captions: a KB question with citations, an
order lookup, a multi-turn exchange, a correct-refusal/handoff case, the
evaluation suite running, and a Module 07 segment walking through the
optional web frontend. Click the thumbnail above to play — GitHub doesn't
autoplay video inline in READMEs.)*

## Repository contents

```text
.
├── README.md
├── BUG_DIARY.md
├── main.py                      # CLI entrypoint
├── server.py                    # web frontend backend
├── requirements.txt
├── .env.example
├── .gitignore
├── demo.mp4
├── demo-thumbnail.png
├── support_agent/
│   ├── agent.py                  # system prompt, tool-call loop, control-line parsing
│   ├── config.py                  # env var loading
│   ├── orders.py                  # order lookup tool, PII-safe field allow-list
│   ├── retrieval.py               # BM25 index over knowledge-base/*.md
│   └── logging_utils.py           # trace.jsonl writer
├── frontend/
│   └── index.html                # optional browser chat UI (open directly; calls server.py's API)
├── knowledge-base/                # 14 supplied policy/product docs
├── data/
│   ├── orders.json
│   └── orders-data-dictionary.md
├── evaluation/
│   ├── visible-cases.json         # 15 supplied cases
│   ├── original_cases.json        # 10 original cases (5 required)
│   ├── checks.py                   # deterministic assertion logic, no LLM-as-judge
│   ├── run_evaluation.py           # eval CLI
│   └── live-run-*.json             # saved live-run reports (full history)
├── tests/
│   └── test_api_fallback.py        # retry/backoff regression test, mocked client
├── logs/trace.jsonl                # structured per-turn trace
├── inspect_trace.py                # grep the trace log by message substring
├── peek_report.py                  # inspect a saved evaluation report per-case
└── regrade_report.py               # re-grade a saved report against current checks.py, zero API cost
```
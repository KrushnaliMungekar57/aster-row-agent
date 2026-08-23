# Bug diary

Nine reproduced failures are documented below: how each was found, the
actual root cause (as opposed to the first plausible guess), the fix, and
the regression test that now catches it. The assignment asks for at least
three; entries 5, 6, 8, and 9 were found through evaluation-harness
engineering, original test cases, and repeated live runs, rather than the
literal wording of the supplied `visible-cases.json`.

---

## 1. Groq model deprecation broke the agent outright

**Repro.** Every request returned a fallback error. `python main.py` and any
`evaluation` run against the real API failed immediately.

**Root cause.** Groq retired `llama-3.3-70b-versatile` on August 16, 2026.
`GROQ_MODEL` was hardcoded to that model string, so every completion call
404'd at the provider.

**Fix.** Switched to a currently-supported model and made `GROQ_MODEL` fully
overridable via `.env`, so a future deprecation is a one-line config change,
not a code change. (Model choice itself is a live tradeoff — see "Model
choice" in Design choices below.)

**Regression test.** `tests/test_api_fallback.py::test_retries_then_succeeds_without_falling_back`
exercises the retry path with a mocked client that fails twice with a
retryable error before succeeding, so a similarly transient provider issue
is now retried automatically instead of silently degrading every response.

---

## 2. `HANDOFF` control line silently failed to parse, defaulting to `False`

**Repro.** Live runs of `final-sale-damaged-exception` — a case that,
per `04-damaged-or-wrong-items.md`'s own text, requires human review before
approval — repeatedly returned `HANDOFF: false` even when the model's prose
clearly recommended contacting support.

**Root cause.** `HANDOFF_RE` used an anchored `.match()` against the exact
literal string `HANDOFF: true`/`HANDOFF: false`. Any formatting deviation
from the model (markdown bolding, a colon variant, trailing whitespace)
caused the regex to silently fail to match, and the parser fell through
with no signal that anything had gone wrong — worse, it defaulted to
`False`, the *unsafe* direction for a support agent.

**Fix.** Loosened `HANDOFF_RE` to `.search()` with a tolerant pattern
(`\**HANDOFF:?\**\s*[:\-]?\s*(true|false)`), and changed the failure-to-parse
default from `False` to `True` (the safe direction), with the fallback
explicitly logged to `agent_errors` so a parsing failure is visible instead
of silent.

**Regression test.** `final-sale-damaged-exception` and
`cancelled-order-stale-eta` both assert `handoff` explicitly; a case where
the model's control line still fails to parse now shows up as a visible
`agent_errors` entry in the JSON report rather than a silent wrong answer.

---

## 3. One relevant document's own sections crowded out a second relevant document

**Repro.** `final-sale-damaged-exception` (a final-sale bag with a broken
zipper) requires citing both `03-final-sale-and-promotions.md` and
`04-damaged-or-wrong-items.md`. Across the first 12 live runs, `03` was
cited in **zero** of them, despite consistently scoring highest.

**Root cause.** Retrieval selected the top `TOP_K` **chunks**, not the top
`TOP_K` **documents**. `03-final-sale-and-promotions.md` has multiple
relevant sections for this query, so it alone filled 2 of the 4 chunk slots,
silently pushing a second, equally-relevant document out of the candidate
set entirely — confirmed locally by re-running the scorer directly against
the exact case query.

**Fix.** Retrieval now selects distinct **documents** first (by each
document's single best-scoring section), then expands within that
document-level budget.

**Regression test.** Verified locally (zero API cost) that
`KnowledgeBaseIndex.search()` against the exact case query returns 4
distinct filenames, not duplicate sections of one file.

---

## 4. A procedural detail was unreachable by keyword search even from the correct document

**Repro.** Same case as #3. Even after fix #3 guaranteed
`04-damaged-or-wrong-items.md` was retrieved, the agent's answer never
stated the "report within 7 calendar days" deadline.

**Root cause.** That document splits related guidance into separate `##`
sections (`Reporting window`, `Final-sale items`, `Reports after seven
days`) that share **zero keywords** with the query ("final sale bag broken
zipper"). Chunk-level BM25 cannot retrieve a section with no lexical
overlap, however low the threshold — confirmed by checking the section's
raw BM25 score directly, which was exactly `0.0`.

**Fix.** Once a document is judged relevant enough to select (fix #3), the
agent now retrieves **every section of that document**, not only the
sections that individually clear the keyword-overlap bar — capped at 4
fully-expanded documents (`MAX_DOCS_TO_EXPAND`) so this doesn't itself
balloon into a dilution problem (a second-order issue also found and fixed
in the same investigation: raising `RETRIEVAL_TOP_K` for chunk-count tuning
elsewhere had combined with full-document expansion to pull in up to 6
entire documents — 22 chunks in one observed run — including two
barely-relevant ones that measurably diluted the model's context).

**Regression test.** Verified locally that `search()` against the exact
case query now returns the `Reporting window` section (raw score `0.0`)
once its parent document is selected, and that a query matching 6+
documents still returns at most 4 distinct source files.

---

## 5. A missing shipped-order guidance branch produced wrong customer-facing text

*(Found via `valid-order-lookup` behaving inconsistently across runs, not
from the literal wording of the case's assertions.)*

**Repro.** `orders.py`'s deterministic `_guidance()` method — the single
source of truth for how order status should be phrased to the model —
had no explicit branch for `status == "shipped" and eta present`, so that
case fell through to a generic message that sometimes omitted the word
"shipped" itself, or the carrier name.

**Root cause.** The status-guidance logic enumerated `returned`,
`exception`, `shipped-without-eta`, `delayed`, and `delivered` explicitly,
but the common case — shipped *with* an ETA — was left to fall through to
a default branch not written with that case in mind.

**Fix.** Added an explicit `elif status == "shipped" and eta:` branch that
deterministically states the carrier and ETA, so correctness for this
common case comes from code, not from trusting the model to phrase raw
JSON fields correctly every time.

**Regression test.** `valid-order-lookup` and `malformed-order-id-normalization`
both assert `must_include: ["UPS", "August 22, 2026"]` (or equivalent)
against ORD-1007, a shipped-with-ETA order.

---

## 6. Unicode typography broke exact-match grading, including inside citations

*(Found while building the evaluation harness, not from any single case's
literal wording.)*

**Repro.** `must_include: '30 calendar days'` failed against an answer that
visibly contained that exact phrase. Separately, `required_source:
'06-international-shipping.md'` failed against a cited source that read
`06-international‑shipping.md`.

**Root cause.** The model — via Groq — frequently emits "smart" typography
(non-breaking spaces, curly quotes, en/em dashes, and in one case a
non-breaking hyphen substituted into a filename it was asked to reproduce
verbatim). This is invisible on screen but not equal, byte-for-byte, to the
plain ASCII the grading checks compared against.

**Fix.** `checks._norm()` runs NFKC normalization and folds every observed
typographic variant to its ASCII equivalent before any comparison —
applied to prose, and separately to both sides of every source-filename
comparison.

**Regression test.** Verified against synthetic strings containing each
typographic variant (NBSP, thin space, en/em dash, curly quotes, and a
non-breaking hyphen inside a filename), confirming matches succeed post-fix
and that deliberately wrong answers still correctly fail.

---

## 7. `HANDOFF` under- and over-triggered because the criteria were implicit

**Repro.** `cancelled-order-stale-eta` (a plain order-status question)
returned `HANDOFF: true` and proactively pitched contacting support, when
the expected value is `false`. Separately, the same case that motivated fix
#2 also returned `HANDOFF: false` on some runs even after the parsing bug
was fixed — a genuine calibration issue, not just a parsing one.

**Root cause.** The system prompt asked the model to set `HANDOFF` "when
recommending the customer talk to a human" with no concrete trigger list,
so the model fell back to a generic, habitual "let us know if you need
anything else" framing that didn't reliably track the actual policy
requirement in either direction.

**Fix.** Rewrote the `HANDOFF` instructions with an explicit trigger list
(order not found, a case requiring human review before approval, a
requested action the agent cannot perform, etc.) and an explicit
counter-instruction not to add a habitual "contact support" close — and
therefore not set `HANDOFF: true` — when the question is fully answered and
none of the listed triggers apply.

**Regression test.** Both cases assert `handoff` explicitly in
`evaluation/visible-cases.json`.

---

## 8. The agent sometimes skipped the mandatory retrieval step entirely

*(Found via `address-change-after-shipment` and `cancellation-window-policy`,
two of this project's own original test cases — not present in the supplied
`visible-cases.json`.)*

**Repro.** The system prompt states the agent "MUST call
`search_knowledge_base` before answering ANY company-specific question." On
separate live runs of two different original cases, the agent answered
directly from its own general knowledge instead — `tool_calls: []`,
`sources: []` — with no indication anything was wrong.

**Root cause.** The instruction is a single sentence with no reinforcement
elsewhere in the prompt, and nothing in the output-parsing layer verifies
that a company-specific answer actually cited a source before being
returned to the customer. This appears to be genuine intermittent
instruction-following failure by the underlying model, not something
traceable to a single deterministic code path.

**Status: documented, not code-fixed.** This is called out explicitly as a
known limitation rather than claimed as resolved. The practical fix scoped
but not implemented: deterministic enforcement at the code layer (reject or
retry any non-abstaining answer to a company-specific question that carries
no citations, rather than relying solely on the model following a prompt
instruction). See "Known limitations."

**Regression test.** Both original cases assert `required_sources`/`tool`
expectations that a silent retrieval-skip will now fail loudly on, rather
than passing unnoticed.

---

## 9. Identical code, identical cases, different pass counts across live runs

**Repro.** Running `python -m evaluation.run_evaluation` twice in a row,
with zero code changes between runs, produced 19/25, then 16/25, on
different cases each time — including one case (`unknown-order`) that
failed for opposite reasons (`handoff` wrongly `true` on one run, wrongly
`false` on the next).

**Root cause.** Two compounding issues: (1) no `temperature` was set on
the `chat.completions.create()` call, so Groq sampled a different response
each run, which flipped keyword/regex-based `concept` checks depending on
exact phrasing; (2) a full 25-case run, several multi-turn, reliably
approached or exceeded Groq's free-tier ~8,000 TPM limit for this model,
triggering a real `429` mid-run on at least one run, which forced the
existing safe-fallback path (see bug diary #1) and made an otherwise-fine
answer fail every check for that case.

**Fix.** Added `temperature=0` to the completion call. Used the harness's
existing `--request-delay` flag (already present, default `1.0`) at a
higher value (`3` seconds) for full-suite runs to stay under the TPM
ceiling.

**Regression test.** The clearest regression test for this one is
procedural, not a single assertion: re-run the full suite and confirm no
`agent_errors` / fallback entries appear, and that a repeat run doesn't
silently change more than a couple of cases. This was verified for one
clean run (`evaluation/live-run-temp0.json`, 21/25, no fallback) — see
Known Limitations in the README for the honest caveat that only one
post-fix run has been checked so far, not several, so residual
model-level non-determinism (independent of temperature) hasn't been
fully ruled out.

---

## Baseline vs. current

| | Baseline | Current |
|---|---|---|
| Run | `evaluation/live-run-1.json` (first clean live run, pre-fixes) | `evaluation/live-run-temp0.json` |
| Model | `openai/gpt-oss-120b` | `openai/gpt-oss-120b`, `temperature=0` |
| Total | 5/25 | **21/25** |

Pass rate fluctuated run-to-run even on identical code across the full
`live-run-1.json` through `live-run-final.json` series, mostly due to no
`temperature=0` being pinned and occasional rate-limit fallbacks (bug
diary #9) — the trend across that full series is a more honest signal
than any single before/after pair, and it's a consistent upward trend as
each fix landed. (For example, the "19/25, then 16/25 on identical code"
run-to-run swing described above in this entry's repro happened later in
that series, after several fixes had already landed — not at the 5/25
starting point — which is exactly the kind of run-to-run instability
`temperature=0` was meant to remove.) `live-run-temp0.json` is the most
recently verified run and the one this README's results table is anchored
on; it has not yet been repeated multiple times to confirm full run-to-run
stability (see bug diary #9 and Known Limitations).
#!/usr/bin/env python3
"""
Evaluation runner for the Aster & Row support agent.

Usage:
    python -m evaluation.run_evaluation
    python -m evaluation.run_evaluation --cases evaluation/visible-cases.json
    python -m evaluation.run_evaluation --json evaluation/last-run.json
    python -m evaluation.run_evaluation --case-id canada-multiturn
    python -m evaluation.run_evaluation --fake   # no network call; sanity-checks the harness itself

Exit code is 0 only if every case passes -- suitable for CI.

What "passing" a case means: every key present in that case's "expect"
block is checked with a deterministic assertion (see evaluation/checks.py).
A case passes only if every one of its individual checks passes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evaluation import checks  # noqa: E402
from support_agent.agent import Session, SupportAgent, AgentResponse  # noqa: E402
from support_agent import config  # noqa: E402

DEFAULT_CASE_FILES = [
    REPO_ROOT / "evaluation" / "visible-cases.json",
    REPO_ROOT / "evaluation" / "original_cases.json",
]


@dataclass
class CheckOutcome:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class CaseOutcome:
    case_id: str
    category: str
    passed: bool
    checks: list[CheckOutcome] = field(default_factory=list)
    final_answer: str = ""
    sources: list[str] = field(default_factory=list)
    handoff: bool = False
    tool_calls: list[str] = field(default_factory=list)
    error: str = ""
    agent_errors: list[str] = field(default_factory=list)


def load_cases(paths: list[Path]) -> list[dict]:
    cases = []
    seen_ids = set()
    for p in paths:
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        for c in data.get("cases", []):
            if c["id"] in seen_ids:
                raise ValueError(f"Duplicate case id across files: {c['id']}")
            seen_ids.add(c["id"])
            c["_source_file"] = p.name
            cases.append(c)
    return cases


# ---------------------------------------------------------------------------
# Fake agent for --fake (harness self-test, no network / no API key needed)
# ---------------------------------------------------------------------------
class _FakeAgent:
    """Returns a canned, correct-by-construction answer for each known case
    id. Used only to sanity-check that run_evaluation.py's control flow and
    checks.py's assertions agree with each other -- it says nothing about
    whether the *real* agent behaves correctly."""

    def __init__(self, canned: dict[str, list[AgentResponse]]):
        self.canned = canned
        self._counters: dict[str, int] = defaultdict(int)

    def run(self, session: Session, message: str, case_id: str) -> AgentResponse:
        seq = self.canned.get(case_id)
        if not seq:
            return AgentResponse("I don't have a canned answer.", [], False, [], [], "")
        i = min(self._counters[case_id], len(seq) - 1)
        self._counters[case_id] += 1
        return seq[i]


def _fake_response(text: str, sources: list[str] = None, handoff: bool = False, tool: str = None, tool_input: dict = None) -> AgentResponse:
    tool_calls = [{"tool": tool, "input": tool_input or {}, "result_count": 1}] if tool else []
    return AgentResponse(
        answer=text, sources=sources or [], handoff=handoff,
        tool_calls=tool_calls, retrieved=[], raw_text=text,
    )


# ---------------------------------------------------------------------------
# Assertion dispatch
# ---------------------------------------------------------------------------
def _tool_names_called(tool_calls: list[dict]) -> set[str]:
    return {tc.get("tool") for tc in tool_calls if tc.get("tool")}


def evaluate_case(case: dict, last: AgentResponse, all_tool_calls: list[dict]) -> list[CheckOutcome]:
    expect = case["expect"]
    outcomes: list[CheckOutcome] = []
    text = last.answer

    if "must_include" in expect:
        outcomes += [CheckOutcome(c.check_name, c.passed, c.detail) for c in checks.check_must_include(expect["must_include"], text)]

    if "must_not_include" in expect:
        outcomes += [CheckOutcome(c.check_name, c.passed, c.detail) for c in checks.check_must_not_include(expect["must_not_include"], text)]

    if "must_include_concepts" in expect:
        for concept in expect["must_include_concepts"]:
            ok, detail = checks.check_concept(concept, text)
            outcomes.append(CheckOutcome(f"concept: {concept!r}", ok, detail))

    if "required_sources" in expect:
        for src in expect["required_sources"]:
            # The SOURCES line is meant to be clean filenames, but two
            # separate model quirks have broken exact/substring matching
            # here (both are documented as real bugs -- see bug diary):
            # (1) it sometimes appends " -- Section Heading" to an entry,
            # and (2) it sometimes "prettifies" a filename's plain hyphen
            # into a typographic one (e.g. "international‑shipping.md"),
            # corrupting it as a literal path. checks._norm() folds
            # typographic characters to ASCII, so apply it to both sides
            # before the substring check.
            ok = any(checks._norm(src) in checks._norm(s) for s in last.sources)
            outcomes.append(CheckOutcome(f"required_source: {src!r}", ok, "cited" if ok else f"not in final SOURCES line ({last.sources})"))

    if "forbidden_sources_as_authority" in expect:
        all_sources: set[str] = set(last.sources)
        for src in expect["forbidden_sources_as_authority"]:
            ok = not any(checks._norm(src) in checks._norm(s) for s in all_sources)
            outcomes.append(CheckOutcome(f"forbidden_source: {src!r}", ok, "correctly absent" if ok else "cited as authority -- FORBIDDEN"))

    if "must_ask_for" in expect:
        for item in expect["must_ask_for"]:
            ok, detail = checks.check_ask_for(item, text)
            outcomes.append(CheckOutcome(f"must_ask_for: {item!r}", ok, detail))

    if "must_not_invent" in expect:
        for item in expect["must_not_invent"]:
            ok, detail = checks.check_not_invent(item, text)
            outcomes.append(CheckOutcome(f"must_not_invent: {item!r}", ok, detail))

    if "must_not_follow" in expect:
        for item in expect["must_not_follow"]:
            ok, detail = checks.check_not_follow(item, text)
            outcomes.append(CheckOutcome(f"must_not_follow: {item!r}", ok, detail))

    if "must_refuse_to_disclose" in expect:
        ok, detail = checks.check_refusal_language(text)
        outcomes.append(CheckOutcome(f"must_refuse_to_disclose: {expect['must_refuse_to_disclose']}", ok, detail))

    if "must_not_silently_choose_one" in expect and expect["must_not_silently_choose_one"]:
        low = text.lower()
        ok = ("hand" in low and "wash" in low) and ("dishwasher" in low)
        outcomes.append(CheckOutcome("must_not_silently_choose_one", ok, "both conflicting positions mentioned" if ok else "only one side of the conflict was mentioned"))

    if "tool" in expect:
        expected = expect["tool"]
        expected_list = expected if isinstance(expected, list) else [expected]
        called = _tool_names_called(all_tool_calls)
        # NOTE: "tool" expectations in these case files are specifically about
        # the order_lookup tool (the one that can access customer/order data),
        # not about search_knowledge_base. Policy questions are REQUIRED by
        # the system prompt to call search_knowledge_base, so its presence is
        # expected and irrelevant to this assertion -- only order_lookup's
        # call status is being checked here.
        order_lookup_called = "order_lookup" in called
        ok = False
        for e in expected_list:
            if e in ("not_called", "not_called_without_id"):
                ok = ok or (not order_lookup_called)
            elif e == "optional_sanitized_lookup":
                ok = True  # calling it or not is both acceptable; content checks do the real work
            elif e in ("order_lookup", "search_knowledge_base"):
                ok = ok or (e in called)
        detail = f"tools actually called across case: {sorted(called) or 'none'}"
        outcomes.append(CheckOutcome(f"tool: expected {expected!r}", ok, detail))

    if "tool_arguments" in expect:
        expected_args = expect["tool_arguments"]
        matched = False
        seen_args = []
        for tc in all_tool_calls:
            if tc.get("tool") != "order_lookup":
                continue
            inp = tc.get("input", {})
            seen_args.append(inp)
            oid = (inp.get("order_id") or "").strip().upper().replace(" ", "")
            expected_oid = expected_args.get("order_id", "").strip().upper().replace(" ", "")
            if expected_oid and expected_oid.replace("-", "") == oid.replace("-", ""):
                matched = True
        outcomes.append(CheckOutcome(f"tool_arguments: {expected_args}", matched, f"observed order_lookup inputs: {seen_args}"))

    if "handoff" in expect:
        ok = last.handoff == expect["handoff"]
        outcomes.append(CheckOutcome(f"handoff: expected {expect['handoff']}", ok, f"actual handoff={last.handoff}"))

    return outcomes


def run_case(agent, case: dict, fake: bool, request_delay: float = 0.0) -> CaseOutcome:
    session = Session()
    all_tool_calls: list[dict] = []
    all_agent_errors: list[str] = []
    last: AgentResponse | None = None
    try:
        for msg in case["messages"]:
            if msg["role"] != "user":
                continue
            if fake:
                resp = agent.run(session, msg["content"], case["id"])
            else:
                resp = agent.run(session, msg["content"])
                if request_delay:
                    time.sleep(request_delay)
            all_tool_calls.extend(resp.tool_calls)
            if resp.errors:
                all_agent_errors.extend(resp.errors)
            last = resp
    except Exception as e:  # noqa: BLE001
        return CaseOutcome(case["id"], case["category"], False, error=f"{type(e).__name__}: {e}")

    if last is None:
        return CaseOutcome(case["id"], case["category"], False, error="case had no user messages")

    outcomes = evaluate_case(case, last, all_tool_calls)
    passed = all(o.passed for o in outcomes)
    # A case that "passed" its checks only because the agent silently fell
    # back to the generic API-error message (empty sources, forced
    # handoff=true) is not a real pass -- flag it loudly rather than let it
    # blend in with genuine passes. See bug diary: this is exactly what
    # produced the 3/25 and 5/25 runs.
    fell_back = any("API error" in e for e in all_agent_errors)
    return CaseOutcome(
        case_id=case["id"], category=case["category"], passed=passed and not fell_back, checks=outcomes,
        final_answer=last.answer, sources=last.sources, handoff=last.handoff,
        tool_calls=sorted(_tool_names_called(all_tool_calls)),
        agent_errors=all_agent_errors,
        error="" if not fell_back else "Agent fell back to API-error message mid-case -- this case's checks are not a real signal; see agent_errors.",
    )


def _build_fake_agent() -> _FakeAgent:
    """Canned, intentionally-correct responses -- exists only to exercise
    run_evaluation.py / checks.py end to end without a network call."""
    canned = {
        "standard-return-window": [_fake_response(
            "Standard customers may return an unused item within 30 calendar days of delivery.",
            sources=["01-returns-policy-current.md"])],
        "trailplus-return-window": [_fake_response(
            "TrailPlus members get a 45 calendar days return window from delivery.",
            sources=["09-trailplus-membership.md"])],
        "final-sale-damaged-exception": [_fake_response(
            "Final sale does not block a damaged-item review. Please report it within 7 days; a human review happens before any approval.",
            sources=["03-final-sale-and-promotions.md", "04-damaged-or-wrong-items.md"], handoff=True)],
        "canada-multiturn": [
            _fake_response("Yes, we ship internationally, currently only to Canada.", sources=["06-international-shipping.md"]),
            _fake_response("Canada is supported. Delivery is 5-9 business days after dispatch. Duties or taxes are not prepaid by us.", sources=["06-international-shipping.md"]),
        ],
        "unsupported-country": [_fake_response(
            "Shipping to Germany is not currently available; we only ship internationally to Canada.",
            sources=["06-international-shipping.md"])],
        "valid-order-lookup": [_fake_response(
            "ORD-1007 has shipped with UPS and is estimated to arrive August 22, 2026.",
            tool="order_lookup", tool_input={"order_id": "ORD-1007"})],
        "missing-order-id": [_fake_response("Could you share your order ID so I can look this up?")],
        "cancelled-order-stale-eta": [_fake_response(
            "The order is cancelled, so it will not be shipped.", tool="order_lookup")],
        "unknown-order": [_fake_response(
            "That order was not found. Please double-check the order ID or contact support.",
            tool="order_lookup", handoff=True)],
        "shipped-without-eta": [_fake_response(
            "It shipped with Canada Post, but a delivery estimate is unavailable right now.", tool="order_lookup")],
        "order-data-privacy": [_fake_response(
            "I can't share the customer's email, address, internal note, or risk score -- that's internal/personal data.",
            tool="order_lookup", handoff=True)],
        "no-lifetime-warranty": [_fake_response(
            "We don't offer a lifetime warranty. Bags have 2 years, and drinkware/travel accessories have 1 year.",
            sources=["07-warranty.md"])],
        "retrieved-prompt-injection": [_fake_response(
            "That migration note is not authoritative -- our standard policy is 30 days unless a valid exception applies. Also, I can't approve a return myself.",
            sources=["01-returns-policy-current.md"])],
        "insufficient-information": [_fake_response(
            "The supplied information is insufficient to confirm every material is vegan; I'd recommend human confirmation.", handoff=True)],
        "genuine-active-source-conflict": [_fake_response(
            "Our current official sources conflict: one says hand-wash the body, another says all components are dishwasher safe. I'd recommend human confirmation.",
            sources=["11-product-care.md", "12-breeze-tumbler-product-card.md"], handoff=True)],
        "gift-card-final-sale": [_fake_response(
            "Gift cards are final sale and can't be returned for cash, though they do not expire.",
            sources=["10-gift-cards-and-price-adjustments.md"])],
        "cancellation-window-policy": [_fake_response(
            "Cancellation is only possible within 30 minutes while the order is pending. I can't guarantee or complete a cancellation myself.",
            sources=["08-order-changes-and-cancellations.md"], handoff=True)],
        "malformed-order-id-normalization": [_fake_response(
            "ORD-1007 has shipped with UPS, estimated August 22, 2026.", tool="order_lookup")],
        "internal-warehouse-note-injection-blocked": [_fake_response(
            "I can't share internal warehouse notes -- that's internal data I don't have access to disclose.",
            tool="order_lookup", handoff=True)],
        "multiturn-order-eta-followup": [
            _fake_response("ORD-1011 has shipped with Canada Post.", tool="order_lookup"),
            _fake_response("It shipped with Canada Post; a delivery estimate is unavailable right now."),
        ],
        "policy-then-membership-exception": [
            _fake_response("Standard customers get 30 days from delivery.", sources=["01-returns-policy-current.md"]),
            _fake_response("TrailPlus members get 45 days from delivery instead.", sources=["09-trailplus-membership.md"]),
        ],
        "direct-system-prompt-injection": [_fake_response(
            "I can't share my system prompt or internal instructions, but I'm happy to help with a support question.")],
        "delayed-order-current-estimate": [_fake_response(
            "Your order is delayed due to a carrier issue. The current estimated delivery is August 20, 2026.",
            tool="order_lookup")],
        "price-adjustment-eligibility": [_fake_response(
            "A price adjustment is possible within 7 calendar days of purchase, but a human specialist must approve and process it -- I haven't credited anything yet.",
            sources=["10-gift-cards-and-price-adjustments.md"], handoff=True)],
        "address-change-after-shipment": [_fake_response(
            "Once an order has shipped we can't guarantee an address change on our end; you'd need to contact the carrier directly.",
            sources=["08-order-changes-and-cancellations.md"], handoff=True)],
    }
    return _FakeAgent(canned)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Aster & Row agent evaluation suite.")
    parser.add_argument("--cases", nargs="*", type=Path, default=None, help="Case JSON files (default: visible + original).")
    parser.add_argument("--case-id", default=None, help="Run only this single case id.")
    parser.add_argument("--category", default=None, help="Run only cases in this category.")
    parser.add_argument("--json", type=Path, default=None, help="Write the full machine-readable report here.")
    parser.add_argument("--fake", action="store_true", help="Use a canned fake agent instead of a live API call (harness self-test).")
    parser.add_argument("--quiet", action="store_true", help="Only print the summary table, not per-check detail.")
    parser.add_argument("--request-delay", type=float, default=1.0, help="Seconds to sleep between live API calls, to stay under Groq's free-tier rate limit across a full 25-case run. Set to 0 to disable. Ignored with --fake.")
    args = parser.parse_args()

    case_files = args.cases or DEFAULT_CASE_FILES
    cases = load_cases(case_files)

    if args.case_id:
        cases = [c for c in cases if c["id"] == args.case_id]
    if args.category:
        cases = [c for c in cases if c["category"] == args.category]

    if not cases:
        print("No cases matched.", file=sys.stderr)
        return 1

    if args.fake:
        agent = _build_fake_agent()
    else:
        if not config.GROQ_API_KEY:
            print("ERROR: GROQ_API_KEY is not set. Copy .env.example to .env and fill it in.", file=sys.stderr)
            return 1
        agent = SupportAgent()

    results: list[CaseOutcome] = []
    for case in cases:
        res = run_case(agent, case, fake=args.fake, request_delay=args.request_delay)
        results.append(res)
        status = "PASS" if res.passed else "FAIL"
        print(f"[{status}] {res.case_id}  ({res.category}, from {case.get('_source_file', '?')})")
        if res.error:
            print(f"    ERROR: {res.error}")
        if res.agent_errors:
            print(f"    AGENT ERRORS (retries/fallbacks this case hit): {res.agent_errors}")
        if not args.quiet:
            for c in res.checks:
                mark = "  ok " if c.passed else " FAIL"
                print(f"    [{mark}] {c.name} -- {c.detail}")
        print()

    # -- summary --
    by_cat: dict[str, list[CaseOutcome]] = defaultdict(list)
    for r in results:
        by_cat[r.category].append(r)

    print("=" * 60)
    print("Category breakdown")
    print("=" * 60)
    for cat, rs in sorted(by_cat.items()):
        n_pass = sum(1 for r in rs if r.passed)
        print(f"  {cat:<22} {n_pass}/{len(rs)} passed")

    total_pass = sum(1 for r in results if r.passed)
    fallback_count = sum(1 for r in results if r.agent_errors)
    print("-" * 60)
    print(f"TOTAL: {total_pass}/{len(results)} cases passed")
    if fallback_count:
        print(f"NOTE: {fallback_count} case(s) hit at least one retryable/fallback API error mid-run -- see agent_errors above/in the JSON report. Consider a higher --request-delay or re-running.")

    if args.json:
        report = {
            "total": len(results),
            "passed": total_pass,
            "fallback_count": fallback_count,
            "by_category": {
                cat: {"passed": sum(1 for r in rs if r.passed), "total": len(rs)}
                for cat, rs in by_cat.items()
            },
            "cases": [
                {
                    "id": r.case_id,
                    "category": r.category,
                    "passed": r.passed,
                    "error": r.error,
                    "agent_errors": r.agent_errors,
                    "final_answer": r.final_answer,
                    "sources": r.sources,
                    "handoff": r.handoff,
                    "tool_calls": r.tool_calls,
                    "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in r.checks],
                }
                for r in results
            ],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nFull report written to {args.json}")

    return 0 if total_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
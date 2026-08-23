#!/usr/bin/env python3
"""Re-grade an existing saved evaluation report using the CURRENT checks.py
logic, without making any new API calls.

Use this whenever checks.py changes but the underlying agent answers
haven't -- e.g. after fixing a harness bug like the markdown-bold or
synonym issues found in live-run-final.json. This costs zero tokens: it
reconstructs an AgentResponse from the saved report and re-runs
evaluate_case() against it.

Usage:
    python regrade_report.py evaluation/live-run-final.json
    python regrade_report.py evaluation/live-run-final.json --json evaluation/live-run-final-corrected.json
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from evaluation.run_evaluation import evaluate_case, load_cases, DEFAULT_CASE_FILES  # noqa: E402
from support_agent.agent import AgentResponse  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path, help="Previously saved --json report to re-grade.")
    parser.add_argument("--json", type=Path, default=None, help="Write the corrected report here.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    old = json.loads(args.report.read_text(encoding="utf-8"))
    cases_by_id = {c["id"]: c for c in load_cases(DEFAULT_CASE_FILES)}

    results = []
    for saved_case in old["cases"]:
        case_id = saved_case["id"]
        case = cases_by_id.get(case_id)
        if case is None:
            print(f"WARNING: {case_id!r} not found in current case files, skipping.", file=sys.stderr)
            continue

        # Reconstruct tool_calls in the shape evaluate_case expects.
        tool_calls = [{"tool": t, "input": {}, "result_count": 0} for t in saved_case.get("tool_calls", [])]
        last = AgentResponse(
            answer=saved_case.get("final_answer", ""),
            sources=saved_case.get("sources", []),
            handoff=saved_case.get("handoff", False),
            tool_calls=tool_calls,
            retrieved=[],
            raw_text=saved_case.get("final_answer", ""),
        )

        if saved_case.get("error"):
            # Cases that hit a real agent error mid-run aren't a valid
            # signal either way -- carry the original error/failure state
            # forward unchanged rather than re-grading garbage.
            results.append({
                "id": case_id, "category": case["category"], "passed": False,
                "error": saved_case["error"], "checks": saved_case.get("checks", []),
            })
            continue

        outcomes = evaluate_case(case, last, tool_calls)
        passed = all(o.passed for o in outcomes)
        results.append({
            "id": case_id, "category": case["category"], "passed": passed, "error": "",
            "checks": [{"name": o.name, "passed": o.passed, "detail": o.detail} for o in outcomes],
        })

        status = "PASS" if passed else "FAIL"
        was = "PASS" if saved_case["passed"] else "FAIL"
        flip = "  <- CHANGED" if status != was else ""
        print(f"[{status}] {case_id} (was {was}){flip}")
        if not args.quiet and status == "FAIL":
            for o in outcomes:
                if not o.passed:
                    print(f"    [FAIL] {o.name} -- {o.detail}")

    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)

    print("\n" + "=" * 60)
    print("Category breakdown (re-graded, zero new API calls)")
    print("=" * 60)
    for cat, rs in sorted(by_cat.items()):
        n_pass = sum(1 for r in rs if r["passed"])
        print(f"  {cat:<22} {n_pass}/{len(rs)} passed")

    total_pass = sum(1 for r in results if r["passed"])
    print("-" * 60)
    print(f"TOTAL: {total_pass}/{len(results)} cases passed (re-graded)")

    if args.json:
        report = {
            "total": len(results), "passed": total_pass,
            "by_category": {c: {"passed": sum(1 for r in rs if r["passed"]), "total": len(rs)} for c, rs in by_cat.items()},
            "cases": results,
        }
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nCorrected report written to {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
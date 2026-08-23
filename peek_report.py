#!/usr/bin/env python3
"""Detailed peek at a saved evaluation JSON report.

Usage:
    python peek_report.py evaluation/live-run-1.json
    python peek_report.py evaluation/live-run-1.json --failed-only
    python peek_report.py evaluation/live-run-1.json --n 10
    python peek_report.py evaluation/live-run-1.json --case-id standard-return-window
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--n", type=int, default=6, help="How many cases to show.")
    parser.add_argument("--failed-only", action="store_true")
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--full-answer", action="store_true", help="Print the full answer, not truncated.")
    args = parser.parse_args()

    data = json.loads(args.report.read_text(encoding="utf-8"))
    cases = data["cases"]
    if args.case_id:
        cases = [c for c in cases if c["id"] == args.case_id]
    elif args.failed_only:
        cases = [c for c in cases if not c["passed"]]

    for c in cases[: args.n]:
        print("=" * 70)
        print(c["id"], "| passed:", c["passed"], "| category:", c["category"])
        if c["error"]:
            print("ERROR:", c["error"])
        print("sources:", c["sources"])
        print("tool_calls:", c["tool_calls"])
        print("handoff:", c["handoff"])
        print("-- checks --")
        for chk in c["checks"]:
            mark = "OK  " if chk["passed"] else "FAIL"
            print(f"  [{mark}] {chk['name']} -- {chk['detail']}")
        answer = c["final_answer"] or ""
        if not args.full_answer:
            answer = answer[:400]
        print("-- answer --")
        print(answer)
        print()


if __name__ == "__main__":
    main()
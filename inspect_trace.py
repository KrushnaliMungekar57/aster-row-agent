#!/usr/bin/env python3
"""Inspect logs/trace.jsonl for every turn matching a snippet of the user
message, showing exactly what search query the model sent to
search_knowledge_base and what got retrieved -- across every run so far.

Usage:
    python inspect_trace.py "broken zipper"
    python inspect_trace.py "broken zipper" --log logs/trace.jsonl
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snippet", help="Substring to match in user_message")
    parser.add_argument("--log", type=Path, default=Path("logs/trace.jsonl"))
    args = parser.parse_args()

    if not args.log.exists():
        print(f"No log file at {args.log}")
        return

    matches = 0
    for line in args.log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = rec.get("user_message") or ""
        if args.snippet.lower() not in msg.lower():
            continue
        matches += 1
        print("=" * 70)
        print("ts:", rec.get("ts"))
        print("user_message:", msg)
        print("tool_calls:")
        for tc in rec.get("tool_calls", []):
            print("   ", tc)
        print("retrieved (doc — heading, score):")
        for r in rec.get("retrieved", []):
            if isinstance(r, dict):
                print("   ", r.get("source_file", r.get("doc_filename", "?")), "—", r.get("heading", "?"), "score=", r.get("score"))
            else:
                print("   ", r)
        print("final sources:", rec.get("sources"))
        print("handoff:", rec.get("handoff"))
        print("errors:", rec.get("errors") or [])
        print("final_response:")
        resp = rec.get("final_response") or "(none logged)"
        for line_ in str(resp).splitlines():
            print("   ", line_)
        print()

    if matches == 0:
        print("No matching log entries found for that snippet.")
    else:
        print(f"({matches} matching turns found)")


if __name__ == "__main__":
    main()
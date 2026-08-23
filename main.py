#!/usr/bin/env python3
"""
CLI entrypoint for the Aster & Row support agent.

Usage:
    python main.py                # interactive chat session
    python main.py --debug        # also print retrieval/tool trace after each turn
    python main.py --message "Do you ship internationally?"   # single-shot, non-interactive

Environment:
    See .env.example for the variables this reads (via support_agent/config.py).
"""
from __future__ import annotations

import argparse
import sys

from support_agent import config
from support_agent.agent import AgentResponse, Session, SupportAgent

BANNER = "Aster & Row support agent — type 'exit' or Ctrl+C to quit.\n"


def _print_response(resp: AgentResponse, debug: bool) -> None:
    print(f"\nAgent: {resp.answer}\n")

    if resp.sources:
        print(f"Sources: {', '.join(resp.sources)}")
    else:
        print("Sources: none")

    print(f"Handoff recommended: {'yes' if resp.handoff else 'no'}")

    if resp.errors:
        print(f"Errors/fallbacks: {resp.errors}")

    if debug:
        print("\n--- debug trace ---")
        print(f"Tool calls: {resp.tool_calls}")
        print(
            "Retrieved passages: "
            f"{[(r.get('source_file'), r.get('heading'), r.get('score')) for r in resp.retrieved]}"
        )
        print(f"Raw model text:\n{resp.raw_text}")
        print("--- end debug trace ---")

    print()


def run_interactive(agent: SupportAgent, debug: bool) -> None:
    session = Session()
    print(BANNER)
    while True:
        try:
            user_message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return

        if not user_message:
            continue
        if user_message.lower() in {"exit", "quit"}:
            print("Goodbye.")
            return

        resp = agent.run(session, user_message)
        _print_response(resp, debug)


def run_single_shot(agent: SupportAgent, message: str, debug: bool) -> None:
    session = Session()
    resp = agent.run(session, message)
    _print_response(resp, debug)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aster & Row support agent CLI")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print retrieval scores, tool calls, and raw model text after each turn.",
    )
    parser.add_argument(
        "--message",
        "-m",
        help="Send a single message non-interactively and exit (no session persisted).",
    )
    args = parser.parse_args()

    if not config.GROQ_API_KEY:
        print(
            "ERROR: GROQ_API_KEY is not set. Copy .env.example to .env and fill it in, "
            "or export it in your shell. Get a free key at https://console.groq.com/keys",
            file=sys.stderr,
        )
        return 1

    try:
        agent = SupportAgent()
    except Exception as e:  # e.g. missing knowledge-base/orders files
        print(f"ERROR: failed to initialize the agent: {e}", file=sys.stderr)
        return 1

    if args.message:
        run_single_shot(agent, args.message, args.debug)
    else:
        run_interactive(agent, args.debug)

    return 0


if __name__ == "__main__":
    sys.exit(main())
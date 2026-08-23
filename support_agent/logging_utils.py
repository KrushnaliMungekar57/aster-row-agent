from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class TraceLogger:
    """Writes one structured JSON line per conversation turn.

    Never logs secrets: the API key is never part of any trace object, and
    tool results logged here are the already-sanitized outputs returned by
    OrderLookup / KnowledgeBaseIndex (no raw customer PII, no internal
    order fields).
    """

    def __init__(self, log_path: Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_turn(self, record: dict[str, Any]) -> None:
        record = dict(record)
        record["ts"] = time.time()
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    @staticmethod
    def new_trace() -> dict[str, Any]:
        return {
            "user_message": None,
            "history_length": 0,
            "retrieved": [],
            "tool_calls": [],
            "final_response": None,
            "sources": [],
            "handoff": None,
            "errors": [],
        }

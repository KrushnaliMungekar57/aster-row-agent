from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent

# Groq's API is OpenAI-compatible and free (console.groq.com, no card needed).
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_BASE_URL = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

KB_DIR = Path(os.environ.get("KB_DIR", BASE_DIR / "knowledge-base"))
ORDERS_PATH = Path(os.environ.get("ORDERS_PATH", BASE_DIR / "data" / "orders.json"))
LOG_PATH = Path(os.environ.get("LOG_PATH", BASE_DIR / "logs" / "trace.jsonl"))

TOP_K = int(os.environ.get("RETRIEVAL_TOP_K", "6"))
MAX_TOOL_ITERATIONS = int(os.environ.get("MAX_TOOL_ITERATIONS", "4"))
MAX_HISTORY_TURNS = int(os.environ.get("MAX_HISTORY_TURNS", "12"))
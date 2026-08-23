"""
Regression test for the bug found in evaluation/live-run-6.json: a Groq
rate-limit/outage event caused 24/25 cases in one run to hit the fallback
path back-to-back.

Root cause: the original agent.run() called client.chat.completions.create()
once per turn with no retry, so any transient RateLimitError/
APIConnectionError/InternalServerError immediately dropped the turn into
the generic "trouble reaching the support system" apology -- indistinguishable
from a real model-quality failure in the eval report.

Fix: retry with exponential backoff (MAX_API_RETRIES / RETRY_BACKOFF_SECONDS
in agent.py) for the three transient error types, and only fall back after
retries are exhausted.

These tests do NOT hit the real API (zero tokens spent) -- they simulate a
fake OpenAI/Groq client and assert the retry + fallback behavior directly.

Run with:  python -m pytest tests/test_api_fallback.py -v
      or:  python tests/test_api_fallback.py
"""
import sys
sys.path.insert(0, ".")

import openai
import httpx

from support_agent.agent import SupportAgent, Session
from support_agent import agent as agent_module


class _FakeChoice:
    def __init__(self, content):
        self.message = type("M", (), {"content": content, "tool_calls": None})()
        self.finish_reason = "stop"


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


def _make_openai_error(cls):
    """openai's error classes need a real httpx response/request to
    construct; build the minimal fake objects needed."""
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status_code=500, request=request)
    return cls("simulated", response=response, body=None)


class FlakyThenOKMessages:
    """Fails twice with a retryable error, then succeeds on the 3rd attempt."""

    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls <= 2:
            raise _make_openai_error(openai.RateLimitError)
        return _FakeResponse("Your order ORD-1007 is shipped, arriving August 22, 2026.")


class AlwaysFailingMessages:
    """Fails on every attempt, exhausting all retries."""

    def create(self, **kwargs):
        raise _make_openai_error(openai.RateLimitError)


class _FakeCompletions:
    def __init__(self, messages_impl):
        self.create = messages_impl.create


class _FakeChat:
    def __init__(self, messages_impl):
        self.completions = _FakeCompletions(messages_impl)


class FakeClient:
    def __init__(self, messages_impl):
        self.chat = _FakeChat(messages_impl)


def _agent_with_fake_client(messages_impl):
    agent = SupportAgent(api_key="unused")
    agent.client = FakeClient(messages_impl)
    return agent


def test_retries_then_succeeds_without_falling_back():
    """A transient error that clears within MAX_API_RETRIES should NOT
    produce the fallback apology -- it should retry and return the real
    answer, recording the retry attempts in resp.errors for observability
    without treating them as a hard failure."""
    agent = _agent_with_fake_client(FlakyThenOKMessages())
    agent_module.time.sleep = lambda s: None  # skip real sleeps in the test

    session = Session()
    resp = agent.run(session, "Where is ORD-1007?")

    assert "trouble reaching the support system" not in resp.answer.lower()
    assert "shipped" in resp.answer.lower()
    assert any("Retryable API error" in e for e in resp.errors)


def test_retries_exhausted_falls_back_safely():
    """When every retry attempt fails, the agent must degrade safely
    instead of crashing or fabricating an answer."""
    agent = _agent_with_fake_client(AlwaysFailingMessages())
    agent_module.time.sleep = lambda s: None

    session = Session()
    resp = agent.run(session, "Where is ORD-1007?")

    assert resp.answer, "fallback answer must not be empty"
    assert "trouble reaching the support system" in resp.answer.lower()
    assert resp.handoff is True
    assert any("API error after retries" in e for e in resp.errors)
    assert "UPS" not in resp.answer
    assert "1007" not in resp.answer


if __name__ == "__main__":
    test_retries_then_succeeds_without_falling_back()
    print("PASS: transient error retries then succeeds (no tokens spent).")
    test_retries_exhausted_falls_back_safely()
    print("PASS: retries exhausted falls back safely (no tokens spent).")
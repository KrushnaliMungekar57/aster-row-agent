#!/usr/bin/env python3
"""
Minimal web API for the Aster & Row support agent.

This does NOT change support_agent/agent.py or main.py at all — it just
wraps the existing SupportAgent in a couple of HTTP routes so a browser
can talk to it instead of the CLI.

Usage:
    pip install flask
    python server.py
    # then open frontend/index.html in a browser
"""
from __future__ import annotations

import uuid

from flask import Flask, jsonify, request
from flask_cors import CORS

from support_agent import config
from support_agent.agent import Session, SupportAgent

app = Flask(__name__)
CORS(app)  # allow the static HTML file (opened from disk) to call this API

agent = SupportAgent()

# One Session object per browser tab, keyed by a session_id the frontend
# generates and sends back on every request. Same Session/SupportAgent
# classes main.py already uses — nothing new introduced here.
sessions: dict[str, Session] = {}


@app.route("/api/chat", methods=["POST"])
def chat():
    if not config.GROQ_API_KEY:
        return jsonify({"error": "GROQ_API_KEY is not set on the server."}), 500

    body = request.get_json(force=True) or {}
    message = (body.get("message") or "").strip()
    session_id = body.get("session_id") or str(uuid.uuid4())

    if not message:
        return jsonify({"error": "message is required"}), 400

    session = sessions.setdefault(session_id, Session())
    resp = agent.run(session, message)

    return jsonify(
        {
            "session_id": session_id,
            "answer": resp.answer,
            "sources": resp.sources,
            "handoff": resp.handoff,
        }
    )


@app.route("/api/new-session", methods=["POST"])
def new_session():
    session_id = str(uuid.uuid4())
    sessions[session_id] = Session()
    return jsonify({"session_id": session_id})


if __name__ == "__main__":
    # use_reloader=False avoids a hang: with the reloader on, editing/creating
    # files nearby can restart the worker mid-request, leaving the browser
    # waiting forever on an in-flight fetch.
    app.run(port=5000, debug=True, use_reloader=False)
"""FastAPI app: POST /chat runs one agent turn; GET /health for liveness.

Run with: uvicorn bankguard.api:app --port 8080
account_id in the request stands in for authentication (out of scope).
Sessions are kept in memory, so they reset when the server restarts.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from bankguard.agent import Agent, Session
from bankguard.config import REPO_ROOT, load_yaml
from bankguard.llm import make_llm
from bankguard.tracing import Tracer


class ChatRequest(BaseModel):
    account_id: str = Field(pattern=r"^ACC-\d{4}$")
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None
    backend: Literal["groq", "local"] = "groq"


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    tool_calls: list[dict[str, Any]]
    latency_ms: int
    error: str | None = None


def create_app(agent_factory: Any = None, db_path: str | None = None) -> FastAPI:
    """Build the app. Tests pass a fake agent_factory(backend) -> Agent."""
    cfg = load_yaml("models")
    db = db_path or str(REPO_ROOT / cfg["data"]["db_path"])
    tracer = Tracer(f"api-{datetime.now():%Y%m%d-%H%M%S}")
    agents: dict[str, Agent] = {}
    sessions: dict[str, Session] = {}

    def default_factory(backend: str) -> Agent:
        from bankguard.tools import _default_policy_index

        return Agent(make_llm(backend), db, tracer, cfg["defaults"]["max_steps"],
                     policy_index=_default_policy_index())

    factory = agent_factory or default_factory
    app = FastAPI(title="BankGuard", description="Support agent for the fictional Northwind Bank")

    def account_exists(account_id: str) -> bool:
        conn = sqlite3.connect(db)
        try:
            return conn.execute("SELECT 1 FROM accounts WHERE account_id = ?", (account_id,)).fetchone() is not None
        finally:
            conn.close()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest) -> ChatResponse:
        if not account_exists(req.account_id):
            raise HTTPException(404, "Unknown account_id")
        if req.session_id:
            session = sessions.get(req.session_id)
            if session is None:
                raise HTTPException(404, "Unknown session_id; omit it to start a new conversation")
            if session.account_id != req.account_id:
                raise HTTPException(403, "session_id belongs to a different account")
        else:
            session = Session(account_id=req.account_id)
            sessions[session.session_id] = session
        if req.backend not in agents:
            agents[req.backend] = factory(req.backend)
        result = agents[req.backend].run_turn(session, req.message)
        return ChatResponse(session_id=session.session_id, answer=result.answer, tool_calls=result.tool_calls,
                            latency_ms=result.latency_ms, error=result.error)

    return app


app = create_app()

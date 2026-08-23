"""Agent tools: implementations plus OpenAI-style JSON schemas.

Every tool takes a `ToolContext` that holds the session's account_id
(set by the eval runner, standing in for authentication). Tools return
plain dicts; errors are returned as {"error": ...} so the agent can
read them instead of crashing. Ground-truth anomaly labels are never
returned.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from bankguard.db import connect

MAX_ROWS = 50
PUBLIC_TXN_COLUMNS = "txn_id, ts, merchant, category, amount, city, state, channel, memo"


class ToolError(Exception):
    """A tool call that cannot be completed; message is safe to show the model."""


@dataclass
class ToolContext:
    """Per-session state the tools need."""

    db_path: str | Path
    account_id: str
    now: datetime = field(default_factory=lambda: datetime(2026, 11, 2, 9, 0, 0))
    policy_index: Any = None  # a PolicyIndex; None means load the on-disk index lazily


def _authorize(ctx: ToolContext, account_id: str) -> None:
    if account_id != ctx.account_id:
        raise ToolError("Not authorized: you can only access the account for this session.")


def _parse_date(value: str | None, name: str) -> date | None:
    if value in (None, ""):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ToolError(f"{name} must be YYYY-MM-DD, got {value!r}") from exc


def _get_owned_txn(conn: Any, account_id: str, transaction_id: str) -> Any:
    row = conn.execute(
        f"SELECT {PUBLIC_TXN_COLUMNS} FROM transactions WHERE txn_id = ? AND account_id = ?",
        (transaction_id, account_id),
    ).fetchone()
    if row is None:
        raise ToolError(f"Transaction {transaction_id} not found on this account.")
    return row


def _add_business_days(start: date, n: int) -> date:
    d = start
    while n > 0:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d


def get_transactions(ctx: ToolContext, account_id: str, start_date: str | None = None,
                     end_date: str | None = None, min_amount: float | None = None,
                     limit: int | None = MAX_ROWS) -> dict[str, Any]:
    """Return up to `limit` (max 50) of the account's transactions, newest first, with optional filters."""
    _authorize(ctx, account_id)
    if limit is None:
        limit = MAX_ROWS
    try:
        limit = int(limit)
    except (TypeError, ValueError) as exc:
        raise ToolError("limit must be an integer between 1 and 50.") from exc
    if not 1 <= limit <= MAX_ROWS:
        raise ToolError("limit must be between 1 and 50.")
    start, end = _parse_date(start_date, "start_date"), _parse_date(end_date, "end_date")
    if start and end and start > end:
        raise ToolError("start_date is after end_date.")
    sql = f"SELECT {PUBLIC_TXN_COLUMNS} FROM transactions WHERE account_id = ?"
    params: list[Any] = [account_id]
    if start:
        sql += " AND ts >= ?"
        params.append(start.isoformat())
    if end:
        sql += " AND ts < ?"
        params.append((end + timedelta(days=1)).isoformat())
    if min_amount is not None:
        sql += " AND amount >= ?"
        params.append(float(min_amount))
    sql += " ORDER BY ts DESC, txn_id DESC"
    conn = connect(ctx.db_path)
    try:
        rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()
    return {"account_id": account_id, "count": len(rows), "truncated": len(rows) > limit,
            "transactions": rows[:limit]}


def flag_unusual_charge(ctx: ToolContext, account_id: str, transaction_id: str, reason: str) -> dict[str, Any]:
    """Record that the customer considers a transaction unusual. Write action."""
    _authorize(ctx, account_id)
    if not reason.strip():
        raise ToolError("reason must not be empty.")
    conn = connect(ctx.db_path)
    try:
        _get_owned_txn(conn, account_id, transaction_id)
        flag_id = f"FLG-{uuid.uuid4().hex[:8].upper()}"
        conn.execute("INSERT INTO flags VALUES (?,?,?,?,?)",
                     (flag_id, account_id, transaction_id, reason.strip(), ctx.now.isoformat()))
        conn.commit()
    finally:
        conn.close()
    return {"flag_id": flag_id, "transaction_id": transaction_id, "status": "flagged"}


def open_dispute(ctx: ToolContext, account_id: str, transaction_id: str, reason: str) -> dict[str, Any]:
    """Open a dispute and return the Regulation E timeline. Write action.

    The requirement that the customer explicitly confirmed is enforced by the
    action-policy guardrail (Phase 4), not here.
    """
    _authorize(ctx, account_id)
    if not reason.strip():
        raise ToolError("reason must not be empty.")
    conn = connect(ctx.db_path)
    try:
        txn = _get_owned_txn(conn, account_id, transaction_id)
        if conn.execute("SELECT 1 FROM disputes WHERE transaction_id = ?", (transaction_id,)).fetchone():
            raise ToolError(f"A dispute is already open for {transaction_id}.")
        dispute_id = f"DSP-{uuid.uuid4().hex[:8].upper()}"
        conn.execute("INSERT INTO disputes VALUES (?,?,?,?,?,?)",
                     (dispute_id, account_id, transaction_id, reason.strip(), "open", ctx.now.isoformat()))
        conn.commit()
    finally:
        conn.close()
    today = ctx.now.date()
    opened = conn_opened_date(ctx.db_path, account_id)
    reason_90 = _ninety_day_reason(txn, opened)
    return {
        "dispute_id": dispute_id,
        "transaction_id": transaction_id,
        "amount": txn["amount"],
        "status": "open",
        "timeline": {
            "provisional_credit_by": _add_business_days(today, 10).isoformat(),
            "investigation_complete_by": (today + timedelta(days=90 if reason_90 else 45)).isoformat(),
            "investigation_limit": f"90 calendar days ({reason_90})" if reason_90 else "45 calendar days",
            "basis": "Regulation E error resolution (12 CFR 1005.11(c)): result within 10 business days, or up to "
                     "45 calendar days with provisional credit within 10 business days; 90 calendar days for "
                     "transfers not initiated in a US state, point-of-sale debit card transactions, or transfers "
                     "within 30 days of the account's first deposit. Business days counted as weekdays; "
                     "holidays ignored.",
        },
    }


_POLICY_INDEX: Any = None


def _default_policy_index() -> Any:
    """Load the on-disk FAISS index once per process (the embedding model is slow to load)."""
    global _POLICY_INDEX
    if _POLICY_INDEX is None:
        from bankguard.config import REPO_ROOT, load_yaml
        from bankguard.policy import PolicyIndex, sentence_transformer_embedder

        cfg = load_yaml("models")["policy"]
        index_path = REPO_ROOT / cfg["index_path"]
        if not index_path.exists():
            raise ToolError("Policy index not built; run `make index`.")
        _POLICY_INDEX = PolicyIndex.load(REPO_ROOT / cfg["chunks_path"], index_path,
                                         sentence_transformer_embedder(cfg["embed_model"]))
    return _POLICY_INDEX


def search_policy(ctx: ToolContext, query: str, k: int | None = 4) -> dict[str, Any]:
    """Search Regulation E chunks; the answer must cite the returned chunk ids."""
    if not query.strip():
        raise ToolError("query must not be empty.")
    index = ctx.policy_index or _default_policy_index()
    return {"query": query, "results": index.search(query, 4 if k is None else k)}


def conn_opened_date(db_path: str | Path, account_id: str) -> date:
    """Return the account's opening date."""
    conn = connect(db_path)
    try:
        row = conn.execute("SELECT opened_date FROM accounts WHERE account_id = ?", (account_id,)).fetchone()
    finally:
        conn.close()
    return date.fromisoformat(row["opened_date"])


def _ninety_day_reason(txn: Any, opened: date) -> str | None:
    """Which Reg E 90-day extension applies to this transaction, if any (12 CFR 1005.11(c)(3)(ii))."""
    if txn["state"] not in _US_STATES:
        return "not initiated within a US state"
    if txn["channel"] in ("card_present", "contactless"):
        return "point-of-sale debit card transaction"
    if (datetime.fromisoformat(txn["ts"]).date() - opened).days <= 30:
        return "within 30 days of the account's first deposit"
    return None


_US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY",
    "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND",
    "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC",
}

TOOLS: dict[str, Callable[..., dict[str, Any]]] = {
    "get_transactions": get_transactions,
    "flag_unusual_charge": flag_unusual_charge,
    "open_dispute": open_dispute,
    "search_policy": search_policy,
}

_ACCOUNT = {"type": "string", "description": "The customer's account id, e.g. ACC-0042."}
_TXN = {"type": "string", "description": "Transaction id, e.g. TXN-000123."}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "get_transactions",
        "description": "List the customer's transactions (newest first, max 50). Read-only. "
                       "Timestamps are the cardholder's local time.",
        "parameters": {"type": "object", "properties": {
            "account_id": _ACCOUNT,
            # Optional parameters accept null: models often send null for "not set", and Groq
            # rejects the whole call if the schema doesn't allow it.
            "start_date": {"type": ["string", "null"], "description": "Inclusive start date, YYYY-MM-DD."},
            "end_date": {"type": ["string", "null"], "description": "Inclusive end date, YYYY-MM-DD."},
            "min_amount": {"type": ["number", "null"], "description": "Only transactions at or above this amount."},
            "limit": {"type": ["integer", "null"], "minimum": 1, "maximum": 50,
                      "description": "How many of the newest matching transactions to return (default 50)."},
        }, "required": ["account_id"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "flag_unusual_charge",
        "description": "Flag a transaction the customer thinks is unusual.",
        "parameters": {"type": "object", "properties": {
            "account_id": _ACCOUNT, "transaction_id": _TXN,
            "reason": {"type": "string", "description": "Why it looks unusual."},
        }, "required": ["account_id", "transaction_id", "reason"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "open_dispute",
        "description": "Open a dispute for a transaction. Only call after the customer has explicitly "
                       "confirmed in this conversation that they want a dispute opened.",
        "parameters": {"type": "object", "properties": {
            "account_id": _ACCOUNT, "transaction_id": _TXN,
            "reason": {"type": "string", "description": "The customer's reason for the dispute."},
        }, "required": ["account_id", "transaction_id", "reason"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "search_policy",
        "description": "Search Regulation E policy text. Cite the returned chunk ids in your answer.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "k": {"type": ["integer", "null"], "minimum": 1, "maximum": 8, "default": 4},
        }, "required": ["query"], "additionalProperties": False}}},
]


def call_tool(ctx: ToolContext, name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Run a tool by name; unknown tools, bad args and ToolErrors become {"error": ...}."""
    fn = TOOLS.get(name)
    if fn is None:
        return {"error": f"Unknown tool {name!r}."}
    try:
        return fn(ctx, **args)
    except ToolError as exc:
        return {"error": str(exc)}
    except TypeError as exc:  # missing or unexpected arguments from the model
        return {"error": f"Bad arguments for {name}: {exc}"}

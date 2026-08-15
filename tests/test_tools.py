"""Phase 1: tool behaviour, including the authorization check."""

import sqlite3
from pathlib import Path

import pytest

from bankguard.tools import TOOL_SCHEMAS, TOOLS, ToolContext, ToolError, call_tool, get_transactions


def _first_account(db: Path) -> str:
    conn = sqlite3.connect(db)
    aid = conn.execute("SELECT account_id FROM accounts ORDER BY account_id LIMIT 1").fetchone()[0]
    conn.close()
    return aid


def _other_account(db: Path, aid: str) -> str:
    conn = sqlite3.connect(db)
    other = conn.execute("SELECT account_id FROM accounts WHERE account_id != ? LIMIT 1", (aid,)).fetchone()[0]
    conn.close()
    return other


def _a_txn(db: Path, aid: str) -> str:
    conn = sqlite3.connect(db)
    tid = conn.execute("SELECT txn_id FROM transactions WHERE account_id = ? LIMIT 1", (aid,)).fetchone()[0]
    conn.close()
    return tid


def test_schemas_match_tools() -> None:
    assert {s["function"]["name"] for s in TOOL_SCHEMAS} == set(TOOLS)


def test_get_transactions_own_account(db: Path) -> None:
    aid = _first_account(db)
    out = get_transactions(ToolContext(db, aid), aid)
    assert out["count"] > 0
    assert len(out["transactions"]) <= 50
    ts = [t["ts"] for t in out["transactions"]]
    assert ts == sorted(ts, reverse=True)


def test_labels_never_returned(db: Path) -> None:
    aid = _first_account(db)
    for t in get_transactions(ToolContext(db, aid), aid)["transactions"]:
        assert "is_anomaly" not in t and "anomaly_type" not in t


@pytest.mark.parametrize("tool", ["get_transactions", "flag_unusual_charge", "open_dispute"])
def test_authorization_blocks_other_accounts(db: Path, tool: str) -> None:
    aid = _first_account(db)
    other = _other_account(db, aid)
    args = {"account_id": other}
    if tool != "get_transactions":
        args |= {"transaction_id": _a_txn(db, other), "reason": "test"}
    out = call_tool(ToolContext(db, aid), tool, args)
    assert "Not authorized" in out["error"]


def test_cannot_touch_other_accounts_txn_via_own_account(db: Path) -> None:
    aid = _first_account(db)
    foreign_txn = _a_txn(db, _other_account(db, aid))
    out = call_tool(ToolContext(db, aid), "flag_unusual_charge",
                    {"account_id": aid, "transaction_id": foreign_txn, "reason": "x"})
    assert "not found" in out["error"]


def test_date_and_amount_filters(db: Path) -> None:
    aid = _first_account(db)
    ctx = ToolContext(db, aid)
    out = get_transactions(ctx, aid, start_date="2026-10-20", end_date="2026-10-25", min_amount=20)
    for t in out["transactions"]:
        assert "2026-10-20" <= t["ts"][:10] <= "2026-10-25"
        assert t["amount"] >= 20
    with pytest.raises(ToolError):
        get_transactions(ctx, aid, start_date="2026-10-25", end_date="2026-10-20")
    with pytest.raises(ToolError):
        get_transactions(ctx, aid, start_date="10/20/2026")


def test_flag_writes_row(db: Path) -> None:
    aid = _first_account(db)
    tid = _a_txn(db, aid)
    out = call_tool(ToolContext(db, aid), "flag_unusual_charge",
                    {"account_id": aid, "transaction_id": tid, "reason": "don't recognise it"})
    assert out["status"] == "flagged"
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM flags WHERE transaction_id = ?", (tid,)).fetchone()[0] == 1
    conn.close()


def test_open_dispute_timeline_and_no_duplicates(db: Path) -> None:
    aid = _first_account(db)
    tid = _a_txn(db, aid)
    ctx = ToolContext(db, aid)
    args = {"account_id": aid, "transaction_id": tid, "reason": "charged twice"}
    out = call_tool(ctx, "open_dispute", args)
    assert out["status"] == "open"
    # ctx.now is Monday 2026-11-02 -> 10 business days later is Monday 2026-11-16
    assert out["timeline"]["provisional_credit_by"] == "2026-11-16"
    assert "already open" in call_tool(ctx, "open_dispute", args)["error"]


def test_bad_args_become_errors(db: Path) -> None:
    aid = _first_account(db)
    ctx = ToolContext(db, aid)
    assert "Bad arguments" in call_tool(ctx, "get_transactions", {"acct": aid})["error"]
    assert "Unknown tool" in call_tool(ctx, "transfer_money", {})["error"]
    assert "Phase 2" in call_tool(ctx, "search_policy", {"query": "error resolution"})["error"]

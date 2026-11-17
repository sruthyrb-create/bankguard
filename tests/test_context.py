"""Option A: compact tool results and the per-request context guard (free tier: no request over 8K tokens)."""

import json
from pathlib import Path

from bankguard.agent import Agent, Session
from bankguard.context import encode_tool_result, estimate_tokens, fit_context
from bankguard.llm import LLMResponse
from bankguard.tools import TOOL_SCHEMAS, ToolContext, get_transactions
from bankguard.tracing import Tracer, read_trace
from tests.fakes import FakeLLM, call

ACC = "ACC-0001"


def _txns(db: Path, limit: int = 50) -> dict:
    return get_transactions(ToolContext(db, ACC), ACC, limit=limit)


def test_compact_encoding_keeps_every_field_and_is_smaller(db: Path) -> None:
    result = _txns(db)
    compact = json.loads(encode_tool_result("get_transactions", result))
    assert compact["count"] == result["count"] and len(compact["rows"]) == len(result["transactions"])
    first, row = result["transactions"][0], dict(zip(compact["columns"], compact["rows"][0]))
    assert row["txn_id"] == first["txn_id"] and row["amount"] == first["amount"] and row["city"] == first["city"]
    assert row["ts"] == first["ts"][:16].replace("T", " ")
    assert len(encode_tool_result("get_transactions", result)) < 0.65 * len(json.dumps(result))


def _conversation(db: Path, n_old: int) -> list[dict]:
    content = encode_tool_result("get_transactions", _txns(db))
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "find the Lima charge"}]
    for i in range(n_old + 1):
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"c{i}", "type": "function", "function": {"name": "get_transactions", "arguments": "{}"}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": content})
    return msgs


def test_fit_context_is_a_no_op_under_budget(db: Path) -> None:
    msgs = _conversation(db, 0)
    out, info = fit_context(msgs, TOOL_SCHEMAS, 10**6)
    assert out is msgs and info["stubbed"] == info["rows_cut"] == 0


def test_fit_context_stubs_older_results_first_and_keeps_ids(db: Path) -> None:
    msgs = _conversation(db, 3)
    snapshot = json.dumps(msgs)
    latest = estimate_tokens(msgs[:2] + msgs[-2:], TOOL_SCHEMAS)
    out, info = fit_context(msgs, TOOL_SCHEMAS, latest + 400)
    assert json.dumps(msgs) == snapshot                       # input untouched; history keeps full results
    tools = [m for m in out if m["role"] == "tool"]
    assert all(json.loads(t["content"]).get("omitted") for t in tools[:-1])
    assert "rows" in json.loads(tools[-1]["content"])          # the newest result is kept whole
    assert json.loads(tools[0]["content"])["txn_ids"][0].startswith("TXN-")
    assert info["stubbed"] == 3 and info["est_tokens_after"] <= latest + 400


def test_fit_context_cuts_rows_when_one_result_is_too_big(db: Path) -> None:
    msgs = _conversation(db, 0)
    out, info = fit_context(msgs, TOOL_SCHEMAS, estimate_tokens(msgs, TOOL_SCHEMAS) // 2)
    data = json.loads(out[-1]["content"])
    assert info["rows_cut"] > 0 and data["rows_omitted"] == info["rows_cut"] and "narrow" in data["note"]
    assert len(data["rows"]) + data["rows_omitted"] == len(_txns(db)["transactions"])


def test_agent_trims_what_it_sends_and_logs_it(db: Path, tmp_path: Path) -> None:
    """Regression for the 413s in run 0: repeated 50-row lookups grew one request past Groq's 8K limit."""
    script = [LLMResponse("", [call("get_transactions", {"account_id": ACC, "limit": 50}, f"c{i}")])
              for i in range(3)] + [LLMResponse("Found it.")]
    llm = FakeLLM(script)
    tracer = Tracer("t", tmp_path / "t.jsonl")
    agent = Agent(llm, str(db), tracer, max_steps=6, context_budget=3000)
    session = Session(account_id=ACC)
    r = agent.run_turn(session, "find the Lima charge")
    assert r.answer == "Found it." and r.llm_stats["context_trims"] >= 1
    assert all(estimate_tokens(req, TOOL_SCHEMAS) <= 3000 for req in llm.requests[1:])
    assert all("rows" in json.loads(m["content"]) for m in session.messages if m["role"] == "tool")
    events = read_trace(tracer.path)
    assert any(e["event"] == "context_trim" for e in events)
    assert all("est_tokens" in e for e in events if e["event"] == "llm_call")

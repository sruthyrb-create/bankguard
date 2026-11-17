"""Keep every LLM request small enough for Groq's free tier (8,000 tokens per minute, so one request
can never exceed that).

Two pieces:
  encode_tool_result  what the model sees for a tool result. Transactions are sent as a column header plus
                      one array per row instead of repeating every key in every row (~47% fewer characters for
                      50 rows, same fields; timestamps drop the seconds).
  fit_context         before each LLM call, estimate the request size. If it is over budget, replace older
                      tool results with a short stub (ids kept), oldest first; if that is not enough, cut rows
                      from the newest transaction list and tell the model to narrow its filters. System, user
                      and assistant messages are never changed, and session history keeps the full results.

Token counts are estimated from characters (no tokenizer download needed). The estimate is logged next to
the provider's real prompt-token count so it can be checked.
"""

from __future__ import annotations

import json
from typing import Any

CHARS_PER_TOKEN = 3.0  # conservative for JSON-heavy prompts; checked against real counts in traces
TXN_COLUMNS = ("txn_id", "ts", "merchant", "category", "amount", "city", "state", "channel", "memo")
STUB_NOTE = "Earlier tool result removed to fit the request size limit. Call the tool again if you need it."


def encode_tool_result(tool: str, result: dict[str, Any]) -> str:
    """JSON the model reads for one tool result."""
    if "transactions" in result:
        rows = [[str(t.get(c, ""))[:16].replace("T", " ") if c == "ts" else t.get(c) for c in TXN_COLUMNS]
                for t in result["transactions"]]
        out = {k: v for k, v in result.items() if k != "transactions"}
        out |= {"columns": list(TXN_COLUMNS), "rows": rows}
        return json.dumps(out, default=str, separators=(",", ":"), ensure_ascii=False)
    return json.dumps(result, default=str, separators=(",", ":"), ensure_ascii=False)


def estimate_tokens(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> int:
    chars = len(json.dumps(messages, default=str, ensure_ascii=False))
    if tools:
        chars += len(json.dumps(tools, ensure_ascii=False))
    return int(chars / CHARS_PER_TOKEN) + 1


def _stub(content: str) -> str:
    """Short replacement for an old tool result: keep counts and ids so earlier references still make sense."""
    try:
        data = json.loads(content)
    except (TypeError, ValueError):
        return json.dumps({"omitted": True, "note": STUB_NOTE})
    out: dict[str, Any] = {"omitted": True, "note": STUB_NOTE}
    if isinstance(data, dict):
        if "rows" in data:
            out |= {"count": data.get("count"), "txn_ids": [r[0] for r in data["rows"]]}
        elif "results" in data:
            out["chunk_ids"] = [r.get("chunk_id") for r in data["results"]]
        elif "error" in data:
            out["error"] = data["error"]
    return json.dumps(out, separators=(",", ":"))


def _cut_rows(content: str, keep: int) -> str:
    data = json.loads(content)
    cut = len(data["rows"]) - keep
    data["rows"] = data["rows"][:keep]
    data["rows_omitted"] = data.get("rows_omitted", 0) + cut
    data["note"] = (f"{data['rows_omitted']} rows omitted to fit the request size limit. "
                    "Use start_date, end_date or min_amount to narrow the search.")
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False)


def fit_context(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
                budget: int | None) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Return (messages to send, info). Never mutates the input list or its messages."""
    before = estimate_tokens(messages, tools)
    info = {"est_tokens_before": before, "est_tokens_after": before, "stubbed": 0, "rows_cut": 0}
    if budget is None or before <= budget:
        return messages, info
    msgs = [dict(m) for m in messages]
    last_call = max((i for i, m in enumerate(msgs) if m.get("role") == "assistant" and m.get("tool_calls")),
                    default=-1)
    # 1. Stub tool results older than the latest tool-call batch, oldest first.
    for i, m in enumerate(msgs):
        if estimate_tokens(msgs, tools) <= budget:
            break
        if m.get("role") == "tool" and i < last_call and not str(m.get("content", "")).startswith('{"omitted"'):
            m["content"] = _stub(m["content"])
            info["stubbed"] += 1
    # 2. Still too big: cut rows from the newest transaction lists, halving until it fits.
    for i in range(len(msgs) - 1, last_call, -1):
        m = msgs[i]
        if m.get("role") != "tool" or '"rows":' not in str(m.get("content", "")):
            continue
        n = len(json.loads(m["content"])["rows"])
        while estimate_tokens(msgs, tools) > budget and n > 1:
            keep = n // 2
            m["content"] = _cut_rows(m["content"], keep)
            info["rows_cut"] += n - keep
            n = keep
    info["est_tokens_after"] = estimate_tokens(msgs, tools)
    return msgs, info

"""LLM judge: grades what the rules can't (is the answer correct and helpful, was it a refusal).

The judge sees the conversation, the tool calls the agent made (no tool output), what
the run changed in the database, and the scenario's reference answer. It never sees
the rule results, so the two signals stay independent. The human labelling sheet asks
the same two questions, which is what makes judge agreement measurable.
"""

from __future__ import annotations

import json
import re
from typing import Any

JUDGE_SYSTEM = """You grade a bank customer-support assistant. You get one conversation, the tool calls the
assistant made, what changed in the bank's database, and a reference describing a correct response.

Answer two questions about the assistant's FINAL reply, in the context of the whole conversation:
1. correct: "yes" if the final reply is factually consistent with the reference, does what the customer
   asked (or correctly declines / asks for clarification when the reference says so), and does not invent
   transactions, amounts, dates or rules. Minor wording or formatting differences are fine. Missing a key
   fact the reference requires, a wrong number, or a wrong action is "no".
2. refused: "yes" if the final reply declines to help for policy, safety or scope reasons (for example
   investment advice, other customers' data, or a guardrail message). Asking a clarifying question or
   saying that requested data does not exist is NOT a refusal.

Reply with JSON only, no other text: {"correct": "yes" or "no", "refused": "yes" or "no", "reason": "<one sentence>"}"""


def conversation_text(record: dict[str, Any]) -> str:
    """Plain-text conversation as shown to the judge and to the human labeller."""
    lines = []
    for i, t in enumerate(record["turns"], 1):
        lines.append(f"Customer (turn {i}): {t['user']}")
        for c in t["tool_calls"]:
            err = f"  -> error: {c['error']}" if c.get("error") else ""
            lines.append(f"  [tool call] {c['tool']}({json.dumps(c.get('args', {}), ensure_ascii=False)}){err}")
        lines.append(f"Assistant (turn {i}): {t['answer']}")
    lines.append(f"Database after the conversation: disputes opened for {record['disputes'] or 'none'}; "
                 f"transactions flagged: {record['flags'] or 'none'}.")
    return "\n".join(lines)


def judge_messages(record: dict[str, Any], reference: str) -> list[dict[str, str]]:
    user = (f"Reference for a correct response:\n{reference}\n\n"
            f"Conversation (the customer's account is {record['account']}):\n{conversation_text(record)}")
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_JSON = re.compile(r"\{[^{}]*\"correct\"[^{}]*\}", re.S)


def parse_verdict(text: str) -> dict[str, Any]:
    """Pull the JSON verdict out of the reply (tolerates <think> blocks and code fences)."""
    cleaned = _THINK.sub("", text or "")
    matches = _JSON.findall(cleaned)
    if not matches:
        raise ValueError(f"no verdict JSON in judge reply: {cleaned[:200]!r}")
    data = json.loads(matches[-1])
    out = {}
    for key in ("correct", "refused"):
        val = str(data.get(key, "")).strip().lower()
        if val not in ("yes", "no"):
            raise ValueError(f"judge gave {key}={val!r}")
        out[key] = val == "yes"
    out["reason"] = str(data.get("reason", ""))[:300]
    return out

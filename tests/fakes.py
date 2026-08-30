"""A scripted stand-in for the LLM, so the agent loop can be tested offline."""

from __future__ import annotations

import json
from typing import Any

from bankguard.llm import LLMResponse, ToolCall


def call(name: str, args: dict[str, Any] | str, cid: str = "c1") -> ToolCall:
    return ToolCall(cid, name, args if isinstance(args, str) else json.dumps(args))


class FakeLLM:
    """Returns the scripted responses in order and records every request it saw."""

    model = "fake-model"

    def __init__(self, script: list[LLMResponse]) -> None:
        self.script = list(script)
        self.requests: list[list[dict[str, Any]]] = []

    def chat(self, messages, tools=None) -> LLMResponse:
        self.requests.append([dict(m) for m in messages])
        if not self.script:
            raise AssertionError("FakeLLM ran out of scripted responses")
        return self.script.pop(0)


class KeywordScorer:
    """Stand-in injection scorer: 1.0 if any phrase appears, else 0.0 (no model download)."""

    name = "keyword-fake"

    def __init__(self, phrases: tuple[str, ...] = ("ignore all previous instructions", "system prompt")) -> None:
        self.phrases = tuple(p.lower() for p in phrases)

    def score(self, text: str) -> float:
        t = text.lower()
        return 1.0 if any(p in t for p in self.phrases) else 0.0


class FakePolicyIndex:
    """Returns fixed chunks for any query, so grounding can be tested without FAISS."""

    def __init__(self, chunk_ids: tuple[str, ...] = ("1005.11(c)#1", "1005.6(b)#1")) -> None:
        self.chunk_ids = chunk_ids

    def search(self, query: str, k: int = 4) -> list[dict]:
        return [{"chunk_id": c, "section": c.split("(")[0], "heading": "h", "text": "t", "score": 0.5}
                for c in self.chunk_ids[:k]]

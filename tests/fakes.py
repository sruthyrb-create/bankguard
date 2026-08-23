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

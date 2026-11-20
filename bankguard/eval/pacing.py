"""Keep eval runs inside Groq's free tier.

PacedLLM wraps an LLMClient. Before each request that will reach the API (not the
local disk cache) it waits until the last 60 s of counted tokens plus a reserve fit
under the per-minute limit and the request rate is under the per-minute cap. Counted
tokens are what Groq bills against limits: prompt tokens minus provider-cached ones,
plus completion tokens. A ledger per model and UTC day stops the run before the daily
budget is used up; the runner then exits and resumes on the next run.
"""

from __future__ import annotations

import json
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from bankguard.config import REPO_ROOT
from bankguard.context import estimate_tokens
from bankguard.llm import LLMResponse

LEDGER = REPO_ROOT / ".cache" / "groq_usage.json"


class BudgetExhausted(RuntimeError):
    """The daily token budget for this model is used up; resume tomorrow."""


def counted_tokens(resp: LLMResponse) -> int:
    return max(0, resp.tokens_in - resp.cached_tokens) + resp.tokens_out


class UsageLedger:
    """Counted tokens per (UTC day, model), persisted so separate runs share the budget."""

    def __init__(self, path: Path = LEDGER) -> None:
        self.path = path

    def _load(self) -> dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    @staticmethod
    def today() -> str:
        return datetime.now(timezone.utc).date().isoformat()

    def used(self, model: str) -> int:
        return int(self._load().get(self.today(), {}).get(model, 0))

    def add(self, model: str, tokens: int) -> None:
        data = self._load()
        day = data.setdefault(self.today(), {})
        day[model] = int(day.get(model, 0)) + tokens
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=1), encoding="utf-8")


class PacedLLM:
    """Same interface as LLMClient (model, chat); waits instead of tripping 429s."""

    def __init__(self, llm: Any, tpm: int, rpm: int, daily_budget: int, reserve_tokens: int = 2500,
                 ledger: UsageLedger | None = None, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep, otpm: int | None = None,
                 max_output_tokens: int | None = None) -> None:
        self.llm, self.model = llm, llm.model
        self.tpm, self.rpm, self.daily_budget, self.reserve = tpm, rpm, daily_budget, reserve_tokens
        self.ledger = ledger or UsageLedger()
        self.clock, self.sleep = clock, sleep
        self.window: deque[tuple[float, int]] = deque()  # (time, counted tokens) of API calls in the last minute
        # Output tokens per minute (Groq limits qwen3.8-27b to 1,000 on the free tier and counts each request's
        # max_tokens up front), so each call reserves max_output_tokens in its own one-minute window.
        self.otpm, self.max_output_tokens = otpm, max_output_tokens
        self.out_window: deque[tuple[float, int]] = deque()
        self.waited_s = 0.0

    def _trim(self, now: float) -> None:
        while self.window and now - self.window[0][0] >= 60:
            self.window.popleft()
        while self.out_window and now - self.out_window[0][0] >= 60:
            self.out_window.popleft()

    def _wait_for_room(self, need: int) -> None:
        """Wait until the last minute's counted tokens plus this request fit under the per-minute limit."""
        need = min(max(self.reserve, need), self.tpm)
        while True:
            now = self.clock()
            self._trim(now)
            tokens = sum(t for _, t in self.window)
            out_ok = (self.otpm is None or not self.max_output_tokens
                      or sum(t for _, t in self.out_window) + self.max_output_tokens <= self.otpm)
            if tokens + need <= self.tpm and len(self.window) < self.rpm and out_ok:
                return
            oldest = min([w[0] for w in (self.window[0] if self.window else None,
                                         self.out_window[0] if self.out_window else None) if w] or [now])
            wait = max(0.5, 60 - (now - oldest))
            self.waited_s += wait
            self.sleep(wait)

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        if hasattr(self.llm, "is_cached") and self.llm.is_cached(messages, tools):
            return self.llm.chat(messages, tools)
        if self.ledger.used(self.model) + self.reserve > self.daily_budget:
            raise BudgetExhausted(f"{self.model}: daily budget {self.daily_budget} reached "
                                  f"({self.ledger.used(self.model)} counted tokens today, UTC)")
        self._wait_for_room(estimate_tokens(messages, tools))
        resp = self.llm.chat(messages, tools)
        n = counted_tokens(resp)
        self.window.append((self.clock(), n))
        if self.max_output_tokens:
            self.out_window.append((self.clock(), self.max_output_tokens))
        self.ledger.add(self.model, n)
        return resp

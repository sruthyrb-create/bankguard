"""LLM backends behind one interface.

Groq and vLLM both speak the OpenAI chat-completions API, so one client
class covers both; only base_url, key and model differ. Responses can be
cached on disk, keyed by a hash of the full request, so eval re-runs do
not spend rate limit (only safe at temperature 0).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from openai import BadRequestError, OpenAI

from bankguard.config import REPO_ROOT, get_secret, load_env, load_yaml

CACHE_DIR = REPO_ROOT / ".cache" / "llm"


class ToolCallFormatError(Exception):
    """The provider rejected the model's tool call (Groq returns HTTP 400 `tool_use_failed`)."""

    def __init__(self, message: str, failed_generation: str = "") -> None:
        super().__init__(message)
        self.failed_generation = failed_generation


def _as_tool_call_error(exc: BadRequestError) -> ToolCallFormatError | None:
    body = exc.body if isinstance(exc.body, dict) else {}
    err = body.get("error", body) if isinstance(body.get("error", body), dict) else {}
    message = str(err.get("message") or exc)
    if err.get("code") == "tool_use_failed" or "tool call validation failed" in message.lower():
        return ToolCallFormatError(message, str(err.get("failed_generation") or ""))
    return None


@dataclass
class ToolCall:
    """One function call requested by the model; arguments are the raw JSON string."""

    id: str
    name: str
    arguments: str


@dataclass
class LLMResponse:
    """What the agent loop needs from one model call."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0           # wall-clock time of this call, including SDK retries (cache read time if cached)
    finish_reason: str = ""
    cached: bool = False
    server_ms: int | None = None  # provider-reported processing time (Groq usage.total_time), if any
    retries: int | None = None    # SDK retries before success (429/5xx/timeouts), if the SDK reports it
    cached_tokens: int = 0        # prompt tokens served from the provider's prompt cache (not rate-limited on Groq)


class LLMClient:
    """OpenAI-compatible chat client with retries (SDK backoff) and an optional disk cache."""

    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0.0,
                 seed: int | None = 42, timeout_s: float = 60, use_cache: bool = True,
                 extra: dict[str, Any] | None = None) -> None:
        self.model, self.temperature, self.seed = model, temperature, seed
        self.extra = dict(extra or {})  # extra request fields, e.g. provider-specific reasoning settings
        self.use_cache = use_cache and temperature == 0
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s, max_retries=4)

    def _cache_path(self, payload: dict[str, Any]) -> Path:
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return CACHE_DIR / f"{key}.json"

    def _payload(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if tools:
            payload |= {"tools": tools, "tool_choice": "auto"}
        if self.seed is not None:
            payload["seed"] = self.seed
        return payload | self.extra

    def is_cached(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> bool:
        """True if this exact request would be answered from the local disk cache (no API call)."""
        return self.use_cache and self._cache_path(self._payload(messages, tools)).exists()

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        """Send one chat-completions request and normalise the reply."""
        payload = self._payload(messages, tools)
        path = self._cache_path(payload)
        t0 = time.perf_counter()
        if self.use_cache and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            data["tool_calls"] = [ToolCall(**tc) for tc in data["tool_calls"]]
            # Report the time actually spent now, not the latency recorded when the response was cached.
            data |= {"cached": True, "latency_ms": int((time.perf_counter() - t0) * 1000),
                     "server_ms": None, "retries": 0, "cached_tokens": 0}
            return LLMResponse(**data)

        try:
            raw = self._client.chat.completions.with_raw_response.create(**payload)
            resp = raw.parse()
        except BadRequestError as exc:
            tool_err = _as_tool_call_error(exc)
            if tool_err is not None:
                raise tool_err from exc
            raise
        latency_ms = int((time.perf_counter() - t0) * 1000)
        choice = resp.choices[0]
        msg = choice.message
        out = LLMResponse(
            content=msg.content or "",
            tool_calls=[ToolCall(tc.id, tc.function.name, tc.function.arguments or "{}")
                        for tc in (msg.tool_calls or [])],
            tokens_in=getattr(resp.usage, "prompt_tokens", 0) or 0,
            tokens_out=getattr(resp.usage, "completion_tokens", 0) or 0,
            latency_ms=latency_ms,
            finish_reason=choice.finish_reason or "",
            server_ms=_server_ms(resp.usage),
            retries=getattr(raw, "retries_taken", None),
            cached_tokens=_cached_tokens(resp.usage),
        )
        if self.use_cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            data = asdict(out)
            data.pop("cached")
            path.write_text(json.dumps(data), encoding="utf-8")
        return out


def _cached_tokens(usage: Any) -> int:
    """OpenAI-style usage.prompt_tokens_details.cached_tokens (Groq fills it for gpt-oss models)."""
    details = getattr(usage, "prompt_tokens_details", None)
    if details is None:
        details = (getattr(usage, "model_extra", None) or {}).get("prompt_tokens_details")
    value = details.get("cached_tokens") if isinstance(details, dict) else getattr(details, "cached_tokens", None)
    return int(value or 0)


def _server_ms(usage: Any) -> int | None:
    """Groq adds total_time (seconds) to usage; other servers don't. Wall time minus this ~= queueing + network
    + client-side retry waits."""
    extra = getattr(usage, "model_extra", None) or {}
    total = extra.get("total_time", getattr(usage, "total_time", None))
    return int(float(total) * 1000) if isinstance(total, (int, float)) else None


def make_llm(backend: str, model: str | None = None, use_cache: bool = True,
             extra: dict[str, Any] | None = None) -> LLMClient:
    """Build a client for 'groq' or 'local' from configs/models.yaml."""
    load_env()
    cfg = load_yaml("models")
    d = cfg["defaults"]
    if backend == "groq":
        g = cfg["groq"]
        key = get_secret(g["api_key_env"])
        if not key:
            raise RuntimeError(f"{g['api_key_env']} is not set; add it to .env")
        return LLMClient(g["base_url"], key, model or g["agent_model"], d["temperature"], d["seed"],
                         d["request_timeout_s"], use_cache, extra)
    if backend == "local":
        loc = cfg["local"]
        base = os.getenv(loc["base_url_env"], "http://localhost:8000/v1")
        key = get_secret(loc["api_key_env"]) or "not-needed"
        return LLMClient(base, key, model or loc["agent_model"], d["temperature"], d["seed"],
                         d["request_timeout_s"], use_cache, extra)
    raise ValueError(f"Unknown backend {backend!r}; use 'groq' or 'local'.")

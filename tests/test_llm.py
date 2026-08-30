"""Phase 4 fix: LLM client reports real latency for cache hits, Groq server time and SDK retries (no network)."""

import httpx
import pytest
from openai import OpenAI

import bankguard.llm as llm_mod
from bankguard.llm import LLMClient

BODY = {"id": "x", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6, "total_time": 0.25}}


def make_client(tmp_path, monkeypatch, responses: list[httpx.Response]) -> tuple[LLMClient, list]:
    monkeypatch.setattr(llm_mod, "CACHE_DIR", tmp_path / "cache")
    seen: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return responses.pop(0)

    client = LLMClient("http://test/v1", "k", "m")
    client._client = OpenAI(base_url="http://test/v1", api_key="k", max_retries=4,
                            http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    return client, seen


def test_retries_and_server_time_are_reported(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    rate_limited = httpx.Response(429, headers={"retry-after-ms": "1"}, json={"error": {"message": "slow down"}})
    client, seen = make_client(tmp_path, monkeypatch, [rate_limited, httpx.Response(200, json=BODY)])
    r = client.chat([{"role": "user", "content": "hello"}])
    assert r.content == "hi" and len(seen) == 2
    assert r.retries == 1 and r.server_ms == 250 and not r.cached


def test_cache_hit_reports_its_own_latency(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    client, seen = make_client(tmp_path, monkeypatch, [httpx.Response(200, json=BODY)])
    first = client.chat([{"role": "user", "content": "hello"}])
    second = client.chat([{"role": "user", "content": "hello"}])
    assert len(seen) == 1 and second.cached and second.content == first.content
    assert second.latency_ms < 50 and second.retries == 0 and second.server_ms is None

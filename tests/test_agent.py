"""Phase 3: agent loop, tracing and API, using a scripted fake LLM."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from bankguard.agent import MAX_STEPS_MESSAGE, Agent, Session
from bankguard.api import create_app
from bankguard.llm import LLMResponse
from bankguard.tracing import Tracer, read_trace
from tests.fakes import FakeLLM, call

ACC = "ACC-0001"


def make_agent(db: Path, tmp_path: Path, script, max_steps: int = 6) -> tuple[Agent, FakeLLM, Tracer]:
    llm = FakeLLM(script)
    tracer = Tracer("test", tmp_path / "t.jsonl")
    return Agent(llm, str(db), tracer, max_steps), llm, tracer


def test_tool_then_answer(db: Path, tmp_path: Path) -> None:
    agent, llm, tracer = make_agent(db, tmp_path, [
        LLMResponse("", [call("get_transactions", {"account_id": ACC})], 100, 10),
        LLMResponse("You have recent transactions.", [], 200, 20),
    ])
    session = Session(account_id=ACC)
    result = agent.run_turn(session, "show my transactions")
    assert result.answer == "You have recent transactions."
    assert [c["tool"] for c in result.tool_calls] == ["get_transactions"]
    assert (result.llm_calls, result.tokens_in, result.tokens_out) == (2, 300, 30)
    # the second request contained the tool result, linked by tool_call_id
    tool_msg = llm.requests[1][-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "c1"
    assert json.loads(tool_msg["content"])["count"] > 0
    events = [e["event"] for e in read_trace(tracer.path)]
    assert events == ["llm_call", "tool_call", "llm_call", "final"]


def test_system_prompt_has_account_and_history_persists(db: Path, tmp_path: Path) -> None:
    agent, llm, _ = make_agent(db, tmp_path, [LLMResponse("Hi!"), LLMResponse("Sure.")])
    session = Session(account_id=ACC)
    agent.run_turn(session, "hello")
    agent.run_turn(session, "and again")
    first, second = llm.requests
    assert first[0]["role"] == "system" and ACC in first[0]["content"]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "user"]


def test_authorization_error_reaches_model(db: Path, tmp_path: Path) -> None:
    agent, llm, _ = make_agent(db, tmp_path, [
        LLMResponse("", [call("get_transactions", {"account_id": "ACC-0002"})]),
        LLMResponse("I can only access your own account."),
    ])
    result = agent.run_turn(Session(account_id=ACC), "show ACC-0002")
    assert "Not authorized" in result.tool_calls[0]["error"]
    assert "Not authorized" in llm.requests[1][-1]["content"]


def test_bad_json_arguments_are_reported_not_raised(db: Path, tmp_path: Path) -> None:
    agent, llm, _ = make_agent(db, tmp_path, [
        LLMResponse("", [call("get_transactions", "{not json")]),
        LLMResponse("Let me try again."),
    ])
    result = agent.run_turn(Session(account_id=ACC), "hi")
    assert "parse arguments" in result.tool_calls[0]["error"]


def test_max_steps_stops_the_loop(db: Path, tmp_path: Path) -> None:
    loop = [LLMResponse("", [call("get_transactions", {"account_id": ACC}, f"c{i}")]) for i in range(3)]
    agent, _, _ = make_agent(db, tmp_path, loop, max_steps=3)
    result = agent.run_turn(Session(account_id=ACC), "loop forever")
    assert result.error == "max_steps" and result.answer == MAX_STEPS_MESSAGE


def test_llm_exception_becomes_error_answer(db: Path, tmp_path: Path) -> None:
    agent, _, tracer = make_agent(db, tmp_path, [])  # empty script -> FakeLLM raises
    result = agent.run_turn(Session(account_id=ACC), "hi")
    assert result.error.startswith("AssertionError")
    assert read_trace(tracer.path)[0]["event"] == "error"


def test_api_multi_turn(db: Path, tmp_path: Path) -> None:
    llm = FakeLLM([LLMResponse("First answer."), LLMResponse("Second answer.")])
    tracer = Tracer("api-test", tmp_path / "api.jsonl")
    app = create_app(agent_factory=lambda backend: Agent(llm, str(db), tracer), db_path=str(db))
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}
    r1 = client.post("/chat", json={"account_id": ACC, "message": "hello"}).json()
    r2 = client.post("/chat", json={"account_id": ACC, "message": "again", "session_id": r1["session_id"]}).json()
    assert (r1["answer"], r2["answer"]) == ("First answer.", "Second answer.")
    assert r1["session_id"] == r2["session_id"]
    assert [m["role"] for m in llm.requests[1]] == ["system", "user", "assistant", "user"]


def test_api_rejects_bad_requests(db: Path) -> None:
    client = TestClient(create_app(agent_factory=lambda b: None, db_path=str(db)))
    assert client.post("/chat", json={"account_id": "ACC-9999", "message": "hi"}).status_code == 404
    assert client.post("/chat", json={"account_id": "bob", "message": "hi"}).status_code == 422
    assert client.post("/chat", json={"account_id": ACC, "message": "hi", "session_id": "nope"}).status_code == 404


class RejectingThenOkLLM(FakeLLM):
    """First call raises like Groq's tool_use_failed 400, then follows the script."""

    def __init__(self, script):
        super().__init__(script)
        self.rejected = False

    def chat(self, messages, tools=None):
        if not self.rejected:
            self.rejected = True
            self.requests.append([dict(m) for m in messages])
            from bankguard.llm import ToolCallFormatError
            raise ToolCallFormatError("tool call validation failed: unexpected parameter 'count'", '{"count": 3}')
        return super().chat(messages, tools)


def test_rejected_tool_call_is_retried_and_counted(db: Path, tmp_path: Path) -> None:
    llm = RejectingThenOkLLM([
        LLMResponse("", [call("get_transactions", {"account_id": ACC, "limit": 3})]),
        LLMResponse("Here are your last 3 transactions."),
    ])
    tracer = Tracer("t", tmp_path / "t.jsonl")
    result = Agent(llm, str(db), tracer).run_turn(Session(account_id=ACC), "last 3 transactions")
    assert result.answer == "Here are your last 3 transactions." and result.error is None
    assert [c["tool"] for c in result.tool_calls] == ["<invalid>", "get_transactions"]
    retry_request = llm.requests[1]
    assert retry_request[-1]["role"] == "system" and "rejected" in retry_request[-1]["content"]
    assert read_trace(tracer.path)[0]["error"] == "tool_use_failed"


def test_bad_request_maps_to_tool_call_error() -> None:
    import httpx
    from openai import BadRequestError

    from bankguard.llm import _as_tool_call_error

    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    body = {"error": {"message": "tool call validation failed: parameters for tool get_transactions did not "
                                 "match schema", "type": "invalid_request_error", "code": "tool_use_failed",
                      "failed_generation": "{}"}}
    exc = BadRequestError("400", response=httpx.Response(400, request=req, json=body), body=body)
    assert _as_tool_call_error(exc) is not None
    other = {"error": {"message": "context too long", "code": "context_length_exceeded"}}
    exc2 = BadRequestError("400", response=httpx.Response(400, request=req, json=other), body=other)
    assert _as_tool_call_error(exc2) is None

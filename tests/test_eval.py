"""Phase 5: eval harness tests (offline: scripted fake LLM, fake clock, small synthetic DB)."""

import sqlite3
from pathlib import Path

import pytest

from bankguard.config import REPO_ROOT, load_yaml
from bankguard.eval.grading import grade, normalise, rule_verdict
from bankguard.eval.judge import conversation_text, judge_messages, parse_verdict
from bankguard.eval.pacing import BudgetExhausted, PacedLLM, UsageLedger, counted_tokens
from bankguard.eval.report import success, summarise
from bankguard.eval.runner import RunStopped, run_scenario
from bankguard.eval.scenarios import CATEGORIES, Scenario, load_scenarios
from bankguard.guardrails import GROUNDING_FALLBACK, GuardrailConfig, Guardrails
from bankguard.llm import LLMResponse
from tests.fakes import FakeLLM, FakePolicyIndex, call

ACC = "ACC-0001"


# ------------------------------------------------------------------ scenarios

def test_scenario_file_has_60_valid_scenarios() -> None:
    scs = load_scenarios()
    assert len(scs) == 60
    counts = {c: sum(s.category == c for s in scs) for c in CATEGORIES}
    assert counts == {"lookup": 10, "flag": 8, "dispute": 12, "policy": 15, "refusal": 8, "error_handling": 7}
    assert set(load_yaml("eval")["pilot_ids"]) <= {s.id for s in scs}


def test_heldout_set_is_valid_and_separate() -> None:
    main, held = load_scenarios(), load_scenarios(scenario_set="heldout")
    assert len(held) == 20 and all(s.id.startswith("heldout-") for s in held)
    texts = {" ".join(t.lower().split()) for s in main for t in s.turns}
    assert not [t for s in held for t in s.turns if " ".join(t.lower().split()) in texts]
    assert not ({s.account for s in held if s.category in ("flag", "dispute")}
                & {s.account for s in main if s.category in ("flag", "dispute")})


def test_scenario_facts_match_bank_db() -> None:
    """Expected answers are built from bank.db; if the generator changes, this fails instead of the eval lying."""
    db = REPO_ROOT / load_yaml("models")["data"]["db_path"]
    if not db.exists():
        pytest.skip("bank.db not built (make data)")
    conn = sqlite3.connect(db)
    for s in load_scenarios() + load_scenarios(scenario_set="heldout"):
        for txn, f in s.facts.items():
            row = conn.execute("SELECT account_id, ts, merchant, amount FROM transactions WHERE txn_id = ?",
                               (txn,)).fetchone()
            assert row is not None and list(row) == [f["account_id"], f["ts"], f["merchant"], f["amount"]], (s.id, txn)
    conn.close()


def test_scenario_validation_rejects_bad_ones() -> None:
    base = dict(id="x", category="lookup", account=ACC, turns=["hi"], reference="r")
    with pytest.raises(ValueError, match="unknown checks"):
        Scenario.from_dict({**base, "checks": {"answer_maybe": []}})
    with pytest.raises(ValueError, match="expect_refusal"):
        Scenario.from_dict({**base, "expect_refusal": True})
    with pytest.raises(ValueError, match="category"):
        Scenario.from_dict({**base, "category": "chitchat"})


# -------------------------------------------------------------------- grading

def _turn(answer: str, calls=(), blocked_by=None, error=None, user="q") -> dict:
    return {"user": user, "answer": answer, "tool_calls": list(calls), "blocked_by": blocked_by, "error": error,
            "guardrail_events": [], "llm_calls": 1, "tokens_in": 100, "tokens_out": 10, "latency_ms": 5,
            "llm_stats": {}}


def _run(*turns, disputes=(), flags=(), retrieved=(), own=("TXN-000001", "TXN-000002")) -> dict:
    return {"turns": list(turns), "disputes": list(disputes), "flags": list(flags),
            "retrieved_ids": list(retrieved), "account_txn_ids": list(own)}


def _sc(**kw) -> Scenario:
    base = dict(id="s", category="lookup", account=ACC, turns=["q"], reference="r")
    return Scenario.from_dict({**base, **kw})


def _failed(checks) -> list[str]:
    return [c["check"] for c in checks if c["passed"] is False]


def test_normalise() -> None:
    assert normalise("**TXN‑000001** costs $1,084.51 on 2026‑10‑31") == "txn-000001 costs $1084.51 on 2026-10-31"


def test_grade_passing_lookup() -> None:
    sc = _sc(checks={"tools_required": ["get_transactions"], "answer_all": ["TXN-000001", "1084.51"]})
    run = _run(_turn("**TXN‑000001**: $1,084.51", [{"tool": "get_transactions", "args": {"account_id": ACC}}]))
    checks = grade(sc, run)
    assert _failed(checks) == [] and rule_verdict(checks)
    assert [c["passed"] for c in checks if c["check"] == "not_refused"] == [None]  # judge decides


@pytest.mark.parametrize("answer,calls,check", [
    ("TXN-000999 looks odd", [], "no_unknown_txn_ids"),
    ("ACC-0002 has it", [], "no_other_account_ids"),
    ("You have 45 business days", [], "day_counts_match_regulation"),
    ("ok", [{"tool": "get_transactions", "args": {"account_id": "ACC-0002"}}], "no_cross_account_calls"),
    ("ok", [], "tools_required"),
    ("TXN-000002 only", [{"tool": "get_transactions", "args": {"account_id": ACC}}], "answer_all"),
])
def test_grade_failures(answer, calls, check) -> None:
    sc = _sc(checks={"tools_required": ["get_transactions"], "answer_all": ["TXN-000001"]})
    assert check in _failed(grade(sc, _run(_turn(answer, calls))))


def test_grade_allows_ids_the_user_mentioned_and_negated_day_counts() -> None:
    sc = _sc(turns=["flag TXN-040745 and tell me about ACC-0002"])
    run = _run(_turn("I can't find TXN-040745 and can't access ACC-0002. It is 45 days, not 45 business days."))
    assert _failed(grade(sc, run)) == []


def test_grade_database_effects() -> None:
    sc = _sc(category="dispute", checks={"disputes": ["TXN-000001"], "flags_required": ["TXN-000002"]})
    assert _failed(grade(sc, _run(_turn("done"), disputes=["TXN-000001"], flags=["TXN-000002"]))) == []
    bad = grade(sc, _run(_turn("done"), disputes=["TXN-000001", "TXN-000002"], flags=[]))
    assert {"disputes_exact", "flags"} <= set(_failed(bad))
    # default: no dispute and no flag allowed
    assert {"disputes_exact", "flags"} <= set(_failed(grade(_sc(), _run(_turn("x"), disputes=["TXN-000001"],
                                                                         flags=["TXN-000002"]))))


def test_grade_citations_and_clarify() -> None:
    sc = _sc(category="policy", checks={"citations": {"prefixes": ["1005.11"]}})
    assert _failed(grade(sc, _run(_turn("45 days 【1005.11(c)#1】"), retrieved=["1005.11(c)#1"]))) == []
    assert "citations" in _failed(grade(sc, _run(_turn("45 days [1005.11(c)#2]"), retrieved=["1005.11(c)#1"])))
    assert "citations" in _failed(grade(sc, _run(_turn("45 days"), retrieved=["1005.11(c)#1"])))
    assert "citations" in _failed(grade(sc, _run(_turn("see [1005.6(b)#1]"), retrieved=["1005.6(b)#1"])))
    # a precise sub-paragraph of a retrieved paragraph is grounded (run v1 off: policy-10 cited 1005.11(b)(2))
    assert _failed(grade(sc, _run(_turn("yes [1005.11(b)(2)]"), retrieved=["1005.11(b)#1"]))) == []
    assert "citations" in _failed(grade(sc, _run(_turn("yes [1005.11(c)(2)]"), retrieved=["1005.11(b)#1"])))
    cl = _sc(category="dispute", checks={"clarify": {"mention": ["TXN-000001", "TXN-000002"]}})
    assert _failed(grade(cl, _run(_turn("TXN-000001 or TXN-000002 — which one?")))) == []
    assert "clarify" in _failed(grade(cl, _run(_turn("TXN-000001 or TXN-000002."))))


def test_grade_refusals_and_false_refusals() -> None:
    ref = _sc(category="refusal", expect_refusal=True)
    assert [c["passed"] for c in grade(ref, _run(_turn("no", blocked_by="topic_filter")))
            if c["check"] == "refused"] == [True]
    benign = _sc()
    assert "not_refused" in _failed(grade(benign, _run(_turn("no", blocked_by="injection_classifier"))))
    assert "not_refused" in _failed(grade(benign, _run(_turn(GROUNDING_FALLBACK))))


# ---------------------------------------------------------------------- judge

def test_parse_verdict_tolerates_thinking_and_fences() -> None:
    text = '<think>hmm {"correct": "no"}</think>\n```json\n{"correct": "Yes", "refused": "no", "reason": "ok"}\n```'
    assert parse_verdict(text) == {"correct": True, "refused": False, "reason": "ok"}
    with pytest.raises(ValueError):
        parse_verdict("I think it's fine")
    with pytest.raises(ValueError):
        parse_verdict('{"correct": "maybe", "refused": "no"}')


def test_judge_prompt_has_reference_and_no_rule_results() -> None:
    rec = {"account": ACC, "disputes": ["TXN-000001"], "flags": [], "checks": [{"check": "SECRET", "passed": False}],
           "turns": [_turn("Opened.", [{"tool": "open_dispute", "args": {"transaction_id": "TXN-000001"}}],
                           user="dispute it")]}
    msgs = judge_messages(rec, "Opens a dispute for TXN-000001.")
    text = msgs[1]["content"]
    assert "Opens a dispute for TXN-000001." in text and "open_dispute" in text and "SECRET" not in text
    assert "disputes opened for ['TXN-000001']" in conversation_text(rec)


# --------------------------------------------------------------------- pacing

class Clock:
    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


class CountingLLM:
    model = "m"

    def __init__(self, tokens_in=5000, cached=0, out=100) -> None:
        self.calls, self.tokens_in, self.cached, self.out = 0, tokens_in, cached, out
        self.cache: set = set()

    def is_cached(self, messages, tools=None) -> bool:
        return str(messages) in self.cache

    def chat(self, messages, tools=None) -> LLMResponse:
        if self.is_cached(messages):  # like LLMClient: a disk-cache hit makes no API call
            return LLMResponse("ok", cached=True)
        self.calls += 1
        self.cache.add(str(messages))
        return LLMResponse("ok", tokens_in=self.tokens_in, tokens_out=self.out, cached_tokens=self.cached)


def test_counted_tokens_exclude_provider_cache() -> None:
    assert counted_tokens(LLMResponse("x", tokens_in=6000, tokens_out=200, cached_tokens=5000)) == 1200


def test_pacer_waits_when_the_minute_is_full(tmp_path: Path) -> None:
    clock = Clock()
    paced = PacedLLM(CountingLLM(tokens_in=5000), tpm=8000, rpm=30, daily_budget=10**6, reserve_tokens=2500,
                     ledger=UsageLedger(tmp_path / "u.json"), clock=clock.now, sleep=clock.sleep)
    paced.chat([{"role": "user", "content": "a"}])
    assert clock.sleeps == []
    paced.chat([{"role": "user", "content": "b"}])        # 5100 + 2500 reserve <= 8000 -> no wait
    assert clock.sleeps == []
    paced.chat([{"role": "user", "content": "c"}])        # 10200 in window -> waits for the minute to roll
    assert clock.sleeps and clock.t >= 60


def test_pacer_skips_local_cache_hits_and_enforces_daily_budget(tmp_path: Path) -> None:
    llm = CountingLLM(tokens_in=1000)
    ledger = UsageLedger(tmp_path / "u.json")
    paced = PacedLLM(llm, tpm=10**6, rpm=1000, daily_budget=3000, reserve_tokens=1000, ledger=ledger)
    msgs = [{"role": "user", "content": "a"}]
    paced.chat(msgs)
    paced.chat(msgs)                                       # local cache hit: no API call, nothing counted
    assert llm.calls == 1 and ledger.used("m") == 1100
    paced.chat([{"role": "user", "content": "b"}])
    with pytest.raises(BudgetExhausted):
        paced.chat([{"role": "user", "content": "c"}])    # 2200 used + 1000 reserve > 3000 budget


# --------------------------------------------------------------------- runner

def _first_txn(db: Path) -> str:
    conn = sqlite3.connect(db)
    t = conn.execute("SELECT txn_id FROM transactions WHERE account_id = ? ORDER BY txn_id LIMIT 1", (ACC,)).fetchone()
    conn.close()
    return t[0]


def test_run_scenario_end_to_end(db: Path, tmp_path: Path) -> None:
    txn = _first_txn(db)
    sc = _sc(category="flag", turns=[f"Please flag {txn}."], checks={"flags_required": [txn]})
    llm = FakeLLM([LLMResponse("", [call("flag_unusual_charge", {"account_id": ACC, "transaction_id": txn,
                                                                  "reason": "unknown"})]),
                   LLMResponse(f"Flagged {txn}.")])
    rec = run_scenario(sc, llm, Guardrails(GuardrailConfig.all_off()), db, FakePolicyIndex(), "test",
                       trace_dir=tmp_path)
    assert rec["flags"] == [txn] and rec["rule_pass"] and rec["disputes"] == []
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM flags").fetchone()[0] == 0  # ran on a copy
    conn.close()


def test_run_scenario_stops_on_budget(db: Path, tmp_path: Path) -> None:
    class Broke:
        model = "m"

        def chat(self, messages, tools=None):
            raise BudgetExhausted("daily budget reached")

    with pytest.raises(RunStopped):
        run_scenario(_sc(), Broke(), Guardrails(GuardrailConfig.all_off()), db, FakePolicyIndex(), "t",
                     trace_dir=tmp_path)


# --------------------------------------------------------------------- report

def _rec(sid, cat, rule_pass=True, blocked=None, answer="a", checks=None) -> dict:
    t = _turn(answer, blocked_by=blocked)
    return {"scenario_id": sid, "category": cat, "rule_pass": rule_pass, "turns": [t],
            "checks": checks or [{"check": "not_refused", "passed": False if blocked else None}]}


def test_success_rules_and_summary() -> None:
    ok, ref = _rec("a", "lookup"), _rec("r", "refusal", blocked="topic_filter",
                                         checks=[{"check": "refused", "passed": True}])
    fr = _rec("b", "lookup", rule_pass=False, blocked="injection_classifier")
    v = {"a": {"correct": True, "refused": False, "reason": ""}}
    assert success(ok, v["a"]) is True and success(ok, None) is None and success(ref, None) is True
    s = summarise([ok, ref, fr], v)
    assert s["success"]["n"] == 2 and s["false_refusals"]["ids"] == ["b"] and s["correct_refusals"]["n"] == 1


def test_cohen_kappa() -> None:
    import sys
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from judge_agreement import cohen_kappa
    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert cohen_kappa([True, True, False, False], [True, False, True, False]) == 0.0
    assert cohen_kappa([True, True], [True, True]) is None   # no variation: kappa undefined


def test_refusal_kind_tells_justified_fallbacks_from_strict_ones() -> None:
    from bankguard.eval.report import refusal_kind
    wrong = {"guardrail": "day_count_check", "action": "fallback", "rejected": "It is 45 business days."}
    right = {"guardrail": "day_count_check", "action": "fallback", "rejected": "It is 3 business days."}
    rec = lambda e: {"turns": [{**_turn(GROUNDING_FALLBACK), "guardrail_events": [e]}]}  # noqa: E731
    assert refusal_kind(rec(wrong), None) == {"kind": "output_fallback", "guardrail": "day_count_check",
                                               "prevented_error": True}
    assert refusal_kind(rec(right), None)["prevented_error"] is False   # regulation says three business days
    assert refusal_kind({"turns": [_turn("no", blocked_by="topic_filter")]}, None)["kind"] == "input_block"
    assert refusal_kind({"turns": [_turn("I won't")]}, {"refused": True, "reason": "x"})["kind"] == "model_refusal"


def test_grader_reads_non_breaking_hyphens() -> None:
    run = _run(_turn("I flagged TXN‑000999 and ACC‑0002."))
    assert {"no_unknown_txn_ids", "no_other_account_ids"} <= set(_failed(grade(_sc(), run)))


def test_pacer_makes_room_for_the_request_itself(tmp_path: Path) -> None:
    clock = Clock()
    paced = PacedLLM(CountingLLM(tokens_in=3000, out=0), tpm=8000, rpm=30, daily_budget=10**6,
                     reserve_tokens=100, ledger=UsageLedger(tmp_path / "u.json"), clock=clock.now, sleep=clock.sleep)
    paced.chat([{"role": "user", "content": "a"}])            # 3000 counted in the window
    big = [{"role": "user", "content": "x" * 18000}]           # ~6000 estimated: 3000 + 6000 > 8000 -> wait
    paced.chat(big)
    assert clock.sleeps and clock.t >= 60

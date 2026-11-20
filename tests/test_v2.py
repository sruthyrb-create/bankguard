"""Agent v2: tool filters, optional reason, action-claim and id checks, graceful step limit.

The claim-detector cases are the real v1 replies that motivated v2 (run gpt-oss-120b_on_v1)."""

import sqlite3
from pathlib import Path

from bankguard.agent import MAX_STEPS_MESSAGE, Agent, Session
from bankguard.guardrails import GuardrailConfig, Guardrails, check_action_claims, claimed_actions, unknown_txn_ids
from bankguard.llm import LLMResponse
from bankguard.tools import ToolContext, flag_unusual_charge, get_transactions, open_dispute
from bankguard.tracing import Tracer, read_trace
from tests.fakes import FakeLLM, FakePolicyIndex, KeywordScorer, call

ACC = "ACC-0001"
NBH = "‑"
V2 = GuardrailConfig(schema_check=True, grounding_check=True, day_count_check=True, id_check=True,
                     action_claim_check=True, pii_leak_check=True, action_policy=True)


def _txns(db: Path, n: int = 2) -> list[dict]:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute("SELECT * FROM transactions WHERE account_id = ? ORDER BY txn_id LIMIT ?",
                                          (ACC, n))]
    conn.close()
    return rows


def _other_txn(db: Path) -> str:
    conn = sqlite3.connect(db)
    t = conn.execute("SELECT txn_id FROM transactions WHERE account_id != ? LIMIT 1", (ACC,)).fetchone()[0]
    conn.close()
    return t


# ----------------------------------------------------------------------- tools

def test_get_transactions_filters(db: Path) -> None:
    t = _txns(db, 1)[0]
    ctx = ToolContext(db, ACC)
    assert [r["txn_id"] for r in get_transactions(ctx, ACC, transaction_id=t["txn_id"])["transactions"]] == [t["txn_id"]]
    assert get_transactions(ctx, ACC, transaction_id=_other_txn(db))["count"] == 0   # not on this account
    by_merchant = get_transactions(ctx, ACC, merchant=t["merchant"][:4].lower())["transactions"]
    assert by_merchant and all(t["merchant"][:4].lower() in r["merchant"].lower() for r in by_merchant)
    by_city = get_transactions(ctx, ACC, city=t["city"].upper())["transactions"]
    assert by_city and all(r["city"] == t["city"] for r in by_city)


def test_reason_is_optional(db: Path) -> None:
    a, b = _txns(db)
    ctx = ToolContext(db, ACC)
    assert flag_unusual_charge(ctx, ACC, a["txn_id"])["status"] == "flagged"
    assert open_dispute(ctx, ACC, b["txn_id"], None)["dispute_id"].startswith("DSP-")


# ------------------------------------------------------------- claim detection

def test_claims_from_v1_flag_04() -> None:
    reply = (f"I've flagged all five Shopwise Online charges from August 8 for you:\n\n"
             f"- TXN{NBH}040733 ( $0.82 )\n- TXN{NBH}040732 ( $1.06 )\n- TXN{NBH}040729 ( $1.98 )\n\n"
             "If you notice any other unfamiliar activity, just let me know!")
    done = {"TXN-040733", "TXN-040732"}
    assert check_action_claims(reply, done, set()) == ["says TXN-040729 was flagged"]


def test_claims_from_v1_dispute_09() -> None:
    reply = ("I've opened disputes for the five Shopwise Online charges you identified:\n\n"
             "| Transaction ID | Amount | Dispute ID |\n|---|---|---|\n"
             f"| TXN{NBH}040729 | $1.98 | DSP{NBH}FB1FB173 |\n| TXN{NBH}040733 | $0.82 | DSP{NBH}FB1FB173* |\n\n"
             f"*The first dispute (DSP{NBH}FB1FB173) was already opened for TXN{NBH}040733 when you flagged it.")
    assert check_action_claims(reply, {"TXN-040733"}, {"TXN-040729"}) == ["says TXN-040733 has a dispute"]


def test_offers_questions_and_listings_are_not_claims() -> None:
    assert claimed_actions("I see TXN-000001 for $5. Would you like me to flag it or open a dispute?") == (set(), set())
    turn1 = ("I've flagged the $0.82 charge (TXN-040733) as unusual. I see a few other small purchases that day:\n\n"
             "| Transaction ID | Amount |\n|---|---|\n| TXN-040732 | $1.06 |\n\nWould you like me to flag them too?")
    assert claimed_actions(turn1) == ({"TXN-040733"}, set())
    assert claimed_actions("I have not flagged TXN-000001 yet.") == (set(), set())
    assert claimed_actions("A dispute is already open for TXN-040760.") == (set(), {"TXN-040760"})


def test_unknown_ids() -> None:
    assert unknown_txn_ids(f"I mean TXN{NBH}030745 and TXN-000001", {"TXN-000001"}, set()) == ["TXN-030745"]
    assert unknown_txn_ids("I can't find TXN-999999 on your account.", set(), {"TXN-999999"}) == []


# ------------------------------------------------------------ agent integration

def make(db: Path, tmp_path: Path, script, **kw):
    llm = FakeLLM(script)
    tracer = Tracer("t", tmp_path / "t.jsonl")
    agent = Agent(llm, str(db), tracer, policy_index=FakePolicyIndex(),
                  guardrails=Guardrails(V2, scorer=KeywordScorer()), **kw)
    return agent, llm, tracer


def test_false_claim_is_repaired_then_summarised(db: Path, tmp_path: Path) -> None:
    a, b = _txns(db)
    lie = f"I've flagged {a['txn_id']} and {b['txn_id']}."
    agent, llm, _ = make(db, tmp_path, [
        LLMResponse("", [call("flag_unusual_charge", {"account_id": ACC, "transaction_id": a["txn_id"]})]),
        LLMResponse(lie), LLMResponse(lie)])
    r = agent.run_turn(Session(account_id=ACC), f"Flag {a['txn_id']} and {b['txn_id']}")
    assert [e["action"] for e in r.guardrail_events] == ["repair", "fallback"]
    assert f"Done so far: flagged ['{a['txn_id']}']" in llm.requests[2][-1]["content"]
    assert r.answer.startswith("To be precise") and a["txn_id"] in r.answer and b["txn_id"] not in r.answer


def test_claim_check_counts_already_open_disputes(db: Path, tmp_path: Path) -> None:
    a = _txns(db, 1)[0]
    args = {"account_id": ACC, "transaction_id": a["txn_id"]}
    agent, _, _ = make(db, tmp_path, [
        LLMResponse("", [call("open_dispute", args)]), LLMResponse("Opened."),
        LLMResponse("", [call("open_dispute", args)]), LLMResponse(f"A dispute is already open for {a['txn_id']}.")])
    session = Session(account_id=ACC)
    agent.run_turn(session, f"Yes, open a dispute for {a['txn_id']}")
    r = agent.run_turn(session, f"Yes, open another dispute for {a['txn_id']}")
    assert "already" in r.tool_calls[0]["error"] and r.guardrail_events == []


def test_id_check_repairs_an_unknown_id(db: Path, tmp_path: Path) -> None:
    a = _txns(db, 1)[0]
    other = _other_txn(db)
    agent, llm, _ = make(db, tmp_path, [LLMResponse(f"Your charge is {other}."), LLMResponse(f"It is {a['txn_id']}.")])
    r = agent.run_turn(Session(account_id=ACC), "which charge was it?")
    assert r.answer == f"It is {a['txn_id']}." and other in llm.requests[1][-1]["content"]
    assert r.guardrail_events[0]["guardrail"] == "id_check"


def test_graceful_step_limit(db: Path, tmp_path: Path) -> None:
    """Regression for run v1 off eval-flag-04: all flags made, no step left to say so."""
    rows = _txns(db, 3)
    script = [LLMResponse("", [call("flag_unusual_charge", {"account_id": ACC, "transaction_id": t["txn_id"]},
                                    f"c{i}")]) for i, t in enumerate(rows)]
    ids = ", ".join(t["txn_id"] for t in rows)
    agent, llm, tracer = make(db, tmp_path, script + [LLMResponse(f"I've flagged {ids}.")], max_steps=3,
                              graceful_step_limit=True)
    r = agent.run_turn(Session(account_id=ACC), f"Flag {ids}")
    assert r.answer == f"I've flagged {ids}." and r.error is None and r.llm_stats["step_limit"] == 1
    assert llm.tools_seen[-1] is None                         # the last call offers no tools
    assert any(e["event"] == "step_limit" for e in read_trace(tracer.path))


def test_without_graceful_limit_v1_behaviour_is_kept(db: Path, tmp_path: Path) -> None:
    rows = _txns(db, 2)
    script = [LLMResponse("", [call("flag_unusual_charge", {"account_id": ACC, "transaction_id": t["txn_id"]},
                                    f"c{i}")]) for i, t in enumerate(rows)]
    agent, _, _ = make(db, tmp_path, script, max_steps=2)
    r = agent.run_turn(Session(account_id=ACC), "flag them")
    assert r.answer == MAX_STEPS_MESSAGE and r.error == "max_steps"

"""Run eval scenarios through the agent and save one JSON record per scenario.

Each scenario runs on its own temporary copy of bank.db, so writes (flags, disputes)
never leak between scenarios or into the real database. Records go to
results/eval/runs/<run_name>/<scenario_id>.json; a scenario with a record is skipped
on the next run, which makes runs resumable across days.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bankguard.agent import Agent, Session
from bankguard.config import REPO_ROOT
from bankguard.eval.grading import grade, rule_verdict
from bankguard.eval.scenarios import Scenario
from bankguard.guardrails import Guardrails
from bankguard.tracing import Tracer

RUNS_DIR = REPO_ROOT / "results" / "eval" / "runs"
STOP_ERRORS = ("BudgetExhausted", "RateLimitError")  # stop the run; the scenario is retried next time


class RunStopped(RuntimeError):
    pass


def _db_state(db: Path, account: str) -> dict[str, Any]:
    conn = sqlite3.connect(db)
    try:
        disputes = [r[0] for r in conn.execute("SELECT transaction_id FROM disputes WHERE account_id = ?", (account,))]
        flags = [r[0] for r in conn.execute("SELECT DISTINCT transaction_id FROM flags WHERE account_id = ?",
                                            (account,))]
        own = [r[0] for r in conn.execute("SELECT txn_id FROM transactions WHERE account_id = ?", (account,))]
    finally:
        conn.close()
    return {"disputes": disputes, "flags": flags, "account_txn_ids": own}


def run_scenario(sc: Scenario, llm: Any, guardrails: Guardrails, base_db: str | Path, policy_index: Any,
                 run_name: str, max_steps: int = 6, trace_dir: Path | None = None,
                 context_budget: int | None = None) -> dict[str, Any]:
    """Run every turn of one scenario and return the graded record (raises RunStopped on budget/429)."""
    tmp = Path(tempfile.mkdtemp(prefix="bankguard-eval-"))
    db = tmp / "bank.db"
    shutil.copy(base_db, db)
    trace_path = (trace_dir or REPO_ROOT / "traces" / "eval" / run_name) / f"{sc.id}.jsonl"
    trace_path.unlink(missing_ok=True)
    tracer = Tracer(f"eval-{run_name}-{sc.id}", trace_path)
    agent = Agent(llm, str(db), tracer, max_steps, policy_index=policy_index, guardrails=guardrails,
                  context_budget=context_budget)
    session = Session(account_id=sc.account)
    turns = []
    try:
        for text in sc.turns:
            r = agent.run_turn(session, text)
            if r.error and r.error.startswith(STOP_ERRORS):
                raise RunStopped(r.error)
            turns.append({"user": text, **asdict(r)})
        state = _db_state(db, sc.account)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    run = {"turns": turns, "retrieved_ids": sorted(session.retrieved_ids), **state}
    checks = grade(sc, run)
    return {
        "scenario_id": sc.id, "category": sc.category, "account": sc.account, "run": run_name,
        "model": llm.model, "guardrails": guardrails.config.enabled(),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "turns": turns, "disputes": state["disputes"], "flags": state["flags"],
        "retrieved_ids": run["retrieved_ids"], "checks": checks, "rule_pass": rule_verdict(checks),
    }


def record_path(run_name: str, scenario_id: str) -> Path:
    return RUNS_DIR / run_name / f"{scenario_id}.json"


def save_record(record: dict[str, Any]) -> Path:
    path = record_path(record["run"], record["scenario_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def load_records(run_name: str) -> list[dict[str, Any]]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((RUNS_DIR / run_name).glob("*.json"))]

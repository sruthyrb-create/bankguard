"""Talk to the agent from the terminal, or run a scripted multi-turn demo.

  python scripts/chat_cli.py --backend groq --account ACC-0001     # interactive
  python scripts/chat_cli.py --backend groq --demo                 # scripted 3-turn demo
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bankguard.agent import Agent, Session  # noqa: E402
from bankguard.config import REPO_ROOT, load_yaml  # noqa: E402
from bankguard.llm import make_llm  # noqa: E402
from bankguard.tracing import Tracer  # noqa: E402


def demo_turns(db_path: str) -> tuple[str, list[str]]:
    """Pick an account with an injected duplicate charge (ground truth used only to choose the demo)."""
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT account_id, merchant, substr(ts, 1, 10), txn_id FROM transactions "
        "WHERE anomaly_type = 'duplicate' ORDER BY txn_id LIMIT 1").fetchone()
    conn.close()
    account, merchant, day, dup_txn = row
    return account, [
        f"I think I was charged twice at {merchant} on {day}. Can you check?",
        f"Yes, please open a dispute for {dup_txn}, the second charge.",
        "Under Regulation E, how long does the bank have to investigate?",
    ]


def show(result) -> None:
    for c in result.tool_calls:
        print(f"   [tool] {c['tool']}({c['args']}){'  ERROR: ' + c['error'] if c['error'] else ''}")
    print(f"agent> {result.answer}")
    print(f"   ({result.llm_calls} LLM calls, {result.tokens_in}+{result.tokens_out} tokens, "
          f"{result.latency_ms} ms{', error: ' + result.error if result.error else ''})\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["groq", "local"], default="groq")
    p.add_argument("--model", default=None, help="override the model from configs/models.yaml")
    p.add_argument("--account", default="ACC-0001")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()

    cfg = load_yaml("models")
    db = str(REPO_ROOT / cfg["data"]["db_path"])
    if not Path(db).exists():
        sys.exit("bank.db not found; run `make data` first")
    llm = make_llm(args.backend, args.model, use_cache=not args.no_cache)
    tracer = Tracer(f"cli-{args.backend}-{datetime.now():%Y%m%d-%H%M%S}")
    from bankguard.tools import _default_policy_index
    policy_index = _default_policy_index()  # load embeddings once, before timing any turn
    agent = Agent(llm, db, tracer, cfg["defaults"]["max_steps"], policy_index=policy_index)

    if args.demo:
        account, turns = demo_turns(db)
        session = Session(account_id=account)
        print(f"Demo on {account} with {llm.model}\n")
        for text in turns:
            print(f"user > {text}")
            show(agent.run_turn(session, text))
    else:
        session = Session(account_id=args.account)
        print(f"Chatting as {args.account} with {llm.model}. Empty line to quit.\n")
        while text := input("user > ").strip():
            show(agent.run_turn(session, text))
    print(f"Trace: {tracer.path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

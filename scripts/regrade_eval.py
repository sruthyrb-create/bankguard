"""Re-apply the current rule graders to saved run records (no API calls).

  python scripts/regrade_eval.py --run gpt-oss-120b_on_v1            # show what would change
  python scripts/regrade_eval.py --run gpt-oss-120b_on_v1 --write    # save the new checks

Use this when a grader is fixed, so every run is graded by the same rules. Each record keeps the
previous result under "checks_before_regrade" the first time it is regraded.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bankguard.config import REPO_ROOT, load_yaml  # noqa: E402
from bankguard.eval.grading import grade, rule_verdict  # noqa: E402
from bankguard.eval.runner import RUNS_DIR, record_path  # noqa: E402
from bankguard.eval.scenarios import SETS, load_scenarios  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--write", action="store_true")
    args = p.parse_args()
    scenarios = {s.id: s for name in SETS for s in load_scenarios(scenario_set=name)}
    conn = sqlite3.connect(REPO_ROOT / load_yaml("models")["data"]["db_path"])
    paths = sorted((RUNS_DIR / args.run).glob("*.json"))
    if not paths:
        sys.exit(f"No records in results/eval/runs/{args.run}/")
    changed = 0
    for path in paths:
        rec = json.loads(path.read_text(encoding="utf-8"))
        sc = scenarios[rec["scenario_id"]]
        own = [r[0] for r in conn.execute("SELECT txn_id FROM transactions WHERE account_id = ?", (sc.account,))]
        run = {"turns": rec["turns"], "retrieved_ids": rec["retrieved_ids"], "disputes": rec["disputes"],
               "flags": rec["flags"], "account_txn_ids": own}
        checks = grade(sc, run)
        old = {c["check"]: c["passed"] for c in rec["checks"]}
        new = {c["check"]: c["passed"] for c in checks}
        if old != new:
            changed += 1
            diff = {k: f"{old.get(k)} -> {new.get(k)}" for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)}
            print(f"{rec['scenario_id']:22s} rule_pass {rec['rule_pass']} -> {rule_verdict(checks)}  {diff}")
            if args.write:
                rec.setdefault("checks_before_regrade", rec["checks"])
                rec["checks"], rec["rule_pass"] = checks, rule_verdict(checks)
                record_path(args.run, rec["scenario_id"]).write_text(
                    json.dumps(rec, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"{changed} of {len(paths)} records change{'d' if args.write else ' (dry run; add --write to save)'}.")


if __name__ == "__main__":
    main()

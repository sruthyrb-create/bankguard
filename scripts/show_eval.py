"""Print eval transcripts: turns, tool calls, guardrail actions (with rejected answers), failed checks, judge.

  python scripts/show_eval.py --run gpt-oss-120b_on                    # every failed scenario
  python scripts/show_eval.py --run gpt-oss-120b_on --ids eval-policy-01 eval-flag-05
  python scripts/show_eval.py --run gpt-oss-120b_on --all
No API calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bankguard.eval.report import success  # noqa: E402
from bankguard.eval.runner import load_records  # noqa: E402
from judge_eval import load_verdicts  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--ids", nargs="*")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()
    verdicts = load_verdicts(args.run)
    for r in load_records(args.run):
        v = verdicts.get(r["scenario_id"])
        if args.ids and r["scenario_id"] not in args.ids:
            continue
        if not (args.ids or args.all) and success(r, v) is not False:
            continue
        print("=" * 100)
        print(r["scenario_id"], "| success:", success(r, v))
        for t in r["turns"]:
            print("USER:", t["user"])
            for c in t["tool_calls"]:
                print("  TOOL:", c["tool"], json.dumps(c["args"], ensure_ascii=False),
                      f"ERROR: {c['error']}" if c.get("error") else "")
            for e in t.get("guardrail_events", []):
                if e.get("action") != "pass":
                    print("  GUARDRAIL:", {k: v for k, v in e.items() if k != "rejected"})
                    if e.get("rejected"):
                        print("    rejected answer:", e["rejected"].replace("\n", " ")[:400])
            print("AGENT:", t["answer"])
        print("DB: disputes", r["disputes"], "flags", r["flags"])
        print("FAILED CHECKS:", [(c["check"], c["detail"]) for c in r["checks"] if c["passed"] is False])
        print("JUDGE:", v)


if __name__ == "__main__":
    main()

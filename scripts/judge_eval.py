"""Phase 5: LLM-judge every record of a run (resumable; verdicts cached per scenario).

  python scripts/judge_eval.py --run gpt-oss-120b_on
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bankguard.config import REPO_ROOT, load_yaml  # noqa: E402
from bankguard.eval.judge import judge_messages, parse_verdict  # noqa: E402
from bankguard.eval.pacing import BudgetExhausted, PacedLLM, UsageLedger  # noqa: E402
from bankguard.eval.runner import load_records  # noqa: E402
from bankguard.eval.scenarios import load_scenarios  # noqa: E402
from bankguard.llm import make_llm  # noqa: E402

JUDGE_DIR = REPO_ROOT / "results" / "eval" / "judge"


def verdict_path(run: str, scenario_id: str) -> Path:
    return JUDGE_DIR / run / f"{scenario_id}.json"


def load_verdicts(run: str) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((JUDGE_DIR / run).glob("*.json"))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    cfg, models = load_yaml("eval"), load_yaml("models")
    refs = {s.id: s.reference for s in load_scenarios(REPO_ROOT / cfg["scenarios"])}
    records = load_records(args.run)
    if not records:
        sys.exit(f"No records in results/eval/runs/{args.run}/")
    todo = [r for r in records if args.force or not verdict_path(args.run, r["scenario_id"]).exists()]
    print(f"Judging {len(todo)} of {len(records)} records of {args.run}")
    llm = make_llm("groq", models["groq"]["judge_model"], extra=cfg["judge"].get("extra") or {})
    ft, pc = cfg["free_tier"], cfg["pacing"]
    paced = PacedLLM(llm, ft["tpm"], ft["rpm"], pc["daily_budget"], pc["reserve_tokens"], UsageLedger())
    for r in todo:
        try:
            resp = paced.chat(judge_messages(r, refs[r["scenario_id"]]))
        except BudgetExhausted as exc:
            print(f"Stopped: {exc}. Run again to resume.")
            break
        try:
            verdict = parse_verdict(resp.content)
        except ValueError as exc:
            print(f"{r['scenario_id']}: could not parse the judge reply ({exc}); skipped")
            continue
        verdict |= {"judge_model": llm.model, "tokens_in": resp.tokens_in, "tokens_out": resp.tokens_out}
        path = verdict_path(args.run, r["scenario_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(verdict, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{r['scenario_id']:18s} correct={verdict['correct']!s:5s} refused={verdict['refused']!s:5s} "
              f"{verdict['reason'][:90]}")


if __name__ == "__main__":
    main()

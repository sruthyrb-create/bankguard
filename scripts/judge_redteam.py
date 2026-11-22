"""Phase 6: LLM-judge the attacks whose goal rules can't decide (success_if has "judge"); resumable.

  python scripts/judge_redteam.py --run gpt-oss-120b_on_v2_redteam

Attacks stopped by an input guardrail are not sent to the judge: no model reply exists, so the verdict is
"no" without an API call (recorded with judge_model "rule:input_block").
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bankguard.config import load_yaml  # noqa: E402
from bankguard.eval.pacing import BudgetExhausted, PacedLLM, UsageLedger  # noqa: E402
from bankguard.eval.redteam import (judge_attack_messages, load_attacks, load_records,  # noqa: E402
                                    parse_attack_verdict, verdict_path)
from bankguard.llm import make_llm  # noqa: E402
from openai import RateLimitError  # noqa: E402


def save(run: str, attack_id: str, verdict: dict) -> None:
    path = verdict_path(run, attack_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(verdict, indent=1, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    attacks = {a.id: a for a in load_attacks()}
    records = [r for r in load_records(args.run) if attacks[r["scenario_id"]].success_if.get("judge")]
    if not records:
        sys.exit(f"No judge-graded records in results/redteam/runs/{args.run}/")
    todo = [r for r in records if args.force or not verdict_path(args.run, r["scenario_id"]).exists()]
    print(f"Judging {len(todo)} of {len(records)} judge-graded attacks of {args.run}")

    cfg, models = load_yaml("eval"), load_yaml("models")
    llm = make_llm("groq", models["groq"]["judge_model"], extra=cfg["judge"].get("extra") or {})
    ft, pc, jc = cfg["free_tier"], cfg["pacing"], cfg["judge"]
    paced = PacedLLM(llm, ft["tpm"], ft["rpm"], pc["daily_budget"], pc["reserve_tokens"], UsageLedger(),
                     otpm=jc.get("otpm"), max_output_tokens=(jc.get("extra") or {}).get("max_tokens"))
    for r in todo:
        aid = r["scenario_id"]
        if all(t.get("blocked_by") for t in r["turns"]):
            save(args.run, aid, {"complied": False, "reason": "every turn was refused by an input guardrail",
                                 "judge_model": "rule:input_block"})
            print(f"{aid:24s} complied=False (input block, no API call)")
            continue
        try:
            resp = paced.chat(judge_attack_messages(r, attacks[aid].goal))
        except (BudgetExhausted, RateLimitError) as exc:
            print(f"Stopped: {str(exc)[:200]}. Run again to resume (verdicts so far are saved).")
            break
        try:
            verdict = parse_attack_verdict(resp.content)
        except ValueError as exc:
            print(f"{aid}: could not parse the judge reply ({exc}); skipped")
            continue
        save(args.run, aid, verdict | {"judge_model": llm.model, "tokens_in": resp.tokens_in,
                                       "tokens_out": resp.tokens_out})
        print(f"{aid:24s} complied={verdict['complied']!s:5s} {verdict['reason'][:90]}")


if __name__ == "__main__":
    main()

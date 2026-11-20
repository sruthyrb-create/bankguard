"""Phase 5: run eval scenarios through the agent on Groq, inside the free tier.

  python scripts/run_eval.py --guardrails on --pilot      # the 10 pilot scenarios
  python scripts/run_eval.py --guardrails on              # all 60 (skips finished ones; resumable)
  python scripts/run_eval.py --guardrails off             # same scenarios, every guardrail off
  python scripts/run_eval.py --guardrails on --set heldout   # the 20 held-out scenarios
  options: --ids eval-policy-01 eval-policy-02   --force (re-run finished ones)

Stops cleanly when the daily token budget in configs/eval.yaml is reached or Groq
keeps returning 429; run the same command again (next UTC day for the budget).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bankguard.config import REPO_ROOT, load_yaml  # noqa: E402
from bankguard.eval.pacing import PacedLLM, UsageLedger  # noqa: E402
from bankguard.eval.runner import RunStopped, record_path, run_scenario, save_record  # noqa: E402
from bankguard.eval.scenarios import load_scenarios  # noqa: E402
from bankguard.guardrails import GuardrailConfig, Guardrails  # noqa: E402
from bankguard.llm import make_llm  # noqa: E402


def run_name(model: str, guardrails: str, version: str, scenario_set: str = "main") -> str:
    suffix = "" if scenario_set == "main" else f"_{scenario_set}"
    return f"{model.split('/')[-1]}_{guardrails}_{version}{suffix}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--guardrails", choices=["on", "off"], required=True)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--set", default="main", choices=["main", "heldout"], dest="scenario_set")
    p.add_argument("--ids", nargs="*")
    p.add_argument("--force", action="store_true", help="re-run scenarios that already have a record")
    p.add_argument("--backend", default="groq")
    args = p.parse_args()

    cfg, models = load_yaml("eval"), load_yaml("models")
    scenarios = load_scenarios(scenario_set=args.scenario_set)
    wanted = set(args.ids or (cfg["pilot_ids"] if args.pilot else [s.id for s in scenarios]))
    unknown = wanted - {s.id for s in scenarios}
    if unknown:
        sys.exit(f"Unknown scenario ids: {sorted(unknown)}")
    scenarios = [s for s in scenarios if s.id in wanted]

    llm = make_llm(args.backend)
    name = run_name(llm.model, args.guardrails, cfg["agent_version"], args.scenario_set)
    todo = [s for s in scenarios if args.force or not record_path(name, s.id).exists()]
    print(f"Run {name}: {len(scenarios)} selected, {len(scenarios) - len(todo)} already done, {len(todo)} to run")
    if not todo:
        return

    from bankguard.tools import _default_policy_index
    guards = Guardrails.from_config() if args.guardrails == "on" else Guardrails(GuardrailConfig.all_off())
    ft, pc = cfg["free_tier"], cfg["pacing"]
    ledger = UsageLedger()
    paced = PacedLLM(llm, ft["tpm"], ft["rpm"], pc["daily_budget"], pc["reserve_tokens"], ledger)
    db = REPO_ROOT / models["data"]["db_path"]
    index = _default_policy_index()
    print(f"Guardrails on: {', '.join(guards.config.enabled()) or 'none'}; "
          f"{ledger.used(llm.model)} counted tokens already used today (UTC) on {llm.model}\n")

    done = 0
    for sc in todo:
        t0 = time.perf_counter()
        used_before = ledger.used(llm.model)
        try:
            rec = run_scenario(sc, paced, guards, db, index, name, models["defaults"]["max_steps"],
                               context_budget=models["defaults"].get("context_budget_tokens"))
        except RunStopped as exc:
            print(f"\nStopped before finishing {sc.id}: {exc}\nRun the same command again to resume.")
            break
        save_record(rec)
        done += 1
        stats = [t.get("llm_stats") or {} for t in rec["turns"]]
        tin = sum(t["tokens_in"] for t in rec["turns"])
        tout = sum(t["tokens_out"] for t in rec["turns"])
        cached = sum(s.get("cached_tokens", 0) for s in stats)
        failed = [c["check"] for c in rec["checks"] if c["passed"] is False]
        print(f"{sc.id:18s} rules {'PASS' if rec['rule_pass'] else 'FAIL'}"
              f"{' ' + str(failed) if failed else '':40s} | {sum(t['llm_calls'] for t in rec['turns'])} calls, "
              f"in {tin} (provider-cached {cached}), out {tout}, counted {ledger.used(llm.model) - used_before} | "
              f"{time.perf_counter() - t0:.1f} s")
    print(f"\n{done} scenarios saved to results/eval/runs/{name}/. Counted tokens today (UTC): "
          f"{ledger.used(llm.model)} / budget {pc['daily_budget']}. Pacing waits: {paced.waited_s:.0f} s.")


if __name__ == "__main__":
    main()

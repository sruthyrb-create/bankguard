"""Phase 6: run the 40 red-team attacks through agent v2 on Groq, inside the free tier.

  python scripts/run_redteam.py --guardrails on     # resumable: skips attacks that already have a record
  python scripts/run_redteam.py --guardrails off
  options: --ids rt-direct-01 rt-pii-02   --force (re-run finished ones)

Records go to results/redteam/runs/<model>_<on|off>_<agent_version>_redteam/. The benign look-alikes run
through the normal eval runner: python scripts/run_eval.py --guardrails on --set redteam_benign
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bankguard.config import REPO_ROOT, load_yaml  # noqa: E402
from bankguard.eval.pacing import PacedLLM, UsageLedger  # noqa: E402
from bankguard.eval.redteam import load_attacks, record_path, run_attack, save_record  # noqa: E402
from bankguard.eval.runner import RunStopped  # noqa: E402
from bankguard.guardrails import GuardrailConfig, Guardrails  # noqa: E402
from bankguard.llm import make_llm  # noqa: E402


def run_name(model: str, guardrails: str, version: str) -> str:
    return f"{model.split('/')[-1]}_{guardrails}_{version}_redteam"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--guardrails", choices=["on", "off"], required=True)
    p.add_argument("--ids", nargs="*")
    p.add_argument("--force", action="store_true")
    p.add_argument("--backend", default="groq")
    args = p.parse_args()

    cfg, models = load_yaml("eval"), load_yaml("models")
    attacks = load_attacks()
    if args.ids:
        unknown = set(args.ids) - {a.id for a in attacks}
        if unknown:
            sys.exit(f"Unknown attack ids: {sorted(unknown)}")
        attacks = [a for a in attacks if a.id in set(args.ids)]

    llm = make_llm(args.backend)
    name = run_name(llm.model, args.guardrails, cfg["agent_version"])
    todo = [a for a in attacks if args.force or not record_path(name, a.id).exists()]
    print(f"Run {name}: {len(attacks)} selected, {len(attacks) - len(todo)} already done, {len(todo)} to run")
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
    for a in todo:
        t0 = time.perf_counter()
        used_before = ledger.used(llm.model)
        try:
            rec = run_attack(a, paced, guards, db, index, name, models["defaults"]["max_steps"],
                             context_budget=models["defaults"].get("context_budget_tokens"),
                             graceful_step_limit=models["defaults"].get("graceful_step_limit", False))
        except RunStopped as exc:
            print(f"\nStopped before finishing {a.id}: {exc}\nRun the same command again to resume.")
            break
        save_record(rec)
        done += 1
        o = rec["outcome"]
        state = {True: "SUCCEEDED", False: "stopped", None: "judge?"}[o["succeeded"]]
        print(f"{a.id:24s} {state:9s} complied={o['complied']!s:5s} {o['stopped_by'] or '':32s} | "
              f"{sum(t['llm_calls'] for t in rec['turns'])} calls, counted {ledger.used(llm.model) - used_before} | "
              f"{time.perf_counter() - t0:.1f} s")
    print(f"\n{done} attacks saved to results/redteam/runs/{name}/. Counted tokens today (UTC): "
          f"{ledger.used(llm.model)} / budget {pc['daily_budget']}. Pacing waits: {paced.waited_s:.0f} s.")


if __name__ == "__main__":
    main()

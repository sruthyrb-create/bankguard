"""Phase 6: aggregate the red-team runs into results/redteam_report.json and print the comparison.

  python scripts/report_redteam.py

Reads attack runs from results/redteam/runs/ (+ judge verdicts), the benign look-alike runs from
results/eval/runs/*_redteam_benign (+ the Phase 5 judge's verdicts) and, if present,
results/redteam_classifier.json. Outcomes are re-graded from the saved records, so no API calls are made.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bankguard.config import REPO_ROOT  # noqa: E402
from bankguard.eval.redteam import RT_RUNS, load_attacks, load_records, load_verdicts, summarise_attacks  # noqa: E402
from bankguard.eval.report import summarise  # noqa: E402
from bankguard.eval.runner import RUNS_DIR  # noqa: E402
from bankguard.eval.runner import load_records as load_eval_records  # noqa: E402
from judge_eval import load_verdicts as load_eval_verdicts  # noqa: E402

OUT = REPO_ROOT / "results" / "redteam_report.json"
CLASSIFIER = REPO_ROOT / "results" / "redteam_classifier.json"


def main() -> None:
    attacks = {a.id: a for a in load_attacks()}
    runs = sorted(p.name for p in RT_RUNS.iterdir() if p.is_dir()) if RT_RUNS.exists() else []
    benign_runs = sorted(p.name for p in RUNS_DIR.glob("*_redteam_benign") if p.is_dir()) if RUNS_DIR.exists() else []
    if not runs and not benign_runs:
        sys.exit("No red-team runs yet; run scripts/run_redteam.py and scripts/run_eval.py --set redteam_benign.")
    report = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "attack_runs": {},
              "benign_runs": {}}
    for run in runs:
        report["attack_runs"][run] = summarise_attacks(load_records(run), attacks, load_verdicts(run))
    for run in benign_runs:
        s = summarise(load_eval_records(run), load_eval_verdicts(run))
        report["benign_runs"][run] = {k: s[k] for k in ("n_scenarios", "false_refusals", "success", "by_category",
                                                        "guardrail_events", "cost", "failed")}
    if CLASSIFIER.exists():
        c = json.loads(CLASSIFIER.read_text(encoding="utf-8"))
        report["classifier"] = {k: c[k] for k in ("classifier", "threshold", "attacks_typed_in_chat",
                                                  "indirect_attacks", "benign_lookalikes", "by_category")}
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if runs:
        print(f"{'':28s}" + "".join(f"{r:>32s}" for r in runs))
        rows = [("attack success (ASR)", lambda s: f"{s['attack_success']['n']}/{s['attack_success']['of']} "
                                                   f"({s['attack_success']['rate']}) pending {s['attack_success']['pending_judge']}"),
                ("model complied", lambda s: f"{s['model_complied']['n']}/{s['model_complied']['of']}"),
                ("complied but stopped", lambda s: len(s["complied_but_stopped"])),
                ("tokens in/out", lambda s: f"{s['tokens']['in']}/{s['tokens']['out']}")]
        for label, fn in rows:
            print(f"{label:28s}" + "".join(f"{str(fn(report['attack_runs'][r])):>32s}" for r in runs))
        cats = sorted({c for r in runs for c in report["attack_runs"][r]["by_category"]})
        print("\nASR by category (succeeded/n, model complied):")
        for c in cats:
            cells = []
            for r in runs:
                v = report["attack_runs"][r]["by_category"].get(c)
                cells.append(f"{v['succeeded']}/{v['n']}, complied {v['complied']}" +
                             (f", pending {v['pending_judge']}" if v["pending_judge"] else "") if v else "-")
            print(f"  {c:26s}" + "".join(f"{x:>32s}" for x in cells))
        for r in runs:
            s = report["attack_runs"][r]
            print(f"\n{r} stopped by: {s['stopped_by']}")
            print(f"{r} succeeded: {s['succeeded_ids']}")
            print(f"{r} complied but stopped by a later layer: {s['complied_but_stopped']}")
            for a in s["attacks"]:
                if a["succeeded"] is not False:
                    fired = {k: v["detail"] for k, v in a["signals"].items() if v["succeeded"] is not False}
                    print(f"  {a['id']:24s} {a['succeeded']}  {fired}")
    for r in benign_runs:
        fr = report["benign_runs"][r]["false_refusals"]
        print(f"\n{r}: false refusals {fr['n']}/{fr['of']} {fr['by_kind'] or ''}; "
              f"success {report['benign_runs'][r]['success']}")
        for d in fr["detail"]:
            print(f"  NO ANSWER {d}")
    if "classifier" in report:
        c = report["classifier"]
        print(f"\nInput guardrails alone (local scoring): typed attacks caught by the classifier "
              f"{c['attacks_typed_in_chat']['classifier_caught']}/{c['attacks_typed_in_chat']['n']}, by the input "
              f"pipeline {c['attacks_typed_in_chat']['input_pipeline_caught']}/{c['attacks_typed_in_chat']['n']}; "
              f"benign look-alikes blocked {c['benign_lookalikes']['input_pipeline_blocked']}/{c['benign_lookalikes']['n']}")
    print(f"\nWrote {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

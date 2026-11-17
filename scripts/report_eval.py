"""Phase 5: aggregate every run into results/eval_report.json and print a comparison.

  python scripts/report_eval.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bankguard.config import REPO_ROOT  # noqa: E402
from bankguard.eval.report import summarise  # noqa: E402
from bankguard.eval.runner import RUNS_DIR, load_records  # noqa: E402
from judge_eval import load_verdicts  # noqa: E402

OUT = REPO_ROOT / "results" / "eval_report.json"


def main() -> None:
    runs = sorted(p.name for p in RUNS_DIR.iterdir() if p.is_dir()) if RUNS_DIR.exists() else []
    if not runs:
        sys.exit("No runs yet; run scripts/run_eval.py first.")
    report = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "runs": {}}
    for run in runs:
        report["runs"][run] = summarise(load_records(run), load_verdicts(run))
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = [("scenarios", lambda s: s["n_scenarios"]),
            ("success", lambda s: f"{s['success']['n']} ({s['success']['rate']}) pending judge {s['success']['pending_judge']}"),
            ("rule pass rate", lambda s: s["rule_pass_rate"]),
            ("false refusals", lambda s: f"{s['false_refusals']['n']}/{s['false_refusals']['of']} "
                                         f"{s['false_refusals']['by_kind'] or ''}"),
            ("correct refusals", lambda s: f"{s['correct_refusals']['n']}/{s['correct_refusals']['of']}"),
            ("tool accuracy", lambda s: s["tool_accuracy"]),
            ("invalid tool calls", lambda s: f"{s['invalid_tool_calls']['n']}/{s['invalid_tool_calls']['of']}"),
            ("grounded policy", lambda s: f"{s['grounded_policy_answers']['n']}/{s['grounded_policy_answers']['of']}"),
            ("day-count errors", lambda s: s["day_count_errors"]["n"]),
            ("tokens/scenario", lambda s: s["cost"]["tokens_per_scenario_mean"]),
            ("provider-cached", lambda s: s["cost"]["provider_cached_tokens"]),
            ("turn LLM p50/p95 ms", lambda s: f"{s['latency_ms']['turn_llm_p50']} / {s['latency_ms']['turn_llm_p95']}")]
    print(f"{'':22s}" + "".join(f"{r:>28s}" for r in runs))
    for label, fn in rows:
        print(f"{label:22s}" + "".join(f"{str(fn(report['runs'][r])):>28s}" for r in runs))
    for run in runs:
        s = report["runs"][run]
        print(f"\n{run} by category: " + ", ".join(f"{k} {v['success']}/{v['n']}" for k, v in s["by_category"].items()))
        print(f"{run} guardrail events: {s['guardrail_events']}")
        for d in s["false_refusals"]["detail"]:
            print(f"  NO ANSWER {d['id']}: {d}")
        for f in s["failed"]:
            print(f"  FAIL {f['id']}: {f['failed_checks']} {('judge: ' + f['judge']['reason']) if f['judge'] else ''}")
    print(f"\nWrote {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

"""Phase 5: export a blind sample of judged items for hand-labelling.

  python scripts/export_labels.py --runs gpt-oss-120b_on gpt-oss-120b_off

Writes data/labels/judge_labels.csv. Fill in human_correct and human_refused with
yes/no (and notes if useful), then run scripts/judge_agreement.py. The sheet does not
show the judge's verdict or the rule results, so your labels stay independent.
The sample is stratified by category and fixed by the seed in configs/eval.yaml.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bankguard.config import REPO_ROOT, load_yaml  # noqa: E402
from bankguard.eval.judge import conversation_text  # noqa: E402
from bankguard.eval.runner import load_records  # noqa: E402
from bankguard.eval.scenarios import load_scenarios  # noqa: E402

FIELDS = ["label_id", "run", "scenario_id", "category", "reference", "conversation",
          "human_correct", "human_refused", "notes"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--force", action="store_true", help="overwrite an existing sheet (loses your labels)")
    args = p.parse_args()
    cfg = load_yaml("eval")["labels"]
    out = REPO_ROOT / cfg["path"]
    if out.exists() and not args.force:
        sys.exit(f"{out.relative_to(REPO_ROOT)} exists; use --force to overwrite it")
    refs = {s.id: s.reference for s in load_scenarios(REPO_ROOT / load_yaml("eval")["scenarios"])}
    pool = [r for run in args.runs for r in load_records(run)]
    rng = random.Random(cfg["seed"])
    by_cat: dict[str, list] = {}
    for r in pool:
        by_cat.setdefault(r["category"], []).append(r)
    for items in by_cat.values():
        rng.shuffle(items)
    sample = []
    while len(sample) < min(cfg["n"], len(pool)):  # round-robin over categories
        for cat in sorted(by_cat):
            if by_cat[cat] and len(sample) < cfg["n"]:
                sample.append(by_cat[cat].pop())
    rng.shuffle(sample)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for i, r in enumerate(sample, 1):
            w.writerow({"label_id": f"L{i:02d}", "run": r["run"], "scenario_id": r["scenario_id"],
                        "category": r["category"], "reference": refs[r["scenario_id"]],
                        "conversation": conversation_text(r), "human_correct": "", "human_refused": "", "notes": ""})
    print(f"Wrote {len(sample)} items to {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

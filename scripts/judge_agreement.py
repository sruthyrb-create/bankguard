"""Phase 5: agreement between your hand labels and the LLM judge.

  python scripts/judge_agreement.py

Reads data/labels/judge_labels.csv (human_correct / human_refused = yes or no) and the
judge verdicts, and writes results/judge_agreement.json with raw agreement, Cohen's
kappa and the confusion matrix for each question. Rows without a human label are skipped.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bankguard.config import REPO_ROOT, load_yaml  # noqa: E402
from judge_eval import load_verdicts  # noqa: E402

OUT = REPO_ROOT / "results" / "judge_agreement.json"


def cohen_kappa(a: list[bool], b: list[bool]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else round((po - pe) / (1 - pe), 4)


def agreement(pairs: list[tuple[bool, bool]]) -> dict:
    h, j = [p[0] for p in pairs], [p[1] for p in pairs]
    return {"n": len(pairs), "raw_agreement": round(sum(x == y for x, y in pairs) / len(pairs), 4) if pairs else None,
            "cohen_kappa": cohen_kappa(h, j),
            "confusion": {"human_yes_judge_yes": sum(x and y for x, y in pairs),
                          "human_yes_judge_no": sum(x and not y for x, y in pairs),
                          "human_no_judge_yes": sum(not x and y for x, y in pairs),
                          "human_no_judge_no": sum(not x and not y for x, y in pairs)}}


def main() -> None:
    path = REPO_ROOT / load_yaml("eval")["labels"]["path"]
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    verdicts = {}
    out: dict = {"labels": str(path.relative_to(REPO_ROOT)), "questions": {}, "disagreements": []}
    for q in ("correct", "refused"):
        pairs = []
        for r in rows:
            human = r[f"human_{q}"].strip().lower()
            if human not in ("yes", "no"):
                continue
            v = verdicts.setdefault(r["run"], load_verdicts(r["run"])).get(r["scenario_id"])
            if v is None:
                continue
            pairs.append((human == "yes", bool(v[q])))
            if (human == "yes") != bool(v[q]):
                out["disagreements"].append({"label_id": r["label_id"], "scenario_id": r["scenario_id"],
                                             "question": q, "human": human, "judge": v[q], "judge_reason": v["reason"]})
        out["questions"][q] = agreement(pairs)
    if not out["questions"]["correct"]["n"]:
        sys.exit("No labelled rows yet: fill human_correct / human_refused with yes or no.")
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for q, a in out["questions"].items():
        print(f"{q:8s} n={a['n']} agreement={a['raw_agreement']} kappa={a['cohen_kappa']} {a['confusion']}")
    for d in out["disagreements"]:
        print(f"  {d['label_id']} {d['scenario_id']} {d['question']}: you={d['human']} judge={d['judge']} "
              f"({d['judge_reason'][:80]})")
    print(f"Wrote {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

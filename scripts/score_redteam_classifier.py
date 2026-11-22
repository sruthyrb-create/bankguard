"""Phase 6: held-out measurement of the input guardrails alone (local, no API calls, no Groq tokens).

  python scripts/score_redteam_classifier.py

Scores every user turn of the 40 attacks and the 20 benign look-alikes with the injection classifier at the
threshold in configs/guardrails.yaml (tuned on the dev set in Phase 4, unchanged here), then runs the full
input pipeline (PII masking -> classifier -> topic filter). A conversation counts as caught if any of its
turns would be blocked, which is what happens at run time.

For indirect attacks the injection sits in a transaction memo that the input guardrails never see; their
memo text is also scored, reported separately as "if tool output were scanned" (hypothetical, not a layer
that exists).

Writes results/redteam_classifier.json.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bankguard.config import REPO_ROOT, load_yaml  # noqa: E402
from bankguard.eval.redteam import load_attacks  # noqa: E402
from bankguard.eval.scenarios import load_scenarios  # noqa: E402
from bankguard.guardrails import GuardrailConfig, Guardrails  # noqa: E402
from bankguard.guardrails.injection import HFInjectionClassifier  # noqa: E402

OUT = REPO_ROOT / "results" / "redteam_classifier.json"


def main() -> None:
    cfg = GuardrailConfig.from_yaml(load_yaml("guardrails"))
    threshold = float(cfg.injection_threshold)  # type: ignore[arg-type]
    scorer = HFInjectionClassifier(cfg.injection_model, cfg.injection_revision)
    pipeline = Guardrails(cfg, scorer)  # masking -> classifier -> topic filter, as at run time

    def conversation(id_: str, account: str, turns: list[str]) -> dict:
        per_turn = []
        for t in turns:
            score = scorer.score(t)
            d = pipeline.check_input(t, account)
            per_turn.append({"score": score, "classifier_block": score >= threshold, "pipeline": d.blocked_by})
        return {"id": id_, "max_score": max(x["score"] for x in per_turn),
                "classifier_caught": any(x["classifier_block"] for x in per_turn),
                "input_caught_by": next((x["pipeline"] for x in per_turn if x["pipeline"]), None),
                "turns": per_turn}

    attacks = load_attacks()
    rows, cat = [], defaultdict(lambda: {"n": 0, "classifier": 0, "input_pipeline": 0})
    for a in attacks:
        r = conversation(a.id, a.account, a.turns) | {"category": a.category, "technique": a.technique}
        memos = (a.seed.get("memos") or {}).values()
        if memos:
            r["memo_scores"] = [scorer.score(m) for m in memos]
            r["memo_would_be_caught"] = any(s >= threshold for s in r["memo_scores"])
        rows.append(r)
        c = cat[a.category]
        c["n"] += 1
        c["classifier"] += r["classifier_caught"]
        c["input_pipeline"] += r["input_caught_by"] is not None

    benign = [conversation(s.id, s.account, s.turns) for s in load_scenarios(scenario_set="redteam_benign")]
    direct = [r for r in rows if r["category"] != "indirect_injection"]
    indirect = [r for r in rows if r["category"] == "indirect_injection"]
    out = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "classifier": scorer.name, "threshold": threshold,
        "attacks_typed_in_chat": {
            "n": len(direct),
            "classifier_caught": sum(r["classifier_caught"] for r in direct),
            "input_pipeline_caught": sum(r["input_caught_by"] is not None for r in direct)},
        "indirect_attacks": {
            "n": len(indirect),
            "input_pipeline_caught": sum(r["input_caught_by"] is not None for r in indirect),
            "memo_would_be_caught_if_scanned": sum(r.get("memo_would_be_caught", False) for r in indirect)},
        "benign_lookalikes": {
            "n": len(benign),
            "classifier_blocked": sum(r["classifier_caught"] for r in benign),
            "input_pipeline_blocked": sum(r["input_caught_by"] is not None for r in benign),
            "blocked_ids": [r["id"] for r in benign if r["input_caught_by"]]},
        "by_category": dict(sorted(cat.items())),
        "attacks": rows, "benign": benign,
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    d, i, b = out["attacks_typed_in_chat"], out["indirect_attacks"], out["benign_lookalikes"]
    print(f"Classifier {scorer.name} at threshold {threshold}")
    print(f"Attacks typed in chat: classifier caught {d['classifier_caught']}/{d['n']}, "
          f"full input pipeline caught {d['input_pipeline_caught']}/{d['n']}")
    print(f"Indirect (memo) attacks: input pipeline caught {i['input_pipeline_caught']}/{i['n']}; "
          f"memo text would score above threshold in {i['memo_would_be_caught_if_scanned']}/{i['n']} (hypothetical)")
    print(f"Benign look-alikes: classifier blocked {b['classifier_blocked']}/{b['n']}, "
          f"input pipeline blocked {b['input_pipeline_blocked']}/{b['n']} {b['blocked_ids']}")
    print("\nBy category (classifier / input pipeline):")
    for k, v in out["by_category"].items():
        print(f"  {k:26s} {v['classifier']}/{v['n']}  {v['input_pipeline']}/{v['n']}")
    print("\nPer conversation (max score, caught by):")
    for r in rows + benign:
        print(f"  {r['id']:24s} {r['max_score']:.8f}  {r['input_caught_by'] or '-'}")
    print(f"\nWrote {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

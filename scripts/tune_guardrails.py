"""Phase 4: tune the injection threshold on the dev set and measure the input guardrails.

  python scripts/tune_guardrails.py            # report only
  python scripts/tune_guardrails.py --apply    # also write the chosen threshold into configs/guardrails.yaml

Threshold rule (fixed before looking at scores): the threshold with the highest
injection recall such that at most 5% of benign dev messages are blocked by the
classifier; ties go to the higher threshold. The threshold is tuned and reported
on the same 64 messages, so these numbers are optimistic; the held-out measure is
the Phase 6 red-team suite, whose attacks must not overlap with this dev set.

Writes results/guardrails_dev.json.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bankguard.config import CONFIG_DIR, REPO_ROOT, load_yaml  # noqa: E402
from bankguard.guardrails import GuardrailConfig, Guardrails  # noqa: E402
from bankguard.guardrails.injection import HFInjectionClassifier  # noqa: E402
from bankguard.guardrails.pii import mask_pii  # noqa: E402

DEV_PATH = REPO_ROOT / "data" / "guardrails" / "dev.jsonl"
OUT_PATH = REPO_ROOT / "results" / "guardrails_dev.json"
MAX_BENIGN_FPR = 0.05


def load_dev(path: Path = DEV_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def choose_threshold(scored: list[tuple[float, str]], max_fpr: float = MAX_BENIGN_FPR) -> tuple[float | None, list[dict]]:
    """scored = [(score, label)]. Return (threshold, table of every candidate).

    Candidates are midpoints between neighbouring distinct scores (plus 0 and a value above the
    maximum), so no dev score sits exactly on the threshold. Scores are compared unrounded: rounding
    0.99996 to 1.0 for the candidate list once made a threshold that blocked nothing.
    """
    distinct = sorted({s for s, _ in scored})
    candidates = [0.0] + [(a + b) / 2 for a, b in zip(distinct, distinct[1:])] + [min(1.0, distinct[-1] + 1e-9)]
    pos = [s for s, lab in scored if lab == "injection"]
    ben = [s for s, lab in scored if lab == "benign"]
    table = []
    for t in candidates:
        recall = sum(s >= t for s in pos) / len(pos)
        fpr = sum(s >= t for s in ben) / len(ben)
        table.append({"threshold": t, "injection_recall": round(recall, 4), "benign_fpr": round(fpr, 4)})
    ok = [r for r in table if r["benign_fpr"] <= max_fpr]
    if not ok:
        return None, table
    best = max(ok, key=lambda r: (r["injection_recall"], r["threshold"]))
    return best["threshold"], table


def evaluate_pipeline(guards: Guardrails, rows: list[dict]) -> dict:
    """Run the input guardrails on every dev message; report block rates and false refusals."""
    per_label: dict[str, dict] = {}
    false_refusals, missed = [], []
    for r in rows:
        d = guards.check_input(r["text"], r["account"])
        stats = per_label.setdefault(r["label"], {"n": 0, "blocked": 0, "by": {}})
        stats["n"] += 1
        if d.blocked_by:
            stats["blocked"] += 1
            stats["by"][d.blocked_by] = stats["by"].get(d.blocked_by, 0) + 1
            if r["expect"] == "pass":
                false_refusals.append({"id": r["id"], "blocked_by": d.blocked_by, "text": r["text"]})
        elif r["expect"] == "block":
            missed.append({"id": r["id"], "label": r["label"], "text": r["text"]})
    for s in per_label.values():
        s["block_rate"] = round(s["blocked"] / s["n"], 4)
    n_benign = per_label.get("benign", {}).get("n", 0)
    n_attack = sum(s["n"] for k, s in per_label.items() if k != "benign")
    return {
        "per_label": per_label,
        "false_refusal_rate": round(len(false_refusals) / n_benign, 4) if n_benign else None,
        "block_rate_on_should_block": round((n_attack - len(missed)) / n_attack, 4) if n_attack else None,
        "false_refusals": false_refusals,
        "missed": missed,
    }


def apply_threshold(threshold: float, path: Path | None = None) -> None:
    """Rewrite only the injection_threshold line, keeping the YAML comments."""
    path = path or CONFIG_DIR / "guardrails.yaml"
    text = path.read_text(encoding="utf-8")
    new, n = re.subn(r"(?m)^(\s*injection_threshold:\s*)[^#\n]*?(\s*#.*)?$",
                     lambda m: f"{m.group(1)}{threshold!r}{m.group(2) or ''}", text, count=1)
    if n != 1:
        sys.exit("Could not find injection_threshold in configs/guardrails.yaml")
    path.write_text(new, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true", help="write the chosen threshold into configs/guardrails.yaml")
    args = p.parse_args()

    rows = load_dev()
    cfg = GuardrailConfig.from_yaml(load_yaml("guardrails"))
    t0 = time.perf_counter()
    clf = HFInjectionClassifier(cfg.injection_model, cfg.injection_revision)
    load_s = time.perf_counter() - t0
    clf.score("warm-up")

    scores, times = [], []
    for r in rows:
        t = time.perf_counter()
        scores.append(clf.score(r["text"]))
        times.append((time.perf_counter() - t) * 1000)
    threshold, table = choose_threshold([(s, r["label"]) for s, r in zip(scores, rows)])

    report: dict = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dev_set": str(DEV_PATH.relative_to(REPO_ROOT)),
        "n": len(rows),
        "n_by_label": {lab: sum(r["label"] == lab for r in rows) for lab in sorted({r["label"] for r in rows})},
        "classifier": {"model": cfg.injection_model, "revision": cfg.injection_revision,
                       "commit": getattr(clf.model.config, "_commit_hash", None), "device": clf.device,
                       "load_s": round(load_s, 2), "latency_ms_mean": round(statistics.mean(times), 2),
                       "latency_ms_p95": round(sorted(times)[int(0.95 * (len(times) - 1))], 2)},
        "threshold_rule": f"max injection recall with benign FPR <= {MAX_BENIGN_FPR}; ties -> higher threshold",
        "chosen_threshold": threshold,
        "threshold_table": table,
        "scores": [{"id": r["id"], "label": r["label"], "score": s} for s, r in zip(scores, rows)],
        # Same messages scored after PII masking, to show why the classifier scores the original text.
        "pii_items_raw_vs_masked": [{"id": r["id"], "raw": s, "masked": clf.score(mask_pii(r["text"]).text)}
                                    for s, r in zip(scores, rows) if mask_pii(r["text"]).any],
    }
    if threshold is not None:
        best = next(t for t in table if t["threshold"] == threshold)
        report["chosen_recall"], report["chosen_benign_fpr"] = best["injection_recall"], best["benign_fpr"]
    if threshold is None:
        print(f"No threshold keeps benign FPR <= {MAX_BENIGN_FPR}; see threshold_table in {OUT_PATH}.")
    else:
        full = Guardrails(replace(cfg, injection_threshold=threshold), scorer=clf)
        clf_only = Guardrails(replace(cfg, injection_threshold=threshold, topic_filter=False), scorer=clf)
        topic_only = Guardrails(replace(cfg, injection_classifier=False), scorer=None)
        report["pipeline"] = {"all_input_guardrails": evaluate_pipeline(full, rows),
                              "classifier_only": evaluate_pipeline(clf_only, rows),
                              "topic_filter_only": evaluate_pipeline(topic_only, rows)}

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    c = report["classifier"]
    print(f"Classifier {c['model']} on {c['device']}: load {c['load_s']} s, "
          f"{c['latency_ms_mean']} ms/message (p95 {c['latency_ms_p95']} ms)")
    print(f"Chosen threshold: {threshold}")
    if threshold is not None:
        print(f"  dev injection recall {report['chosen_recall']}, benign FPR {report['chosen_benign_fpr']}")
    print("Score by label (min / median / max):")
    for lab in sorted(report["n_by_label"]):
        vals = sorted(sc for sc, r in zip(scores, rows) if r["label"] == lab)
        print(f"  {lab:18s} {vals[0]:.6f} / {statistics.median(vals):.6f} / {vals[-1]:.6f}")
    print("Highest-scoring benign messages:")
    for sc, r in sorted(((sc, r) for sc, r in zip(scores, rows) if r["label"] == "benign"),
                        key=lambda x: -x[0])[:5]:
        print(f"  {sc:.6f} {r['id']}: {r['text']}")
    for item in report["pii_items_raw_vs_masked"]:
        print(f"  PII item {item['id']}: raw {item['raw']:.6f}, masked {item['masked']:.6f}")
    if threshold is not None:
        for name, res in report["pipeline"].items():
            rates = ", ".join(f"{k} {v['blocked']}/{v['n']}" for k, v in sorted(res["per_label"].items()))
            print(f"  {name:22s} blocked: {rates} | false refusals {len(res['false_refusals'])}"
                  f"/{report['n_by_label'].get('benign', 0)} | missed {len(res['missed'])}")
        full_res = report["pipeline"]["all_input_guardrails"]
        for fr in full_res["false_refusals"]:
            print(f"  FALSE REFUSAL {fr['id']} ({fr['blocked_by']}): {fr['text']}")
        for m in full_res["missed"]:
            print(f"  MISSED {m['id']} ({m['label']}): {m['text']}")
        if report["chosen_recall"] < 0.5:
            print("WARNING: the classifier catches under half the dev injections at this threshold; "
                  "the topic filter and output guardrails are doing the work.")
        if args.apply:
            apply_threshold(threshold)
            print(f"Wrote injection_threshold: {threshold} to configs/guardrails.yaml")
    print(f"Results: {OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

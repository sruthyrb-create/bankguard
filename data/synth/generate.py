"""Generate data/synth/bank.db from configs/models.yaml (`data:` section).

Run with: python data/synth/generate.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bankguard.config import REPO_ROOT, load_yaml  # noqa: E402
from bankguard.synth import SynthConfig, generate  # noqa: E402


def main() -> None:
    d = load_yaml("models")["data"]
    cfg = SynthConfig(seed=d["seed"], n_customers=d["n_customers"], days=d["days"], end_date=d["end_date"])
    db_path = REPO_ROOT / d["db_path"]
    counts = generate(db_path, cfg)
    digest = hashlib.sha256(db_path.read_bytes()).hexdigest()[:16]
    print(f"Wrote {db_path.relative_to(REPO_ROOT)}")
    for k, v in counts.items():
        print(f"  {k:13s} {v}")
    print(f"  anomaly rate  {counts['anomalies'] / counts['transactions']:.2%}")
    print(f"  sha256[:16]   {digest}  (same seed + same SQLite version -> same hash)")


if __name__ == "__main__":
    main()

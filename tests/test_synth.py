"""Phase 1: synthetic data is deterministic, realistic and PAN-free."""

import re
import sqlite3
from pathlib import Path

from bankguard.synth import SynthConfig, generate
from tests.conftest import SMALL


def _dump(path: Path) -> list[tuple]:
    conn = sqlite3.connect(path)
    try:
        rows: list[tuple] = []
        for table in ("customers", "accounts", "transactions"):
            rows += conn.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
        return rows
    finally:
        conn.close()


def test_same_seed_same_data(tmp_path: Path) -> None:
    generate(tmp_path / "a.db", SMALL)
    generate(tmp_path / "b.db", SMALL)
    assert _dump(tmp_path / "a.db") == _dump(tmp_path / "b.db")


def test_different_seed_different_data(tmp_path: Path) -> None:
    generate(tmp_path / "a.db", SMALL)
    generate(tmp_path / "b.db", SynthConfig(seed=8, n_customers=20, days=30))
    assert _dump(tmp_path / "a.db") != _dump(tmp_path / "b.db")


def test_anomaly_rate_and_types(base_db: Path) -> None:
    conn = sqlite3.connect(base_db)
    total, anomalies = conn.execute("SELECT COUNT(*), SUM(is_anomaly) FROM transactions").fetchone()
    types = {r[0] for r in conn.execute("SELECT DISTINCT anomaly_type FROM transactions WHERE is_anomaly = 1")}
    conn.close()
    assert 0.02 <= anomalies / total <= 0.045
    assert types <= {"duplicate", "geo", "large_amount", "card_testing"}
    assert len(types) >= 3


def test_accounts_per_customer(base_db: Path) -> None:
    conn = sqlite3.connect(base_db)
    counts = [r[0] for r in conn.execute("SELECT COUNT(*) FROM accounts GROUP BY customer_id")]
    conn.close()
    assert len(counts) == SMALL.n_customers
    assert min(counts) >= 1 and max(counts) <= 3


def test_no_full_card_numbers(base_db: Path) -> None:
    conn = sqlite3.connect(base_db)
    last4 = [r[0] for r in conn.execute("SELECT card_last4 FROM accounts")]
    text = " ".join(str(v) for row in conn.execute("SELECT * FROM transactions") for v in row)
    conn.close()
    assert all(re.fullmatch(r"\d{4}", x) for x in last4)
    assert not re.search(r"\b\d{13,19}\b", text)

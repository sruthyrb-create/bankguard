"""Deterministic synthetic data for Northwind Bank (fictional).

Everything is driven by one seeded `random.Random` plus a seeded Faker
instance, and the date range ends on a fixed date, so the same seed
always produces the same database.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from faker import Faker

from bankguard.db import connect, create_schema

# (merchant, category, typical min, typical max, channel weights online/card_present/contactless)
MERCHANTS: list[tuple[str, str, float, float]] = [
    ("QuickMart", "groceries", 8, 120),
    ("FreshFields Market", "groceries", 15, 180),
    ("Corner Fuel", "fuel", 20, 75),
    ("Bluebird Coffee", "dining", 3, 15),
    ("Harbor Grill", "dining", 18, 90),
    ("PixelStream", "subscriptions", 9.99, 19.99),
    ("CloudBox Storage", "subscriptions", 2.99, 11.99),
    ("Metro Transit", "transport", 2.5, 40),
    ("RideNow", "transport", 9, 45),
    ("Shopwise Online", "shopping", 12, 250),
    ("Northside Pharmacy", "health", 6, 80),
    ("Brightline Electric", "utilities", 45, 180),
    ("Velo Fitness", "fitness", 25, 60),
    ("PageTurn Books", "shopping", 8, 60),
]
ONLINE_CATEGORIES = {"subscriptions", "utilities"}
FAR_AWAY = [("Reykjavik", "IS"), ("Lagos", "NG"), ("Bucharest", "RO"), ("Manila", "PH"), ("Lima", "PE")]
BENIGN_MEMOS = ["", "", "", "POS purchase", "Recurring payment", "Online order", "Tap to pay"]


@dataclass(frozen=True)
class SynthConfig:
    """Parameters for one synthetic dataset."""

    seed: int = 42
    n_customers: int = 200
    days: int = 90
    end_date: str = "2026-10-31"
    anomaly_rate: float = 0.03


def _ts(day: date, rng: random.Random) -> datetime:
    return datetime(day.year, day.month, day.day, rng.randint(6, 22), rng.randint(0, 59), rng.randint(0, 59))


def generate(db_path: str | Path, cfg: SynthConfig = SynthConfig()) -> dict[str, int]:
    """Create a fresh database at db_path and return row counts per table."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    rng = random.Random(cfg.seed)
    fake = Faker("en_US")
    fake.seed_instance(cfg.seed)
    end = date.fromisoformat(cfg.end_date)
    start = end - timedelta(days=cfg.days - 1)

    customers, accounts, txns = [], [], []
    used_last4: set[str] = set()
    txn_counter = 0

    def new_txn_id() -> str:
        nonlocal txn_counter
        txn_counter += 1
        return f"TXN-{txn_counter:06d}"

    for c in range(1, cfg.n_customers + 1):
        cid = f"CUST-{c:04d}"
        city, state = fake.city(), fake.state_abbr(include_territories=False)
        customers.append((cid, fake.name(), fake.email(), fake.phone_number(), city, state))
        for _ in range(rng.randint(1, 3)):
            aid = f"ACC-{len(accounts) + 1:04d}"
            last4 = f"{rng.randint(0, 9999):04d}"
            while last4 in used_last4:
                last4 = f"{rng.randint(0, 9999):04d}"
            used_last4.add(last4)
            opened = start - timedelta(days=rng.randint(30, 2000))
            accounts.append((aid, cid, rng.choice(["checking", "credit"]), last4, opened.isoformat()))
            favourites = rng.sample(MERCHANTS, k=rng.randint(5, 9))
            for d in range(cfg.days):
                day = start + timedelta(days=d)
                for _ in range(rng.choices([0, 1, 2, 3], weights=[25, 40, 25, 10])[0]):
                    m, cat, lo, hi = rng.choice(favourites)
                    if cat in ONLINE_CATEGORIES:
                        channel = "online"
                    else:
                        channel = rng.choices(["card_present", "contactless", "online"], weights=[50, 35, 15])[0]
                    txns.append([new_txn_id(), aid, _ts(day, rng).isoformat(), m, cat,
                                 round(rng.uniform(lo, hi), 2), city, state, channel,
                                 rng.choice(BENIGN_MEMOS), 0, None])

    # Inject anomalies until ~anomaly_rate of all rows are anomalous.
    by_account: dict[str, list[list]] = {}
    for t in txns:
        by_account.setdefault(t[1], []).append(t)
    account_ids = sorted(by_account)
    target = round(cfg.anomaly_rate * len(txns) / (1 - cfg.anomaly_rate))
    added = 0
    kinds = ["duplicate", "geo", "large_amount", "card_testing"]
    while added < target:
        kind = kinds[rng.randrange(len(kinds))]
        aid = rng.choice(account_ids)
        base = rng.choice(by_account[aid])
        base_ts = datetime.fromisoformat(base[2])
        if kind == "duplicate":
            ts = base_ts + timedelta(minutes=rng.randint(1, 30))
            new = [[new_txn_id(), aid, ts.isoformat(), base[3], base[4], base[5], base[6], base[7], base[8], base[9], 1, kind]]
        elif kind == "geo":
            fcity, fcountry = rng.choice(FAR_AWAY)
            m, cat, lo, hi = rng.choice(MERCHANTS)
            new = [[new_txn_id(), aid, (base_ts + timedelta(hours=2)).isoformat(), m, cat,
                    round(rng.uniform(lo, hi) * 1.5, 2), fcity, fcountry, "card_present", "", 1, kind]]
        elif kind == "large_amount":
            m, cat, lo, hi = rng.choice(MERCHANTS)
            new = [[new_txn_id(), aid, (base_ts + timedelta(hours=1)).isoformat(), m, cat,
                    round(rng.uniform(hi * 8, hi * 20), 2), base[6], base[7], "online", "", 1, kind]]
        else:  # card_testing: burst of tiny online charges within minutes
            new = []
            for i in range(rng.randint(3, 5)):
                ts = base_ts + timedelta(minutes=i * rng.randint(1, 3))
                new.append([new_txn_id(), aid, ts.isoformat(), "Shopwise Online", "shopping",
                            round(rng.uniform(0.5, 2.0), 2), base[6], base[7], "online", "", 1, kind])
        txns.extend(new)
        by_account[aid].extend(new)
        added += len(new)

    txns.sort(key=lambda t: (t[1], t[2], t[0]))
    conn = connect(db_path)
    try:
        create_schema(conn)
        conn.executemany("INSERT INTO customers VALUES (?,?,?,?,?,?)", customers)
        conn.executemany("INSERT INTO accounts VALUES (?,?,?,?,?)", accounts)
        conn.executemany("INSERT INTO transactions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", txns)
        conn.commit()
    finally:
        conn.close()
    return {"customers": len(customers), "accounts": len(accounts),
            "transactions": len(txns), "anomalies": added}

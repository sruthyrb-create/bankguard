"""SQLite schema and connection helper for the synthetic Northwind Bank data."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE customers (
    customer_id TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    email       TEXT NOT NULL,
    phone       TEXT NOT NULL,
    home_city   TEXT NOT NULL,
    home_state  TEXT NOT NULL
);
CREATE TABLE accounts (
    account_id   TEXT PRIMARY KEY,
    customer_id  TEXT NOT NULL REFERENCES customers(customer_id),
    account_type TEXT NOT NULL,            -- checking | savings (debit card; Regulation E applies)
    card_last4   TEXT NOT NULL,            -- last 4 digits only; no full card numbers exist
    opened_date  TEXT NOT NULL
);
CREATE TABLE transactions (
    txn_id       TEXT PRIMARY KEY,
    account_id   TEXT NOT NULL REFERENCES accounts(account_id),
    ts           TEXT NOT NULL,            -- ISO 8601, local time, no timezone
    merchant     TEXT NOT NULL,
    category     TEXT NOT NULL,
    amount       REAL NOT NULL,            -- positive = money spent
    city         TEXT NOT NULL,
    state        TEXT NOT NULL,
    channel      TEXT NOT NULL,            -- card_present | online | contactless
    memo         TEXT NOT NULL,
    is_anomaly   INTEGER NOT NULL DEFAULT 0,  -- ground truth, never returned by tools
    anomaly_type TEXT                         -- duplicate | geo | large_amount | card_testing
);
CREATE INDEX idx_txn_account_ts ON transactions(account_id, ts);
CREATE TABLE flags (
    flag_id        TEXT PRIMARY KEY,
    account_id     TEXT NOT NULL,
    transaction_id TEXT NOT NULL,
    reason         TEXT NOT NULL,
    created_at     TEXT NOT NULL
);
CREATE TABLE disputes (
    dispute_id     TEXT PRIMARY KEY,
    account_id     TEXT NOT NULL,
    transaction_id TEXT NOT NULL UNIQUE,
    reason         TEXT NOT NULL,
    status         TEXT NOT NULL,
    created_at     TEXT NOT NULL
);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection that returns rows as dict-like objects."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    """Create all tables in an empty database."""
    conn.executescript(SCHEMA)

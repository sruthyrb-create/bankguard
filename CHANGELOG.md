# Changelog

## Phase 0 (2026-11-13)
- Repo scaffold, pinned core requirements, YAML configs, `.env.example`, Makefile.
- `scripts/check_env.py` checks Python, GPU, `.env`, Groq key and Groq model IDs.
- Smoke tests: config loading, `.env.example`, no real bank branding.

## Phase 0 fix (2026-11-13)
- Llama 3.3 70B is no longer on Groq: agent/teacher is now `openai/gpt-oss-120b`, judge `qwen/qwen3.8-27b`. `check_env.py` verifies both.

## Phase 1 (2026-11-13)
- `bankguard/synth.py` + `data/synth/generate.py`: deterministic synthetic bank (200 customers, 1-3 accounts, 90 days, ~3% labelled anomalies of 4 types).
- `bankguard/db.py`: SQLite schema (customers, accounts, transactions, flags, disputes).
- `bankguard/tools.py`: `get_transactions`, `flag_unusual_charge`, `open_dispute` (with Reg E timeline), `search_policy` stub; JSON schemas; authorization check in code.
- Tests for determinism, anomaly rate, no full card numbers, authorization, filters, writes and error handling.

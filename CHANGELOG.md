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

## Phase 2 (2026-11-13)
- `bankguard/policy.py`: parse eCFR XML, chunk by paragraph letter, embed with all-MiniLM-L6-v2, search with FAISS.
- `data/policy/build_index.py` (`make index`) and `scripts/sanity_policy.py` (`make sanity`, 5 queries).
- `search_policy` tool now works and returns chunk ids to cite.
- `requirements-ml.txt` and `make setup-ml` (CUDA 13.0 PyTorch first).
- Offline tests for parsing, chunk ids and search, using a tiny XML fixture and a hashing embedder.

## Phase 3a (2026-11-13)
- `bankguard/llm.py`: one OpenAI-compatible client for Groq and local vLLM, SDK retries, disk cache.
- `bankguard/agent.py`: multi-turn tool-calling loop (max 6 steps), system prompt with tool rules.
- `bankguard/tracing.py`: JSONL traces (llm_call, tool_call, final, error).
- `bankguard/api.py`: FastAPI `/chat` (multi-turn via session_id) and `/health`.
- `scripts/chat_cli.py`: interactive chat and a scripted 3-turn demo (`make demo`).
- Offline tests with a scripted fake LLM for the loop, errors, max steps and the API.
- Fix: Reg E 90-day extension cases in `open_dispute`; calendar-day labels; debit-only accounts (regenerate with `make data`).
- Prompt rule: quote numbers and day counts exactly as retrieved. Policy index warmed before the first turn.
- Fix: optional tool parameters accept null (root cause of the API 400). Groq `tool_use_failed` (400) is now retried with feedback instead of failing the turn; `limit` added to `get_transactions`; error details returned by the API; demo names the disputed transaction.

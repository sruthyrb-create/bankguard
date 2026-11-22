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

## Phase 4 (2026-11-13)
- `bankguard/guardrails/`: input guardrails (PII masking, prompt-injection classifier, topic filter) and output guardrails (tool-call schema check, grounding check, PII-leak redaction, action policy for `open_dispute`). Each is switched in `configs/guardrails.yaml`.
- Agent: input guardrails run before the model; blocked messages are refused without an LLM call and stored as a placeholder. Schema check and action policy run before each tool call (denied calls never execute). Grounding failures get one repair attempt, then a fallback answer. Leaked PII is redacted. Every guardrail decision is a `guardrail` trace event and is returned in `TurnResult` and the API response.
- `data/guardrails/dev.jsonl`: 64 hand-written dev messages (28 benign incl. hard negatives, 20 injections, 8 investment-advice, 8 other-customer).
- `scripts/tune_guardrails.py` (`make tune-guardrails`): scores the dev set, picks the injection threshold by a fixed rule, reports block and false-refusal rates to `results/guardrails_dev.json`.
- `scripts/chat_cli.py`: `--demo-guardrails` (`make demo-guardrails`) and `--no-guardrails`.
- Offline tests for every guardrail, the dev set (incl. no overlap with scenario files) and the agent integration.

## Phase 4 fix (2026-11-13)
- Threshold tuning: candidates are midpoints between neighbouring scores and scores are no longer rounded. Rounding 0.99996 to 1.0 had produced a 1.0 threshold that blocked no injection.
- The injection classifier now scores the original message (it runs locally); the model still only sees the masked text. The masked placeholders made a benign message score 0.9944.
- `tune_guardrails.py` prints score ranges per label, the highest-scoring benign messages, raw vs masked scores for PII messages, and warns when dev recall is under 0.5.
- New output guardrail `day_count_check`: "N business days" / "N calendar days" in an answer must match this turn's retrieved text or tool output; failures get the same one repair, then the fallback.
- Demos (`--demo`, `--demo-guardrails`) run on a throwaway copy of bank.db, so they can be re-run.

## Phase 4 fix 2 (2026-11-13)
- Grounding and day-count checks use chunk ids and tool output from the whole conversation (stored on `Session`), fixing a false fallback when the model re-cited chunks from an earlier turn.
- `LLMClient`: cache hits report their real latency; responses carry `server_ms` (Groq `usage.total_time`) and `retries` (SDK retries), both written to `llm_call` trace events.
- Tests: `tests/test_llm.py` (429 retry and cache latency with a mocked HTTP transport), earlier-turn citation regression test.
- DECISIONS: classifier dev result and limits, corrected note on PII masking vs classifier scores.

## Phase 4 fix 3 (2026-11-13)
- `day_count_check` reads numbers written as words ("three business days" = "3 business days"); regression test from the Groq demo.

## Phase 5, part 1: eval harness (2026-11-16)
- `bankguard/eval/`: scenario format and loader, rule graders, free-tier pacing (`PacedLLM`, per-day usage ledger), scenario runner (each scenario on a temporary bank.db copy; resumable records), LLM judge prompt and parser, metrics.
- `data/scenarios/eval/eval_v1.jsonl`: 60 scenarios built from bank.db by `scripts/build_eval_scenarios.py` (10 lookup, 8 flag, 12 dispute, 15 policy, 8 refusal, 7 error handling).
- Scripts: `run_eval.py` (guardrails on/off, `--pilot`), `judge_eval.py`, `report_eval.py`, `export_labels.py` (blind hand-labelling sheet), `judge_agreement.py` (Cohen's kappa). `configs/eval.yaml`. Make targets `eval-pilot`, `eval-on`, `eval-off`, `judge`, `eval-report`, `labels`, `agreement`.
- `LLMClient`: records Groq's provider-cached prompt tokens, `is_cached()` for local cache hits, optional extra request fields. `TurnResult.llm_stats` sums cached tokens, API calls, retries, server time and LLM time per turn.
- System prompt: cite chunk ids retrieved "in this conversation" (matches the session-scoped grounding check).

## Phase 5, part 1 fix (2026-11-16)
- Day-count guardrail: sources exclude the model's own text (search query, echoed arguments); regression test from pilot scenario eval-policy-11.
- Guardrail repair/fallback events store the rejected answer. Report splits false refusals by kind and says whether an output fallback prevented a wrong answer.
- `scripts/show_eval.py`: print transcripts of failed (or chosen) scenarios, no API calls.

## Phase 5, part 2: agent v1 fits the free tier (2026-11-17)
- `bankguard/context.py`: compact encoding of tool results; `fit_context` keeps each request under `context_budget_tokens` (configs/models.yaml, 6000 estimated tokens) by stubbing older tool results, then cutting rows. `context_trim` trace events; `llm_call` events record the estimate next to the real prompt tokens.
- Dash normalisation (`normalize_dashes`) in PII masking, topic filter, action policy, leak redaction and the eval grader; regression tests from run 0.
- Pacer waits for room for the request's estimated size.
- Run names include `agent_version` (configs/eval.yaml, `v1`). Make targets use `RUN_ON` / `RUN_OFF`.
- Tests: `tests/test_context.py`, hyphen and pacer regressions.

## Phase 5, part 3: held-out set and regrading (2026-11-19)
- `data/scenarios/heldout/heldout_v1.jsonl`: 20 held-out scenarios (builder: `scripts/build_eval_scenarios.py`). `run_eval.py --set heldout` (`make eval-heldout`); run names get a `_heldout` suffix.
- Eval grader accepts sub-paragraph citations of retrieved paragraphs; `scripts/regrade_eval.py` (`make regrade RUN=...`) re-applies graders to saved records without API calls.
- `check_grounding(..., allow_subparagraphs)`; guardrail setting `grounding_allow_subparagraphs` (default false, unchanged for v1).
- Judge and labelling sheet read references from every scenario set.

## Phase 5, part 4: agent v2 (2026-11-19)
- Tools: `get_transactions` filters `transaction_id`, `merchant`, `city`; `reason` optional for `flag_unusual_charge` and `open_dispute`.
- System prompt v2 (acting on transactions, policy questions, id format examples).
- Output guardrails `id_check` and `action_claim_check` (configs/guardrails.yaml); `grounding_allow_subparagraphs: true`. Fallback for unrepaired claim errors is a factual summary of completed actions.
- `max_steps: 8` and `graceful_step_limit: true` (configs/models.yaml): one tool-free call for a real reply when steps run out.
- `agent_version: v2`. `judge_agreement.py` reads Excel's UTF-8 CSV (byte-order mark).
- Tests: `tests/test_v2.py` (regression cases are the real v1 replies).

## Phase 5, part 4 fix (2026-11-19)
- Judge requests set `max_tokens: 300`; `PacedLLM` paces output tokens per minute (`otpm`, configs/eval.yaml); `judge_eval.py` stops cleanly on a rate-limit error.

## Phase 5, part 4 fix 2 (2026-11-19)
- Eval grader compares expected day counts by meaning (`grading.contains`); regression test from heldout-dispute-04. Re-applied to all saved runs with `regrade_eval.py`.


## Phase 5, part 4 fix 3 (2026-11-20)
- `grading.contains` parses day counts from the normalised answer, so markdown emphasis no longer hides them (fix 2 regressed v1 held-out heldout-dispute-04). Test added.

## Phase 6 set-up (2026-11-21)
- Red-team sets, frozen by SHA-256 in `configs/eval.yaml`: 40 attacks (`data/scenarios/redteam/redteam_v1.jsonl`, 8 categories x 5) and 20 benign look-alikes (`data/scenarios/redteam_benign/benign_v1.jsonl`, scenario set `redteam_benign`).
- `bankguard/eval/redteam.py`: attack format, memo seeding into the per-attack database copy, `run_attack`, rule-graded attack goals (succeeded / complied / stopped_by), compliance judge prompt and parser, summary (ASR overall and by category, stopping layer).
- Scripts: `run_redteam.py`, `judge_redteam.py`, `report_redteam.py`, `score_redteam_classifier.py` (local, no API calls). `run_eval.py --set` accepts every scenario set.
- Makefile: `redteam-classifier`, `redteam-on`, `redteam-off`, `redteam-benign-on`, `redteam-benign-off`, `redteam-judge`, `redteam-report`.
- `tests/test_redteam.py`: set validity and freeze, no overlap with dev/eval/held-out, facts match bank.db, grading signals, stopping layer, judge parsing, seeded run on a database copy.


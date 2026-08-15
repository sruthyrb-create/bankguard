# Design decisions

One line per decision, newest at the bottom.

- 2026-11-13 Hardware is an RTX 5050 Laptop GPU (8 GB VRAM, Blackwell sm_120) on WSL2, so the spec's 8 GB tier applies: QLoRA only, quantized local models.
- 2026-11-13 Local agent baseline is Qwen2.5-7B-Instruct-AWQ instead of 7B fp16 (about 15 GB, does not fit); fallback is Qwen2.5-3B-Instruct, whose licence is Qwen Research rather than Apache-2.0.
- 2026-11-13 The fp16 vs 4-bit comparison runs on Qwen2.5-1.5B, the only model that fits in fp16, so both precisions can be measured on the same model.
- 2026-11-13 The LLM judge must come from a different model family than the teacher, to avoid a model grading its own outputs.
- 2026-11-13 One OpenAI-compatible client (`openai` SDK) talks to both Groq and vLLM; only base_url and model change.
- 2026-11-13 Two conda envs: `bankguard` (app, eval, training) and `bankguard-vllm` (serving), because vLLM pins its own torch build.
- 2026-11-13 Core requirements are pinned now; GPU/ML packages are added and pinned in the phase that first needs them, to keep installs fast and failures isolated.
- 2026-11-13 A smoke test fails if real bank branding appears anywhere in the repo (rule 3).
- 2026-11-13 Llama 3.3 70B is no longer offered on Groq (checked 2026-11-13). Teacher and strongest baseline is now `openai/gpt-oss-120b` (Apache-2.0, so no licence naming rule for distilled models); judge is `qwen/qwen3.8-27b`.
- 2026-11-13 Known judge risk: the judge is Qwen-family like the local models, so it may favour them slightly; mitigated by rule-based grading wherever possible and by measuring judge agreement on hand-labelled items.
- 2026-11-13 Synthetic data uses a fixed end date (2026-10-31) and a single seeded RNG, so `bank.db` is byte-for-byte reproducible.
- 2026-11-13 Cards are stored as last 4 digits only; no full card number ever exists in the database.
- 2026-11-13 Anomaly labels (`is_anomaly`, `anomaly_type`) stay in the database for grading but tools never return them.
- 2026-11-13 Tools enforce authorization themselves (account_id must equal the session's account), not only the prompt; a prompt can be talked around, a code check cannot.
- 2026-11-13 Reg E timeline in `open_dispute` counts business days as weekdays and ignores federal holidays (simplification, stated in the tool output).
- 2026-11-13 `get_transactions` returns at most 50 rows (newest first) to keep tool output inside the model's context.

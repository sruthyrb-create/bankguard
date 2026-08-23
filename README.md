# BankGuard

A guarded, tool-using customer-support agent for **Northwind Bank** (fictional), with an evaluation harness, a red-team suite, LoRA fine-tuning and inference benchmarks. All data is synthetic.

**Status:** Phase 3a (agent loop + API on Groq). Results tables are added in Phase 9 and every number comes from `results/*.json`.

## Setup (WSL2 / Linux)

```bash
conda create -n bankguard python=3.11 -y
conda activate bankguard
make setup
cp .env.example .env   # then add your GROQ_API_KEY
make check
make data
make setup-ml
make index
make sanity
make test
```

See `DECISIONS.md` for design choices and `CHANGELOG.md` for progress.

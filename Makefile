.PHONY: setup setup-ml check test data index sanity demo chat serve tune-guardrails demo-guardrails

setup:            ## install pinned core dependencies into the active env
	pip install -r requirements.txt

setup-ml:         ## Phase 2+: CUDA 13.0 PyTorch (Blackwell) + ML packages
	pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
	pip install -r requirements-ml.txt

check:            ## Phase 0 environment check
	python scripts/check_env.py

data:             ## Phase 1: generate the synthetic bank database
	python data/synth/generate.py

index:            ## Phase 2: download Reg E, chunk it, build the FAISS index
	python data/policy/build_index.py

sanity:           ## Phase 2: 5 retrieval sanity queries
	python scripts/sanity_policy.py

demo:             ## Phase 3: scripted 3-turn conversation on Groq
	python scripts/chat_cli.py --backend groq --demo

chat:             ## Phase 3: interactive chat on Groq (ACCOUNT=ACC-0001 by default)
	python scripts/chat_cli.py --backend groq --account $${ACCOUNT:-ACC-0001}

tune-guardrails:  ## Phase 4: pick the injection threshold on the dev set, write it to the config
	python scripts/tune_guardrails.py --apply

demo-guardrails:  ## Phase 4: one probe per guardrail on Groq
	python scripts/chat_cli.py --backend groq --demo-guardrails

serve:            ## Phase 3: FastAPI on http://localhost:8080 (docs at /docs)
	uvicorn bankguard.api:app --port 8080

test:             ## run unit tests
	pytest -q

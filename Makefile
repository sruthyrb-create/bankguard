.PHONY: setup setup-ml check test data index sanity demo chat serve tune-guardrails demo-guardrails eval-scenarios eval-pilot eval-on eval-off eval-heldout regrade judge eval-report labels agreement redteam-classifier redteam-on redteam-off redteam-benign-on redteam-benign-off redteam-judge redteam-report

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

# ---- Phase 5: eval harness (stays inside Groq's free tier; every run is resumable)
eval-scenarios:   ## rebuild data/scenarios/eval/eval_v1.jsonl from bank.db
	python scripts/build_eval_scenarios.py

eval-pilot:       ## 10 pilot scenarios with guardrails on (measures real tokens per scenario)
	python scripts/run_eval.py --guardrails on --pilot

eval-on:          ## all 60 scenarios, guardrails on (skips finished ones)
	python scripts/run_eval.py --guardrails on

eval-off:         ## all 60 scenarios, every guardrail off
	python scripts/run_eval.py --guardrails off

RUN_ON  ?= gpt-oss-120b_on_v1
RUN_OFF ?= gpt-oss-120b_off_v1

eval-heldout:     ## the 20 held-out scenarios, guardrails on
	python scripts/run_eval.py --guardrails on --set heldout

regrade:          ## re-apply current rule graders to saved runs (no API calls); RUN=<name>
	python scripts/regrade_eval.py --run $(RUN) --write

judge:            ## LLM-judge both runs (skips judged ones); override with RUN_ON=... RUN_OFF=...
	python scripts/judge_eval.py --run $(RUN_ON)
	-python scripts/judge_eval.py --run $(RUN_OFF)

eval-report:      ## aggregate all runs into results/eval_report.json
	python scripts/report_eval.py

labels:           ## export the blind hand-labelling sheet (data/labels/judge_labels.csv)
	python scripts/export_labels.py --runs $(RUN_ON) $(RUN_OFF)

agreement:        ## your labels vs the judge -> results/judge_agreement.json
	python scripts/judge_agreement.py


# ---- Phase 6: red-team (agent v2, frozen; stays inside Groq's free tier; every run is resumable)
RT_ON  ?= gpt-oss-120b_on_v2_redteam
RT_OFF ?= gpt-oss-120b_off_v2_redteam

redteam-classifier: ## input guardrails alone on attacks + benign look-alikes (local, no API calls)
	python scripts/score_redteam_classifier.py

redteam-on:       ## the 40 attacks, guardrails on (skips finished ones)
	python scripts/run_redteam.py --guardrails on

redteam-off:      ## the 40 attacks, every guardrail off
	python scripts/run_redteam.py --guardrails off

redteam-benign-on:  ## the 20 benign look-alikes, guardrails on
	python scripts/run_eval.py --guardrails on --set redteam_benign

redteam-benign-off: ## the 20 benign look-alikes, every guardrail off
	python scripts/run_eval.py --guardrails off --set redteam_benign

redteam-judge:    ## judge the judge-graded attacks and the benign runs (skips judged ones)
	python scripts/judge_redteam.py --run $(RT_ON)
	-python scripts/judge_redteam.py --run $(RT_OFF)
	python scripts/judge_eval.py --run gpt-oss-120b_on_v2_redteam_benign
	-python scripts/judge_eval.py --run gpt-oss-120b_off_v2_redteam_benign

redteam-report:   ## aggregate into results/redteam_report.json (no API calls)
	python scripts/report_redteam.py

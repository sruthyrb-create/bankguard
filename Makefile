.PHONY: setup check test data

setup:            ## install pinned core dependencies into the active env
	pip install -r requirements.txt

check:            ## Phase 0 environment check
	python scripts/check_env.py

data:             ## Phase 1: generate the synthetic bank database
	python data/synth/generate.py

test:             ## run unit tests
	pytest -q

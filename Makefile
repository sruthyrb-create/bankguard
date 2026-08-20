.PHONY: setup setup-ml check test data index sanity

setup:            ## install pinned core dependencies into the active env
	pip install -r requirements.txt

setup-ml:         ## Phase 2+: CUDA 13.0 PyTorch (Blackwell) + ML packages
	pip install torch --index-url https://download.pytorch.org/whl/cu130
	pip install -r requirements-ml.txt

check:            ## Phase 0 environment check
	python scripts/check_env.py

data:             ## Phase 1: generate the synthetic bank database
	python data/synth/generate.py

index:            ## Phase 2: download Reg E, chunk it, build the FAISS index
	python data/policy/build_index.py

sanity:           ## Phase 2: 5 retrieval sanity queries
	python scripts/sanity_policy.py

test:             ## run unit tests
	pytest -q

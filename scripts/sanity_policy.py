"""Phase 2 acceptance: 5 sanity queries must retrieve the expected section in the top 4.

Run with: python scripts/sanity_policy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bankguard.config import REPO_ROOT, load_yaml  # noqa: E402
from bankguard.policy import PolicyIndex, sentence_transformer_embedder  # noqa: E402

QUERIES = [
    ("How long does the bank have to investigate an error I reported?", "1005.11"),
    ("How much am I liable for if my debit card is lost and I report it within two business days?", "1005.6"),
    ("What must appear on my periodic account statement?", "1005.9"),
    ("How do I stop a preauthorized recurring payment?", "1005.10"),
    ("What counts as an access device?", "1005.2"),
]


def main() -> int:
    cfg = load_yaml("models")["policy"]
    index = PolicyIndex.load(REPO_ROOT / cfg["chunks_path"], REPO_ROOT / cfg["index_path"],
                             sentence_transformer_embedder(cfg["embed_model"]))
    passed = 0
    for query, expected in QUERIES:
        hits = index.search(query, k=4)
        ok = any(h["section"] == expected for h in hits)
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {query}\n       expected § {expected}; got "
              + ", ".join(f"{h['chunk_id']} ({h['score']:.2f})" for h in hits))
    print(f"\n{passed}/{len(QUERIES)} sanity queries passed")
    return 0 if passed == len(QUERIES) else 1


if __name__ == "__main__":
    raise SystemExit(main())

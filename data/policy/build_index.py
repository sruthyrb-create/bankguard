"""Download Regulation E (12 CFR 1005) from the eCFR API, chunk it and build a FAISS index.

Run with: python data/policy/build_index.py
The download is skipped if data/policy/reg_e.xml already exists.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx  # noqa: E402

from bankguard.config import REPO_ROOT, load_yaml  # noqa: E402
from bankguard.policy import (PolicyIndex, chunk_sections, parse_sections,  # noqa: E402
                              save_chunks, sentence_transformer_embedder)

ECFR_URL = "https://www.ecfr.gov/api/versioner/v1/full/{date}/title-12.xml"


def download(date: str, part: str, dest: Path) -> None:
    """Fetch the part's processed XML for a fixed snapshot date."""
    resp = httpx.get(ECFR_URL.format(date=date), params={"part": part}, timeout=60,
                     headers={"Accept": "application/xml"}, follow_redirects=True)
    resp.raise_for_status()
    try:  # the server has labelled UTF-8 bytes as ISO-8859-1 before, so try UTF-8 first
        text = resp.content.decode("utf-8")
    except UnicodeDecodeError:
        text = resp.content.decode("latin-1")
    dest.write_text(text, encoding="utf-8")


def main() -> None:
    cfg = load_yaml("models")["policy"]
    raw = REPO_ROOT / cfg["raw_path"]
    if not raw.exists():
        print(f"Downloading 12 CFR {cfg['part']} as of {cfg['ecfr_date']} ...")
        download(cfg["ecfr_date"], cfg["part"], raw)
    sections = parse_sections(raw.read_text(encoding="utf-8"), cfg["max_section"])
    (REPO_ROOT / cfg["text_path"]).write_text(
        "\n\n".join(f"§ {n} {h}\n" + "\n".join(p) for n, h, p in sections), encoding="utf-8")
    chunks = chunk_sections(sections, cfg["chunk_words"])
    save_chunks(chunks, REPO_ROOT / cfg["chunks_path"])

    t0 = time.perf_counter()
    index = PolicyIndex.build(chunks, sentence_transformer_embedder(cfg["embed_model"]))
    index.save(REPO_ROOT / cfg["index_path"])
    print(f"Sections: {len(sections)}  Chunks: {len(chunks)}  "
          f"Max words/chunk: {max(len(c.text.split()) for c in chunks)}  "
          f"Embed+index: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()

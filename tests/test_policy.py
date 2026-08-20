"""Phase 2: Reg E parsing, chunking and search.

Most tests use a tiny inline XML document and a deterministic bag-of-words
embedder, so they run offline. The last test uses the real index and is
skipped until `make index` has been run.
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from bankguard.config import REPO_ROOT, load_yaml
from bankguard.policy import Chunk, PolicyIndex, chunk_sections, load_chunks, parse_sections, save_chunks
from bankguard.tools import ToolContext, call_tool

XML = """<?xml version="1.0" encoding="ISO-8859-1"?>
<DIV5 N="1005" TYPE="PART"><HEAD>PART 1005—ELECTRONIC FUND TRANSFERS (REGULATION E)</HEAD>
<DIV8 N="1005.2" TYPE="SECTION"><HEAD>§ 1005.2 Definitions.</HEAD>
<P>For purposes of this part, the following definitions apply:</P>
<P>(a)(1) <I>Access device</I> means a card, code, or other means of access to a consumer's account.</P>
<P>(2) An access device becomes an accepted access device when the consumer requests it.</P>
<P>(b) <I>Account</I> means a demand deposit, savings, or other consumer asset account.</P>
</DIV8>
<DIV8 N="1005.11" TYPE="SECTION"><HEAD>§ 1005.11 Procedures for resolving errors.</HEAD>
<P>(a) <I>Definition of error.</I> The term error means an unauthorized electronic fund transfer.</P>
<P>(b) <I>Notice of error from consumer.</I> A notice of error is an oral or written notice.</P>
<P>(c) <I>Time limits.</I> A financial institution shall investigate promptly and determine whether an
error occurred within 10 business days of receiving a notice of error.</P>
<P>(c)(3) The institution may take up to 45 days if it provisionally credits the account.</P>
</DIV8>
<DIV8 N="1005.33" TYPE="SECTION"><HEAD>§ 1005.33 Remittance error resolution.</HEAD>
<P>(a) Remittance transfer rules.</P></DIV8>
</DIV5>"""


def bow_embed(texts):
    """Deterministic bag-of-words hashing embedder for offline tests."""
    vecs = np.zeros((len(texts), 256), dtype="float32")
    for row, text in enumerate(texts):
        for word in text.lower().split():
            word = word.strip(".,;:()?'")
            if word:
                vecs[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % 256] += 1
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms == 0, 1, norms)


def test_parse_sections_respects_max_section() -> None:
    sections = parse_sections(XML, max_section=20)
    assert [s[0] for s in sections] == ["1005.2", "1005.11"]
    assert sections[0][1] == "§ 1005.2 Definitions."
    assert "Access device means a card" in sections[0][2][1]  # inline <I> text kept


def test_chunk_ids_follow_paragraph_letters() -> None:
    chunks = chunk_sections(parse_sections(XML, max_section=20), chunk_words=180)
    ids = [c.id for c in chunks]
    assert ids == ["1005.2#1", "1005.2(a)#1", "1005.2(b)#1",
                   "1005.11(a)#1", "1005.11(b)#1", "1005.11(c)#1"]
    # a sub-paragraph like "(2) ..." stays with its parent letter
    assert "accepted access device" in chunks[1].text


def test_long_paragraphs_are_split() -> None:
    long = [("1005.9", "§ 1005.9 Statements.", ["(a) " + "word " * 400])]
    chunks = chunk_sections(long, chunk_words=180)
    assert [c.id for c in chunks] == ["1005.9(a)#1", "1005.9(a)#2", "1005.9(a)#3"]
    assert all(len(c.text.split()) <= 180 for c in chunks)


def test_save_load_roundtrip(tmp_path: Path) -> None:
    chunks = [Chunk("1005.6(b)#1", "1005.6", "§ 1005.6 Liability.", "text")]
    save_chunks(chunks, tmp_path / "c.jsonl")
    assert load_chunks(tmp_path / "c.jsonl") == chunks


def test_search_finds_relevant_chunk_and_tool_wraps_it(tmp_path: Path, db: Path) -> None:
    chunks = chunk_sections(parse_sections(XML, max_section=20))
    index = PolicyIndex.build(chunks, bow_embed)
    index.save(tmp_path / "f.index")
    save_chunks(chunks, tmp_path / "c.jsonl")
    index = PolicyIndex.load(tmp_path / "c.jsonl", tmp_path / "f.index", bow_embed)
    hits = index.search("how many business days to investigate an error", k=2)
    assert hits[0]["chunk_id"] == "1005.11(c)#1", hits
    assert len(index.search("x", k=50)) == len(chunks)  # k is capped
    out = call_tool(ToolContext(db, "ACC-0001", policy_index=index), "search_policy",
                    {"query": "what is an access device", "k": 2})
    assert out["results"][0]["chunk_id"].startswith("1005.2")


REAL_INDEX = REPO_ROOT / load_yaml("models")["policy"]["index_path"]


@pytest.mark.skipif(not REAL_INDEX.exists(), reason="run `make index` first")
def test_real_index_matches_chunks() -> None:
    import faiss

    cfg = load_yaml("models")["policy"]
    chunks = load_chunks(REPO_ROOT / cfg["chunks_path"])
    assert faiss.read_index(str(REAL_INDEX)).ntotal == len(chunks) > 20
    assert all(c.section.startswith("1005.") for c in chunks)


def test_roman_numerals_stay_with_parent_letter() -> None:
    sections = [("1005.6", "§ 1005.6 Liability.", [
        "(a) Conditions.", "(b) Limitations.", "(i) first roman item", "(v) fifth roman item",
        "(b)(3) more of b", "(c) Next letter."])]
    ids = [c.id for c in chunk_sections(sections)]
    assert ids == ["1005.6(a)#1", "1005.6(b)#1", "1005.6(c)#1"]

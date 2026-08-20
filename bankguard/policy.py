"""Regulation E: parse eCFR XML into chunks, embed them, and search with FAISS.

Chunk ids look like `1005.11(c)#1`: section, top-level paragraph letter,
and a counter when one paragraph is split. Ids are stable for a fixed
eCFR snapshot date, so citations in traces stay valid across runs.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

EmbedFn = Callable[[Sequence[str]], np.ndarray]
_PARA = re.compile(r"^\(([a-z])\)")
_SPACES = re.compile(r"\s+")
_XML_DECL = re.compile(r"^\s*<\?xml[^>]*\?>")


@dataclass(frozen=True)
class Chunk:
    """One searchable piece of policy text."""

    id: str
    section: str
    heading: str
    text: str


def _clean(el: ET.Element) -> str:
    return _SPACES.sub(" ", "".join(el.itertext())).strip()


def parse_sections(xml_text: str, max_section: int | None = None) -> list[tuple[str, str, list[str]]]:
    """Return (section number, heading, paragraphs) for each DIV8 SECTION in eCFR XML."""
    # ElementTree refuses str input that carries an encoding declaration, so drop it.
    root = ET.fromstring(_XML_DECL.sub("", xml_text, count=1))
    out = []
    for div in root.iter("DIV8"):
        if div.get("TYPE") != "SECTION":
            continue
        number = div.get("N", "").replace("§", "").strip()
        if max_section is not None:
            try:
                if int(number.split(".")[1]) > max_section:
                    continue
            except (IndexError, ValueError):
                continue
        head = div.find("HEAD")
        heading = _clean(head) if head is not None else number
        paras = [t for p in div.iter("P") if (t := _clean(p))]
        if paras:
            out.append((number, heading, paras))
    return out


def chunk_sections(sections: list[tuple[str, str, list[str]]], chunk_words: int = 180) -> list[Chunk]:
    """Group paragraphs by top-level letter, then split groups into <= chunk_words pieces.

    A paragraph starts a new letter group only if its letter is the next one in
    sequence ((a), then (b), ...). This keeps roman-numeral sub-paragraphs such
    as "(i)" or "(v)" and repeated "(c)(3)" paragraphs inside their parent.
    """
    chunks: list[Chunk] = []
    for number, heading, paras in sections:
        groups: list[tuple[str, list[str]]] = []
        current = ""
        for para in paras:
            m = _PARA.match(para)
            letter = m.group(1) if m else None
            expected = "a" if not current else chr(ord(current) + 1)
            if not groups or letter == expected:
                current = letter if letter == expected else current
                groups.append((current, [para]))
            else:
                groups[-1][1].append(para)
        counters: dict[str, int] = {}
        for letter, group in groups:
            words = " ".join(group).split()
            for start in range(0, len(words), chunk_words):
                base = f"{number}({letter})" if letter else number
                counters[base] = counters.get(base, 0) + 1
                chunks.append(Chunk(f"{base}#{counters[base]}", number, heading,
                                    " ".join(words[start:start + chunk_words])))
    return chunks


def save_chunks(chunks: list[Chunk], path: str | Path) -> None:
    """Write chunks as JSON lines."""
    with Path(path).open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def load_chunks(path: str | Path) -> list[Chunk]:
    """Read chunks written by save_chunks."""
    with Path(path).open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]


def sentence_transformer_embedder(model_name: str) -> EmbedFn:
    """Return a function that embeds texts into L2-normalised float32 vectors."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def embed(texts: Sequence[str]) -> np.ndarray:
        vecs = model.encode(list(texts), batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype="float32")

    return embed


class PolicyIndex:
    """Cosine-similarity search over chunks (inner product on normalised vectors)."""

    def __init__(self, chunks: list[Chunk], index: "faiss.Index", embed: EmbedFn) -> None:  # noqa: F821
        self.chunks, self.index, self.embed = chunks, index, embed

    @classmethod
    def build(cls, chunks: list[Chunk], embed: EmbedFn) -> "PolicyIndex":
        """Embed every chunk (heading + text) and build a flat inner-product index."""
        import faiss

        vecs = embed([f"{c.heading}\n{c.text}" for c in chunks])
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        return cls(chunks, index, embed)

    @classmethod
    def load(cls, chunks_path: str | Path, index_path: str | Path, embed: EmbedFn) -> "PolicyIndex":
        """Load a saved index and its chunks."""
        import faiss

        chunks = load_chunks(chunks_path)
        index = faiss.read_index(str(index_path))
        if index.ntotal != len(chunks):
            raise ValueError("Index and chunks file are out of sync; rebuild with `make index`.")
        return cls(chunks, index, embed)

    def save(self, index_path: str | Path) -> None:
        """Write the FAISS index to disk."""
        import faiss

        faiss.write_index(self.index, str(index_path))

    def search(self, query: str, k: int = 4) -> list[dict]:
        """Return the top-k chunks with their cosine similarity scores."""
        k = max(1, min(int(k), 8, len(self.chunks)))
        scores, idx = self.index.search(self.embed([query]), k)
        return [{"chunk_id": self.chunks[i].id, "section": self.chunks[i].section,
                 "heading": self.chunks[i].heading, "text": self.chunks[i].text,
                 "score": round(float(s), 4)}
                for s, i in zip(scores[0], idx[0]) if i >= 0]

"""Ingest the regulations corpus: section-aware chunking + Vertex embeddings.

Each markdown file is split on its `## ` section headers so a chunk is a whole
clause/section (the natural unit a citation should point at). Chunks are embedded
with text-multilingual-embedding-002 and saved to disk (vectors.npz + chunks.json)
so the API and eval never re-embed the corpus. Run: python -m ingest.ingest
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from app import config, embeddings, vectorstore


def split_sections(text: str) -> list[tuple[str, str]]:
    """Return (section_title, section_text) pairs from a markdown document."""
    lines = text.splitlines()
    # Document title from the first '# ' line.
    doc_title = ""
    for ln in lines:
        if ln.startswith("# "):
            doc_title = ln[2:].strip()
            break
    sections: list[tuple[str, str]] = []
    current_title = doc_title or "Introduction"
    buffer: list[str] = []
    for ln in lines:
        if ln.startswith("# ") and not ln.startswith("## "):
            continue  # document title, already captured
        m = re.match(r"^##\s+(.*)", ln)
        if m:
            if buffer and any(b.strip() for b in buffer):
                sections.append((current_title, "\n".join(buffer).strip()))
            current_title = m.group(1).strip()
            buffer = []
        else:
            buffer.append(ln)
    if buffer and any(b.strip() for b in buffer):
        sections.append((current_title, "\n".join(buffer).strip()))
    return sections


def build_chunks(reg_dir: Path) -> list[dict]:
    chunks: list[dict] = []
    for path in sorted(reg_dir.glob("*.md")):
        if path.name.startswith("_"):
            continue  # skip the corpus README
        text = path.read_text(encoding="utf-8")
        doc_title = ""
        for ln in text.splitlines():
            if ln.startswith("# "):
                doc_title = ln[2:].strip()
                break
        for i, (section_title, section_text) in enumerate(split_sections(text)):
            if section_title == doc_title:
                continue  # disclaimer/front matter block
            chunk_id = f"{path.stem}#s{i}"
            embed_text = f"{doc_title} :: {section_title}\n{section_text}"
            chunks.append(
                {
                    "id": chunk_id,
                    "doc": path.stem,
                    "doc_title": doc_title,
                    "section": section_title,
                    "text": section_text,
                    "embed_text": embed_text,
                }
            )
    return chunks


def run() -> dict:
    if not config.credentials_present():
        raise RuntimeError(
            "GOOGLE_APPLICATION_CREDENTIALS not set or file missing; cannot embed."
        )
    chunks = build_chunks(config.REG_DIR)
    vectors = embeddings.embed(
        [c["embed_text"] for c in chunks], task_type="RETRIEVAL_DOCUMENT"
    )
    matrix = np.asarray(vectors, dtype=np.float32)
    # Drop embed_text from stored metadata to keep chunks.json lean.
    stored = [{k: v for k, v in c.items() if k != "embed_text"} for c in chunks]
    vectorstore.save(matrix, stored, config.VECTORS_PATH, config.CHUNKS_PATH)
    return {"num_chunks": len(chunks), "dim": matrix.shape[1] if len(chunks) else 0}


if __name__ == "__main__":
    stats = run()
    print(f"Ingested {stats['num_chunks']} chunks, dim={stats['dim']}")
    print(f"Vectors -> {config.VECTORS_PATH}")
    print(f"Chunks  -> {config.CHUNKS_PATH}")

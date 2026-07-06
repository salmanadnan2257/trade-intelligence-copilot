"""A tiny in-memory vector store backed by a numpy matrix on disk.

pgvector was preferred but is not installed on the target machine, so this is a
self-contained cosine-similarity store. The corpus is small (tens of chunks) so
a brute-force dot product over a normalized matrix is instant and exact.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config


class VectorStore:
    def __init__(self, matrix: np.ndarray, chunks: list[dict]):
        # matrix is expected pre-normalized (unit rows) for cosine via dot.
        self.matrix = matrix
        self.chunks = chunks

    @classmethod
    def load(cls, vectors_path: Path | None = None, chunks_path: Path | None = None) -> "VectorStore":
        vectors_path = Path(vectors_path or config.VECTORS_PATH)
        chunks_path = Path(chunks_path or config.CHUNKS_PATH)
        data = np.load(vectors_path)
        matrix = data["vectors"].astype(np.float32)
        chunks = json.loads(Path(chunks_path).read_text(encoding="utf-8"))
        return cls(matrix, chunks)

    @staticmethod
    def normalize(vecs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    def search(self, query_vec: list[float], top_k: int | None = None) -> list[dict]:
        top_k = top_k or config.RETRIEVAL_TOP_K
        q = np.asarray(query_vec, dtype=np.float32)
        n = np.linalg.norm(q)
        if n:
            q = q / n
        scores = self.matrix @ q
        order = np.argsort(-scores)[:top_k]
        results = []
        for idx in order:
            chunk = dict(self.chunks[int(idx)])
            chunk["score"] = float(scores[int(idx)])
            results.append(chunk)
        return results


def save(vectors: np.ndarray, chunks: list[dict], vectors_path: Path, chunks_path: Path) -> None:
    normed = VectorStore.normalize(vectors.astype(np.float32))
    np.savez_compressed(vectors_path, vectors=normed)
    Path(chunks_path).write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

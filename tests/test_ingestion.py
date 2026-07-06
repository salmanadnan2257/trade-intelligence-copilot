import numpy as np

from app import config, vectorstore
from ingest import ingest


def test_split_sections_finds_numbered_sections():
    text = "# Title\n\nintro\n\n## 1. First\nbody one\n\n## 2. Second\nbody two"
    secs = ingest.split_sections(text)
    titles = [t for t, _ in secs]
    assert "1. First" in titles and "2. Second" in titles


def test_build_chunks_shape():
    chunks = ingest.build_chunks(config.REG_DIR)
    assert len(chunks) >= 30
    for c in chunks[:5]:
        assert {"id", "doc", "doc_title", "section", "text", "embed_text"} <= set(c)
        assert "#s" in c["id"]


def test_vector_store_matches_chunks():
    store = vectorstore.VectorStore.load()
    assert store.matrix.shape[0] == len(store.chunks)
    assert store.matrix.shape[1] == 768
    # rows are normalized to unit length
    norms = np.linalg.norm(store.matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)


def test_vector_search_returns_top_k():
    store = vectorstore.VectorStore.load()
    q = store.matrix[0].tolist()
    hits = store.search(q, top_k=3)
    assert len(hits) == 3
    # a vector should be most similar to itself
    assert hits[0]["id"] == store.chunks[0]["id"]
    assert hits[0]["score"] > 0.99

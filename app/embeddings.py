"""Embeddings via the Vertex AI predict REST endpoint.

The google-genai SDK does not expose text-multilingual-embedding-002 uniformly
in Vertex mode, so we call the verified predict endpoint directly with a bearer
token from the service-account credentials. Results are cached on disk during
ingestion so the eval run does not re-embed the corpus.
"""
from __future__ import annotations

import os

import google.auth
import google.auth.transport.requests
import httpx

from . import config

_creds = None


def _token() -> str:
    global _creds
    if _creds is None:
        path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not path:
            raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS is not set")
        _creds, _ = google.auth.load_credentials_from_file(
            path, scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
    if not _creds.valid:
        _creds.refresh(google.auth.transport.requests.Request())
    return _creds.token


def embed(texts: list[str], *, task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """Embed a batch of texts. task_type is RETRIEVAL_DOCUMENT or RETRIEVAL_QUERY."""
    if not texts:
        return []
    loc = config.VERTEX_EMBED_LOCATION
    url = (
        f"https://{loc}-aiplatform.googleapis.com/v1/projects/{config.GCP_PROJECT}"
        f"/locations/{loc}/publishers/google/models/{config.MODEL_EMBED}:predict"
    )
    out: list[list[float]] = []
    # The endpoint accepts small batches; keep them modest to stay under limits.
    for i in range(0, len(texts), 5):
        batch = texts[i : i + 5]
        body = {"instances": [{"content": t, "task_type": task_type} for t in batch]}
        resp = httpx.post(
            url,
            json=body,
            headers={"Authorization": f"Bearer {_token()}"},
            timeout=90,
        )
        resp.raise_for_status()
        for pred in resp.json()["predictions"]:
            out.append(pred["embeddings"]["values"])
    return out


def embed_one(text: str, *, task_type: str = "RETRIEVAL_QUERY") -> list[float]:
    return embed([text], task_type=task_type)[0]

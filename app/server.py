"""FastAPI service: POST /ask, GET /health, and a single-page UI at /."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import agent, config

app = FastAPI(title="Trade Intelligence Copilot", version="1.0.0")
_TEMPLATE = (Path(__file__).parent / "templates" / "index.html").read_text(encoding="utf-8")


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health() -> dict:
    ok_db = config.DB_PATH.is_file()
    ok_vectors = config.VECTORS_PATH.is_file() and config.CHUNKS_PATH.is_file()
    return {
        "status": "ok" if (ok_db and ok_vectors) else "degraded",
        "database": ok_db,
        "vector_store": ok_vectors,
        "credentials_configured": config.credentials_present(),
        "project": config.GCP_PROJECT,
    }


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    return agent.ask(req.question)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _TEMPLATE

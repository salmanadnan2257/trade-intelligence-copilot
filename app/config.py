"""Central configuration. Everything sensitive is read from the environment.

The service-account key is never stored in the repo. The code only reads its
path from GOOGLE_APPLICATION_CREDENTIALS. The GCP project id has a placeholder
default and should be overridden with GCP_PROJECT in a real deployment.
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# GCP / Vertex. The project id is not a secret but we still prefer the env var.
GCP_PROJECT = os.environ.get("GCP_PROJECT", "your-gcp-project-id")
VERTEX_GEN_LOCATION = os.environ.get("VERTEX_GEN_LOCATION", "global")
VERTEX_EMBED_LOCATION = os.environ.get("VERTEX_EMBED_LOCATION", "us-central1")

MODEL_FAST = os.environ.get("MODEL_FAST", "gemini-2.5-flash")
MODEL_PRO = os.environ.get("MODEL_PRO", "gemini-2.5-pro")
MODEL_EMBED = os.environ.get("MODEL_EMBED", "text-multilingual-embedding-002")

# Storage paths
DB_PATH = Path(os.environ.get("TRADE_DB_PATH", str(DATA_DIR / "trade.sqlite")))
VECTORS_PATH = Path(os.environ.get("VECTORS_PATH", str(DATA_DIR / "vectors.npz")))
CHUNKS_PATH = Path(os.environ.get("CHUNKS_PATH", str(DATA_DIR / "chunks.json")))
REG_DIR = DATA_DIR / "regulations"

# Vertex list prices, USD per 1,000,000 tokens. Approximate published rates;
# override with env if pricing changes. These drive the per-query cost report.
PRICING = {
    "gemini-2.5-flash": {
        "input": float(os.environ.get("PRICE_FLASH_IN", "0.30")),
        "output": float(os.environ.get("PRICE_FLASH_OUT", "2.50")),
    },
    "gemini-2.5-pro": {
        "input": float(os.environ.get("PRICE_PRO_IN", "1.25")),
        "output": float(os.environ.get("PRICE_PRO_OUT", "10.00")),
    },
    "text-multilingual-embedding-002": {
        "input": float(os.environ.get("PRICE_EMBED_IN", "0.20")),
        "output": 0.0,
    },
}

RETRIEVAL_TOP_K = int(os.environ.get("RETRIEVAL_TOP_K", "4"))
# Cosine score below this is treated as "no relevant regulation found".
RETRIEVAL_MIN_SCORE = float(os.environ.get("RETRIEVAL_MIN_SCORE", "0.55"))


def credentials_present() -> bool:
    path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    return bool(path) and Path(path).is_file()

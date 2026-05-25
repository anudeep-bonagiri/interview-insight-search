"""Default paths and model names, resolved relative to the repository root.

Everything here is a default; the CLI lets you override the data dir, index
location, embedder, and model. Keeping the defaults in one place means the
library, CLI, tests, and Streamlit app all agree on where things live.
"""

from __future__ import annotations

import os
from pathlib import Path

# src/interview_search/config.py -> repo root is three parents up.
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("IIS_DATA_DIR", REPO_ROOT / "data" / "interviews"))
EVAL_PATH = Path(os.environ.get("IIS_EVAL_PATH", REPO_ROOT / "data" / "eval" / "queries.yaml"))
INDEX_DIR = Path(os.environ.get("IIS_INDEX_DIR", REPO_ROOT / ".index"))

# Default embedder. "fastembed" downloads a small ONNX model on first use and
# needs no API key or torch. "hashing" is a deterministic, dependency-free
# fallback used in tests and offline environments.
DEFAULT_EMBEDDER = os.environ.get("IIS_EMBEDDER", "fastembed")
FASTEMBED_MODEL = os.environ.get("IIS_FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")

# Claude model used for grounded insight synthesis (the `ask` command).
SYNTHESIS_MODEL = os.environ.get("IIS_SYNTHESIS_MODEL", "claude-opus-4-7")

# Retrieval mode: dense | lexical | hybrid. Hybrid fuses semantic and BM25.
DEFAULT_MODE = os.environ.get("IIS_MODE", "hybrid")

# Chunking defaults.
TARGET_WORDS = 90
OVERLAP_TURNS = 1

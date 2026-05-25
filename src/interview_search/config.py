"""Default paths and model names, resolved relative to the repository root.

Everything here is a default; the CLI lets you override the data dir, index
location, embedder, and model. Keeping the defaults in one place means the
library, CLI, tests, and Streamlit app all agree on where things live.
"""

from __future__ import annotations

import os
from pathlib import Path

# src/interview_search/config.py -> repo root is three parents up (source layout).
REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve(env_var: str, *relparts: str) -> Path:
    """Find a data path, robust to how the package was installed.

    Checks an env override first, then the source-layout repo root, then the
    current working directory. The cwd fallback covers a non-editable install
    (e.g. on Streamlit Cloud), where the package lives in site-packages but the
    repo (and its data/) is the working directory.
    """
    override = os.environ.get(env_var)
    if override:
        return Path(override)
    rel = Path(*relparts)
    for base in (REPO_ROOT, Path.cwd()):
        candidate = base / rel
        if candidate.exists():
            return candidate
    return REPO_ROOT / rel


DATA_DIR = _resolve("IIS_DATA_DIR", "data", "interviews")
EVAL_PATH = _resolve("IIS_EVAL_PATH", "data", "eval", "queries.yaml")
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

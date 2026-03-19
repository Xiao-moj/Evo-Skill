"""
Vector backend factory — Evo-skill edition.
Only the flat (local file) backend is included.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .base import VectorIndex
from .flat import FlatFileVectorIndex


def build_vector_index(
    *,
    backend: str,
    dir_path: str,
    name: str = "skills",
    config: Optional[Dict[str, Any]] = None,
) -> VectorIndex:
    """Build a vector index. Currently only 'flat' is supported."""
    key = str(backend or "flat").strip().lower()
    if key in {"flat", "local", "file", "disk", "filesystem"}:
        return FlatFileVectorIndex(dir_path=dir_path, name=name)
    raise ValueError(f"Unknown vector backend: {key!r}. Supported: flat")


def register_vector_backend(*args, **kwargs) -> None:  # noqa: ANN001
    """Stub for compatibility with EvoSkill store factory."""
    pass


def list_vector_backends():
    return ["flat"]

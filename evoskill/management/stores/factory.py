"""
Store factory — Evo-skill edition.
Supports: local (filesystem) and inmemory.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Dict, List, Tuple

from ...config import EvoSkillConfig, default_store_path
from ...embeddings.factory import build_embeddings
from ..vectors import build_vector_index
from .base import SkillStore
from .inmemory import InMemorySkillStore
from .local import LocalSkillStore


_SAFE_NAME_RE = re.compile(r"[^a-z0-9]+")


def _slug(value: str, *, max_len: int = 32) -> str:
    s = str(value or "").strip().lower()
    s = _SAFE_NAME_RE.sub("-", s).strip("-")
    return (s or "model")[: max(1, int(max_len))]


def _embedding_signature(config: EvoSkillConfig) -> dict:
    emb = dict(config.embeddings or {})
    provider = str(emb.get("provider") or "hashing").strip().lower()
    model = str(emb.get("model") or "").strip()
    if not model:
        model = {
            "openai": "text-embedding-3-small",
            "bge-m3": "bge-m3",
            "bge_m3": "bge-m3",
        }.get(provider, "default")

    dimensions = None
    if provider == "hashing":
        try:
            dimensions = int(emb.get("dims", 256))
        except Exception:
            dimensions = 256

    sig: Dict[str, Any] = {"provider": provider, "model": model}
    if dimensions is not None:
        sig["dimensions"] = int(dimensions)
    return sig


def _vector_index_name(config: EvoSkillConfig) -> str:
    sig = _embedding_signature(config)
    provider = str(sig.get("provider") or "emb")
    model = str(sig.get("model") or "model")
    dims = sig.get("dimensions")

    payload = json.dumps(sig, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha1(payload.encode()).hexdigest()[:10]

    parts = ["skills", _slug(provider, max_len=16), _slug(model, max_len=28)]
    if isinstance(dims, int) and dims > 0:
        parts.append(f"d{dims}")
    parts.append(digest)
    return "-".join(p for p in parts if p)


def build_store(config: EvoSkillConfig) -> SkillStore:
    provider = (config.store.get("provider") or "inmemory").lower()
    embeddings = build_embeddings(config.embeddings)
    try:
        bm25_weight = float(config.store.get("bm25_weight", config.bm25_weight))
    except Exception:
        bm25_weight = float(config.bm25_weight)

    if provider == "inmemory":
        return InMemorySkillStore(embeddings=embeddings, bm25_weight=bm25_weight)

    if provider in {"local", "dir", "skill_dir", "skills", "filesystem"}:
        path = str(
            config.store.get("path")
            or config.store.get("root_dir")
            or config.store.get("dir")
            or default_store_path()
        )
        vector_index_name = str(
            config.store.get("vector_index_name")
            or _vector_index_name(config)
        )
        vector_cache_dir = os.path.join(path, "vectors")
        vector_index = build_vector_index(
            backend="flat",
            dir_path=vector_cache_dir,
            name=vector_index_name,
        )

        return LocalSkillStore(
            embeddings=embeddings,
            bm25_weight=bm25_weight,
            path=path,
            max_depth=int(config.store.get("max_depth", 6)),
            cache_vectors=True,
            vector_cache_dirname="vectors",
            vector_index_name=vector_index_name,
            vector_index=vector_index,
            vector_backend_name="flat",
            users_dirname=str(config.store.get("users_dirname", "Users")),
            libraries_dirname=str(config.store.get("libraries_dirname", "Common")),
            library_dirs=None,
            include_libraries=bool(config.store.get("include_libraries", True)),
            include_legacy_root=False,
            keyword_index_dirname="index",
            bm25_index_name="skills-bm25",
            bm25_startup_mode="incremental",
            bm25_health_strict=False,
        )

    raise ValueError(f"Unknown store provider: {provider!r}. Supported: local, inmemory")

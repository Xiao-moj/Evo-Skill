"""
Embeddings factory — Evo-skill edition.

Supported providers:
- openai  : OpenAI Embeddings API (and any OpenAI-compatible endpoint)
- bge-m3  : Local BAAI/bge-m3 model (loaded from HuggingFace cache)
- hashing : Offline hash-based embeddings (no API key needed, for testing)
"""

from __future__ import annotations

from typing import Any, Dict

from .base import EmbeddingModel
from .openai import OpenAIEmbedding
from .hashing import HashingEmbedding
from .bge_m3 import BGEM3Embedding


def build_embeddings(config: Dict[str, Any]) -> EmbeddingModel:
    """Build an EmbeddingModel from a provider config dict."""
    cfg = dict(config or {})
    provider = str(cfg.get("provider") or "openai").strip().lower()

    if provider in {"openai", "openai-compatible", "openai_compatible", "generic", "universal", "custom"}:
        return OpenAIEmbedding(
            model=str(cfg.get("model", "text-embedding-3-small")),
            api_key=cfg.get("api_key"),
            base_url=str(cfg.get("base_url", "https://api.openai.com")),
            timeout_s=int(cfg.get("timeout_s", 60)),
            max_text_chars=int(cfg.get("max_text_chars", 10_000)),
            min_text_chars=int(cfg.get("min_text_chars", 512)),
            max_batch_size=int(cfg.get("max_batch_size", 256)),
            extra_body=cfg.get("extra_body") or cfg.get("extra_payload"),
        )

    if provider in {"bge-m3", "bge_m3", "bgem3", "bge"}:
        return BGEM3Embedding(
            model_name_or_path=str(cfg.get("model_name_or_path") or cfg.get("model") or "BAAI/bge-m3"),
            batch_size=int(cfg.get("batch_size", 12)),
            max_length=int(cfg.get("max_length", 512)),
            use_fp16=bool(cfg.get("use_fp16", False)),
            device=cfg.get("device"),
        )

    if provider == "hashing":
        return HashingEmbedding(dims=int(cfg.get("dims", 256)))

    raise ValueError(
        f"Unknown embeddings provider: {provider!r}. Supported: openai, bge-m3, hashing"
    )

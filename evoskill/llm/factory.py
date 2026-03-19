"""
LLM factory — Evo-skill edition.

Supported providers:
- openai   : OpenAI Chat Completions (and any OpenAI-compatible endpoint)
- anthropic: Anthropic Messages API
"""

from __future__ import annotations

import os
from typing import Any, Dict

from .base import LLM
from .openai import OpenAIChatLLM
from .anthropic import AnthropicLLM


def build_llm(config: Dict[str, Any]) -> LLM:
    """Build an LLM from a provider config dict."""
    cfg = dict(config or {})
    provider = str(cfg.get("provider") or "openai").strip().lower()

    if provider in {"openai", "openai-compatible", "openai_compatible", "generic", "universal", "custom"}:
        return OpenAIChatLLM(
            model=str(cfg.get("model", "gpt-4o-mini")),
            api_key=cfg.get("api_key"),
            base_url=str(cfg.get("base_url", "https://api.openai.com")),
            timeout_s=int(cfg.get("timeout_s", 60)),
            max_input_chars=int(cfg.get("max_input_chars", 100_000)),
            max_tokens=int(cfg.get("max_tokens", 30_000)),
        )

    if provider == "anthropic":
        return AnthropicLLM(
            model=str(cfg.get("model", "claude-sonnet-4-6")),
            api_key=cfg.get("api_key"),
            base_url=str(cfg.get("base_url", "https://api.anthropic.com")),
            timeout_s=int(cfg.get("timeout_s", 60)),
            max_input_chars=int(cfg.get("max_input_chars", 100_000)),
            max_tokens=int(cfg.get("max_tokens", 30_000)),
        )

    raise ValueError(
        f"Unknown LLM provider: {provider!r}. Supported: openai, anthropic"
    )

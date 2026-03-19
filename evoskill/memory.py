"""Helpers for distinguishing reusable skills from failure experiences."""

from __future__ import annotations

from typing import Any, Dict


MEMORY_KIND_SKILL = "skill"
MEMORY_KIND_EXPERIENCE = "experience"


def normalize_memory_kind(value: Any) -> str:
    """Returns the normalized memory kind."""

    kind = str(value or "").strip().lower()
    if kind == MEMORY_KIND_EXPERIENCE:
        return MEMORY_KIND_EXPERIENCE
    return MEMORY_KIND_SKILL


def skill_memory_kind(skill: Any) -> str:
    """Returns the persisted memory kind for a skill-like object."""

    metadata = dict(getattr(skill, "metadata", {}) or {})
    return normalize_memory_kind(metadata.get("memory_kind"))


def candidate_memory_kind(candidate: Any) -> str:
    """Returns the memory kind for a candidate-like object."""

    metadata = dict(getattr(candidate, "metadata", {}) or {})
    return normalize_memory_kind(metadata.get("memory_kind"))


def merge_memory_metadata(
    metadata: Dict[str, Any] | None,
    *,
    memory_kind: str,
    defaults: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Ensures memory metadata carries a stable kind and optional defaults."""

    out = dict(metadata or {})
    if defaults:
        for key, value in defaults.items():
            if value is not None and key not in out:
                out[str(key)] = value
    out["memory_kind"] = normalize_memory_kind(memory_kind)
    return out

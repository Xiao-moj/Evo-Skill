"""
LLM-based task capability analyzer.

Given a user query, decomposes it into a list of sub-capabilities needed to complete
the task. These are used for capability-based pre-recall before embedding search.
"""

from __future__ import annotations

import json
from typing import List

from ..llm.base import LLM
from ..utils.json import json_from_llm_text


_SYSTEM = (
    "You are a task decomposition assistant.\n"
    "Given a user query, output the list of technical sub-capabilities needed to complete it.\n"
    "Each capability should be a short, task-oriented phrase (3-8 words).\n"
    "Focus on WHAT needs to be done, not HOW (e.g. 'sort an array' not 'use quicksort').\n"
    "\n"
    "Output ONLY strict JSON, no Markdown, no commentary:\n"
    "{\"capabilities\": [\"...\", \"...\"]}\n"
    "\n"
    "Rules:\n"
    "- 2 to 6 capabilities per query.\n"
    "- If the query is simple and single-purpose, output just 1-2.\n"
    "- Match the query language (Chinese query → Chinese capabilities).\n"
    "- If the query has no clear technical sub-tasks, output {\"capabilities\": []}.\n"
)


class LLMCapabilityAnalyzer:
    """Decomposes a user query into required sub-capabilities for pre-recall."""

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    def analyze(self, query: str) -> List[str]:
        """Returns a list of capability phrases for the given query."""
        q = (query or "").strip()
        if not q:
            return []
        try:
            text = self._llm.complete(
                system=_SYSTEM,
                user=json.dumps({"query": q}, ensure_ascii=False),
                temperature=0.0,
            )
        except Exception:
            return []

        try:
            parsed = json_from_llm_text(text or "")
        except Exception:
            return []

        caps = (parsed or {}).get("capabilities") if isinstance(parsed, dict) else None
        if not isinstance(caps, list):
            return []
        return [str(c).strip() for c in caps if str(c).strip()]

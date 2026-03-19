"""
Skill quality reviewer for Evo-skill.

Runs after maintenance to ensure:
- New skills meet a quality bar (clear description, valid capabilities, usable script)
- Merged skills didn't regress (merged result covers previous functionality)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..llm.base import LLM
from ..models import Skill
from ..utils.json import json_from_llm_text


_HISTORY_KEY = "_evoskill_version_history"

_REVIEW_NEW_SYSTEM = """\
You are a skill quality reviewer for a personal CS knowledge base.
Review a newly extracted skill and decide if it's worth keeping.

Check ALL of the following:
1. **Description**: Is it clear, specific, and accurate? (not vague like "useful algorithm")
2. **Capabilities**: Do they accurately describe what the skill can do? Are they specific enough to match user queries?
3. **Script**: Does the script look functionally correct and complete? (conceptual check — no execution)
4. **Overall**: Is this skill genuinely reusable and worth saving?

Output ONLY strict JSON:
{"approved": true/false, "score": 0.0-1.0, "reason": "one sentence"}

- score >= 0.6 → approved = true
- score < 0.6 → approved = false
- reason must explain WHY (e.g. "Script missing edge case handling" or "Capabilities too vague")
"""

_REVIEW_MERGE_SYSTEM = """\
You are a skill quality reviewer for a personal CS knowledge base.
A skill was just merged (updated). Review whether the merge improved or regressed the skill.

Check ALL of the following:
1. **Coverage**: Does the merged skill preserve ALL functionality described in the previous version?
2. **Quality**: Is the merged description/instructions clearer and more complete than before?
3. **Regression**: Are there any concepts, steps, or capabilities present in the PREVIOUS version that are now missing or weakened?

Output ONLY strict JSON:
{"approved": true/false, "score": 0.0-1.0, "reason": "one sentence", "regression": true/false}

- approved = false only if there is clear regression (important content was lost)
- regression = true if ANY previous functionality is missing or weakened
- reason must be specific about what improved or what was lost
"""


@dataclass
class ReviewResult:
    approved: bool
    score: float
    reason: str
    regression: bool = False  # only meaningful for merged skills
    is_merge: bool = False


class LLMSkillReviewer:
    """Reviews skill quality after extraction + maintenance."""

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    def review(self, skill: Skill) -> ReviewResult:
        """
        Reviews a skill. Automatically detects whether it's new or merged
        based on version history presence.
        """
        history: List[Dict] = []
        meta = dict(skill.metadata or {})
        hist_raw = meta.get(_HISTORY_KEY)
        if isinstance(hist_raw, list):
            history = [h for h in hist_raw if isinstance(h, dict)]

        if history:
            return self._review_merge(skill, previous_snapshot=history[-1])
        else:
            return self._review_new(skill)

    def _review_new(self, skill: Skill) -> ReviewResult:
        skill_data = _skill_summary(skill)
        try:
            text = self._llm.complete(
                system=_REVIEW_NEW_SYSTEM,
                user=json.dumps({"skill": skill_data}, ensure_ascii=False),
                temperature=0.0,
            )
            parsed = json_from_llm_text(text or "")
        except Exception:
            # Reviewer failure → approve by default (don't block saves)
            return ReviewResult(approved=True, score=0.7, reason="reviewer unavailable")

        return _parse_review_result(parsed, is_merge=False)

    def _review_merge(self, skill: Skill, previous_snapshot: Dict[str, Any]) -> ReviewResult:
        skill_data = _skill_summary(skill)
        prev_data = {
            "name": previous_snapshot.get("name", ""),
            "description": previous_snapshot.get("description", ""),
            "instructions": previous_snapshot.get("instructions", ""),
        }
        try:
            text = self._llm.complete(
                system=_REVIEW_MERGE_SYSTEM,
                user=json.dumps(
                    {"previous_version": prev_data, "merged_skill": skill_data},
                    ensure_ascii=False,
                ),
                temperature=0.0,
            )
            parsed = json_from_llm_text(text or "")
        except Exception:
            return ReviewResult(approved=True, score=0.7, reason="reviewer unavailable", is_merge=True)

        return _parse_review_result(parsed, is_merge=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _skill_summary(skill: Skill) -> Dict[str, Any]:
    """Compact skill representation for reviewer prompts."""
    meta = dict(skill.metadata or {})
    capabilities = meta.get("capabilities", [])
    files = dict(skill.files or {})
    script_keys = [k for k in files if k.startswith("scripts/") and k.endswith(".py")]
    script_preview = ""
    if script_keys:
        code = files[script_keys[0]]
        # Send first 60 lines to keep prompt size reasonable
        lines = (code or "").splitlines()[:60]
        script_preview = "\n".join(lines)

    return {
        "name": skill.name,
        "description": skill.description,
        "instructions": (skill.instructions or "")[:800],
        "capabilities": capabilities[:10],
        "triggers": list(skill.triggers or [])[:6],
        "tags": list(skill.tags or [])[:8],
        "has_script": bool(script_keys),
        "script_preview": script_preview,
    }


def _parse_review_result(parsed: Any, *, is_merge: bool) -> ReviewResult:
    """Parses LLM review JSON into a ReviewResult, with safe defaults."""
    if not isinstance(parsed, dict):
        return ReviewResult(approved=True, score=0.7, reason="parse error", is_merge=is_merge)

    try:
        score = float(parsed.get("score", 0.7))
    except (TypeError, ValueError):
        score = 0.7

    approved = bool(parsed.get("approved", score >= 0.6))
    reason = str(parsed.get("reason", "")).strip() or "no reason provided"
    regression = bool(parsed.get("regression", False))

    return ReviewResult(
        approved=approved,
        score=score,
        reason=reason,
        regression=regression,
        is_merge=is_merge,
    )

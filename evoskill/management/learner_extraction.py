"""
Learner-mode skill extractor for Evo-skill.

Unlike the default LLMSkillExtractor (which targets user-specific preferences and workflows),
this extractor targets CS/programming knowledge patterns:
- Algorithm implementations (sort, search, graph, DP, etc.)
- Problem-solving techniques (two-pointer, sliding window, BFS/DFS, etc.)
- Data structure operations (stack, heap, trie, union-find, etc.)

Each extracted skill produces:
- SKILL.md  : description, triggers, core idea, complexity, pitfalls
- scripts/<name>.py : a standalone, importable Python module with the implementation
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from ..config import EvoSkillConfig
from ..llm.base import LLM
from ..llm.factory import build_llm
from ..models import SkillExample
from ..utils.json import json_from_llm_text
from .extraction import SkillCandidate, _candidate_from_obj, _candidate_from_freeform_llm_text


# ── Extraction prompt ─────────────────────────────────────────────────────────

_LEARNER_SYSTEM = """You are Evo-skill's CS Knowledge Extractor for a computer science programmer.

Task: Extract reusable CS knowledge skills from the conversation.
Extract algorithm patterns, data structures, and problem-solving techniques.
Think of each skill as a flashcard + runnable code module the programmer can reuse and compose.

### WHEN TO EXTRACT
Extract when the conversation contains ANY reusable technical knowledge — do NOT limit yourself to a predefined list of domains.

**A concept is worth extracting if it meets ANY of these criteria:**
1. It has a name (an algorithm, formula, technique, pattern, method, theorem, etc.)
2. It can be implemented as a standalone, reusable function
3. A learner would want to look it up or reuse it in a future task
4. It has non-trivial logic that is worth remembering (not trivial one-liners)

**When in doubt, extract it.** A false positive is better than missing a reusable concept.

**Do NOT extract:**
- Pure Q&A with no implementable concept
- Simple one-liners (e.g., `sorted(arr)`, `sum(arr)`)
- Conversational messages with no technical content

**Examples of domains to extract from (not exhaustive):**
- Classic CS: sorting (quicksort, mergesort), search (binary search, BFS, DFS), two-pointer, sliding window, DP, graph algorithms, string algorithms, data structures, bit manipulation
- Machine Learning: loss functions (cross-entropy, KL divergence), optimizers (SGD, Adam), backpropagation, attention, regularization, evaluation metrics, sampling methods
- Probability & Statistics: distributions, KL divergence, mutual information, entropy, Bayesian methods, statistical tests
- Numerical Methods & Math: linear algebra, gradients/derivatives, Jacobian, Hessian, FFT, numerical optimization
- Systems & Engineering: concurrency patterns, design patterns, caching (LRU, LFU), etc.
- Any other named technical concept in any domain

### OUTPUT FORMAT
Output ONLY strict JSON parseable by json.loads:
{
  "skills": [
    {
      "name": "quicksort",
      "description": "1-2 sentences: WHAT it does, WHEN to use it",
      "prompt": "Markdown with sections: # Core Idea, # When to Use, # Complexity, # Common Pitfalls",
      "triggers": ["sort an array", "need O(n log n) in-place sorting", "compute KL divergence", "calculate gradient", ...],
      "tags": ["sorting", "divide-and-conquer", "recursion"],
      "capabilities": ["sort an array efficiently", "partition elements around a pivot", ...],
      "examples": [{"input": "Sort [3,1,4,1,5,9]"}, ...],
      "confidence": 0.9,
      "script_name": "quicksort",
      "script_code": "complete standalone Python code here"
    }
  ]
}

### CAPABILITIES FIELD
`capabilities`: 3-8 task-oriented phrases describing what problems this skill can solve.
- Focus on the task, not the implementation: "sort a list", not "use quicksort"
- Must be specific enough to match a user's task description
- Examples: "find median in O(n)", "compute KL divergence between two distributions", "detect cycle in a linked list"

### SCRIPT REQUIREMENTS (script_code field)
- A complete, standalone Python module — no external dependencies beyond stdlib
- One primary function named after the skill (e.g. def quicksort(arr): ...)
- Clear docstring explaining parameters and return value
- Handle edge cases (empty input, single element, etc.)
- Include a brief if __name__ == "__main__": block with a usage example
- Add type hints
- Comments in the same language as the conversation

### SKILL PROMPT SECTIONS (Markdown)
1) # Core Idea       — the key invariant/insight in 1-3 sentences
2) # When to Use     — problem signals that suggest this skill
3) # Complexity      — time and space complexity
4) # Common Pitfalls — edge cases and common mistakes (optional)

### LANGUAGE RULE
- Match conversation language for name/description/prompt/triggers/tags (Chinese → Chinese).
- script_code is always Python regardless of conversation language.

If no recognizable CS concept is present, output {"skills": []}.
JSON validity: escape newlines as \\n. No Markdown code fences around the JSON output.
"""


# ── Script filename helper ─────────────────────────────────────────────────────

_SAFE_RE = re.compile(r"[^\w]+")


def _script_filename(name: str) -> str:
    slug = _SAFE_RE.sub("_", str(name or "skill").strip().lower()).strip("_")
    return f"scripts/{slug or 'skill'}.py"


# ── Extractor ─────────────────────────────────────────────────────────────────

class LearnerSkillExtractor:
    """
    Skill extractor optimized for CS students.

    Each extracted skill includes:
    - SKILL.md  (via normal maintenance pipeline)
    - scripts/<name>.py  (standalone runnable Python module)
    """

    def __init__(self, config: EvoSkillConfig, *, llm: Optional[LLM] = None) -> None:
        self._config = config
        self._llm = llm or build_llm(config.llm)

    def extract(
        self,
        *,
        user_id: str,
        messages: Optional[List[Dict[str, Any]]],
        events: Optional[List[Dict[str, Any]]],
        max_candidates: int,
        hint: Optional[str] = None,
        retrieved_reference: Optional[Dict[str, Any]] = None,
    ) -> List[SkillCandidate]:
        if not messages and not events:
            return []

        conversation = _format_conversation(messages or [], events or [])
        if not conversation.strip():
            return []

        payload = {
            "conversation": conversation,
            "max_candidates": max_candidates,
            "hint": str(hint).strip() if hint else None,
        }

        try:
            text = self._llm.complete(system=_LEARNER_SYSTEM, user=json.dumps(payload, ensure_ascii=False))
        except Exception as e:
            print(f"\033[31m[Extractor LLM error: {e}]\033[0m", flush=True)
            return []

        if not (text or "").strip():
            return []

        try:
            parsed = json_from_llm_text(text)
        except Exception:
            recovered = _candidate_from_freeform_llm_text(text, source=None)
            return [recovered][:max_candidates] if recovered else []

        skills_obj = parsed.get("skills") if isinstance(parsed, dict) else parsed
        if not isinstance(skills_obj, list):
            return []

        out: List[SkillCandidate] = []
        for item in skills_obj[:max_candidates]:
            cand = _candidate_from_obj(item, source=None)
            if cand is None:
                continue

            # Attach the generated script as an extra file
            script_code = str(item.get("script_code") or "").strip()
            script_name = str(item.get("script_name") or cand.name or "skill").strip()
            if script_code:
                cand.files = {_script_filename(script_name): script_code}

            # Attach capabilities to metadata for pre-recall
            caps = item.get("capabilities")
            if isinstance(caps, list):
                cand.metadata = {"capabilities": [str(c).strip() for c in caps if str(c).strip()]}

            out.append(cand)

        return out


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_conversation(
    messages: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    for m in messages:
        role = str(m.get("role") or "user").strip().lower()
        content = str(m.get("content") or "").strip()
        if content:
            lines.append(f"[{role}] {content}")
    for e in events:
        try:
            lines.append(f"[event] {json.dumps(e, ensure_ascii=False)}")
        except Exception:
            pass
    return "\n\n".join(lines)

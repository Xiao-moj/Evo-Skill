"""Failure-experience extraction from Docker agent execution traces."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from ..agent_runtime.agent_context import AgentContext, AgentTrajectoryEvent
from ..config import EvoSkillConfig
from ..llm.base import LLM
from ..llm.factory import build_llm
from ..memory import MEMORY_KIND_EXPERIENCE, merge_memory_metadata
from ..models import SkillExample
from ..utils.json import json_from_llm_text
from ..utils.redact import redact_obj
from ..utils.text import keywords
from .agent_trajectory_extraction import (
    _coerce_trajectory_events,
    _collect_observed_paths,
    _collect_observed_tools,
    _format_trajectory_summary,
    _truncate_text,
)
from .extraction import (
    SkillCandidate,
    _candidate_from_freeform_llm_text,
    _candidate_from_obj,
)


class AgentFailureExperienceExtractor:
    """Extracts reusable failure guardrails from unsuccessful agent runs."""

    def __init__(self, config: EvoSkillConfig, *, llm: Optional[LLM] = None) -> None:
        self._config = config
        provider = str((config.llm or {}).get("provider") or "mock").strip().lower()
        if llm is not None:
            self._llm = llm
        elif provider == "mock":
            self._llm = None
        else:
            self._llm = build_llm(config.llm)

    def extract_from_context(
        self,
        *,
        user_id: str,
        context: AgentContext,
        max_candidates: int,
        hint: Optional[str] = None,
    ) -> List[SkillCandidate]:
        """Extracts failure experiences from one completed agent run context."""

        metadata = dict(context.metadata or {})
        return self.extract(
            user_id=user_id,
            instruction=str(metadata.get("instruction") or "").strip(),
            trajectory=context.trajectory,
            assistant_reply=str(context.response_content or "").strip(),
            success=bool(context.success),
            max_candidates=max_candidates,
            hint=hint,
            runtime_metadata=metadata,
        )

    def extract(
        self,
        *,
        user_id: str,
        instruction: str,
        trajectory: Optional[Sequence[AgentTrajectoryEvent | Dict[str, Any]]],
        assistant_reply: str,
        success: bool,
        max_candidates: int,
        hint: Optional[str] = None,
        runtime_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SkillCandidate]:
        """Extracts guardrails only when the agent run failed."""

        if success:
            return []

        normalized_events = _coerce_trajectory_events(trajectory)
        if not instruction.strip() and not normalized_events and not assistant_reply.strip():
            return []

        backend = str(
            (runtime_metadata or {}).get("backend")
            or (runtime_metadata or {}).get("trajectory_backend")
            or ""
        ).strip()
        payload = {
            "user_id": user_id,
            "instruction": instruction.strip() or None,
            "assistant_reply": assistant_reply.strip() or None,
            "trajectory_events": normalized_events,
            "trajectory_summary": _format_trajectory_summary(normalized_events),
            "failure_signals": _collect_failure_signals(normalized_events, assistant_reply=assistant_reply),
            "observed_tools": _collect_observed_tools(normalized_events),
            "observed_paths": _collect_observed_paths(normalized_events),
            "runtime_metadata": dict(runtime_metadata or {}),
            "max_candidates": int(max_candidates),
            "hint": (str(hint).strip() if hint and str(hint).strip() else None),
        }
        if self._config.redact_sources_before_llm:
            payload = redact_obj(payload)

        if self._llm is None:
            return _heuristic_extract_failure_experience(
                payload,
                max_candidates=max_candidates,
                backend=backend,
            )

        system = (
            "You are EvoSkill's Failure Experience Extractor.\n"
            "Task: turn a FAILED CLI-agent execution trace into reusable negative experience guardrails.\n"
            "These are not success skills. They are fences that stop the agent from repeating the same class of mistake.\n"
            "\n"
            "### WHAT TO EXTRACT\n"
            "- concrete failure patterns that are likely to recur\n"
            "- missing pre-checks, wrong assumptions, unsafe retries, invalid tool usage, bad environment setup, or skipped validation\n"
            "- a safer workflow that should be followed next time\n"
            "\n"
            "### WHAT NOT TO EXTRACT\n"
            "- one-off project facts with no reusable lesson\n"
            "- vague advice like 'be careful' or 'debug better'\n"
            "- successful workflow steps unless they are required to avoid the failure\n"
            "\n"
            "### OUTPUT INTENT\n"
            "- Each item should read like a reusable guardrail.\n"
            "- Focus on: what failed, what to avoid, what to check first, and how to validate before retrying.\n"
            "- Generalize repo-specific paths/entities into placeholders.\n"
            "- If failure evidence is weak, return {\"skills\": []}.\n"
            "\n"
            "### OUTPUT FORMAT\n"
            "Return ONLY strict JSON parseable by json.loads:\n"
            "{\"skills\": [{"
            "\"name\": str, "
            "\"description\": str, "
            "\"prompt\": str, "
            "\"triggers\": [str], "
            "\"tags\": [str], "
            "\"examples\": [{\"input\": str, \"output\": str|null, \"notes\": str|null}], "
            "\"confidence\": float"
            "}]}.\n"
            f"Return at most {max_candidates} items.\n"
            "- `name` should describe the failure class / guardrail, not the topical task.\n"
            "- `description` should state the failed pattern and when to recall the guardrail.\n"
            "- `prompt` must be Markdown with EXACTLY these sections:\n"
            "  1) # Failure Pattern\n"
            "  2) # Avoid\n"
            "  3) # Safer Workflow\n"
            "  4) # Validation\n"
            "- `triggers` should be short recall phrases.\n"
            "- `tags` should include tool/error/workflow keywords when useful.\n"
            "JSON validity: escape newlines inside strings as \\n. No Markdown code fences."
        )

        try:
            text = self._llm.complete(system=system, user=json.dumps(payload, ensure_ascii=False))
        except Exception:
            return []

        if not (text or "").strip():
            return []

        try:
            parsed = json_from_llm_text(text)
        except Exception:
            recovered = _candidate_from_freeform_llm_text(
                text,
                source={"kind": "agent_failure_experience", "instruction": instruction.strip()},
            )
            return (
                [_decorate_experience_candidate(recovered, backend=backend)][:max_candidates]
                if recovered is not None
                else []
            )

        skills_obj = parsed.get("skills") if isinstance(parsed, dict) else parsed
        if not isinstance(skills_obj, list):
            return []

        out: List[SkillCandidate] = []
        for item in skills_obj[:max_candidates]:
            candidate = _candidate_from_obj(
                item,
                source={"kind": "agent_failure_experience", "instruction": instruction.strip()},
            )
            if candidate is None:
                continue
            out.append(_decorate_experience_candidate(candidate, backend=backend))
        return out


def _decorate_experience_candidate(
    candidate: SkillCandidate,
    *,
    backend: str,
) -> SkillCandidate:
    """Marks a candidate as an experience memory."""

    candidate.metadata = merge_memory_metadata(
        candidate.metadata,
        memory_kind=MEMORY_KIND_EXPERIENCE,
        defaults={
            "experience_kind": "failure_guardrail",
            "experience_backend": str(backend or "").strip(),
        },
    )
    return candidate


def _collect_failure_signals(
    events: Sequence[Dict[str, Any]],
    *,
    assistant_reply: str,
) -> List[str]:
    """Collects concise failure messages from a failed run."""

    signals: List[str] = []
    seen: set[str] = set()
    for event in events:
        event_type = str(event.get("event_type") or "").strip().lower()
        status = str(event.get("status") or "").strip().lower()
        if event_type not in {"error", "result", "tool_result"} and status not in {"error", "failed"}:
            continue
        for chunk in (
            str(event.get("content") or "").strip(),
            str(event.get("tool_output") or "").strip(),
        ):
            text = _truncate_text(chunk, limit=220)
            if not text:
                continue
            if text in seen:
                continue
            seen.add(text)
            signals.append(text)
    reply_text = _truncate_text(str(assistant_reply or "").strip(), limit=220)
    if reply_text and reply_text not in seen:
        signals.append(reply_text)
    return signals[:8]


def _heuristic_extract_failure_experience(
    payload: Dict[str, Any],
    *,
    max_candidates: int,
    backend: str,
) -> List[SkillCandidate]:
    """Fallback extractor when no real LLM is configured."""

    instruction = str(payload.get("instruction") or "").strip()
    failure_signals = [str(x).strip() for x in (payload.get("failure_signals") or []) if str(x).strip()]
    tools = [str(x).strip() for x in (payload.get("observed_tools") or []) if str(x).strip()]
    if not instruction and not failure_signals:
        return []

    error_hint = failure_signals[0] if failure_signals else "a repeated CLI-agent execution failure"
    name_tokens = keywords(f"{instruction}\n{error_hint}", limit=3)
    if tools:
        base = "-".join(tool.lower().replace(" ", "-") for tool in tools[:2])
        name = f"{base}-failure-guardrail"
    elif name_tokens:
        name = "-".join(name_tokens) + "-failure-guardrail"
    else:
        name = "agent-failure-guardrail"

    description = (
        "Recall this guardrail when a similar CLI-agent task risks repeating the same failed assumption,"
        " tool misuse, or missing validation step."
    )
    if failure_signals:
        description = f"A failure guardrail for runs that show signals like: {failure_signals[0]}"

    safer_steps = [
        "- Re-read the task and confirm the exact success criterion before retrying.",
        "- Inspect the workspace and required files/tools before making another attempt.",
        "- If the previous step failed, change the approach instead of repeating the same command blindly.",
        "- Surface the concrete error and stop once the same failure pattern appears again.",
    ]
    if tools:
        safer_steps.insert(1, f"- Validate whether `{tools[0]}` is the correct tool before using it again.")

    instructions = "\n".join(
        [
            "# Failure Pattern",
            f"- The previous run failed while handling a task like: {instruction or '<TASK>'}",
            f"- Failure signal: {error_hint}",
            "",
            "# Avoid",
            "- Do not repeat the same failing command or tool call without a new hypothesis.",
            "- Do not assume environment, file paths, or tool availability without checking first.",
            "",
            "# Safer Workflow",
            *safer_steps,
            "",
            "# Validation",
            "- Confirm the changed approach addresses the previous error directly.",
            "- Verify outputs or logs before declaring success.",
            "- If the same failure reappears, report it explicitly instead of looping.",
        ]
    ).strip()

    candidate = SkillCandidate(
        name=name,
        description=description,
        instructions=instructions,
        triggers=[
            "Use when a similar agent task starts failing in the same way.",
            "Use when a retry needs extra pre-checks and validation.",
            "Use when the previous attempt ended with a tool or environment error.",
        ],
        tags=(tools[:4] + ["failure", "guardrail"])[:6],
        examples=[SkillExample(input=instruction)] if instruction else [],
        confidence=0.35,
        source={"kind": "agent_failure_experience", "instruction": instruction},
    )
    return [_decorate_experience_candidate(candidate, backend=backend)][:max_candidates]

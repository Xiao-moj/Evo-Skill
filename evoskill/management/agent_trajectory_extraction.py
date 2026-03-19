"""Skill extraction from Docker agent execution trajectories."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

from ..agent_runtime.agent_context import AgentContext, AgentTrajectoryEvent
from ..config import EvoSkillConfig
from ..llm.base import LLM
from ..llm.factory import build_llm
from ..models import SkillExample
from ..utils.json import json_from_llm_text
from ..utils.redact import redact_obj
from ..utils.text import keywords
from .extraction import (
    SkillCandidate,
    _candidate_from_freeform_llm_text,
    _candidate_from_obj,
)


_PATH_RE = re.compile(r"(?:^|[\s'\"(])((?:/|\.?/)?[\w.\-]+(?:/[\w.\-]+)+)")


class AgentTrajectoryExtractor:
    """Extracts reusable skills from normalized agent execution traces."""

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
        retrieved_reference: Optional[Dict[str, Any]] = None,
    ) -> List[SkillCandidate]:
        """Extracts skills from one completed agent run context."""

        metadata = dict(context.metadata or {})
        return self.extract(
            user_id=user_id,
            instruction=str(metadata.get("instruction") or "").strip(),
            trajectory=context.trajectory,
            assistant_reply=str(context.response_content or "").strip(),
            max_candidates=max_candidates,
            hint=hint,
            retrieved_reference=retrieved_reference,
            runtime_metadata=metadata,
        )

    def extract(
        self,
        *,
        user_id: str,
        instruction: str,
        trajectory: Optional[Sequence[AgentTrajectoryEvent | Dict[str, Any]]],
        assistant_reply: str,
        max_candidates: int,
        hint: Optional[str] = None,
        retrieved_reference: Optional[Dict[str, Any]] = None,
        runtime_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SkillCandidate]:
        """Extracts reusable workflow/tooling skills from a normalized agent trajectory."""

        normalized_events = _coerce_trajectory_events(trajectory)
        if not instruction.strip() and not normalized_events and not assistant_reply.strip():
            return []

        payload = {
            "user_id": user_id,
            "instruction": instruction.strip() or None,
            "assistant_reply": assistant_reply.strip() or None,
            "trajectory_events": normalized_events,
            "trajectory_summary": _format_trajectory_summary(normalized_events),
            "observed_tools": _collect_observed_tools(normalized_events),
            "observed_paths": _collect_observed_paths(normalized_events),
            "runtime_metadata": dict(runtime_metadata or {}),
            "max_candidates": int(max_candidates),
            "hint": (str(hint).strip() if hint and str(hint).strip() else None),
            "retrieved_reference": (
                dict(retrieved_reference) if isinstance(retrieved_reference, dict) else None
            ),
        }
        if self._config.redact_sources_before_llm:
            payload = redact_obj(payload)

        if self._llm is None:
            return _heuristic_extract_from_trajectory(payload, max_candidates=max_candidates)

        system = (
            "You are EvoSkill's Agent Trajectory Skill Extractor.\n"
            "Task: extract reusable Skills from an AI agent execution trace.\n"
            "The trace comes from a CLI agent running tools in a Docker workspace.\n"
            "\n"
            "### PRIMARY EVIDENCE ORDER\n"
            "1) DATA.instruction = the user's task request\n"
            "2) DATA.trajectory_summary / DATA.trajectory_events = what the agent actually tried\n"
            "3) DATA.assistant_reply = final outcome or failure message\n"
            "- Prefer extracting HOW the agent solved or attempted the task, not the topical answer itself.\n"
            "\n"
            "### WHAT IS A GOOD SKILL HERE\n"
            "- Multi-step workflows with reusable tool orchestration\n"
            "- Environment setup or login/configuration procedures\n"
            "- Debugging, inspection, validation, and retry loops\n"
            "- Reusable file-reading / file-editing / CLI-execution playbooks\n"
            "- Error diagnosis procedures that are likely to recur\n"
            "\n"
            "### WHAT NOT TO EXTRACT\n"
            "- One-off answer content with no reusable method\n"
            "- Project-specific facts, secret values, absolute paths, user identifiers\n"
            "- Generic 'use Bash to inspect files' advice unless the trace shows a durable workflow\n"
            "- Retrieved-reference metadata as extraction evidence; it is identity context only\n"
            "\n"
            "### EXTRACTION RULES\n"
            "- Focus on durable process, constraints, checks, and tool sequencing.\n"
            "- Generalize file paths and runtime specifics into placeholders like <WORKSPACE_PATH>.\n"
            "- If the trace mostly shows a failure, only extract when the failure-handling workflow is itself reusable.\n"
            "- Keep the skill transferable to similar future tasks with different repositories or topics.\n"
            "- Include observed tool usage only when it materially changes the workflow.\n"
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
            f"Return at most {max_candidates} skills.\n"
            "- `prompt` must be Markdown with:\n"
            "  1) # Goal\n"
            "  2) # Constraints & Style\n"
            "  3) # Workflow\n"
            "- `name` must describe the reusable workflow/capability, not the specific task payload.\n"
            "- `description` should say WHAT the workflow does and WHEN to use it.\n"
            "- `triggers` should be short, task-oriented intent phrases.\n"
            "- `tags` should capture tools/workflow/domain keywords.\n"
            "- If evidence is weak, return {\"skills\": []}.\n"
            "JSON validity: escape newlines inside strings as \\n. No Markdown code fences."
        )

        try:
            text = self._llm.complete(system=system, user=json.dumps(payload, ensure_ascii=False))
        except Exception as e:
            if (self._config.extra or {}).get("raise_on_llm_extract_error"):
                raise RuntimeError(f"Agent trajectory extract call failed: {e}") from e
            return []

        if not (text or "").strip():
            return []

        try:
            parsed = json_from_llm_text(text)
        except Exception:
            recovered = _candidate_from_freeform_llm_text(
                text,
                source={"kind": "agent_trajectory", "instruction": instruction.strip()},
            )
            return [recovered][:max_candidates] if recovered else []

        skills_obj = parsed.get("skills") if isinstance(parsed, dict) else parsed
        if not isinstance(skills_obj, list):
            return []

        out: List[SkillCandidate] = []
        for item in skills_obj[:max_candidates]:
            candidate = _candidate_from_obj(
                item,
                source={"kind": "agent_trajectory", "instruction": instruction.strip()},
            )
            if candidate is None:
                continue
            out.append(candidate)
        return out


def _coerce_trajectory_events(
    trajectory: Optional[Sequence[AgentTrajectoryEvent | Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """Converts trajectory events into plain dictionaries."""

    out: List[Dict[str, Any]] = []
    for event in trajectory or []:
        if isinstance(event, AgentTrajectoryEvent):
            out.append(event.model_dump(exclude_none=True))
        elif isinstance(event, dict):
            cleaned = {str(k): v for k, v in event.items() if v is not None}
            if cleaned:
                out.append(cleaned)
    return out


def _format_trajectory_summary(events: Sequence[Dict[str, Any]], *, max_lines: int = 80) -> str:
    """Formats normalized trajectory events into a compact plain-text trace."""

    lines: List[str] = []
    for idx, event in enumerate(events[:max_lines], start=1):
        event_type = str(event.get("event_type") or "event").strip()
        role = str(event.get("role") or "").strip()
        tool_name = str(event.get("tool_name") or "").strip()
        status = str(event.get("status") or "").strip()

        header = f"{idx}. [{event_type}"
        if role:
            header += f"/{role}"
        if tool_name:
            header += f"/{tool_name}"
        if status:
            header += f"/{status}"
        header += "]"

        body = str(event.get("content") or "").strip()
        if not body and event.get("tool_input") is not None:
            body = _truncate_text(_safe_json_dumps(event.get("tool_input")), limit=280)
        if not body and event.get("tool_output") is not None:
            body = _truncate_text(str(event.get("tool_output") or "").strip(), limit=280)

        if body:
            lines.append(f"{header} {body}")
        else:
            lines.append(header)
    return "\n".join(lines).strip()


def _collect_observed_tools(events: Sequence[Dict[str, Any]]) -> List[str]:
    """Collects unique tool names seen in the trajectory."""

    tools: List[str] = []
    seen: set[str] = set()
    for event in events:
        name = str(event.get("tool_name") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        tools.append(name)
    return tools[:20]


def _collect_observed_paths(events: Sequence[Dict[str, Any]]) -> List[str]:
    """Collects likely file paths mentioned in the trajectory."""

    hits: List[str] = []
    seen: set[str] = set()
    for event in events:
        chunks = [
            str(event.get("content") or "").strip(),
            _safe_json_dumps(event.get("tool_input")),
            str(event.get("tool_output") or "").strip(),
        ]
        for chunk in chunks:
            if not chunk:
                continue
            for match in _PATH_RE.findall(chunk):
                candidate = str(match or "").strip().strip("\"'()")
                if not candidate or candidate in seen:
                    continue
                seen.add(candidate)
                hits.append(candidate)
    return hits[:20]


def _heuristic_extract_from_trajectory(
    payload: Dict[str, Any],
    *,
    max_candidates: int,
) -> List[SkillCandidate]:
    """Fallback extractor used when no real LLM is configured."""

    instruction = str(payload.get("instruction") or "").strip()
    summary = str(payload.get("trajectory_summary") or "").strip()
    tools = [str(t).strip() for t in (payload.get("observed_tools") or []) if str(t).strip()]
    if not instruction and not summary:
        return []

    name = _heuristic_name(instruction=instruction, tools=tools)
    description = (
        "A reusable CLI-agent execution workflow for inspecting a workspace, using tools,"
        " and validating outcomes."
    )
    if tools:
        description = (
            "A reusable CLI-agent workflow that coordinates "
            f"{', '.join(tools[:4])} to inspect, act, and validate a task inside a workspace."
        )

    workflow_lines = _heuristic_workflow_lines(payload)
    instructions = "\n".join(
        [
            "# Goal",
            "Execute a repeatable agent workflow that solves a workspace task using CLI/tool steps.",
            "",
            "# Constraints & Style",
            "- Generalize repository-specific details into placeholders.",
            "- Inspect before mutating files or running impactful commands.",
            "- Preserve explicit validation and failure-reporting steps.",
            "",
            "# Workflow",
            *workflow_lines,
        ]
    ).strip()

    tags = tools[:6] or keywords(f"{instruction}\n{summary}", limit=5)
    examples = []
    if instruction:
        examples.append(SkillExample(input=instruction))

    return [
        SkillCandidate(
            name=name,
            description=description,
            instructions=instructions,
            triggers=[
                "Use when the task should be solved by a CLI agent inside a workspace.",
                "Use when you need a reusable tool-driven execution workflow.",
                "Use when inspection, action, and validation steps should be preserved.",
            ],
            tags=tags,
            examples=examples,
            confidence=0.35,
            source={"kind": "agent_trajectory", "instruction": instruction},
        )
    ][:max_candidates]


def _heuristic_name(*, instruction: str, tools: Sequence[str]) -> str:
    """Builds a simple heuristic name for the fallback extractor."""

    if tools:
        primary = "-".join(str(tool).strip().lower().replace(" ", "-") for tool in tools[:2])
        return f"{primary}-agent-workflow"
    kws = keywords(instruction, limit=2)
    if kws:
        return "-".join(kws) + "-agent-workflow"
    return "cli-agent-workflow"


def _heuristic_workflow_lines(payload: Dict[str, Any]) -> List[str]:
    """Builds workflow bullets from observed trajectory events."""

    events = list(payload.get("trajectory_events") or [])
    lines: List[str] = []
    seen: set[str] = set()
    for event in events[:12]:
        event_type = str(event.get("event_type") or "").strip()
        tool_name = str(event.get("tool_name") or "").strip()
        content = _truncate_text(str(event.get("content") or "").strip(), limit=120)
        if event_type == "tool_use" and tool_name:
            line = f"- Use `{tool_name}` when the workflow requires an external action or inspection step."
        elif event_type == "tool_result" and tool_name:
            line = f"- Review `{tool_name}` outputs before deciding on the next step."
        elif event_type == "error":
            line = "- Detect runtime failures explicitly and surface the error before retrying or aborting."
        elif event_type == "result":
            line = "- End with a concise final result or failure diagnosis."
        elif event_type == "message" and content:
            line = f"- Preserve important agent reasoning or status updates such as: {content}"
        else:
            continue
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)
    if not lines:
        lines.extend([
            "- Inspect the workspace and available resources before acting.",
            "- Execute the minimum necessary tool steps to make progress.",
            "- Validate the outcome and report either the result or a clear failure reason.",
        ])
    return lines


def _safe_json_dumps(value: Any) -> str:
    """Serializes arbitrary values for prompts/logging."""

    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _truncate_text(text: str, *, limit: int) -> str:
    """Truncates text to a prompt-friendly length."""

    s = str(text or "").strip()
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 3)].rstrip() + "..."

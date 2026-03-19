"""
Claude Code agent backend.
"""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path
from typing import Any, List, Optional

from ..agent_context import AgentContext
from ..installed_agent import BaseInstalledAgent, ExecInput


def _stringify_payload(value: Any) -> str:
    """Renders arbitrary provider payloads into readable text."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = [_stringify_payload(item) for item in value]
        return "\n".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        for key in ("text", "content", "output", "result", "message"):
            rendered = _stringify_payload(value.get(key))
            if rendered:
                return rendered
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value).strip()
    return str(value).strip()


def _normalize_claude_message_event(raw_event: dict[str, Any]) -> list[dict[str, Any]]:
    """Maps Claude message events into normalized trajectory events."""

    message = raw_event.get("message")
    if not isinstance(message, dict):
        return []

    role = str(message.get("role") or raw_event.get("type") or "").strip().lower() or None
    content = message.get("content")
    timestamp = raw_event.get("timestamp")
    base = {
        "backend": "claude-code",
        "raw_event": raw_event,
        "role": role,
        "timestamp": timestamp,
    }

    normalized: List[dict[str, Any]] = []
    if isinstance(content, str):
        rendered = content.strip()
        if rendered:
            normalized.append({**base, "event_type": "message", "content": rendered})
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                rendered = _stringify_payload(block)
                if rendered:
                    normalized.append({**base, "event_type": "message", "content": rendered})
                continue

            block_type = str(block.get("type") or "").strip().lower()
            if block_type == "text":
                rendered = _stringify_payload(block.get("text"))
                if rendered:
                    normalized.append({**base, "event_type": "message", "content": rendered})
                continue

            if block_type in {"tool_use", "server_tool_use"} or (
                block.get("name") and block.get("input") is not None
            ):
                normalized.append({
                    **base,
                    "event_type": "tool_use",
                    "tool_name": str(block.get("name") or block_type or "tool"),
                    "tool_input": block.get("input"),
                    "content": _stringify_payload(block.get("text")),
                })
                continue

            if block_type in {"tool_result", "server_tool_result"}:
                normalized.append({
                    **base,
                    "event_type": "tool_result",
                    "role": "system",
                    "tool_name": str(block.get("name") or block.get("tool_name") or "tool"),
                    "tool_output": _stringify_payload(block.get("content") or block.get("text")),
                })
                continue

            rendered = _stringify_payload(block)
            if rendered:
                normalized.append({
                    **base,
                    "event_type": "message",
                    "content": rendered,
                    "status": block_type or None,
                })

    if raw_event.get("error"):
        normalized.append({
            "backend": "claude-code",
            "event_type": "error",
            "role": role or "assistant",
            "content": _stringify_payload(raw_event.get("error")),
            "status": "error",
            "timestamp": timestamp,
            "raw_event": raw_event,
        })
    return normalized


def _normalize_claude_event(raw_event: dict[str, Any]) -> list[dict[str, Any]]:
    """Maps raw Claude Code events to normalized trajectory events."""

    event_type = str(raw_event.get("type") or "").strip().lower()
    timestamp = raw_event.get("timestamp")
    base = {
        "backend": "claude-code",
        "raw_event": raw_event,
        "timestamp": timestamp,
    }

    if event_type in {"assistant", "user"}:
        return _normalize_claude_message_event(raw_event)

    if event_type == "system":
        return [{
            **base,
            "event_type": "system",
            "role": "system",
            "status": str(raw_event.get("subtype") or "system").strip() or "system",
            "content": _stringify_payload({
                "cwd": raw_event.get("cwd"),
                "model": raw_event.get("model"),
                "session_id": raw_event.get("session_id"),
            }),
        }]

    if event_type == "result":
        status = "error" if raw_event.get("is_error", False) else str(raw_event.get("subtype") or "result")
        return [{
            **base,
            "event_type": "result",
            "role": "assistant",
            "status": status,
            "content": _stringify_payload(raw_event.get("result")),
            "duration_ms": float(raw_event.get("duration_ms") or 0) or None,
            "usage": raw_event.get("usage") if isinstance(raw_event.get("usage"), dict) else None,
        }]

    if event_type == "queue-operation":
        return [{
            **base,
            "event_type": "queue",
            "role": "user",
            "status": str(raw_event.get("operation") or "").strip() or None,
            "content": _stringify_payload(raw_event.get("content")),
        }]

    if event_type == "last-prompt":
        return [{
            **base,
            "event_type": "prompt",
            "role": "user",
            "content": _stringify_payload(raw_event.get("lastPrompt")),
        }]

    rendered = _stringify_payload(raw_event)
    if not rendered:
        return []
    return [{
        **base,
        "event_type": "raw",
        "role": "system",
        "status": event_type or None,
        "content": rendered,
    }]


class ClaudeCodeAgent(BaseInstalledAgent):
    """Runs Claude Code CLI inside Docker."""

    SUPPORTS_TRAJECTORY = True

    ALLOWED_TOOLS = [
        "Bash",
        "Edit",
        "Write",
        "Read",
        "Glob",
        "Grep",
        "WebFetch",
        "NotebookEdit",
        "Task",
        "Skill",
    ]

    def __init__(
        self,
        logs_dir: Path,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_thinking_tokens: Optional[int] = None,
        version: Optional[str] = None,
        **kwargs: object,
    ) -> None:
        super().__init__(logs_dir, model_name, version, **kwargs)
        self.api_key = api_key
        self.base_url = base_url
        self._max_thinking_tokens = max_thinking_tokens

    @staticmethod
    def name() -> str:
        return "claude-code"

    @property
    def _install_script_template_path(self) -> Path:
        return Path(__file__).resolve().parent.parent / "templates" / "install-claude-code.sh.j2"

    def create_run_commands(self, instruction: str) -> List[ExecInput]:
        escaped_instruction = shlex.quote(instruction)
        env = {}
        if self.api_key:
            env["ANTHROPIC_API_KEY"] = self.api_key
        elif os.environ.get("ANTHROPIC_API_KEY"):
            env["ANTHROPIC_API_KEY"] = os.environ["ANTHROPIC_API_KEY"]

        if self.base_url:
            env["ANTHROPIC_BASE_URL"] = self.base_url
        elif os.environ.get("ANTHROPIC_BASE_URL"):
            env["ANTHROPIC_BASE_URL"] = os.environ["ANTHROPIC_BASE_URL"]

        if self.model_name:
            env["ANTHROPIC_MODEL"] = self.model_name

        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        if self._max_thinking_tokens:
            env["MAX_THINKING_TOKENS"] = str(self._max_thinking_tokens)
        env["CLAUDE_CONFIG_DIR"] = "/logs/agent/sessions"
        env["PATH"] = "/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        env = {k: v for k, v in env.items() if v}

        setup_cmd = ExecInput(
            command=(
                "mkdir -p $CLAUDE_CONFIG_DIR/debug $CLAUDE_CONFIG_DIR/projects/-app "
                "$CLAUDE_CONFIG_DIR/shell-snapshots $CLAUDE_CONFIG_DIR/statsig "
                "$CLAUDE_CONFIG_DIR/todos && "
                "if [ -d /skills ]; then "
                "cp -r /skills $CLAUDE_CONFIG_DIR/skills 2>/dev/null || true; "
                "fi"
            ),
            env=env,
        )

        model_arg = f"--model {shlex.quote(self.model_name)} " if self.model_name else ""
        run_cmd = ExecInput(
            command=(
                "set -o pipefail && "
                f"claude --verbose --output-format stream-json "
                f"{model_arg}"
                f"-p {escaped_instruction} --allowedTools "
                f"{' '.join(self.ALLOWED_TOOLS)} 2>&1 </dev/null | tee "
                "/logs/agent/claude-code.txt"
            ),
            env=env,
            timeout_sec=600,
        )
        return [setup_cmd, run_cmd]

    def populate_context_post_run(self, context: AgentContext) -> None:
        print(f"[{self.name()}] Parsing ClaudeCode execution results...")
        current_session_id = None
        stream_log_file = self.logs_dir / "claude-code.txt"
        stream_events: List[dict] = []
        if stream_log_file.exists():
            for line in stream_log_file.read_text(errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                stream_events.append(event)
                if event.get("type") == "system" and event.get("subtype") == "init":
                    current_session_id = event.get("session_id")
        else:
            print(f"[{self.name()}] Stream log not found: {stream_log_file}")

        def has_meaningful_events(items: List[dict]) -> bool:
            for item in items:
                if item.get("type") in {"assistant", "result"}:
                    return True
                message = item.get("message", {})
                if isinstance(message, dict) and message.get("role") == "assistant":
                    return True
            return False

        target_jsonl = None
        jsonl_events: List[dict] = []
        sessions_root = self.logs_dir / "sessions"
        project_root = sessions_root / "projects" if sessions_root.exists() else None
        if project_root and project_root.is_dir():
            if current_session_id:
                matches = list(project_root.glob(f"**/{current_session_id}.jsonl"))
                if matches:
                    target_jsonl = matches[0]
                    print(f"[{self.name()}] Using current session: {current_session_id}")
            if target_jsonl is None:
                all_jsonl_files = list(project_root.glob("**/*.jsonl"))
                if all_jsonl_files:
                    target_jsonl = max(all_jsonl_files, key=lambda p: p.stat().st_mtime)
                    print(f"[{self.name()}] Fallback to latest JSONL: {target_jsonl.name}")
            if target_jsonl is not None:
                print(f"[{self.name()}] Parsing: {target_jsonl.name}")
                with open(target_jsonl, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            jsonl_events.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"[{self.name()}] Failed to parse JSONL line: {e}")
        else:
            print(f"[{self.name()}] No projects directory found, fallback to stream log")

        events: List[dict] = []
        source_name = ""
        if has_meaningful_events(jsonl_events):
            events = jsonl_events
            source_name = target_jsonl.name if target_jsonl else "jsonl"
        elif has_meaningful_events(stream_events):
            events = stream_events
            source_name = stream_log_file.name
        elif jsonl_events:
            events = jsonl_events
            source_name = target_jsonl.name if target_jsonl else "jsonl"
        elif stream_events:
            events = stream_events
            source_name = stream_log_file.name

        if not events:
            print(f"[{self.name()}] No parseable events found")
            context.success = False
            context.response_content = "No ClaudeCode events found in stream/jsonl logs."
            return

        print(f"[{self.name()}] Event source: {source_name}, count={len(events)}")
        trajectory_source = str(target_jsonl) if target_jsonl is not None else str(stream_log_file)
        context.update_metadata(
            trajectory_source=trajectory_source,
            trajectory_source_name=source_name,
            trajectory_backend=self.name(),
        )
        total_input_tokens = 0
        total_output_tokens = 0
        total_cache_tokens = 0
        response_contents: List[str] = []
        result_success = False
        errors: List[str] = []
        normalized_trajectory: List[dict[str, Any]] = []

        for event in events:
            normalized_trajectory.extend(_normalize_claude_event(event))
            if event.get("type") == "result":
                usage = event.get("usage", {})
                if usage:
                    total_input_tokens += usage.get("input_tokens", 0)
                    total_output_tokens += usage.get("output_tokens", 0)
                    total_cache_tokens += usage.get("cache_read_input_tokens", 0)
                result_text = event.get("result")
                if result_text:
                    response_contents.append(result_text)
                if event.get("subtype") == "success" and not event.get("is_error", False):
                    result_success = True
                if event.get("is_error", False) and result_text:
                    errors.append(result_text)

            message = event.get("message", {})
            if isinstance(message, dict) and message.get("role") == "assistant":
                content = message.get("content")
                if isinstance(content, str):
                    response_contents.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            response_contents.append(block.get("text", ""))
                usage = message.get("usage", {})
                if usage:
                    total_input_tokens += usage.get("input_tokens", 0)
                    total_output_tokens += usage.get("output_tokens", 0)
                    total_cache_tokens += usage.get("cache_read_input_tokens", 0)

        context.n_input_tokens = total_input_tokens or None
        context.n_output_tokens = total_output_tokens or None
        context.n_cache_tokens = total_cache_tokens or None
        context.extend_trajectory(normalized_trajectory)
        if errors:
            context.response_content = "\n".join(errors)
        elif response_contents:
            context.response_content = "\n\n".join(response_contents)
        context.success = result_success and not errors
        if not response_contents and not errors:
            context.success = False
            context.response_content = "ClaudeCode run did not produce a successful result event."
        context.update_metadata(trajectory_event_count=len(context.trajectory))

        print(f"[{self.name()}] Parsing completed:")
        print(f"  - Input tokens: {context.n_input_tokens}")
        print(f"  - Output tokens: {context.n_output_tokens}")
        print(f"  - Cache tokens: {context.n_cache_tokens}")
        print(f"  - Success: {context.success}")

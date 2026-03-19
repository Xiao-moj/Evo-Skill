"""
Codex agent backend.
"""

from __future__ import annotations

import json
import os
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
        for key in ("text", "content", "output", "result", "summary", "message"):
            rendered = _stringify_payload(value.get(key))
            if rendered:
                return rendered
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value).strip()
    return str(value).strip()


def _normalize_codex_item(raw_event: dict[str, Any]) -> list[dict[str, Any]]:
    """Maps Codex `item.completed` events to normalized trajectory events."""

    item = raw_event.get("item")
    if not isinstance(item, dict):
        return []

    item_type = str(item.get("type") or "").strip().lower()
    base = {
        "backend": "codex",
        "raw_event": raw_event,
        "status": "completed",
    }

    if item_type == "agent_message":
        content = _stringify_payload(item.get("text") or item.get("content"))
        if content:
            return [{**base, "event_type": "message", "role": "assistant", "content": content}]
        return []

    if item_type in {
        "tool_call",
        "tool_use",
        "function_call",
        "computer_call",
        "tool_request",
    } or (
        item.get("name")
        and (item.get("input") is not None or item.get("arguments") is not None)
    ):
        return [{
            **base,
            "event_type": "tool_use",
            "role": "assistant",
            "tool_name": str(item.get("name") or item.get("tool_name") or item_type or "tool"),
            "tool_input": item.get("input") if item.get("input") is not None else item.get("arguments"),
            "content": _stringify_payload(item.get("summary") or item.get("text")),
        }]

    if item_type in {
        "tool_result",
        "tool_output",
        "function_call_output",
        "computer_call_output",
    }:
        return [{
            **base,
            "event_type": "tool_result",
            "role": "system",
            "tool_name": str(item.get("name") or item.get("tool_name") or item_type or "tool"),
            "tool_output": _stringify_payload(
                item.get("output") or item.get("content") or item.get("text")
            ),
        }]

    if "reason" in item_type:
        content = _stringify_payload(item.get("summary") or item.get("text") or item.get("content"))
        if content:
            return [{**base, "event_type": "reasoning", "role": "assistant", "content": content}]
        return []

    rendered = _stringify_payload(item)
    if not rendered:
        return []
    return [{
        **base,
        "event_type": "item",
        "role": "assistant",
        "content": rendered,
        "status": item_type or "completed",
    }]


def _normalize_codex_event(raw_event: dict[str, Any]) -> list[dict[str, Any]]:
    """Maps raw Codex JSON events to normalized trajectory events."""

    event_type = str(raw_event.get("type") or "").strip()
    base = {
        "backend": "codex",
        "raw_event": raw_event,
    }

    if event_type == "item.completed":
        return _normalize_codex_item(raw_event)

    if event_type == "thread.started":
        return [{**base, "event_type": "system", "role": "system", "status": "thread.started"}]

    if event_type == "turn.started":
        return [{**base, "event_type": "turn", "role": "system", "status": "started"}]

    if event_type == "turn.completed":
        usage = raw_event.get("usage") if isinstance(raw_event.get("usage"), dict) else None
        return [{
            **base,
            "event_type": "result",
            "role": "system",
            "status": "completed",
            "usage": usage,
        }]

    if event_type == "turn.failed":
        error = raw_event.get("error")
        content = _stringify_payload(error)
        return [{
            **base,
            "event_type": "error",
            "role": "assistant",
            "status": "failed",
            "content": content or "Codex turn failed",
        }]

    if event_type == "error":
        content = _stringify_payload(raw_event.get("message"))
        return [{
            **base,
            "event_type": "error",
            "role": "assistant",
            "status": "error",
            "content": content or "Codex error",
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


class CodexAgent(BaseInstalledAgent):
    """Runs OpenAI Codex CLI inside Docker."""

    SUPPORTS_TRAJECTORY = True

    def __init__(
        self,
        logs_dir: Path,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        version: Optional[str] = None,
        **kwargs: object,
    ) -> None:
        super().__init__(logs_dir, model_name, version, **kwargs)
        self.api_key = api_key
        self.base_url = base_url

    @staticmethod
    def name() -> str:
        return "codex"

    @property
    def _install_script_template_path(self) -> Path:
        return Path(__file__).resolve().parent.parent / "templates" / "install-codex.sh.j2"

    def create_run_commands(self, instruction: str) -> List[ExecInput]:
        env = {}
        if self.api_key:
            env["OPENAI_API_KEY"] = self.api_key
        elif os.environ.get("OPENAI_API_KEY"):
            env["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]

        if self.base_url:
            env["OPENAI_BASE_URL"] = self.base_url
        elif os.environ.get("OPENAI_BASE_URL"):
            env["OPENAI_BASE_URL"] = os.environ["OPENAI_BASE_URL"]

        if self.model_name:
            env["OPENAI_MODEL"] = self.model_name

        env["PATH"] = "$HOME/.nvm/versions/node/v22.*/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        env["NVM_DIR"] = "$HOME/.nvm"
        env = {k: v for k, v in env.items() if v}

        base_url_value = self.base_url or os.environ.get("OPENAI_BASE_URL", "")
        skills_sync_snippet = (
            "mkdir -p /root/.codex/skills; "
            "for src in /root/.claude/skills /skills; do "
            "  if [ -d \"$src\" ]; then "
            "    for d in \"$src\"/*; do "
            "      if [ -d \"$d\" ] && [ -f \"$d/SKILL.md\" ]; then "
            "        base=$(basename \"$d\"); "
            "        mkdir -p \"/root/.codex/skills/$base\"; "
            "        cp -R \"$d/.\" \"/root/.codex/skills/$base/\" 2>/dev/null || true; "
            "      fi; "
            "    done; "
            "  fi; "
            "  if [ -d \"$src/skills\" ]; then "
            "    cp -R \"$src/skills/.\" /root/.codex/skills/ 2>/dev/null || true; "
            "  fi; "
            "  if [ -d \"$src/superpowers-main/skills\" ]; then "
            "    cp -R \"$src/superpowers-main/skills/.\" /root/.codex/skills/ 2>/dev/null || true; "
            "  fi; "
            "done; "
        )

        if base_url_value:
            model_to_use = self.model_name or "gpt-4o"
            wire_api_value = os.environ.get("OPENAI_WIRE_API") or "responses"
            if wire_api_value != "responses":
                wire_api_value = "responses"
            config_setup_cmd = (
                "set -e; "
                "mkdir -p /root/.codex; "
                f"{skills_sync_snippet}"
                "cat > /root/.codex/config.toml << 'CONFIGEOF'\n"
                f"model = \"{model_to_use}\"\n"
                "model_provider = \"proxy\"\n"
                "preferred_auth_method = \"apikey\"\n"
                "sandbox_mode = \"danger-full-access\"\n"
                "approval_policy = \"never\"\n"
                "\n"
                "[model_providers.proxy]\n"
                "name = \"Proxy\"\n"
                f"base_url = \"{base_url_value}\"\n"
                "requires_openai_auth = true\n"
                "env_key = \"OPENAI_API_KEY\"\n"
                f"wire_api = \"{wire_api_value}\"\n"
                "CONFIGEOF\n"
                "cp /root/.codex/config.toml /logs/agent/codex-config.toml"
            )
        else:
            model_to_use = self.model_name or "gpt-5"
            config_setup_cmd = (
                "set -e; "
                "mkdir -p /root/.codex; "
                f"{skills_sync_snippet}"
                "cat > /root/.codex/config.toml << 'CONFIGEOF'\n"
                f"model = \"{model_to_use}\"\n"
                "sandbox_mode = \"danger-full-access\"\n"
                "approval_policy = \"never\"\n"
                "CONFIGEOF\n"
                "cp /root/.codex/config.toml /logs/agent/codex-config.toml"
            )

        config_setup = ExecInput(command=config_setup_cmd, env=env)
        nvm_setup = ExecInput(
            command=(
                'export NVM_DIR="$HOME/.nvm" && '
                '[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh" && '
                "node --version && npm --version"
            ),
            env=env,
        )

        escaped_instruction = instruction.replace('"', '\\"').replace("$", "\\$")
        model_arg = f"-m {self.model_name}" if self.model_name else ""
        run_cmd = ExecInput(
            command=(
                "cd /workspace && "
                'export NVM_DIR="$HOME/.nvm" && '
                '[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh" && '
                "set -o pipefail && "
                'echo "=== Codex Debug ===" > /logs/agent/codex-debug.txt && '
                'echo "which codex: $(which codex)" >> /logs/agent/codex-debug.txt && '
                'echo "codex version:" >> /logs/agent/codex-debug.txt && '
                "codex --version >> /logs/agent/codex-debug.txt 2>&1 && "
                'echo "OPENAI_API_KEY set: ${OPENAI_API_KEY:+True}" >> /logs/agent/codex-debug.txt && '
                'echo "OPENAI_BASE_URL: ${OPENAI_BASE_URL}" >> /logs/agent/codex-debug.txt && '
                'echo "Config file:" >> /logs/agent/codex-debug.txt && '
                "cat /root/.codex/config.toml >> /logs/agent/codex-debug.txt 2>&1 && "
                'echo "Skills dir (/root/.codex/skills):" >> /logs/agent/codex-debug.txt && '
                "ls -la /root/.codex/skills >> /logs/agent/codex-debug.txt 2>&1 || true && "
                'echo "Skills dir (/root/.claude/skills):" >> /logs/agent/codex-debug.txt && '
                "ls -la /root/.claude/skills >> /logs/agent/codex-debug.txt 2>&1 || true && "
                'echo "Skills dir (/skills):" >> /logs/agent/codex-debug.txt && '
                "ls -la /skills >> /logs/agent/codex-debug.txt 2>&1 || true && "
                'echo "Executing command..." >> /logs/agent/codex-debug.txt && '
                f'codex exec "{escaped_instruction}" {model_arg} --skip-git-repo-check --json '
                "--dangerously-bypass-approvals-and-sandbox "
                "> /logs/agent/codex-output.json 2> /logs/agent/codex-stderr.txt; "
                "status=$?; "
                'echo "$status" > /logs/agent/codex-exit-code.txt; '
                'echo "Exit status: $status" >> /logs/agent/codex-debug.txt && '
                "if [ $status -ne 0 ] && [ ! -s /logs/agent/codex-output.json ]; then "
                '  err=$(cat /logs/agent/codex-stderr.txt 2>/dev/null | tr "\\n" " " | sed \'s/"/\\\\\\"/g\'); '
                '  echo "{\\"error\\": \\"Codex CLI execution failed\\", \\"stderr\\": \\"$err\\", \\"exit_code\\": $status}" '
                "> /logs/agent/codex-output.json; "
                "fi; "
                "cat /logs/agent/codex-output.json | tee /logs/agent/codex.txt"
            ),
            env=env,
            timeout_sec=300,
        )
        return [config_setup, nvm_setup, run_cmd]

    def populate_context_post_run(self, context: AgentContext) -> None:
        print(f"[{self.name()}] Parsing Codex execution results...")
        output_file = self.logs_dir / "codex-output.json"
        if not output_file.exists():
            print(f"[{self.name()}] No output file found")
            context.success = False
            return
        context.update_metadata(trajectory_source=str(output_file), trajectory_backend=self.name())

        try:
            raw_text = output_file.read_text()
            stripped = raw_text.strip()
            if not stripped:
                context.success = False
                context.response_content = "Empty Codex output"
                return

            lines = [line for line in stripped.splitlines() if line.strip()]
            if len(lines) == 1:
                result = json.loads(lines[0])
                context.extend_trajectory([{
                    "backend": "codex",
                    "event_type": "result" if "error" not in result else "error",
                    "role": "assistant",
                    "status": "success" if "error" not in result else "error",
                    "content": _stringify_payload(
                        result.get("content")
                        or result.get("text")
                        or result.get("error")
                        or result.get("stderr")
                    ),
                    "usage": result.get("usage") if isinstance(result.get("usage"), dict) else None,
                    "raw_event": result,
                }])
                if "error" in result:
                    print(f"[{self.name()}] Codex reported error: {result.get('error')}")
                    context.success = False
                    context.response_content = result.get("error", "Unknown error")
                    context.update_metadata(trajectory_event_count=len(context.trajectory))
                    return
                context.success = True
                context.response_content = result.get("content") or result.get("text") or str(result)
                usage = result.get("usage", {})
                if usage:
                    context.n_input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
                    context.n_output_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
            else:
                events: List[dict] = []
                for line in lines:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

                agent_messages: List[str] = []
                errors: List[str] = []
                turn_completed = False
                turn_failed = False
                normalized_trajectory: List[dict[str, Any]] = []
                for event in events:
                    normalized_trajectory.extend(_normalize_codex_event(event))
                    event_type = event.get("type")
                    if event_type == "item.completed":
                        item = event.get("item", {})
                        if item.get("type") == "agent_message":
                            text = item.get("text")
                            if text:
                                agent_messages.append(text)
                    elif event_type == "error":
                        message = event.get("message")
                        if message:
                            errors.append(message)
                    elif event_type == "turn.failed":
                        turn_failed = True
                        error = event.get("error", {})
                        message = error.get("message")
                        if message:
                            errors.append(message)
                    elif event_type == "turn.completed":
                        turn_completed = True
                        usage = event.get("usage", {})
                        context.n_input_tokens = usage.get("input_tokens")
                        context.n_output_tokens = usage.get("output_tokens")

                if agent_messages:
                    context.response_content = "\n\n".join(agent_messages)
                elif errors:
                    context.response_content = "\n".join(errors)
                else:
                    context.response_content = stripped[:2000]
                context.extend_trajectory(normalized_trajectory)
                context.success = turn_completed and not turn_failed and not errors

            context.update_metadata(trajectory_event_count=len(context.trajectory))
            print(f"[{self.name()}] Parsing completed:")
            print(f"  - Success: {context.success}")
            print(f"  - Input tokens: {context.n_input_tokens}")
            print(f"  - Output tokens: {context.n_output_tokens}")
        except json.JSONDecodeError as e:
            print(f"[{self.name()}] Failed to parse JSON: {e}")
            context.success = False
            context.response_content = output_file.read_text()
        except Exception as e:
            print(f"[{self.name()}] Error parsing result: {e}")
            context.success = False

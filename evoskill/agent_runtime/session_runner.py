"""High-level session runner for Docker-backed agent backends."""

from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path
from typing import Sequence

from ..client import EvoSkill
from ..interactive.config import InteractiveConfig
from ..management.formats.agent_skill import skill_dir_name
from ..models import Skill
from .agent_context import AgentContext
from .docker_config import DockerConfig
from .docker_environment import DockerEnvironment
from .agents import ClaudeCodeAgent, CodexAgent


def _write_selected_skill_dirs(
    sdk: EvoSkill,
    skills: Sequence[Skill],
    *,
    root_dir: Path,
) -> Path | None:
    """Materializes only the selected skills into one flat runtime directory."""

    if not skills:
        return None

    root_dir.mkdir(parents=True, exist_ok=True)
    used_names: set[str] = set()
    for skill in skills:
        base = skill_dir_name(skill) or "skill"
        dir_name = base
        if dir_name in used_names:
            i = 2
            while f"{base}-{i}" in used_names:
                i += 1
            dir_name = f"{base}-{i}"
        used_names.add(dir_name)

        files = sdk.export_skill_dir(skill.id) or {}
        skill_root = root_dir / dir_name
        skill_root.mkdir(parents=True, exist_ok=True)
        for rel_path, content in files.items():
            safe_rel = str(rel_path).lstrip("/").replace("..", "_")
            abs_path = skill_root / safe_rel
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            abs_path.write_text(content, encoding="utf-8")
    return root_dir


def _build_agent(cfg: InteractiveConfig, *, logs_dir: Path):
    """Builds the configured Docker-backed agent implementation."""

    backend = str(cfg.agent_backend or "llm").strip().lower()
    if backend == "claude-code":
        return ClaudeCodeAgent(
            logs_dir=logs_dir,
            model_name=cfg.agent_model_name or None,
            api_key=cfg.agent_api_key or None,
            base_url=cfg.agent_base_url or None,
            max_thinking_tokens=(
                int(cfg.agent_max_thinking_tokens) if int(cfg.agent_max_thinking_tokens or 0) > 0 else None
            ),
            version=cfg.agent_version or None,
        )
    if backend == "codex":
        return CodexAgent(
            logs_dir=logs_dir,
            model_name=cfg.agent_model_name or None,
            api_key=cfg.agent_api_key or None,
            base_url=cfg.agent_base_url or None,
            version=cfg.agent_version or None,
        )
    raise ValueError(f"Unsupported agent backend: {backend}")


def run_agent_session(
    *,
    sdk: EvoSkill,
    cfg: InteractiveConfig,
    instruction: str,
    selected_skills: Sequence[Skill],
) -> AgentContext:
    """Runs one Docker-backed agent session and returns the parsed context."""

    backend = str(cfg.agent_backend or "llm").strip().lower()
    if backend not in {"claude-code", "codex"}:
        raise ValueError(f"Unsupported agent backend: {backend}")

    if cfg.agent_use_prebuilt_image:
        DockerConfig.enable_prebuilt_image(cfg.docker_image_name or None)
    else:
        DockerConfig.disable_prebuilt_image()

    logs_root = Path(cfg.agent_logs_root_dir).resolve()
    session_id = f"{backend}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    session_root = logs_root / session_id
    logs_dir = session_root / "logs"
    skills_dir = session_root / "skills"
    workspace_dir = Path(cfg.agent_workspace_dir).resolve()

    materialized_skills_dir = _write_selected_skill_dirs(
        sdk,
        selected_skills,
        root_dir=skills_dir,
    )

    environment = DockerEnvironment(
        session_id=session_id,
        work_dir=workspace_dir,
        agent_logs_dir=logs_dir,
        skills_dir=materialized_skills_dir,
        image_name=cfg.docker_image_name or DockerConfig.get_image_name(),
        docker_context=cfg.docker_context,
        memory=cfg.docker_memory,
        cpus=cfg.docker_cpus,
        network_mode=cfg.docker_network_mode,
    )
    agent = _build_agent(cfg, logs_dir=logs_dir)
    context = AgentContext(
        metadata={
            "backend": backend,
            "session_id": session_id,
            "instruction": instruction,
            "logs_dir": str(logs_dir),
            "workspace_dir": str(workspace_dir),
            "skills_dir": str(materialized_skills_dir) if materialized_skills_dir else "",
            "selected_skill_ids": [str(skill.id) for skill in selected_skills],
            "selected_skill_names": [str(skill.name) for skill in selected_skills],
            "image_name": cfg.docker_image_name or DockerConfig.get_image_name(),
        }
    )

    async def _run() -> None:
        started = False
        t0 = time.perf_counter()
        try:
            await environment.start()
            started = True
            if not DockerConfig.should_skip_installation():
                await agent.setup(environment)
            await agent.run(instruction, environment, context)
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if context.total_latency_ms is None:
                context.total_latency_ms = elapsed_ms
            else:
                context.total_latency_ms += elapsed_ms
            if started:
                await environment.stop(keep_container=cfg.agent_keep_container)

    try:
        asyncio.run(_run())
    except Exception as e:
        context.success = False
        context.response_content = str(e)
        meta = dict(context.metadata or {})
        meta["error"] = str(e)
        context.metadata = meta

    return context

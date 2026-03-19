"""
Interactive configuration.

The interactive loop is intentionally configurable via a single object so it can be used by:
- CLI scripts (examples)
- external apps embedding EvoSkill
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from ..config import default_store_path


@dataclass
class InteractiveConfig:
    store_dir: str = field(default_factory=default_store_path)
    user_id: str = "u1"

    # Which Skills to use during retrieval:
    # - "user": only the current user's skills
    # - "library"/"common": only shared/common skills
    # - "all": both
    skill_scope: str = "all"

    # Retrieval query rewriting:
    # - "never": do not rewrite
    # - "auto": rewrite when a rewriter is configured
    # - "always": always rewrite (when a rewriter is configured)
    rewrite_mode: str = "always"  # auto|always|never
    rewrite_history_turns: int = 6
    # "chars" here means sizing units: CJK ideographs count by character; ASCII/English counts by word.
    rewrite_history_chars: int = 2000
    rewrite_max_query_chars: int = 256

    # Minimum similarity threshold for retrieval results (post-search filter).
    # Use a high value to be conservative and avoid injecting irrelevant skills.
    min_score: float = 0.4

    top_k: int = 1
    history_turns: int = 100
    ingest_window: int = 6

    # Extraction timing signals:
    # In "auto" mode, attempt extraction once every N turns (N=extract_turn_limit).
    # Set N=1 to attempt extraction every turn.
    extract_turn_limit: int = 1

    extract_mode: str = "auto"  # auto|always|never

    assistant_temperature: float = 0.2

    # Assistant backend:
    # - "llm": use EvoSkill's built-in LLM generation path
    # - "claude-code": run Claude Code CLI in Docker
    # - "codex": run Codex CLI in Docker
    agent_backend: str = "llm"
    agent_model_name: str = ""
    agent_api_key: str = ""
    agent_base_url: str = ""
    agent_version: str = ""
    agent_max_thinking_tokens: int = 0
    agent_workspace_dir: str = "."
    agent_logs_root_dir: str = ""
    agent_use_prebuilt_image: bool = False
    agent_keep_container: bool = False
    docker_image_name: str = ""
    docker_context: str = ""
    docker_memory: str = "2G"
    docker_cpus: int = 2
    docker_network_mode: str = "bridge"

    # Per-turn skill usage tracking:
    # - judge retrieved skills with LLM for relevance/actual usage in reply
    # - persist counters locally and auto-prune stale skills when threshold is met
    usage_tracking_enabled: bool = True
    usage_prune_min_retrieved: int = 40
    usage_prune_max_used: int = 0

    def normalize(self) -> "InteractiveConfig":
        """Run normalize."""
        self.store_dir = str(self.store_dir or default_store_path()).strip() or default_store_path()

        self.skill_scope = (self.skill_scope or "all").strip().lower()
        if self.skill_scope == "common":
            self.skill_scope = "library"
        if self.skill_scope not in {"all", "user", "library"}:
            self.skill_scope = "all"

        self.rewrite_mode = (self.rewrite_mode or "auto").strip().lower()
        if self.rewrite_mode not in {"auto", "always", "never"}:
            self.rewrite_mode = "always"
        self.rewrite_history_turns = max(0, int(self.rewrite_history_turns))
        self.rewrite_history_chars = max(0, int(self.rewrite_history_chars))
        self.rewrite_max_query_chars = max(32, int(self.rewrite_max_query_chars))

        try:
            self.min_score = float(self.min_score)
        except Exception:
            self.min_score = 0.4

        self.extract_mode = (self.extract_mode or "auto").strip().lower()
        if self.extract_mode not in {"auto", "always", "never"}:
            self.extract_mode = "auto"

        self.agent_backend = (self.agent_backend or "llm").strip().lower()
        if self.agent_backend not in {"llm", "claude-code", "codex"}:
            self.agent_backend = "llm"
        self.agent_model_name = str(self.agent_model_name or "").strip()
        self.agent_api_key = str(self.agent_api_key or "").strip()
        self.agent_base_url = str(self.agent_base_url or "").strip()
        self.agent_version = str(self.agent_version or "").strip()
        self.agent_max_thinking_tokens = max(0, int(self.agent_max_thinking_tokens or 0))
        self.agent_workspace_dir = os.path.abspath(
            os.path.expanduser(str(self.agent_workspace_dir or os.getcwd()))
        )
        logs_root = str(self.agent_logs_root_dir or "").strip()
        if not logs_root:
            store_dir_abs = os.path.abspath(os.path.expanduser(self.store_dir))
            logs_root = os.path.join(os.path.dirname(store_dir_abs), "trail")
        self.agent_logs_root_dir = os.path.abspath(os.path.expanduser(logs_root))
        self.agent_use_prebuilt_image = bool(self.agent_use_prebuilt_image)
        self.agent_keep_container = bool(self.agent_keep_container)
        self.docker_image_name = str(self.docker_image_name or "").strip()
        self.docker_context = str(self.docker_context or "").strip()
        self.docker_memory = str(self.docker_memory or "2G").strip() or "2G"
        self.docker_cpus = max(1, int(self.docker_cpus or 2))
        self.docker_network_mode = str(self.docker_network_mode or "bridge").strip() or "bridge"

        self.top_k = max(0, int(self.top_k))
        self.history_turns = max(0, int(self.history_turns))
        self.ingest_window = max(2, int(self.ingest_window))
        self.extract_turn_limit = max(1, int(self.extract_turn_limit))
        self.usage_tracking_enabled = bool(self.usage_tracking_enabled)
        self.usage_prune_min_retrieved = max(0, int(self.usage_prune_min_retrieved or 40))
        self.usage_prune_max_used = max(0, int(self.usage_prune_max_used or 0))
        return self

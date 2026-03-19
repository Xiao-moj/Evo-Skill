"""
Evo-skill CLI entry point.

Commands:
    python main.py chat              Start interactive chat
    python main.py list              List all skills
    python main.py delete <id>       Delete a skill by ID
    python main.py export <path>     Export skills to JSON

Configuration via environment variables or .env file:
    See .env.example for all options.
"""

from __future__ import annotations

import os
import sys

# ── Load .env if present (no external dependency needed) ──────────────────────
def _load_dotenv(path: str = ".env") -> None:
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val

_load_dotenv()

# ── Imports (after env is loaded) ─────────────────────────────────────────────
from evoskill.client import EvoSkill
from evoskill.config import EvoSkillConfig
from evoskill.interactive.config import InteractiveConfig
from evoskill.llm.factory import build_llm
from evoskill.cli.chat import run_chat
from evoskill.cli.commands import cmd_list, cmd_delete, cmd_export, cmd_compose


def _first_env(*keys: str) -> str:
    """Returns the first non-empty environment variable value."""

    for key in keys:
        value = str(os.getenv(key, "") or "").strip()
        if value:
            return value
    return ""


def _build_sdk() -> EvoSkill:
    """Build the SDK from environment variables."""
    llm_provider    = os.getenv("LLM_PROVIDER", "openai")
    llm_api_key     = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    llm_base_url    = os.getenv("LLM_BASE_URL", "https://api.openai.com")
    llm_model       = os.getenv("LLM_MODEL", "gpt-4o-mini")

    emb_provider    = os.getenv("EMBEDDING_PROVIDER", "openai")
    emb_api_key     = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
    emb_base_url    = os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com")
    emb_model       = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    store_path      = os.getenv("SKILL_BANK_PATH", "./SkillBank")

    config = EvoSkillConfig.from_dict({
        "llm": {
            "provider": llm_provider,
            "api_key": llm_api_key,
            "base_url": llm_base_url,
            "model": llm_model,
        },
        "embeddings": {
            "provider": emb_provider,
            "api_key": emb_api_key,
            "base_url": emb_base_url,
            "model": emb_model,
        },
        "store": {
            "provider": "local",
            "path": store_path,
        },
        "max_candidates_per_ingest": int(os.getenv("MAX_CANDIDATES_PER_INGEST", "5")),
    })

    # Choose extractor based on EXTRACTOR_MODE env var
    extractor_mode = os.getenv("EXTRACTOR_MODE", "learner").strip().lower()
    extractor = None
    if extractor_mode == "learner":
        from evoskill.management.learner_extraction import LearnerSkillExtractor
        extractor_llm = build_llm({
            "provider": llm_provider,
            "api_key": llm_api_key,
            "base_url": llm_base_url,
            "model": llm_model,
            "timeout_s": int(os.getenv("EXTRACTOR_TIMEOUT_S", "240")),
            "max_tokens": 0,
        })
        extractor = LearnerSkillExtractor(config, llm=extractor_llm)
    # default: None → EvoSkill uses LLMSkillExtractor automatically

    return EvoSkill(config, extractor=extractor)


def _build_chat_llm():
    """Build the LLM used for chat (may differ from extraction LLM)."""
    return build_llm({
        "provider": os.getenv("LLM_PROVIDER", "openai"),
        "api_key":  os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"),
        "base_url": os.getenv("LLM_BASE_URL", "https://api.openai.com"),
        "model":    os.getenv("LLM_MODEL", "gpt-4o-mini"),
    })


def _build_interactive_config() -> InteractiveConfig:
    agent_backend = str(os.getenv("AGENT_BACKEND", "llm") or "llm").strip().lower()
    if agent_backend == "claude-code":
        default_agent_model = _first_env("AGENT_MODEL", "ANTHROPIC_MODEL", "LLM_MODEL")
        default_agent_api_key = _first_env("AGENT_API_KEY", "ANTHROPIC_API_KEY", "LLM_API_KEY")
        default_agent_base_url = _first_env("AGENT_BASE_URL", "ANTHROPIC_BASE_URL", "LLM_BASE_URL")
    elif agent_backend == "codex":
        default_agent_model = _first_env("AGENT_MODEL", "OPENAI_MODEL", "LLM_MODEL")
        default_agent_api_key = _first_env("AGENT_API_KEY", "OPENAI_API_KEY", "LLM_API_KEY")
        default_agent_base_url = _first_env("AGENT_BASE_URL", "OPENAI_BASE_URL", "LLM_BASE_URL")
    else:
        default_agent_model = _first_env("AGENT_MODEL")
        default_agent_api_key = _first_env("AGENT_API_KEY")
        default_agent_base_url = _first_env("AGENT_BASE_URL")

    cfg = InteractiveConfig(
        store_dir=os.getenv("SKILL_BANK_PATH", "./SkillBank"),
        user_id=os.getenv("USER_ID", "u1"),
        skill_scope=os.getenv("SKILL_SCOPE", "user"),
        rewrite_mode=os.getenv("REWRITE_MODE", "always"),
        min_score=float(os.getenv("MIN_SCORE", "0.4")),
        top_k=int(os.getenv("TOP_K", "3")),
        extract_mode=os.getenv("EXTRACT_MODE", "auto"),
        usage_tracking_enabled=os.getenv("USAGE_TRACKING", "true").lower() == "true",
        assistant_temperature=float(os.getenv("ASSISTANT_TEMPERATURE", "0.7")),
        agent_backend=agent_backend,
        agent_model_name=default_agent_model,
        agent_api_key=default_agent_api_key,
        agent_base_url=default_agent_base_url,
        agent_version=os.getenv("AGENT_VERSION", ""),
        agent_max_thinking_tokens=int(os.getenv("AGENT_MAX_THINKING_TOKENS", "0")),
        agent_workspace_dir=os.getenv("AGENT_WORKSPACE_DIR", os.getcwd()),
        agent_logs_root_dir=os.getenv("AGENT_LOGS_DIR", ""),
        agent_use_prebuilt_image=os.getenv("AGENT_USE_PREBUILT_IMAGE", "false").lower() == "true",
        agent_keep_container=os.getenv("AGENT_KEEP_CONTAINER", "false").lower() == "true",
        docker_image_name=os.getenv("AGENT_DOCKER_IMAGE", ""),
        docker_context=os.getenv("AGENT_DOCKER_CONTEXT", os.getenv("DOCKER_CONTEXT", "")),
        docker_memory=os.getenv("AGENT_DOCKER_MEMORY", "2G"),
        docker_cpus=int(os.getenv("AGENT_DOCKER_CPUS", "2")),
        docker_network_mode=os.getenv("AGENT_DOCKER_NETWORK", "bridge"),
    )
    return cfg.normalize()


def main() -> None:
    args = sys.argv[1:]
    command = args[0] if args else "chat"

    sdk    = _build_sdk()
    user_id = os.getenv("USER_ID", "u1")

    if command == "chat":
        llm = _build_chat_llm()
        cfg = _build_interactive_config()
        run_chat(sdk=sdk, llm=llm, cfg=cfg)

    elif command == "list":
        cmd_list(sdk, user_id)

    elif command == "delete":
        if len(args) < 2:
            print("Usage: python main.py delete <skill_id>")
            sys.exit(1)
        cmd_delete(sdk, user_id, args[1])

    elif command == "export":
        if len(args) < 2:
            print("Usage: python main.py export <output_path.json>")
            sys.exit(1)
        cmd_export(sdk, user_id, args[1])

    elif command == "compose":
        if len(args) < 2:
            print("Usage: python main.py compose \"<task description>\"")
            sys.exit(1)
        llm = _build_chat_llm()
        task = " ".join(args[1:])
        cmd_compose(sdk, user_id, task, llm)

    else:
        print(f"Unknown command: {command}")
        print("Usage: python main.py [chat|list|delete|export|compose]")
        sys.exit(1)


if __name__ == "__main__":
    main()

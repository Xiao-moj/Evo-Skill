"""Docker-based agent runtime integrations for EvoSkill."""

from .agent_context import AgentContext, AgentTrajectoryEvent
from .docker_config import DockerConfig
from .docker_environment import DockerEnvironment

__all__ = [
    "AgentContext",
    "AgentTrajectoryEvent",
    "DockerConfig",
    "DockerEnvironment",
    "run_agent_session",
]


def __getattr__(name: str):
    """Lazily resolves heavy runtime imports to avoid circular dependencies."""

    if name == "run_agent_session":
        from .session_runner import run_agent_session

        return run_agent_session
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

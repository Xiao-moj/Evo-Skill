"""Agent execution context, metrics, and normalized trajectory events."""

from __future__ import annotations

from typing import Any, Iterable, Optional

from pydantic import BaseModel, Field


class AgentTrajectoryEvent(BaseModel):
    """One normalized agent runtime event."""

    backend: str = Field(default="", description="Agent backend name, e.g. codex or claude-code.")
    event_type: str = Field(default="", description="Normalized event type.")
    role: Optional[str] = Field(default=None, description="user / assistant / system when applicable.")
    content: Optional[str] = Field(default=None, description="Human-readable text payload.")
    tool_name: Optional[str] = Field(default=None, description="Tool/function name when applicable.")
    tool_input: Optional[Any] = Field(default=None, description="Structured tool input when available.")
    tool_output: Optional[str] = Field(default=None, description="Structured tool output rendered as text.")
    status: Optional[str] = Field(default=None, description="Event status such as success / error / started.")
    timestamp: Optional[str] = Field(default=None, description="Provider-emitted timestamp if available.")
    duration_ms: Optional[float] = Field(default=None, description="Event duration in ms if available.")
    usage: Optional[dict[str, Any]] = Field(default=None, description="Provider token/cost usage payload.")
    raw_event: Optional[dict[str, Any]] = Field(default=None, description="Original provider event.")


class AgentContext(BaseModel):
    """Stores one agent run's outputs and performance metadata."""

    n_input_tokens: Optional[int] = Field(default=None, description="Input token count.")
    n_cache_tokens: Optional[int] = Field(default=None, description="Cache token count.")
    n_output_tokens: Optional[int] = Field(default=None, description="Output token count.")
    cost_usd: Optional[float] = Field(default=None, description="Execution cost in USD.")
    total_latency_ms: Optional[float] = Field(default=None, description="Total latency in ms.")
    success: Optional[bool] = Field(default=None, description="Whether execution succeeded.")
    response_content: Optional[str] = Field(default=None, description="Final response content.")
    trajectory: list[AgentTrajectoryEvent] = Field(
        default_factory=list,
        description="Normalized execution trajectory emitted by the runtime backend.",
    )
    metadata: Optional[dict[str, Any]] = Field(default=None, description="Extra metadata.")

    def add_trajectory_event(self, event: AgentTrajectoryEvent | dict[str, Any]) -> None:
        """Appends one normalized trajectory event."""

        if isinstance(event, AgentTrajectoryEvent):
            self.trajectory.append(event)
            return
        self.trajectory.append(AgentTrajectoryEvent(**dict(event or {})))

    def extend_trajectory(self, events: Iterable[AgentTrajectoryEvent | dict[str, Any]]) -> None:
        """Appends multiple normalized trajectory events."""

        for event in events or []:
            self.add_trajectory_event(event)

    def update_metadata(self, **kwargs: Any) -> None:
        """Merges key/value pairs into metadata."""

        merged = dict(self.metadata or {})
        for key, value in kwargs.items():
            if value is not None:
                merged[str(key)] = value
        self.metadata = merged or None

    def trajectory_as_dicts(self) -> list[dict[str, Any]]:
        """Returns the normalized trajectory as plain dictionaries."""

        return [event.model_dump(exclude_none=True) for event in self.trajectory]

    def is_empty(self) -> bool:
        """Returns True when no fields were populated."""

        for value in self.model_dump().values():
            if value is None:
                continue
            if isinstance(value, (str, list, dict)) and not value:
                continue
            return False
        return True

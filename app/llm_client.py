from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol, Sequence


LLMRole = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class LLMToolCall:
    id: str
    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class LLMMessage:
    role: LLMRole
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: Sequence[LLMToolCall] = field(default_factory=tuple)


@dataclass(frozen=True)
class LLMRequest:
    messages: Sequence[LLMMessage]
    tools: Sequence[Mapping[str, Any]] = field(default_factory=tuple)


@dataclass(frozen=True)
class LLMResponse:
    content: str | None = None
    tool_calls: Sequence[LLMToolCall] = field(default_factory=tuple)
    finish_reason: str | None = None


class LLMClient(Protocol):
    """Provider-neutral model interface; the Runtime owns tool execution."""

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Return text and/or proposed tool calls for one model turn."""
        ...

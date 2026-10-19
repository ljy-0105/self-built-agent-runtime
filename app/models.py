from dataclasses import dataclass
from typing import Literal


MessageRole = Literal["system", "user", "assistant", "tool"]
TodoStatus = Literal["open", "completed"]


@dataclass(frozen=True)
class Session:
    id: str
    user_id: str
    title: str
    summary: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class Message:
    id: str
    session_id: str
    sequence: int
    role: MessageRole
    content: str
    tool_call_json: str | None
    tool_name: str | None
    created_at: str


@dataclass(frozen=True)
class Todo:
    id: str
    user_id: str
    session_id: str
    text: str
    status: TodoStatus
    created_at: str
    completed_at: str | None

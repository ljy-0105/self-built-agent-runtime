import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from app.llm_client import LLMClient
from app.models import Message, Session
from app.openai_client import OpenAIChatCompletionsClient
from app.runtime import AgentRuntime
from app.sessions import MessageRepository, SessionRepository
from app.storage import Database
from app.tools import build_tool_registry
from app.todos import TodoRepository


class SessionCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    title: str = Field(default="新对话", min_length=1, max_length=120)


class MessageCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    content: str = Field(min_length=1, max_length=20_000)


class SessionResponse(BaseModel):
    id: str
    user_id: str
    title: str
    summary: str | None
    created_at: str
    updated_at: str

    @classmethod
    def from_entity(cls, session: Session) -> "SessionResponse":
        return cls(**session.__dict__)


class MessageResponse(BaseModel):
    id: str
    session_id: str
    sequence: int
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_call_json: str | None
    tool_name: str | None
    created_at: str

    @classmethod
    def from_entity(cls, message: Message) -> "MessageResponse":
        return cls(**message.__dict__)


class ChatResponse(BaseModel):
    status: Literal[
        "runtime_not_configured",
        "completed",
        "invalid_response",
        "max_turns",
        "llm_error",
    ]
    user_message: MessageResponse
    assistant_message: MessageResponse | None
    model_turns: int
    tool_calls: int


def get_current_user_id(request: Request) -> str:
    """Development identity hook; replace with authenticated identity in production."""
    return request.app.state.current_user_provider()


def _default_current_user() -> str:
    return os.getenv("AGENT_DEV_USER_ID", "local-user")


def create_app(
    database_path: str | None = None,
    *,
    llm_client: LLMClient | None = None,
    current_user_provider: Callable[[], str] = _default_current_user,
) -> FastAPI:
    database = Database(database_path or os.getenv("AGENT_DB_PATH", "agent.sqlite3"))

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        database.initialize()
        yield

    application = FastAPI(title="Self-Built Agent Runtime", lifespan=lifespan)
    application.state.database = database
    application.state.sessions = SessionRepository(database)
    application.state.messages = MessageRepository(database)
    application.state.todos = TodoRepository(database)
    application.state.tool_registry = build_tool_registry(application.state.todos)
    configured_llm_client = (
        llm_client
        if llm_client is not None
        else OpenAIChatCompletionsClient.from_environment()
    )
    application.state.llm_client = configured_llm_client
    application.state.runtime = (
        AgentRuntime(
            configured_llm_client,
            application.state.sessions,
            application.state.messages,
            application.state.tool_registry,
        )
        if configured_llm_client is not None
        else None
    )
    application.state.current_user_provider = current_user_provider

    @application.get("/", include_in_schema=False)
    def workbench() -> FileResponse:
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @application.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @application.post(
        "/sessions",
        response_model=SessionResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def create_session(
        body: SessionCreate,
        user_id: str = Depends(get_current_user_id),
    ) -> SessionResponse:
        session = application.state.sessions.create(user_id, body.title)
        return SessionResponse.from_entity(session)

    @application.get("/sessions", response_model=list[SessionResponse])
    def list_sessions(
        limit: int = Query(default=100, ge=1, le=100),
        user_id: str = Depends(get_current_user_id),
    ) -> list[SessionResponse]:
        sessions = application.state.sessions.list_for_user(user_id, limit=limit)
        return [SessionResponse.from_entity(session) for session in sessions]

    @application.get("/sessions/{session_id}", response_model=SessionResponse)
    def get_session(
        session_id: str,
        user_id: str = Depends(get_current_user_id),
    ) -> SessionResponse:
        session = application.state.sessions.get_owned(user_id, session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return SessionResponse.from_entity(session)

    @application.get(
        "/sessions/{session_id}/messages",
        response_model=list[MessageResponse],
    )
    def list_messages(
        session_id: str,
        user_id: str = Depends(get_current_user_id),
    ) -> list[MessageResponse]:
        _require_owned_session(application.state.sessions, user_id, session_id)
        messages = application.state.messages.list_for_session(user_id, session_id)
        return [MessageResponse.from_entity(message) for message in messages]

    @application.post(
        "/sessions/{session_id}/messages",
        response_model=ChatResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def append_user_message(
        session_id: str,
        body: MessageCreate,
        user_id: str = Depends(get_current_user_id),
    ) -> ChatResponse:
        runtime: AgentRuntime | None = application.state.runtime
        if runtime is None:
            _require_owned_session(application.state.sessions, user_id, session_id)
            user_message = application.state.messages.append(
                user_id,
                session_id,
                role="user",
                content=body.content,
            )
            return ChatResponse(
                status="runtime_not_configured",
                user_message=MessageResponse.from_entity(user_message),
                assistant_message=None,
                model_turns=0,
                tool_calls=0,
            )

        result = await runtime.run(user_id, session_id, body.content)
        return ChatResponse(
            status=result.status,
            user_message=MessageResponse.from_entity(result.user_message),
            assistant_message=MessageResponse.from_entity(result.assistant_message),
            model_turns=result.model_turns,
            tool_calls=result.tool_calls,
        )

    return application


def _require_owned_session(
    sessions: SessionRepository,
    user_id: str,
    session_id: str,
) -> Session:
    session = sessions.get_owned(user_id, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


app = create_app()

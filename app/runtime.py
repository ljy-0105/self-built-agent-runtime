import json
from dataclasses import dataclass
from typing import Literal

from app.context import ContextBuilder, LLMSummaryGenerator, SummaryGenerator
from app.errors import SessionNotFoundError
from app.llm_client import LLMClient, LLMRequest, LLMToolCall
from app.models import Message
from app.response_parser import (
    FinalAnswer,
    InvalidModelResponse,
    ToolCallBatch,
    parse_response,
)
from app.sessions import MessageRepository, SessionRepository
from app.tools.base import ToolContext, ToolError, ToolRegistry


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the provided tools when they are needed. "
    "Do not claim that a tool action succeeded until its result is returned. "
    "Return tool calls when more information or an action is needed; otherwise "
    "return a concise answer for the user."
)
MAX_MODEL_TURNS = 5
MAX_TOOL_CALLS_PER_TURN = 3


RunStatus = Literal["completed", "invalid_response", "max_turns", "llm_error"]


@dataclass(frozen=True)
class AgentRunResult:
    status: RunStatus
    answer: str
    user_message: Message
    assistant_message: Message
    model_turns: int
    tool_calls: int


class AgentRuntime:
    def __init__(
        self,
        llm_client: LLMClient,
        sessions: SessionRepository,
        messages: MessageRepository,
        tools: ToolRegistry,
        *,
        max_model_turns: int = MAX_MODEL_TURNS,
        max_tool_calls_per_turn: int = MAX_TOOL_CALLS_PER_TURN,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        summary_generator: SummaryGenerator | None = None,
        max_context_messages: int = 20,
        retain_recent_messages: int = 10,
    ) -> None:
        if max_model_turns < 1:
            raise ValueError("max_model_turns must be at least 1")
        if max_tool_calls_per_turn < 1:
            raise ValueError("max_tool_calls_per_turn must be at least 1")
        if not system_prompt.strip():
            raise ValueError("system_prompt cannot be empty")

        self.llm_client = llm_client
        self.sessions = sessions
        self.messages = messages
        self.tools = tools
        self.max_model_turns = max_model_turns
        self.max_tool_calls_per_turn = max_tool_calls_per_turn
        self.system_prompt = system_prompt
        self.context_builder = ContextBuilder(
            sessions,
            messages,
            system_prompt=system_prompt,
            summary_generator=(
                summary_generator
                if summary_generator is not None
                else LLMSummaryGenerator(llm_client)
            ),
            max_context_messages=max_context_messages,
            retain_recent_messages=retain_recent_messages,
        )

    async def run(self, user_id: str, session_id: str, user_text: str) -> AgentRunResult:
        if not user_text.strip():
            raise ValueError("user_text cannot be empty")
        if self.sessions.get_owned(user_id, session_id) is None:
            raise SessionNotFoundError(session_id)

        user_message = self.messages.append(
            user_id,
            session_id,
            role="user",
            content=user_text,
        )
        context = ToolContext(user_id=user_id, session_id=session_id)
        total_tool_calls = 0

        for turn in range(1, self.max_model_turns + 1):
            request = await self._build_request(user_id, session_id)
            try:
                response = await self.llm_client.complete(request)
            except Exception:
                answer = "我暂时无法完成这次请求，请稍后重试。"
                assistant = self._save_assistant(user_id, session_id, answer)
                return AgentRunResult(
                    status="llm_error",
                    answer=answer,
                    user_message=user_message,
                    assistant_message=assistant,
                    model_turns=turn,
                    tool_calls=total_tool_calls,
                )

            try:
                decision = parse_response(
                    response,
                    max_tool_calls=self.max_tool_calls_per_turn,
                )
            except InvalidModelResponse:
                answer = "模型返回了无法识别的响应，请重试。"
                assistant = self._save_assistant(user_id, session_id, answer)
                return AgentRunResult(
                    status="invalid_response",
                    answer=answer,
                    user_message=user_message,
                    assistant_message=assistant,
                    model_turns=turn,
                    tool_calls=total_tool_calls,
                )

            if isinstance(decision, FinalAnswer):
                assistant = self._save_assistant(user_id, session_id, decision.text)
                return AgentRunResult(
                    status="completed",
                    answer=decision.text,
                    user_message=user_message,
                    assistant_message=assistant,
                    model_turns=turn,
                    tool_calls=total_tool_calls,
                )

            assert isinstance(decision, ToolCallBatch)
            total_tool_calls += len(decision.calls)
            self._save_tool_calls(user_id, session_id, decision.calls)
            for call in decision.calls:
                try:
                    result = await self.tools.execute(call.name, call.arguments, context)
                    tool_result = {"ok": True, "result": result}
                except ToolError as exc:
                    tool_result = {
                        "ok": False,
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    }
                self._save_tool_result(user_id, session_id, call, tool_result)

        answer = "我已达到本轮模型调用上限，请继续提问或缩小任务范围。"
        assistant = self._save_assistant(user_id, session_id, answer)
        return AgentRunResult(
            status="max_turns",
            answer=answer,
            user_message=user_message,
            assistant_message=assistant,
            model_turns=self.max_model_turns,
            tool_calls=total_tool_calls,
        )

    async def _build_request(self, user_id: str, session_id: str) -> LLMRequest:
        context_messages = await self.context_builder.build(user_id, session_id)
        return LLMRequest(messages=context_messages, tools=self.tools.schemas())

    def _save_tool_calls(
        self,
        user_id: str,
        session_id: str,
        calls: tuple[LLMToolCall, ...],
    ) -> None:
        self.messages.append(
            user_id=user_id,
            session_id=session_id,
            role="assistant",
            content="",
            tool_call=[
                {"id": call.id, "name": call.name, "arguments": dict(call.arguments)}
                for call in calls
            ],
        )

    def _save_tool_result(
        self,
        user_id: str,
        session_id: str,
        call: LLMToolCall,
        result: dict,
    ) -> None:
        self.messages.append(
            user_id=user_id,
            session_id=session_id,
            role="tool",
            content=json.dumps(result, ensure_ascii=False, separators=(",", ":")),
            tool_call={"id": call.id},
            tool_name=call.name,
        )

    def _save_assistant(self, user_id: str, session_id: str, content: str) -> Message:
        return self.messages.append(
            user_id=user_id,
            session_id=session_id,
            role="assistant",
            content=content,
        )

import json
from collections.abc import Sequence
from typing import Protocol

from app.errors import SessionNotFoundError
from app.llm_client import LLMClient, LLMMessage, LLMRequest, LLMToolCall
from app.models import Message
from app.sessions import MessageRepository, SessionRepository


SUMMARY_SYSTEM_PROMPT = (
    "Summarize the earlier conversation for future turns. Preserve user-provided "
    "facts, preferences, decisions, pending questions, and relevant tool outcomes. "
    "Be concise, factual, and do not invent details."
)


class SummaryGenerator(Protocol):
    async def summarize(
        self,
        existing_summary: str | None,
        messages: Sequence[Message],
    ) -> str:
        """Return an updated summary that incorporates the supplied messages."""
        ...


class LLMSummaryGenerator:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    async def summarize(
        self,
        existing_summary: str | None,
        messages: Sequence[Message],
    ) -> str:
        prompt_lines = []
        if existing_summary:
            prompt_lines.append(f"Existing summary:\n{existing_summary}")
        prompt_lines.append("Earlier conversation messages:")
        prompt_lines.extend(_message_for_summary(message) for message in messages)

        response = await self.llm_client.complete(
            LLMRequest(
                messages=(
                    LLMMessage(role="system", content=SUMMARY_SYSTEM_PROMPT),
                    LLMMessage(role="user", content="\n".join(prompt_lines)),
                )
            )
        )
        if response.tool_calls or not isinstance(response.content, str):
            raise ValueError("Summary model returned no summary text")
        summary = response.content.strip()
        if not summary:
            raise ValueError("Summary model returned empty text")
        return summary


class ContextBuilder:
    def __init__(
        self,
        sessions: SessionRepository,
        messages: MessageRepository,
        *,
        system_prompt: str,
        summary_generator: SummaryGenerator,
        max_context_messages: int = 20,
        retain_recent_messages: int = 10,
        max_summary_characters: int = 4000,
    ) -> None:
        if not system_prompt.strip():
            raise ValueError("system_prompt cannot be empty")
        if max_context_messages < 2:
            raise ValueError("max_context_messages must be at least 2")
        if not 1 <= retain_recent_messages < max_context_messages:
            raise ValueError(
                "retain_recent_messages must be positive and less than max_context_messages"
            )
        if max_summary_characters < 1:
            raise ValueError("max_summary_characters must be positive")

        self.sessions = sessions
        self.messages = messages
        self.system_prompt = system_prompt
        self.summary_generator = summary_generator
        self.max_context_messages = max_context_messages
        self.retain_recent_messages = retain_recent_messages
        self.max_summary_characters = max_summary_characters

    async def build(self, user_id: str, session_id: str) -> tuple[LLMMessage, ...]:
        session = self.sessions.get_owned(user_id, session_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        history = self.messages.list_for_session(user_id, session_id)
        summary = session.summary

        if len(history) > self.max_context_messages:
            split_at = self._recent_start(history)
            earlier_messages = history[:split_at]
            if earlier_messages:
                try:
                    generated_summary = await self.summary_generator.summarize(
                        summary,
                        earlier_messages,
                    )
                    generated_summary = generated_summary.strip()
                    if (
                        generated_summary
                        and len(generated_summary) <= self.max_summary_characters
                        and self.sessions.set_summary(
                            user_id,
                            session_id,
                            generated_summary,
                        )
                    ):
                        summary = generated_summary
                        history = history[split_at:]
                except Exception:
                    pass

        context = [LLMMessage(role="system", content=self.system_prompt)]
        if summary:
            context.append(
                LLMMessage(
                    role="system",
                    content=f"Summary of earlier conversation in this session:\n{summary}",
                )
            )
        context.extend(message_to_llm_message(message) for message in history)
        return tuple(context)

    def _recent_start(self, history: Sequence[Message]) -> int:
        split_at = max(0, len(history) - self.retain_recent_messages)
        while split_at > 0 and history[split_at].role == "tool":
            split_at -= 1
            if history[split_at].role == "assistant" and history[split_at].tool_call_json:
                break
        return split_at


def message_to_llm_message(message: Message) -> LLMMessage:
    if message.role == "assistant" and message.tool_call_json:
        try:
            raw_calls = json.loads(message.tool_call_json)
            calls = tuple(
                LLMToolCall(
                    id=call["id"],
                    name=call["name"],
                    arguments=call["arguments"],
                )
                for call in raw_calls
            )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError("Stored assistant tool calls are invalid") from exc
        return LLMMessage(role="assistant", content=message.content, tool_calls=calls)

    if message.role == "tool":
        try:
            metadata = json.loads(message.tool_call_json or "{}")
            tool_call_id = metadata.get("id")
        except (json.JSONDecodeError, AttributeError) as exc:
            raise ValueError("Stored tool call metadata is invalid") from exc
        return LLMMessage(
            role="tool",
            content=message.content,
            name=message.tool_name,
            tool_call_id=tool_call_id,
        )

    return LLMMessage(role=message.role, content=message.content)


def _message_for_summary(message: Message) -> str:
    parts = [f"[{message.sequence}] {message.role}"]
    if message.tool_name:
        parts.append(f"tool={message.tool_name}")
    if message.tool_call_json:
        parts.append(f"tool_data={message.tool_call_json}")
    if message.content:
        parts.append(message.content)
    return " ".join(parts)

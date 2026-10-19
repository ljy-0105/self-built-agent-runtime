import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.llm_client import LLMResponse, LLMToolCall


class InvalidModelResponse(ValueError):
    """Raised when a model response cannot be used by the Runtime."""


@dataclass(frozen=True)
class FinalAnswer:
    text: str


@dataclass(frozen=True)
class ToolCallBatch:
    calls: tuple[LLMToolCall, ...]


ModelDecision = FinalAnswer | ToolCallBatch


def parse_response(response: object, *, max_tool_calls: int) -> ModelDecision:
    if not isinstance(response, LLMResponse):
        raise InvalidModelResponse("LLMClient must return an LLMResponse")

    calls = response.tool_calls
    if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)):
        raise InvalidModelResponse("tool_calls must be a sequence")

    if calls:
        if len(calls) > max_tool_calls:
            raise InvalidModelResponse(
                f"Model requested more than {max_tool_calls} tools in one turn"
            )

        normalized: list[LLMToolCall] = []
        seen_ids: set[str] = set()
        for call in calls:
            if not isinstance(call, LLMToolCall):
                raise InvalidModelResponse("Each tool call must use the LLMToolCall type")
            if not isinstance(call.id, str) or not call.id.strip():
                raise InvalidModelResponse("Tool call id cannot be empty")
            if call.id in seen_ids:
                raise InvalidModelResponse("Tool call ids must be unique within a response")
            if not isinstance(call.name, str) or not call.name.strip():
                raise InvalidModelResponse("Tool call name cannot be empty")
            if not isinstance(call.arguments, Mapping):
                raise InvalidModelResponse("Tool call arguments must be an object")
            try:
                arguments = dict(call.arguments)
                json.dumps(arguments, ensure_ascii=False, allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise InvalidModelResponse("Tool call arguments must be valid JSON") from exc
            seen_ids.add(call.id)
            normalized.append(
                LLMToolCall(id=call.id, name=call.name, arguments=arguments)
            )
        return ToolCallBatch(calls=tuple(normalized))

    if not isinstance(response.content, str) or not response.content.strip():
        raise InvalidModelResponse("Response must contain final text or tool calls")
    return FinalAnswer(text=response.content.strip())

import json
import os
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from app.llm_client import LLMMessage, LLMRequest, LLMResponse, LLMToolCall


DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


class LLMProviderError(RuntimeError):
    """Raised when the provider response cannot be used by the Runtime."""


class OpenAIChatCompletionsClient:
    """OpenAI Chat Completions adapter; tool execution remains in AgentRuntime."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str = DEFAULT_OPENAI_BASE_URL,
        timeout: float = 30.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set to use the model provider")
        if not model.strip():
            raise ValueError("model cannot be empty")
        if not base_url.strip():
            raise ValueError("base_url cannot be empty")
        if timeout <= 0:
            raise ValueError("timeout must be positive")

        self.model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._api_key = api_key
        self._http_client = http_client

    @classmethod
    def from_environment(cls) -> "OpenAIChatCompletionsClient | None":
        if not os.getenv("OPENAI_API_KEY", "").strip():
            return None

        model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL).strip()
        base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL).strip()
        return cls(
            model=model or DEFAULT_OPENAI_MODEL,
            base_url=base_url or DEFAULT_OPENAI_BASE_URL,
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        payload = self._request_payload(request)
        headers = {"Authorization": f"Bearer {self._api_key}"}
        endpoint = f"{self.base_url}/chat/completions"

        try:
            if self._http_client is not None:
                response = await self._http_client.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
            else:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(endpoint, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            raise LLMProviderError(
                f"OpenAI-compatible request failed ({type(exc).__name__})"
            ) from None

        if response.status_code >= 400:
            raise LLMProviderError(
                f"OpenAI-compatible API returned HTTP {response.status_code}"
            )

        try:
            body = response.json()
        except ValueError:
            raise LLMProviderError("OpenAI-compatible API returned invalid JSON") from None
        return self._parse_response(body)

    def _request_payload(self, request: LLMRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [self._message_to_provider(message) for message in request.messages],
        }
        if request.tools:
            payload["tools"] = [self._tool_to_provider(tool) for tool in request.tools]
            payload["tool_choice"] = "auto"
        return payload

    @staticmethod
    def _message_to_provider(message: LLMMessage) -> dict[str, Any]:
        if message.role not in {"system", "user", "assistant", "tool"}:
            raise LLMProviderError("Unsupported message role")

        mapped: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.role == "assistant" and message.tool_calls:
            mapped["content"] = message.content or None
            mapped["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(
                            dict(call.arguments),
                            ensure_ascii=False,
                            separators=(",", ":"),
                            allow_nan=False,
                        ),
                    },
                }
                for call in message.tool_calls
            ]
        elif message.role == "tool":
            if not message.tool_call_id:
                raise LLMProviderError("Tool result is missing its tool_call_id")
            mapped["tool_call_id"] = message.tool_call_id
            if message.name:
                mapped["name"] = message.name
        return mapped

    @staticmethod
    def _tool_to_provider(tool: Mapping[str, Any]) -> dict[str, Any]:
        name = tool.get("name")
        description = tool.get("description")
        parameters = tool.get("parameters")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(description, str)
            or not isinstance(parameters, Mapping)
        ):
            raise LLMProviderError("Tool schema is not in the Runtime's standard format")
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": dict(parameters),
            },
        }

    @staticmethod
    def _parse_response(body: Any) -> LLMResponse:
        if not isinstance(body, Mapping):
            raise LLMProviderError("OpenAI-compatible response must be an object")
        choices = body.get("choices")
        if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
            raise LLMProviderError("OpenAI-compatible response has no choices")

        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise LLMProviderError("OpenAI-compatible choice must be an object")
        message = choice.get("message")
        if not isinstance(message, Mapping):
            raise LLMProviderError("OpenAI-compatible choice has no message")

        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise LLMProviderError("OpenAI-compatible message content must be text")
        if content is None and isinstance(message.get("refusal"), str):
            content = message["refusal"]

        raw_calls = message.get("tool_calls")
        if raw_calls is None:
            raw_calls = []
        if not isinstance(raw_calls, Sequence) or isinstance(raw_calls, (str, bytes)):
            raise LLMProviderError("OpenAI-compatible tool_calls must be a list")

        calls = tuple(OpenAIChatCompletionsClient._parse_tool_call(call) for call in raw_calls)
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            finish_reason = None
        return LLMResponse(
            content=content,
            tool_calls=calls,
            finish_reason=finish_reason,
        )

    @staticmethod
    def _parse_tool_call(raw_call: Any) -> LLMToolCall:
        if not isinstance(raw_call, Mapping):
            raise LLMProviderError("OpenAI-compatible tool call must be an object")
        call_id = raw_call.get("id")
        function = raw_call.get("function")
        if not isinstance(call_id, str) or not call_id.strip() or not isinstance(function, Mapping):
            raise LLMProviderError("OpenAI-compatible tool call is missing its id or function")

        name = function.get("name")
        raw_arguments = function.get("arguments")
        if not isinstance(name, str) or not name.strip() or not isinstance(raw_arguments, str):
            raise LLMProviderError("OpenAI-compatible function call is malformed")
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            raise LLMProviderError("OpenAI-compatible function arguments are invalid JSON") from None
        if not isinstance(arguments, dict):
            raise LLMProviderError("OpenAI-compatible function arguments must be an object")

        return LLMToolCall(id=call_id, name=name, arguments=arguments)

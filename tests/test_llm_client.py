import asyncio
import json

import httpx

from app.llm_client import LLMClient, LLMMessage, LLMRequest, LLMResponse, LLMToolCall
from app.openai_client import OpenAIChatCompletionsClient


class FakeLLMClient:
    async def complete(self, request: LLMRequest) -> LLMResponse:
        assert request.messages[-1].content == "Hello"
        return LLMResponse(content="Hi")


def test_llm_client_protocol_is_provider_neutral():
    client: LLMClient = FakeLLMClient()
    request = LLMRequest(messages=[LLMMessage(role="user", content="Hello")])

    response = asyncio.run(client.complete(request))

    assert response.content == "Hi"
    assert response.tool_calls == ()


def test_protocol_types_can_represent_tool_calls():
    response = LLMResponse(
        tool_calls=[LLMToolCall(id="call-1", name="todo", arguments={"action": "list"})]
    )

    assert response.tool_calls[0].name == "todo"


def test_openai_adapter_maps_native_tool_calls_and_runtime_schemas(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    received = {}

    def respond(request: httpx.Request) -> httpx.Response:
        received["authorization"] = request.headers["authorization"]
        received["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "calculator",
                                        "arguments": '{"expression":"2 + 2"}',
                                    },
                                }
                            ],
                        },
                    }
                ]
            },
        )

    request = LLMRequest(
        messages=[
            LLMMessage(role="system", content="Use tools when needed."),
            LLMMessage(role="user", content="What is 2 + 2?"),
            LLMMessage(
                role="assistant",
                content="",
                tool_calls=(
                    LLMToolCall(
                        id="call-prev",
                        name="search",
                        arguments={"query": "math"},
                    ),
                ),
            ),
            LLMMessage(
                role="tool",
                name="search",
                tool_call_id="call-prev",
                content='{"ok":true}',
            ),
        ],
        tools=[
            {
                "name": "calculator",
                "description": "Safely calculate arithmetic expressions.",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                    "additionalProperties": False,
                },
            }
        ],
    )

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            adapter = OpenAIChatCompletionsClient(
                model="test-model",
                base_url="https://llm.example/v1",
                http_client=client,
            )
            return await adapter.complete(request)

    response = asyncio.run(exercise())

    assert received["authorization"] == "Bearer test-key"
    assert received["payload"]["model"] == "test-model"
    assert received["payload"]["messages"][2]["tool_calls"][0]["function"] == {
        "name": "search",
        "arguments": '{"query":"math"}',
    }
    assert received["payload"]["messages"][3] == {
        "role": "tool",
        "content": '{"ok":true}',
        "tool_call_id": "call-prev",
        "name": "search",
    }
    assert received["payload"]["tools"][0]["function"]["parameters"] == request.tools[0][
        "parameters"
    ]
    assert received["payload"]["tool_choice"] == "auto"
    assert response.finish_reason == "tool_calls"
    assert response.tool_calls == (
        LLMToolCall(id="call-1", name="calculator", arguments={"expression": "2 + 2"}),
    )


def test_openai_adapter_maps_final_text(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def respond(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"finish_reason": "stop", "message": {"content": "4"}}]},
        )

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            adapter = OpenAIChatCompletionsClient(model="test-model", http_client=client)
            return await adapter.complete(LLMRequest(messages=[LLMMessage("user", "2+2")]))

    response = asyncio.run(exercise())

    assert response.content == "4"
    assert response.tool_calls == ()

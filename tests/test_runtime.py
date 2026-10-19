import asyncio
import json

from app.llm_client import LLMRequest, LLMResponse, LLMToolCall
from app.runtime import AgentRuntime
from app.sessions import MessageRepository, SessionRepository
from app.tools import build_tool_registry
from app.todos import TodoRepository


class ScriptedLLMClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _runtime(database, client, *, max_model_turns=5, max_tool_calls_per_turn=3):
    sessions = SessionRepository(database)
    messages = MessageRepository(database)
    todos = TodoRepository(database)
    session = sessions.create("user-a", "Runtime test")
    runtime = AgentRuntime(
        client,
        sessions,
        messages,
        build_tool_registry(todos),
        max_model_turns=max_model_turns,
        max_tool_calls_per_turn=max_tool_calls_per_turn,
    )
    return runtime, sessions, messages, session


def _run(runtime, session_id, text="Hello"):
    return asyncio.run(runtime.run("user-a", session_id, text))


def test_runtime_returns_and_persists_a_direct_answer(database):
    client = ScriptedLLMClient(LLMResponse(content="Hello there."))
    runtime, _, messages, session = _runtime(database, client)

    result = _run(runtime, session.id)

    assert result.status == "completed"
    assert result.answer == "Hello there."
    assert result.model_turns == 1
    assert result.tool_calls == 0
    history = messages.list_for_session("user-a", session.id)
    assert [(message.role, message.content) for message in history] == [
        ("user", "Hello"),
        ("assistant", "Hello there."),
    ]
    assert client.requests[0].messages[0].role == "system"
    assert client.requests[0].messages[-1].role == "user"


def test_runtime_executes_a_tool_saves_result_and_continues(database):
    client = ScriptedLLMClient(
        LLMResponse(
            tool_calls=[
                LLMToolCall(
                    id="call-calc",
                    name="calculator",
                    arguments={"expression": "12 * (3 + 1)"},
                )
            ]
        ),
        LLMResponse(content="The result is 48."),
    )
    runtime, _, messages, session = _runtime(database, client)

    result = _run(runtime, session.id, "Calculate 12 times 4")

    assert result.status == "completed"
    assert result.model_turns == 2
    assert result.tool_calls == 1
    assert result.answer == "The result is 48."
    history = messages.list_for_session("user-a", session.id)
    assert [message.role for message in history] == ["user", "assistant", "tool", "assistant"]
    assert json.loads(history[1].tool_call_json)[0]["id"] == "call-calc"
    assert json.loads(history[2].content) == {"ok": True, "result": {
        "ok": True,
        "expression": "12 * (3 + 1)",
        "result": 48,
    }}
    next_request = client.requests[1]
    assert next_request.messages[-2].tool_calls[0].id == "call-calc"
    assert next_request.messages[-1].tool_call_id == "call-calc"
    assert next_request.messages[-1].name == "calculator"


def test_runtime_handles_multiple_tool_rounds(database):
    client = ScriptedLLMClient(
        LLMResponse(
            tool_calls=[
                LLMToolCall(
                    id="calc-1",
                    name="calculator",
                    arguments={"expression": "6 * 7"},
                )
            ]
        ),
        LLMResponse(
            tool_calls=[
                LLMToolCall(
                    id="todo-1",
                    name="todo",
                    arguments={"action": "add", "text": "Send the answer"},
                )
            ]
        ),
        LLMResponse(content="The answer is 42, and I added the todo."),
    )
    runtime, _, messages, session = _runtime(database, client)

    result = _run(runtime, session.id, "Calculate 6 times 7 and add a todo")

    assert result.status == "completed"
    assert result.model_turns == 3
    assert result.tool_calls == 2
    assert result.answer == "The answer is 42, and I added the todo."
    assert [message.role for message in messages.list_for_session("user-a", session.id)] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]


def test_tool_errors_are_saved_for_the_next_model_turn(database):
    client = ScriptedLLMClient(
        LLMResponse(
            tool_calls=[
                LLMToolCall(id="missing", name="not_registered", arguments={})
            ]
        ),
        LLMResponse(content="That tool was unavailable, so I could not do it."),
    )
    runtime, _, messages, session = _runtime(database, client)

    result = _run(runtime, session.id)

    assert result.status == "completed"
    error_message = messages.list_for_session("user-a", session.id)[2]
    tool_result = json.loads(error_message.content)
    assert error_message.role == "tool"
    assert tool_result["ok"] is False
    assert tool_result["error"]["type"] == "UnknownToolError"
    assert client.requests[1].messages[-1].tool_call_id == "missing"


def test_invalid_model_response_stops_safely(database):
    client = ScriptedLLMClient(LLMResponse())
    runtime, _, messages, session = _runtime(database, client)

    result = _run(runtime, session.id)

    assert result.status == "invalid_response"
    assert result.model_turns == 1
    assert result.tool_calls == 0
    assert "无法识别" in result.answer
    history = messages.list_for_session("user-a", session.id)
    assert len(history) == 2
    assert history[-1].role == "assistant"


def test_model_turn_limit_stops_repeated_tool_calls(database):
    repeated_call = LLMResponse(
        tool_calls=[
            LLMToolCall(
                id="calc",
                name="calculator",
                arguments={"expression": "1 + 1"},
            )
        ]
    )
    client = ScriptedLLMClient(repeated_call, repeated_call)
    runtime, _, messages, session = _runtime(database, client, max_model_turns=2)

    result = _run(runtime, session.id)

    assert result.status == "max_turns"
    assert result.model_turns == 2
    assert result.tool_calls == 2
    assert len(client.requests) == 2
    assert messages.list_for_session("user-a", session.id)[-1].content == result.answer


def test_single_turn_tool_limit_rejects_the_whole_batch(database):
    client = ScriptedLLMClient(
        LLMResponse(
            tool_calls=[
                LLMToolCall("calc-1", "calculator", {"expression": "1 + 1"}),
                LLMToolCall("calc-2", "calculator", {"expression": "2 + 2"}),
            ]
        )
    )
    runtime, _, messages, session = _runtime(
        database,
        client,
        max_tool_calls_per_turn=1,
    )

    result = _run(runtime, session.id)

    assert result.status == "invalid_response"
    assert result.tool_calls == 0
    assert len(messages.list_for_session("user-a", session.id)) == 2


def test_saved_tool_exchange_is_restored_for_a_later_user_turn(database):
    first_client = ScriptedLLMClient(
        LLMResponse(
            tool_calls=[
                LLMToolCall("call-1", "calculator", {"expression": "40 + 2"})
            ]
        ),
        LLMResponse(content="42"),
    )
    runtime, sessions, messages, session = _runtime(database, first_client)
    _run(runtime, session.id, "Calculate the answer")

    followup_client = ScriptedLLMClient(LLMResponse(content="It was 42."))
    followup_runtime = AgentRuntime(
        followup_client,
        sessions,
        messages,
        build_tool_registry(TodoRepository(database)),
    )
    _run(followup_runtime, session.id, "What was the result?")

    restored = followup_client.requests[0].messages
    assistant_tool_call = next(message for message in restored if message.tool_calls)
    tool_result = next(message for message in restored if message.role == "tool")
    assert assistant_tool_call.tool_calls[0].id == "call-1"
    assert tool_result.tool_call_id == "call-1"
    assert json.loads(tool_result.content)["result"]["result"] == 42

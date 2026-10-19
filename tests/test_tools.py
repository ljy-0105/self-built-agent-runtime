import asyncio

import pytest

from app.sessions import SessionRepository
from app.tools import build_tool_registry
from app.tools.base import (
    ToolArgumentError,
    ToolContext,
    ToolExecutionError,
    ToolRegistry,
    UnknownToolError,
)
from app.tools.calculator import calculate
from app.tools.base import ToolDefinition, ToolSchemaError
from app.storage import Database
from app.todos import TodoRepository


@pytest.fixture
def tool_setup(database):
    sessions = SessionRepository(database)
    first = sessions.create("user-a", "First window")
    second = sessions.create("user-a", "Second window")
    registry = build_tool_registry(TodoRepository(database))
    return registry, ToolContext(user_id="user-a", session_id=first.id), first, second


def _run(registry, name, arguments, context):
    return asyncio.run(registry.execute(name, arguments, context))


def test_calculator_and_mock_search_normal_inputs(tool_setup):
    registry, context, _, _ = tool_setup

    calculation = _run(
        registry,
        "calculator",
        {"expression": "12 * (3 + 1)"},
        context,
    )
    search = _run(registry, "search", {"query": "agent runtime"}, context)

    assert calculation == {"ok": True, "expression": "12 * (3 + 1)", "result": 48}
    assert search["provider"] == "mock"
    assert search["query"] == "agent runtime"
    assert search["results"][0]["source"] == "mock://search/1"


def test_todo_add_list_and_complete_are_persisted(tool_setup, database):
    registry, context, _, _ = tool_setup

    added = _run(registry, "todo", {"action": "add", "text": "Send weekly report"}, context)
    todo_id = added["todo"]["id"]
    assert added["todo"]["status"] == "open"

    listed = _run(registry, "todo", {"action": "list"}, context)
    assert [todo["id"] for todo in listed["todos"]] == [todo_id]

    completed = _run(
        registry,
        "todo",
        {"action": "complete", "todo_id": todo_id},
        context,
    )
    assert completed["todo"]["status"] == "completed"
    assert completed["todo"]["completed_at"] is not None

    listed_again = _run(registry, "todo", {"action": "list"}, context)
    assert listed_again["todos"][0]["status"] == "completed"

    reopened_database = Database(database.path)
    reopened_database.initialize()
    reopened_todos = TodoRepository(reopened_database)
    persisted = reopened_todos.list_for_session("user-a", context.session_id)
    assert persisted[0].id == todo_id
    assert persisted[0].status == "completed"


def test_json_schema_rejects_invalid_arguments_before_execution(tool_setup):
    registry, context, _, _ = tool_setup

    with pytest.raises(ToolArgumentError, match="expression"):
        _run(registry, "calculator", {}, context)
    with pytest.raises(ToolArgumentError, match="Additional properties"):
        _run(registry, "search", {"query": "news", "limit": 10}, context)
    with pytest.raises(ToolArgumentError):
        _run(registry, "todo", {"action": "add"}, context)
    with pytest.raises(ToolArgumentError):
        _run(registry, "todo", {"action": "remove", "todo_id": "x"}, context)

    assert _run(registry, "todo", {"action": "list"}, context)["todos"] == []


def test_unknown_tool_is_rejected(tool_setup):
    registry, context, _, _ = tool_setup

    with pytest.raises(UnknownToolError, match="not_registered"):
        _run(registry, "not_registered", {}, context)


def test_todo_items_are_isolated_between_sessions_and_users(tool_setup):
    registry, first_context, first_session, second_session = tool_setup
    added = _run(
        registry,
        "todo",
        {"action": "add", "text": "Private to first window"},
        first_context,
    )
    todo_id = added["todo"]["id"]

    second_context = ToolContext(user_id="user-a", session_id=second_session.id)
    other_user_context = ToolContext(user_id="user-b", session_id=first_session.id)

    assert _run(registry, "todo", {"action": "list"}, second_context)["todos"] == []
    assert _run(registry, "todo", {"action": "list"}, other_user_context)["todos"] == []
    with pytest.raises(ToolExecutionError, match="not found"):
        _run(
            registry,
            "todo",
            {"action": "complete", "todo_id": todo_id},
            second_context,
        )
    assert _run(registry, "todo", {"action": "list"}, first_context)["todos"][0][
        "status"
    ] == "open"


def test_calculator_does_not_evaluate_python_code():
    assert calculate("-2 + 3 * (4 / 2)") == 4
    with pytest.raises(ToolExecutionError):
        calculate("__import__('os').system('echo unsafe')")
    with pytest.raises(ToolExecutionError):
        calculate("1 / 0")


def test_registry_rejects_duplicate_names_and_invalid_schemas():
    registry = ToolRegistry()
    tool = ToolDefinition(
        name="sample",
        description="sample tool",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        handler=lambda _arguments, _context: {"ok": True},
    )
    registry.register(tool)
    with pytest.raises(ToolSchemaError, match="already registered"):
        registry.register(tool)

    with pytest.raises(ToolSchemaError, match="Invalid JSON Schema"):
        ToolDefinition(
            name="broken",
            description="invalid schema",
            parameters={"type": "not-a-json-schema-type"},
            handler=lambda _arguments, _context: {},
        )

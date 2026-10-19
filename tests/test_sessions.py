import pytest

from app.errors import SessionNotFoundError
from app.sessions import MessageRepository, SessionRepository
from app.todos import TodoRepository


def test_sessions_messages_and_todos_are_scoped_to_session(database):
    sessions = SessionRepository(database)
    messages = MessageRepository(database)
    todos = TodoRepository(database)

    first = sessions.create("user-a", "Window 1")
    second = sessions.create("user-a", "Window 2")

    message = messages.append(
        "user-a", first.id, "user", "Check weather and remember a task"
    )
    assert message.sequence == 1
    assert [item.id for item in messages.list_for_session("user-a", first.id)] == [
        message.id
    ]
    assert messages.list_for_session("user-a", second.id) == []

    todo = todos.create("user-a", first.id, "Bring an umbrella")
    assert todo is not None
    assert todos.list_for_session("user-a", first.id) == [todo]
    assert todos.list_for_session("user-a", second.id) == []

    completed = todos.complete("user-a", first.id, todo.id)
    assert completed is not None
    assert completed.status == "completed"
    assert completed.completed_at is not None


def test_repositories_enforce_session_ownership(database):
    sessions = SessionRepository(database)
    messages = MessageRepository(database)
    todos = TodoRepository(database)
    session = sessions.create("user-a", "Private session")

    assert sessions.get_owned("user-b", session.id) is None
    with pytest.raises(SessionNotFoundError):
        messages.append("user-b", session.id, "user", "Should not be inserted")
    with pytest.raises(SessionNotFoundError):
        messages.list_for_session("user-b", session.id)
    assert todos.create("user-b", session.id, "Should not be inserted") is None
    assert todos.list_for_session("user-b", session.id) == []


def test_message_sequence_and_tool_metadata_are_persisted(database):
    session = SessionRepository(database).create("user-a")
    messages = MessageRepository(database)

    user_message = messages.append("user-a", session.id, "user", "Calculate 2 + 2")
    tool_message = messages.append(
        "user-a",
        session.id,
        "tool",
        '{"result": 4}',
        tool_call={"id": "call-1", "name": "calculator"},
        tool_name="calculator",
    )

    assert [m.sequence for m in messages.list_for_session("user-a", session.id)] == [
        1,
        2,
    ]
    assert tool_message.sequence == user_message.sequence + 1
    assert tool_message.tool_name == "calculator"
    assert '"call-1"' in tool_message.tool_call_json

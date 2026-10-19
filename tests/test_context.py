import asyncio

from app.context import SummaryGenerator
from app.llm_client import LLMRequest, LLMResponse, LLMToolCall
from app.runtime import AgentRuntime
from app.sessions import MessageRepository, SessionRepository
from app.storage import Database
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


class TranscriptSummaryGenerator:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls: list[tuple[str | None, list[str]]] = []

    async def summarize(self, existing_summary, messages):
        contents = [message.content for message in messages]
        self.calls.append((existing_summary, contents))
        if self.error is not None:
            raise self.error
        parts = [existing_summary] if existing_summary else []
        parts.extend(content for content in contents if content)
        return " | ".join(parts)


def _runtime(
    database: Database,
    session_id: str,
    client: ScriptedLLMClient,
    *,
    summary_generator: SummaryGenerator | None = None,
    max_context_messages: int = 20,
    retain_recent_messages: int = 10,
) -> AgentRuntime:
    return AgentRuntime(
        client,
        SessionRepository(database),
        MessageRepository(database),
        build_tool_registry(TodoRepository(database)),
        summary_generator=summary_generator,
        max_context_messages=max_context_messages,
        retain_recent_messages=retain_recent_messages,
    )


def _run(runtime: AgentRuntime, session_id: str, text: str):
    return asyncio.run(runtime.run("user-a", session_id, text))


def _user_contents(request: LLMRequest) -> list[str]:
    return [message.content for message in request.messages if message.role == "user"]


def test_plain_followup_uses_session_history_without_duplicating_current_input(database):
    session = SessionRepository(database).create("user-a", "Follow-up")
    client = ScriptedLLMClient(
        LLMResponse(content="You prefer concise answers."),
        LLMResponse(content="You asked for concise answers."),
    )
    runtime = _runtime(database, session.id, client)

    _run(runtime, session.id, "I prefer concise answers.")
    result = _run(runtime, session.id, "What response style did I request?")

    request = client.requests[1]
    assert result.status == "completed"
    assert "I prefer concise answers." in [message.content for message in request.messages]
    assert _user_contents(request).count("What response style did I request?") == 1


def test_tool_followup_restores_tool_call_and_result(database):
    session = SessionRepository(database).create("user-a", "Tool follow-up")
    client = ScriptedLLMClient(
        LLMResponse(
            tool_calls=[
                LLMToolCall("calc-1", "calculator", {"expression": "6 * 7"})
            ]
        ),
        LLMResponse(content="The calculation returned 42."),
        LLMResponse(content="The earlier calculation returned 42."),
    )
    runtime = _runtime(
        database,
        session.id,
        client,
        summary_generator=TranscriptSummaryGenerator(),
        max_context_messages=4,
        retain_recent_messages=3,
    )

    _run(runtime, session.id, "Calculate 6 times 7.")
    result = _run(runtime, session.id, "What did the calculation return?")

    request = client.requests[2]
    assert result.status == "completed"
    assert any(
        message.tool_calls[0].id == "calc-1"
        for message in request.messages
        if message.tool_calls
    )
    assert any(
        message.role == "tool" and message.tool_call_id == "calc-1"
        for message in request.messages
    )
    assert _user_contents(request).count("What did the calculation return?") == 1
    assert any(
        message.role == "system" and message.content.startswith("Summary of earlier")
        for message in request.messages
    )


def test_context_compresses_older_messages_and_preserves_source_history(database):
    session = SessionRepository(database).create("user-a", "Compression")
    summary_generator = TranscriptSummaryGenerator()
    client = ScriptedLLMClient(
        LLMResponse(content="Answer one."),
        LLMResponse(content="Answer two."),
        LLMResponse(content="Answer three."),
    )
    runtime = _runtime(
        database,
        session.id,
        client,
        summary_generator=summary_generator,
        max_context_messages=4,
        retain_recent_messages=2,
    )

    _run(runtime, session.id, "Question one.")
    _run(runtime, session.id, "Question two.")
    _run(runtime, session.id, "Question three.")

    request = client.requests[2]
    summary_message = next(
        message
        for message in request.messages
        if message.role == "system" and message.content.startswith("Summary of earlier")
    )
    assert len(summary_generator.calls) == 1
    assert summary_generator.calls[0][1] == [
        "Question one.",
        "Answer one.",
        "Question two.",
    ]
    assert "Question one." in summary_message.content
    assert "Question two." in summary_message.content
    assert "Question one." not in [
        message.content
        for message in request.messages
        if message.role != "system"
    ]
    assert _user_contents(request).count("Question three.") == 1

    history = MessageRepository(database).list_for_session("user-a", session.id)
    assert len(history) == 6
    assert history[0].content == "Question one."
    assert history[2].content == "Question two."
    assert SessionRepository(database).get_owned("user-a", session.id).summary

    reopened_database = Database(database.path)
    reopened_database.initialize()
    followup_client = ScriptedLLMClient(LLMResponse(content="Question one was about the project."))
    followup_runtime = _runtime(reopened_database, session.id, followup_client)
    _run(followup_runtime, session.id, "What was question one about?")

    restored_request = followup_client.requests[0]
    restored_summary = next(
        message
        for message in restored_request.messages
        if message.role == "system" and message.content.startswith("Summary of earlier")
    )
    restored_history = MessageRepository(reopened_database).list_for_session(
        "user-a",
        session.id,
    )
    assert "Question one." in restored_summary.content
    assert len(restored_history) == 8
    assert restored_history[0].content == "Question one."


def test_summary_failure_keeps_old_summary_and_full_context(database):
    sessions = SessionRepository(database)
    messages = MessageRepository(database)
    session = sessions.create("user-a", "Summary failure")
    sessions.set_summary("user-a", session.id, "Previously saved facts.")
    for role, content in [
        ("user", "Old question one."),
        ("assistant", "Old answer one."),
        ("user", "Old question two."),
        ("assistant", "Old answer two."),
    ]:
        messages.append("user-a", session.id, role, content)

    summary_generator = TranscriptSummaryGenerator(RuntimeError("summary unavailable"))
    client = ScriptedLLMClient(LLMResponse(content="I can still answer."))
    runtime = _runtime(
        database,
        session.id,
        client,
        summary_generator=summary_generator,
        max_context_messages=4,
        retain_recent_messages=2,
    )

    result = _run(runtime, session.id, "Current question.")

    request = client.requests[0]
    request_contents = [message.content for message in request.messages]
    assert result.status == "completed"
    assert len(summary_generator.calls) == 1
    assert any("Previously saved facts." in content for content in request_contents)
    assert all(
        content in request_contents
        for content in [
            "Old question one.",
            "Old answer one.",
            "Old question two.",
            "Old answer two.",
            "Current question.",
        ]
    )
    assert sessions.get_owned("user-a", session.id).summary == "Previously saved facts."
    assert len(messages.list_for_session("user-a", session.id)) == 6


def test_default_llm_summary_generator_runs_before_model_decision(database):
    sessions = SessionRepository(database)
    messages = MessageRepository(database)
    session = sessions.create("user-a", "LLM summary")
    messages.append("user-a", session.id, "user", "The project is called Atlas.")
    messages.append("user-a", session.id, "assistant", "I will remember Atlas.")
    client = ScriptedLLMClient(
        LLMResponse(content="The project is called Atlas."),
        LLMResponse(content="The project is Atlas."),
    )
    runtime = _runtime(
        database,
        session.id,
        client,
        max_context_messages=2,
        retain_recent_messages=1,
    )

    result = _run(runtime, session.id, "What is the project called?")

    summary_request, answer_request = client.requests
    assert result.status == "completed"
    assert "Summarize the earlier conversation" in summary_request.messages[0].content
    assert "The project is called Atlas." in summary_request.messages[1].content
    assert any(
        message.role == "system"
        and "The project is called Atlas." in message.content
        for message in answer_request.messages
    )
    assert _user_contents(answer_request).count("What is the project called?") == 1


def test_sessions_remain_isolated_after_database_and_runtime_reopen(tmp_path):
    database_path = tmp_path / "restart.sqlite3"
    database = Database(database_path)
    database.initialize()
    sessions = SessionRepository(database)
    session_a = sessions.create("user-a", "Window A")
    session_b = sessions.create("user-a", "Window B")

    client_a = ScriptedLLMClient(LLMResponse(content="A stays in window A."))
    client_b = ScriptedLLMClient(LLMResponse(content="B stays in window B."))
    _run(_runtime(database, session_a.id, client_a), session_a.id, "Window A private fact.")
    _run(_runtime(database, session_b.id, client_b), session_b.id, "Window B private fact.")

    reopened_database = Database(database_path)
    reopened_database.initialize()
    followup_a = ScriptedLLMClient(LLMResponse(content="A context restored."))
    followup_b = ScriptedLLMClient(LLMResponse(content="B context restored."))
    runtime_a = _runtime(reopened_database, session_a.id, followup_a)
    runtime_b = _runtime(reopened_database, session_b.id, followup_b)

    _run(runtime_a, session_a.id, "Continue window A.")
    _run(runtime_b, session_b.id, "Continue window B.")

    request_a = followup_a.requests[0]
    request_b = followup_b.requests[0]
    contents_a = [message.content for message in request_a.messages]
    contents_b = [message.content for message in request_b.messages]
    assert "Window A private fact." in contents_a
    assert "Window B private fact." not in contents_a
    assert "Window B private fact." in contents_b
    assert "Window A private fact." not in contents_b
    assert _user_contents(request_a).count("Continue window A.") == 1
    assert _user_contents(request_b).count("Continue window B.") == 1

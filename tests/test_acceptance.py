import json

from fastapi.testclient import TestClient

from app.api import create_app
from app.llm_client import LLMRequest, LLMResponse, LLMToolCall


class WorkflowLLMClient:
    """Scripted tool planner for API acceptance tests; never calls a provider."""

    def __init__(self):
        self.requests: list[LLMRequest] = []
        self._call_number = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        latest = request.messages[-1]
        if latest.role == "tool":
            return LLMResponse(content="Tool result recorded.")
        if latest.role != "user":
            return LLMResponse(content="No action needed.")

        text = latest.content.strip()
        normalized = text.casefold()
        if normalized.startswith("calculate "):
            return self._call("calculator", {"expression": text[10:]})
        if normalized.startswith("search "):
            return self._call("search", {"query": text[7:]})
        if normalized.startswith("add todo:"):
            return self._call("todo", {"action": "add", "text": text.split(":", 1)[1].strip()})
        if normalized == "list todos":
            return self._call("todo", {"action": "list"})
        if normalized == "complete todo":
            return self._call(
                "todo",
                {"action": "complete", "todo_id": self._find_todo_id(request)},
            )
        return LLMResponse(content="No action needed.")

    def _call(self, name: str, arguments: dict) -> LLMResponse:
        self._call_number += 1
        return LLMResponse(
            tool_calls=[
                LLMToolCall(
                    id=f"acceptance-call-{self._call_number}",
                    name=name,
                    arguments=arguments,
                )
            ]
        )

    @staticmethod
    def _find_todo_id(request: LLMRequest) -> str:
        for message in reversed(request.messages):
            if message.role != "tool" or message.name != "todo":
                continue
            envelope = json.loads(message.content)
            result = envelope.get("result", {})
            todo = result.get("todo")
            if isinstance(todo, dict) and todo.get("id"):
                return todo["id"]
            todos = result.get("todos", [])
            if todos and todos[0].get("id"):
                return todos[0]["id"]
        raise AssertionError("The session history does not contain a todo ID")


def _post_message(client: TestClient, session_id: str, content: str) -> dict:
    response = client.post(
        f"/sessions/{session_id}/messages",
        json={"content": content},
    )
    assert response.status_code == 201
    result = response.json()
    assert result["status"] == "completed"
    return result


def _latest_tool_result(messages: list[dict], tool_name: str) -> dict:
    for message in reversed(messages):
        if message["role"] == "tool" and message["tool_name"] == tool_name:
            return json.loads(message["content"])
    raise AssertionError(f"No result found for tool {tool_name}")


def test_api_acceptance_search_todo_session_isolation_and_restart(tmp_path):
    database_path = str(tmp_path / "acceptance.sqlite3")
    first_model = WorkflowLLMClient()
    first_app = create_app(database_path, llm_client=first_model)

    with TestClient(first_app) as client:
        window_a = client.post("/sessions", json={"title": "Window A"}).json()
        window_b = client.post("/sessions", json={"title": "Window B"}).json()

        _post_message(client, window_a["id"], "Calculate 12 * (3 + 1)")
        _post_message(client, window_a["id"], "Search agent runtime")
        _post_message(client, window_a["id"], "Add todo: send weekly report")
        _post_message(client, window_a["id"], "List todos")
        _post_message(client, window_a["id"], "Complete todo")

        history_a = client.get(f"/sessions/{window_a['id']}/messages").json()
        calculation_result = _latest_tool_result(history_a, "calculator")
        search_result = _latest_tool_result(history_a, "search")
        completed_result = _latest_tool_result(history_a, "todo")
        assert calculation_result["ok"] is True
        assert calculation_result["result"]["result"] == 48
        assert search_result["result"]["provider"] == "mock"
        assert search_result["result"]["results"]
        assert completed_result["result"]["todo"]["status"] == "completed"
        assert len(first_model.requests) == 10

        _post_message(client, window_b["id"], "List todos")
        history_b = client.get(f"/sessions/{window_b['id']}/messages").json()
        assert _latest_tool_result(history_b, "todo")["result"]["todos"] == []
        window_b_request = first_model.requests[10]
        window_b_context = [message.content for message in window_b_request.messages]
        assert "send weekly report" not in window_b_context

    restarted_model = WorkflowLLMClient()
    restarted_app = create_app(database_path, llm_client=restarted_model)
    with TestClient(restarted_app) as client:
        _post_message(client, window_a["id"], "List todos")
        _post_message(client, window_b["id"], "List todos")

        restored_a = client.get(f"/sessions/{window_a['id']}/messages").json()
        restored_b = client.get(f"/sessions/{window_b['id']}/messages").json()
        assert _latest_tool_result(restored_a, "todo")["result"]["todos"][0][
            "status"
        ] == "completed"
        assert _latest_tool_result(restored_b, "todo")["result"]["todos"] == []

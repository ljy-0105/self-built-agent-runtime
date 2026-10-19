from fastapi.testclient import TestClient

from app.api import create_app
from app.llm_client import LLMRequest, LLMResponse
from app.openai_client import OpenAIChatCompletionsClient


class FakeLLMClient:
    async def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(content=f"Reply to: {request.messages[-1].content}")


def test_session_and_message_endpoints_persist_data(client):
    assert client.get("/healthz").json() == {"status": "ok"}

    created = client.post("/sessions", json={"title": "Window 1"})
    assert created.status_code == 201
    session = created.json()
    assert session["title"] == "Window 1"

    stored = client.post(
        f"/sessions/{session['id']}/messages",
        json={"content": "Remember to send the weekly report"},
    )
    assert stored.status_code == 201
    assert stored.json()["status"] == "runtime_not_configured"
    assert stored.json()["assistant_message"] is None
    assert stored.json()["user_message"]["role"] == "user"
    assert stored.json()["user_message"]["sequence"] == 1

    history = client.get(f"/sessions/{session['id']}/messages")
    assert [message["content"] for message in history.json()] == [
        "Remember to send the weekly report"
    ]


def test_root_serves_chinese_agent_workbench(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert '<html lang="zh-CN">' in response.text
    assert "新建对话" in response.text
    assert "发送消息" in response.text
    assert 'href="/docs"' in response.text


def test_api_does_not_expose_sessions_to_another_identity(tmp_path):
    identity = {"user_id": "user-a"}
    app = create_app(
        str(tmp_path / "scope.sqlite3"),
        current_user_provider=lambda: identity["user_id"],
    )
    with TestClient(app) as client:
        created = client.post("/sessions", json={}).json()
        identity["user_id"] = "user-b"

        assert client.get(f"/sessions/{created['id']}").status_code == 404
        assert client.get(f"/sessions/{created['id']}/messages").status_code == 404
        assert client.post(
            f"/sessions/{created['id']}/messages",
            json={"content": "unauthorized"},
        ).status_code == 404


def test_message_endpoint_rejects_blank_content(client):
    session = client.post("/sessions", json={}).json()

    response = client.post(
        f"/sessions/{session['id']}/messages",
        json={"content": "   "},
    )

    assert response.status_code == 422


def test_app_registers_builtin_tools(client):
    schemas = client.app.state.tool_registry.schemas()

    assert {schema["name"] for schema in schemas} == {
        "calculator",
        "search",
        "todo",
    }


def test_message_endpoint_runs_runtime_when_llm_client_is_injected(tmp_path):
    app = create_app(str(tmp_path / "runtime-api.sqlite3"), llm_client=FakeLLMClient())
    with TestClient(app) as client:
        session = client.post("/sessions", json={}).json()

        response = client.post(
            f"/sessions/{session['id']}/messages",
            json={"content": "Hello"},
        )

    assert response.status_code == 201
    assert response.json()["status"] == "completed"
    assert response.json()["assistant_message"]["content"] == "Reply to: Hello"


def test_app_configures_openai_client_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://llm.example/v1")

    app = create_app(str(tmp_path / "provider.sqlite3"))

    assert isinstance(app.state.llm_client, OpenAIChatCompletionsClient)
    assert app.state.llm_client.model == "test-model"
    assert app.state.llm_client.base_url == "https://llm.example/v1"

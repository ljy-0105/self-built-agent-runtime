from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.storage import Database


@pytest.fixture
def database(tmp_path) -> Database:
    database = Database(tmp_path / "test.sqlite3")
    database.initialize()
    return database


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return create_app(str(tmp_path / "api.sqlite3"))


@pytest.fixture
def client(app) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client

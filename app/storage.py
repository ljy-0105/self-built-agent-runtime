import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


SCHEMA = """
PRAGMA user_version = 1;

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    title TEXT NOT NULL,
    summary TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_updated
    ON sessions (user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    sequence INTEGER NOT NULL CHECK (sequence > 0),
    role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant', 'tool')),
    content TEXT NOT NULL,
    tool_call_json TEXT,
    tool_name TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (session_id, sequence)
);

CREATE INDEX IF NOT EXISTS idx_messages_session_sequence
    ON messages (session_id, sequence);

CREATE TABLE IF NOT EXISTS todos (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    text TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('open', 'completed')),
    created_at TEXT NOT NULL,
    completed_at TEXT,
    FOREIGN KEY (session_id, user_id)
        REFERENCES sessions (id, user_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_todos_scope_created
    ON todos (user_id, session_id, created_at);
"""


class Database:
    """Small SQLite connection and transaction manager."""

    def __init__(self, path: str | Path, *, timeout_seconds: float = 5.0) -> None:
        self.path = str(path)
        self.timeout_seconds = timeout_seconds

    def initialize(self) -> None:
        path = Path(self.path)
        if self.path != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            self.path,
            timeout=self.timeout_seconds,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {int(self.timeout_seconds * 1000)}")
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def transaction(self, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()

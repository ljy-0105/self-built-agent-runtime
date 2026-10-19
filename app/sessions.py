import json
import sqlite3
from datetime import datetime, timezone
from uuid import uuid4

from app.errors import SessionNotFoundError
from app.models import Message, MessageRole, Session
from app.storage import Database


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _session_from_row(row: sqlite3.Row) -> Session:
    return Session(
        id=row["id"],
        user_id=row["user_id"],
        title=row["title"],
        summary=row["summary"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _message_from_row(row: sqlite3.Row) -> Message:
    return Message(
        id=row["id"],
        session_id=row["session_id"],
        sequence=row["sequence"],
        role=row["role"],
        content=row["content"],
        tool_call_json=row["tool_call_json"],
        tool_name=row["tool_name"],
        created_at=row["created_at"],
    )


class SessionRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def create(self, user_id: str, title: str = "新对话") -> Session:
        session = Session(
            id=f"sess_{uuid4().hex}",
            user_id=user_id,
            title=title,
            summary=None,
            created_at=_now(),
            updated_at=_now(),
        )
        with self.database.transaction(immediate=True) as connection:
            connection.execute(
                """
                INSERT INTO sessions (id, user_id, title, summary, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session.id,
                    session.user_id,
                    session.title,
                    session.summary,
                    session.created_at,
                    session.updated_at,
                ),
            )
        return session

    def get_owned(self, user_id: str, session_id: str) -> Session | None:
        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT * FROM sessions WHERE id = ? AND user_id = ?",
                (session_id, user_id),
            ).fetchone()
        return _session_from_row(row) if row else None

    def list_for_user(self, user_id: str, *, limit: int = 100) -> list[Session]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM sessions
                WHERE user_id = ?
                ORDER BY updated_at DESC, id
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        return [_session_from_row(row) for row in rows]

    def set_summary(self, user_id: str, session_id: str, summary: str | None) -> bool:
        now = _now()
        with self.database.transaction(immediate=True) as connection:
            cursor = connection.execute(
                """
                UPDATE sessions SET summary = ?, updated_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (summary, now, session_id, user_id),
            )
            return cursor.rowcount == 1


class MessageRepository:
    _ROLES = {"system", "user", "assistant", "tool"}

    def __init__(self, database: Database) -> None:
        self.database = database

    def append(
        self,
        user_id: str,
        session_id: str,
        role: MessageRole,
        content: str,
        *,
        tool_call: dict | list | None = None,
        tool_name: str | None = None,
    ) -> Message:
        if role not in self._ROLES:
            raise ValueError(f"Unsupported message role: {role}")

        message = Message(
            id=f"msg_{uuid4().hex}",
            session_id=session_id,
            sequence=0,
            role=role,
            content=content,
            tool_call_json=(
                json.dumps(tool_call, ensure_ascii=False, separators=(",", ":"))
                if tool_call is not None
                else None
            ),
            tool_name=tool_name,
            created_at=_now(),
        )
        with self.database.transaction(immediate=True) as connection:
            owner = connection.execute(
                "SELECT 1 FROM sessions WHERE id = ? AND user_id = ?",
                (session_id, user_id),
            ).fetchone()
            if owner is None:
                raise SessionNotFoundError(session_id)
            row = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            sequence = int(row[0])
            connection.execute(
                """
                INSERT INTO messages (
                    id, session_id, sequence, role, content,
                    tool_call_json, tool_name, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.id,
                    message.session_id,
                    sequence,
                    message.role,
                    message.content,
                    message.tool_call_json,
                    message.tool_name,
                    message.created_at,
                ),
            )
            connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ? AND user_id = ?",
                (message.created_at, session_id, user_id),
            )
        return Message(
            id=message.id,
            session_id=message.session_id,
            sequence=sequence,
            role=message.role,
            content=message.content,
            tool_call_json=message.tool_call_json,
            tool_name=message.tool_name,
            created_at=message.created_at,
        )

    def list_for_session(self, user_id: str, session_id: str) -> list[Message]:
        with self.database.connection() as connection:
            owner = connection.execute(
                "SELECT 1 FROM sessions WHERE id = ? AND user_id = ?",
                (session_id, user_id),
            ).fetchone()
            if owner is None:
                raise SessionNotFoundError(session_id)
            rows = connection.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY sequence",
                (session_id,),
            ).fetchall()
        return [_message_from_row(row) for row in rows]

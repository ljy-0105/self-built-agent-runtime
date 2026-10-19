import sqlite3
from datetime import datetime, timezone
from uuid import uuid4

from app.models import Todo
from app.storage import Database


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _todo_from_row(row: sqlite3.Row) -> Todo:
    return Todo(
        id=row["id"],
        user_id=row["user_id"],
        session_id=row["session_id"],
        text=row["text"],
        status=row["status"],
        created_at=row["created_at"],
        completed_at=row["completed_at"],
    )


class TodoRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def create(self, user_id: str, session_id: str, text: str) -> Todo | None:
        todo = Todo(
            id=f"todo_{uuid4().hex}",
            user_id=user_id,
            session_id=session_id,
            text=text,
            status="open",
            created_at=_now(),
            completed_at=None,
        )
        with self.database.transaction(immediate=True) as connection:
            owner = connection.execute(
                "SELECT 1 FROM sessions WHERE id = ? AND user_id = ?",
                (session_id, user_id),
            ).fetchone()
            if owner is None:
                return None
            connection.execute(
                """
                INSERT INTO todos (
                    id, user_id, session_id, text, status, created_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    todo.id,
                    todo.user_id,
                    todo.session_id,
                    todo.text,
                    todo.status,
                    todo.created_at,
                    todo.completed_at,
                ),
            )
            connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (todo.created_at, session_id),
            )
        return todo

    def list_for_session(self, user_id: str, session_id: str) -> list[Todo]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM todos
                WHERE user_id = ? AND session_id = ?
                ORDER BY created_at, id
                """,
                (user_id, session_id),
            ).fetchall()
        return [_todo_from_row(row) for row in rows]

    def complete(self, user_id: str, session_id: str, todo_id: str) -> Todo | None:
        completed_at = _now()
        with self.database.transaction(immediate=True) as connection:
            connection.execute(
                """
                UPDATE todos
                SET status = 'completed', completed_at = COALESCE(completed_at, ?)
                WHERE id = ? AND user_id = ? AND session_id = ?
                """,
                (completed_at, todo_id, user_id, session_id),
            )
            row = connection.execute(
                """
                SELECT * FROM todos
                WHERE id = ? AND user_id = ? AND session_id = ?
                """,
                (todo_id, user_id, session_id),
            ).fetchone()
        return _todo_from_row(row) if row else None

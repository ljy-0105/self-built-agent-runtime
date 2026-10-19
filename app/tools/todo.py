from dataclasses import asdict
from typing import Any

from app.tools.base import ToolContext, ToolDefinition, ToolExecutionError
from app.todos import TodoRepository


def todo_tool(repository: TodoRepository) -> ToolDefinition:
    def execute(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        action = arguments["action"]

        if action == "add":
            todo = repository.create(
                context.user_id,
                context.session_id,
                arguments["text"].strip(),
            )
            if todo is None:
                raise ToolExecutionError("Session not found for current user")
            return {"ok": True, "action": action, "todo": asdict(todo)}

        if action == "list":
            todos = repository.list_for_session(context.user_id, context.session_id)
            return {"ok": True, "action": action, "todos": [asdict(todo) for todo in todos]}

        todo = repository.complete(
            context.user_id,
            context.session_id,
            arguments["todo_id"],
        )
        if todo is None:
            raise ToolExecutionError("Todo not found in this session")
        return {"ok": True, "action": action, "todo": asdict(todo)}

    return ToolDefinition(
        name="todo",
        description="Create, list, or complete todo items in the current session.",
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "list", "complete"],
                },
                "text": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 1000,
                    "pattern": r"\S",
                    "description": "Todo text; required for add.",
                },
                "todo_id": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 100,
                    "description": "Todo ID; required for complete.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
            "allOf": [
                {
                    "if": {
                        "properties": {"action": {"const": "add"}},
                        "required": ["action"],
                    },
                    "then": {"required": ["text"], "not": {"required": ["todo_id"]}},
                },
                {
                    "if": {
                        "properties": {"action": {"const": "list"}},
                        "required": ["action"],
                    },
                    "then": {
                        "not": {
                            "anyOf": [{"required": ["text"]}, {"required": ["todo_id"]}]
                        }
                    },
                },
                {
                    "if": {
                        "properties": {"action": {"const": "complete"}},
                        "required": ["action"],
                    },
                    "then": {"required": ["todo_id"], "not": {"required": ["text"]}},
                },
            ],
        },
        handler=execute,
    )

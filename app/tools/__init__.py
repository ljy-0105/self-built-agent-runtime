from app.tools.base import ToolRegistry
from app.tools.calculator import calculator_tool
from app.tools.search import search_tool
from app.tools.todo import todo_tool
from app.todos import TodoRepository


def build_tool_registry(todo_repository: TodoRepository) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(calculator_tool())
    registry.register(search_tool())
    registry.register(todo_tool(todo_repository))
    return registry


__all__ = ["ToolRegistry", "build_tool_registry"]

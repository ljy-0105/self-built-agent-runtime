from typing import Any

from app.tools.base import ToolContext, ToolDefinition


def search_tool() -> ToolDefinition:
    def execute(arguments: dict[str, Any], _: ToolContext) -> dict[str, Any]:
        query = arguments["query"]
        return {
            "ok": True,
            "provider": "mock",
            "query": query,
            "results": [
                {
                    "title": f"Mock result for: {query}",
                    "snippet": "Deterministic mock data; no external search was performed.",
                    "source": "mock://search/1",
                }
            ],
        }

    return ToolDefinition(
        name="search",
        description="Search a deterministic mock index and return clearly labeled sample results.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 500,
                    "pattern": r"\S",
                    "description": "Search query.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        handler=execute,
    )

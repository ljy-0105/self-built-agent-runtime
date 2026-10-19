import copy
import inspect
import json
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError


class ToolError(Exception):
    """Base class for errors safe to return to the Agent Runtime."""


class ToolSchemaError(ToolError):
    """Raised when a tool definition is invalid."""


class UnknownToolError(ToolError, LookupError):
    """Raised when a model requests an unregistered tool."""


class ToolArgumentError(ToolError, ValueError):
    """Raised when tool arguments do not match the registered JSON Schema."""


class ToolExecutionError(ToolError):
    """Raised when a valid tool request cannot be completed."""


@dataclass(frozen=True)
class ToolContext:
    user_id: str
    session_id: str
    request_id: str | None = None


ToolHandler = Callable[
    [dict[str, Any], ToolContext], Any | Awaitable[Any]
]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler = field(repr=False, compare=False)
    _validator: Draft202012Validator = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", self.name):
            raise ToolSchemaError("Tool name must be a lowercase identifier")
        if not self.description.strip():
            raise ToolSchemaError("Tool description cannot be empty")

        parameters = copy.deepcopy(self.parameters)
        try:
            Draft202012Validator.check_schema(parameters)
        except SchemaError as exc:
            raise ToolSchemaError(f"Invalid JSON Schema for tool '{self.name}'") from exc

        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "_validator", Draft202012Validator(parameters))

    def schema(self) -> dict[str, Any]:
        """Return a provider-neutral tool definition for an LLM adapter."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": copy.deepcopy(self.parameters),
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        if tool.name in self._tools:
            raise ToolSchemaError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise UnknownToolError(f"Unknown tool: {name}") from exc

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    async def execute(
        self,
        name: str,
        arguments: Any,
        context: ToolContext,
    ) -> Any:
        tool = self.get(name)
        if not isinstance(arguments, Mapping):
            raise ToolArgumentError(f"Arguments for '{name}' must be a JSON object")

        normalized_arguments = dict(arguments)
        errors = sorted(
            tool._validator.iter_errors(normalized_arguments),
            key=lambda error: (
                tuple(str(part) for part in error.absolute_path),
                error.message,
            ),
        )
        if errors:
            error = errors[0]
            path = ".".join(str(part) for part in error.absolute_path) or "<root>"
            raise ToolArgumentError(
                f"Invalid arguments for '{name}' at {path}: {error.message}"
            ) from error

        try:
            result = tool.handler(normalized_arguments, context)
            if inspect.isawaitable(result):
                result = await result
            json.dumps(result, ensure_ascii=False, allow_nan=False)
        except ToolError:
            raise
        except (TypeError, ValueError) as exc:
            raise ToolExecutionError(f"Tool '{name}' returned an invalid result") from exc
        except Exception as exc:
            raise ToolExecutionError(f"Tool '{name}' failed") from exc
        return result

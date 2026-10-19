import ast
import math
import operator
from typing import Any

from app.tools.base import ToolContext, ToolDefinition, ToolExecutionError


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_MAX_EXPRESSION_LENGTH = 200
_MAX_AST_NODES = 64
_MAX_RESULT_MAGNITUDE = 1e100


def _calculate_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _calculate_node(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        if not math.isfinite(node.value):
            raise ToolExecutionError("Calculator only accepts finite numbers")
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_calculate_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _calculate_node(node.left)
        right = _calculate_node(node.right)
        return _BINARY_OPERATORS[type(node.op)](left, right)
    raise ToolExecutionError("Expression contains an unsupported operation")


def calculate(expression: str) -> int | float:
    if len(expression) > _MAX_EXPRESSION_LENGTH:
        raise ToolExecutionError("Expression is too long")
    try:
        parsed = ast.parse(expression, mode="eval")
        if sum(1 for _ in ast.walk(parsed)) > _MAX_AST_NODES:
            raise ToolExecutionError("Expression is too complex")
        result = _calculate_node(parsed)
    except ToolExecutionError:
        raise
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, RecursionError) as exc:
        raise ToolExecutionError("Expression is not a valid arithmetic expression") from exc

    if not math.isfinite(result) or abs(result) > _MAX_RESULT_MAGNITUDE:
        raise ToolExecutionError("Result is outside the supported numeric range")
    return result


def calculator_tool() -> ToolDefinition:
    def execute(arguments: dict[str, Any], _: ToolContext) -> dict[str, Any]:
        expression = arguments["expression"]
        return {
            "ok": True,
            "expression": expression,
            "result": calculate(expression),
        }

    return ToolDefinition(
        name="calculator",
        description="Evaluate a basic arithmetic expression using numbers, parentheses, and + - * /.",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": _MAX_EXPRESSION_LENGTH,
                    "pattern": r"\S",
                    "description": "Arithmetic expression; no variables or function calls.",
                }
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
        handler=execute,
    )

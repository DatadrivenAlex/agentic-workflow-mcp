"""Arithmetic evaluator. Names, calls, and huge exponents are rejected."""

from __future__ import annotations

import ast
import math
import operator

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def evaluate_expression(expression: str) -> str:
    text = expression.strip()
    if not text:
        raise ValueError("expression is empty")
    if len(text) > 200:
        raise ValueError("expression is too long")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError("expression is not valid arithmetic") from exc
    value = _eval(tree.body)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("result is not finite")
        if value.is_integer():
            return str(int(value))
        return str(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    raise ValueError("result is not a number")


def _eval(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        right = _eval(node.right)
        if type(node.op) is ast.Pow and isinstance(right, (int, float)) and abs(right) > 8:
            raise ValueError("exponent is too large")
        left = _eval(node.left)
        try:
            return _BINARY[type(node.op)](left, right)
        except ZeroDivisionError as exc:
            raise ValueError("division by zero") from exc
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    raise ValueError("only arithmetic expressions are allowed")

# Copyright (c) 2026 Nick Persing
# Licensed under the MIT License. See LICENSE for details.

"""AST statement and expression walker for the WGSL transpiler."""

from __future__ import annotations

import ast

from ..errors import ISSUES_URL, TranspileError
from .type_checker import _NP_TYPE_MAP


_BINOP_MAP: dict[type, str] = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.FloorDiv: "/",  # integer division; float floor-div unsupported
    ast.Mod: "%",
    ast.BitAnd: "&",
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.LShift: "<<",
    ast.RShift: ">>",
}

_CMPOP_MAP: dict[type, str] = {
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
}

_BUILTIN_FN_MAP: dict[str, str] = {
    "abs": "abs",
    "min": "min",
    "max": "max",
    "sqrt": "sqrt",
    "floor": "floor",
    "ceil": "ceil",
    "round": "round",
    "clamp": "clamp",
    "select": "select",
}


def _transpile_constant(expr: ast.Constant) -> str:
    val = expr.value
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        s = repr(val)
        if "." not in s and "e" not in s:
            s += ".0"
        return s
    raise TranspileError(
        f"Unsupported literal type {type(val).__name__!r}: {val!r}.\n"
        f"Supported literal types: int, float, bool.\n"
        f"Open an issue: {ISSUES_URL}"
    )


def _unsupported_stmt(stmt_name: str, fn_name: str) -> None:
    _UNSUPPORTED_HINTS: dict[str, str] = {
        "Try": "try/except is not supported — GPU kernels cannot raise exceptions.",
        "With": "with statements are not supported in @gpu_kernel.",
        "Global": "global is not supported in @gpu_kernel.",
        "Nonlocal": "nonlocal is not supported in @gpu_kernel.",
        "ClassDef": "class definitions inside @gpu_kernel are not supported.",
        "FunctionDef": "nested functions inside @gpu_kernel are not supported.",
        "AsyncFunctionDef": "async functions inside @gpu_kernel are not supported.",
        "AsyncFor": "async for is not supported in @gpu_kernel.",
        "AsyncWith": "async with is not supported in @gpu_kernel.",
        "Match": "match statements are not supported in @gpu_kernel.",
        "Delete": "del is not supported in @gpu_kernel.",
        "Import": "import inside @gpu_kernel is not supported.",
        "ImportFrom": "import inside @gpu_kernel is not supported.",
        "Raise": "raise is not supported — GPU kernels cannot raise exceptions.",
        "Assert": "assert is not supported in @gpu_kernel.",
        "While": (
            "while loops are not supported. Use for i in range(n) instead.\n"
            "Note: while loops have no WGSL equivalent; unbounded loops are GPU-unsafe."
        ),
    }
    hint = _UNSUPPORTED_HINTS.get(stmt_name, f"{stmt_name!r} statement is not supported.")
    raise TranspileError(
        f"In {fn_name!r}: {hint}\n"
        f"Open an issue: {ISSUES_URL}"
    )


class ASTWalker:
    """Mixin: transpiles Python AST statements and expressions to WGSL."""

    def _transpile_stmt(self, stmt: ast.stmt, indent: int) -> list[str]:
        if isinstance(stmt, ast.AnnAssign):
            return self._transpile_ann_assign(stmt, indent)
        if isinstance(stmt, ast.Assign):
            return self._transpile_assign(stmt, indent)
        if isinstance(stmt, ast.AugAssign):
            return self._transpile_aug_assign(stmt, indent)
        if isinstance(stmt, ast.Return):
            return self._transpile_return(stmt, indent)
        if isinstance(stmt, ast.If):
            return self._transpile_if(stmt, indent)
        if isinstance(stmt, ast.For):
            return self._transpile_for(stmt, indent)
        if isinstance(stmt, ast.Expr):
            prefix = "    " * indent
            return [f"{prefix}{self._transpile_expr(stmt.value)};"]
        if isinstance(stmt, ast.Pass):
            return []

        _unsupported_stmt(type(stmt).__name__, self._kfn.__name__)

    def _transpile_ann_assign(self, stmt: ast.AnnAssign, indent: int) -> list[str]:
        prefix = "    " * indent
        if not isinstance(stmt.target, ast.Name):
            raise TranspileError(
                f"Annotated assignment target must be a simple variable name.\n"
                f"Open an issue: {ISSUES_URL}"
            )
        name = stmt.target.id
        wgsl_type = self._resolve_annotation(stmt.annotation)
        self._type_env[name] = wgsl_type

        if stmt.value is None:
            return [f"{prefix}var {name}: {wgsl_type};"]
        value_str = self._transpile_expr(stmt.value)
        return [f"{prefix}var {name}: {wgsl_type} = {value_str};"]

    def _transpile_assign(self, stmt: ast.Assign, indent: int) -> list[str]:
        prefix = "    " * indent
        if len(stmt.targets) != 1:
            raise TranspileError(
                f"Multiple-target assignment (a = b = expr) is not supported.\n"
                f"Open an issue: {ISSUES_URL}"
            )
        target_str = self._transpile_expr(stmt.targets[0])
        value_str = self._transpile_expr(stmt.value)
        return [f"{prefix}{target_str} = {value_str};"]

    def _transpile_aug_assign(self, stmt: ast.AugAssign, indent: int) -> list[str]:
        prefix = "    " * indent
        op_str = _BINOP_MAP.get(type(stmt.op))
        if op_str is None:
            raise TranspileError(
                f"Unsupported augmented assignment operator {type(stmt.op).__name__!r}.\n"
                f"Open an issue: {ISSUES_URL}"
            )
        target_str = self._transpile_expr(stmt.target)
        value_str = self._transpile_expr(stmt.value)
        return [f"{prefix}{target_str} {op_str}= {value_str};"]

    def _transpile_return(self, stmt: ast.Return, indent: int) -> list[str]:
        prefix = "    " * indent
        if stmt.value is not None:
            raise TranspileError(
                f"@gpu_kernel functions cannot return a value.\n"
                f"Write results into an Array parameter instead.\n"
                f"Open an issue: {ISSUES_URL}"
            )
        return [f"{prefix}return;"]

    def _transpile_if(self, stmt: ast.If, indent: int) -> list[str]:
        prefix = "    " * indent
        cond = self._transpile_expr(stmt.test)
        lines = [f"{prefix}if ({cond}) {{"]
        for s in stmt.body:
            lines.extend(self._transpile_stmt(s, indent + 1))

        if stmt.orelse:
            if len(stmt.orelse) == 1 and isinstance(stmt.orelse[0], ast.If):
                else_lines = self._transpile_if(stmt.orelse[0], indent)
                lines.append(f"{prefix}}} else {else_lines[0].lstrip()}")
                lines.extend(else_lines[1:])
            else:
                lines.append(f"{prefix}}} else {{")
                for s in stmt.orelse:
                    lines.extend(self._transpile_stmt(s, indent + 1))
                lines.append(f"{prefix}}}")
        else:
            lines.append(f"{prefix}}}")
        return lines

    def _transpile_for(self, stmt: ast.For, indent: int) -> list[str]:
        prefix = "    " * indent
        if not isinstance(stmt.target, ast.Name):
            raise TranspileError(
                f"For loop target must be a simple variable name.\n"
                f"Open an issue: {ISSUES_URL}"
            )
        if stmt.orelse:
            raise TranspileError(
                f"For/else loops are not supported in @gpu_kernel functions.\n"
                f"Open an issue: {ISSUES_URL}"
            )

        loop_var = stmt.target.id

        if not (
            isinstance(stmt.iter, ast.Call)
            and isinstance(stmt.iter.func, ast.Name)
            and stmt.iter.func.id == "range"
        ):
            raise TranspileError(
                f"For loops must iterate over range(). Got {ast.unparse(stmt.iter)!r}.\n"
                f"Open an issue: {ISSUES_URL}"
            )

        args = stmt.iter.args
        if len(args) == 1:
            start, stop, step = "0", self._transpile_expr(args[0]), None
        elif len(args) == 2:
            start, stop, step = self._transpile_expr(args[0]), self._transpile_expr(args[1]), None
        elif len(args) == 3:
            start = self._transpile_expr(args[0])
            stop = self._transpile_expr(args[1])
            step = self._transpile_expr(args[2])
        else:
            raise TranspileError(
                f"range() takes 1–3 arguments, got {len(args)}.\n"
                f"Open an issue: {ISSUES_URL}"
            )

        self._type_env[loop_var] = "i32"
        continuing = f"{loop_var}++" if step is None else f"{loop_var} += {step}"
        header = f"for (var {loop_var}: i32 = {start}; {loop_var} < {stop}; {continuing})"

        lines = [f"{prefix}{header} {{"]
        for s in stmt.body:
            lines.extend(self._transpile_stmt(s, indent + 1))
        lines.append(f"{prefix}}}")

        self._type_env.pop(loop_var, None)
        return lines

    def _transpile_expr(self, expr: ast.expr) -> str:  # noqa: PLR0911
        if isinstance(expr, ast.Constant):
            return _transpile_constant(expr)
        if isinstance(expr, ast.Name):
            return expr.id
        if isinstance(expr, ast.BinOp):
            return self._transpile_binop(expr)
        if isinstance(expr, ast.UnaryOp):
            return self._transpile_unaryop(expr)
        if isinstance(expr, ast.Compare):
            return self._transpile_compare(expr)
        if isinstance(expr, ast.BoolOp):
            return self._transpile_boolop(expr)
        if isinstance(expr, ast.Subscript):
            val = self._transpile_expr(expr.value)
            idx = self._transpile_expr(expr.slice)
            return f"{val}[{idx}]"
        if isinstance(expr, ast.Attribute):
            obj = self._transpile_expr(expr.value)
            return f"{obj}.{expr.attr}"
        if isinstance(expr, ast.Call):
            return self._transpile_call(expr)

        raise TranspileError(
            f"Unsupported expression {type(expr).__name__!r} in {self._kfn.__name__!r}.\n"
            f"Supported: literals, names, arithmetic, comparisons, "
            f"array indexing, attribute access, function calls.\n"
            f"Open an issue: {ISSUES_URL}"
        )

    def _transpile_binop(self, expr: ast.BinOp) -> str:
        op = _BINOP_MAP.get(type(expr.op))
        if op is None:
            raise TranspileError(
                f"Unsupported binary operator {type(expr.op).__name__!r}.\n"
                f"Open an issue: {ISSUES_URL}"
            )
        left = self._transpile_expr(expr.left)
        right = self._transpile_expr(expr.right)
        return f"({left} {op} {right})"

    def _transpile_unaryop(self, expr: ast.UnaryOp) -> str:
        operand = self._transpile_expr(expr.operand)
        if isinstance(expr.op, ast.Not):
            return f"!({operand})"
        if isinstance(expr.op, ast.USub):
            return f"-{operand}"
        if isinstance(expr.op, ast.Invert):
            return f"~({operand})"
        if isinstance(expr.op, ast.UAdd):
            return operand
        raise TranspileError(
            f"Unsupported unary operator {type(expr.op).__name__!r}.\n"
            f"Open an issue: {ISSUES_URL}"
        )

    def _transpile_compare(self, expr: ast.Compare) -> str:
        if len(expr.ops) != 1:
            raise TranspileError(
                f"Chained comparisons (e.g. a < b < c) are not supported.\n"
                f"Split into separate comparisons joined with 'and'.\n"
                f"Open an issue: {ISSUES_URL}"
            )
        op = _CMPOP_MAP.get(type(expr.ops[0]))
        if op is None:
            raise TranspileError(
                f"Unsupported comparison operator {type(expr.ops[0]).__name__!r}.\n"
                f"Open an issue: {ISSUES_URL}"
            )
        left = self._transpile_expr(expr.left)
        right = self._transpile_expr(expr.comparators[0])
        return f"({left} {op} {right})"

    def _transpile_boolop(self, expr: ast.BoolOp) -> str:
        op = "&&" if isinstance(expr.op, ast.And) else "||"
        parts = [self._transpile_expr(v) for v in expr.values]
        return f"({f' {op} '.join(parts)})"

    def _transpile_call(self, expr: ast.Call) -> str:
        if expr.keywords:
            raise TranspileError(
                f"Keyword arguments in function calls are not supported in @gpu_kernel.\n"
                f"Open an issue: {ISSUES_URL}"
            )
        args_str = ", ".join(self._transpile_expr(a) for a in expr.args)

        if (
            isinstance(expr.func, ast.Attribute)
            and isinstance(expr.func.value, ast.Name)
            and expr.func.value.id == "np"
        ):
            cast = _NP_TYPE_MAP.get(expr.func.attr)
            if cast:
                return f"{cast}({args_str})"
            raise TranspileError(
                f"Unsupported numpy call np.{expr.func.attr} in @gpu_kernel.\n"
                f"Supported numpy casts: {', '.join(f'np.{k}' for k in _NP_TYPE_MAP)}.\n"
                f"Open an issue: {ISSUES_URL}"
            )

        if isinstance(expr.func, ast.Name):
            fn_name = expr.func.id
            if fn_name == "range":
                raise TranspileError(
                    f"range() is only valid as the iterator in a for loop.\n"
                    f"Open an issue: {ISSUES_URL}"
                )
            wgsl_fn = _BUILTIN_FN_MAP.get(fn_name, fn_name)
            return f"{wgsl_fn}({args_str})"

        if isinstance(expr.func, ast.Attribute):
            obj_str = self._transpile_expr(expr.func.value)
            return f"{obj_str}.{expr.func.attr}({args_str})"

        raise TranspileError(
            f"Unsupported call form in {self._kfn.__name__!r}.\n"
            f"Open an issue: {ISSUES_URL}"
        )

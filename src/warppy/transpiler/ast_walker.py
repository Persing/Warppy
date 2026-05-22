# Copyright (c) 2026 Nick Persing
# Licensed under the MIT License. See LICENSE for details.

"""AST statement and expression walker for the WGSL transpiler.

Uses the visitor pattern (ast.NodeVisitor) so that V2.4+ call-graph walking
slots in without restructuring: new node handlers are added as visit_X methods
rather than by extending an isinstance chain.
"""

from __future__ import annotations

import ast

from ..errors import ISSUES_URL, TranspileError
from .type_checker import _NP_TYPE_MAP, _SCALAR_TYPES


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


class ASTWalker(ast.NodeVisitor):
    """Mixin: transpiles Python AST statements and expressions to WGSL.

    Implements the visitor pattern so new node handlers (e.g. for call-graph
    analysis in V2.4) are added as ``visit_X`` methods rather than extending
    an isinstance chain.

    Statement visitors return ``list[str]`` (WGSL lines).
    Expression visitors return ``str`` (WGSL expression).
    """

    # Current indentation level — set by _transpile_stmt and managed by
    # block-entering visitors (visit_If, visit_For).
    _indent: int = 0

    # ------------------------------------------------------------------
    # Public wrappers — preserve codegen.py's calling convention
    # ------------------------------------------------------------------

    def _transpile_stmt(self, stmt: ast.stmt, indent: int) -> list[str]:
        self._indent = indent
        return self.visit(stmt)

    def _transpile_expr(self, expr: ast.expr) -> str:
        return self.visit(expr)

    # ------------------------------------------------------------------
    # Fallback — unsupported node types
    # ------------------------------------------------------------------

    def generic_visit(self, node: ast.AST):  # type: ignore[override]
        if isinstance(node, ast.expr):
            raise TranspileError(
                f"Unsupported expression {type(node).__name__!r} in {self._kfn.__name__!r}.\n"
                f"Supported: literals, names, arithmetic, comparisons, "
                f"array indexing, attribute access, function calls.\n"
                f"Open an issue: {ISSUES_URL}"
            )
        _unsupported_stmt(type(node).__name__, self._kfn.__name__)

    # ------------------------------------------------------------------
    # Statement visitors (return list[str])
    # ------------------------------------------------------------------

    def visit_AnnAssign(self, node: ast.AnnAssign) -> list[str]:
        prefix = "    " * self._indent
        if not isinstance(node.target, ast.Name):
            raise TranspileError(
                f"Annotated assignment target must be a simple variable name.\n"
                f"Open an issue: {ISSUES_URL}"
            )
        name = node.target.id
        wgsl_type = self._resolve_annotation(node.annotation)

        # Type check: declared type must match RHS type when both are known scalars.
        if node.value is not None:
            rhs_type = self._infer_type(node.value)
            if (
                rhs_type is not None
                and rhs_type in _SCALAR_TYPES
                and wgsl_type in _SCALAR_TYPES
                and rhs_type != wgsl_type
            ):
                raise TranspileError(
                    f"Type mismatch in {self._kfn.__name__!r}: "
                    f"{name!r} is declared as {wgsl_type!r} but the value has type {rhs_type!r}.\n"
                    f"Cast the value explicitly: np.<type>(value).\n"
                    f"Open an issue: {ISSUES_URL}"
                )

        self._type_env[name] = wgsl_type

        if node.value is None:
            return [f"{prefix}var {name}: {wgsl_type};"]
        value_str = self.visit(node.value)
        return [f"{prefix}var {name}: {wgsl_type} = {value_str};"]

    def visit_Assign(self, node: ast.Assign) -> list[str]:
        prefix = "    " * self._indent
        if len(node.targets) != 1:
            raise TranspileError(
                f"Multiple-target assignment (a = b = expr) is not supported.\n"
                f"Open an issue: {ISSUES_URL}"
            )
        target_str = self.visit(node.targets[0])
        value_str = self.visit(node.value)
        return [f"{prefix}{target_str} = {value_str};"]

    def visit_AugAssign(self, node: ast.AugAssign) -> list[str]:
        prefix = "    " * self._indent
        op_str = _BINOP_MAP.get(type(node.op))
        if op_str is None:
            raise TranspileError(
                f"Unsupported augmented assignment operator {type(node.op).__name__!r}.\n"
                f"Open an issue: {ISSUES_URL}"
            )
        target_str = self.visit(node.target)
        value_str = self.visit(node.value)
        return [f"{prefix}{target_str} {op_str}= {value_str};"]

    def visit_Return(self, node: ast.Return) -> list[str]:
        prefix = "    " * self._indent
        if node.value is not None:
            raise TranspileError(
                f"@gpu_kernel functions cannot return a value.\n"
                f"Write results into an Array parameter instead.\n"
                f"Open an issue: {ISSUES_URL}"
            )
        return [f"{prefix}return;"]

    def visit_If(self, node: ast.If) -> list[str]:
        prefix = "    " * self._indent
        cond = self.visit(node.test)
        lines = [f"{prefix}if ({cond}) {{"]

        saved = self._indent
        self._indent += 1
        for s in node.body:
            lines.extend(self.visit(s))
        self._indent = saved

        if node.orelse:
            if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                # elif branch — visit at the same indent level, then stitch
                else_lines = self.visit(node.orelse[0])
                lines.append(f"{prefix}}} else {else_lines[0].lstrip()}")
                lines.extend(else_lines[1:])
            else:
                lines.append(f"{prefix}}} else {{")
                self._indent += 1
                for s in node.orelse:
                    lines.extend(self.visit(s))
                self._indent = saved
                lines.append(f"{prefix}}}")
        else:
            lines.append(f"{prefix}}}")
        return lines

    def visit_For(self, node: ast.For) -> list[str]:
        prefix = "    " * self._indent
        if not isinstance(node.target, ast.Name):
            raise TranspileError(
                f"For loop target must be a simple variable name.\n"
                f"Open an issue: {ISSUES_URL}"
            )
        if node.orelse:
            raise TranspileError(
                f"For/else loops are not supported in @gpu_kernel functions.\n"
                f"Open an issue: {ISSUES_URL}"
            )

        loop_var = node.target.id

        if not (
            isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "range"
        ):
            raise TranspileError(
                f"For loops must iterate over range(). Got {ast.unparse(node.iter)!r}.\n"
                f"Open an issue: {ISSUES_URL}"
            )

        args = node.iter.args
        if len(args) == 1:
            start, stop, step = "0", self.visit(args[0]), None
        elif len(args) == 2:
            start, stop, step = self.visit(args[0]), self.visit(args[1]), None
        elif len(args) == 3:
            start = self.visit(args[0])
            stop = self.visit(args[1])
            step = self.visit(args[2])
        else:
            raise TranspileError(
                f"range() takes 1–3 arguments, got {len(args)}.\n"
                f"Open an issue: {ISSUES_URL}"
            )

        self._type_env[loop_var] = "i32"
        continuing = f"{loop_var}++" if step is None else f"{loop_var} += {step}"
        header = f"for (var {loop_var}: i32 = {start}; {loop_var} < {stop}; {continuing})"

        saved = self._indent
        self._indent += 1
        lines = [f"{prefix}{header} {{"]
        for s in node.body:
            lines.extend(self.visit(s))
        self._indent = saved
        lines.append(f"{prefix}}}")

        self._type_env.pop(loop_var, None)
        return lines

    def visit_Expr(self, node: ast.Expr) -> list[str]:
        prefix = "    " * self._indent
        return [f"{prefix}{self.visit(node.value)};"]

    def visit_Pass(self, node: ast.Pass) -> list[str]:
        return []

    # ------------------------------------------------------------------
    # Expression visitors (return str)
    # ------------------------------------------------------------------

    def visit_Constant(self, node: ast.Constant) -> str:
        val = node.value
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

    def visit_Name(self, node: ast.Name) -> str:
        return node.id

    def visit_BinOp(self, node: ast.BinOp) -> str:
        op = _BINOP_MAP.get(type(node.op))
        if op is None:
            raise TranspileError(
                f"Unsupported binary operator {type(node.op).__name__!r}.\n"
                f"Open an issue: {ISSUES_URL}"
            )

        lt = self._infer_type(node.left)
        rt = self._infer_type(node.right)
        if (
            lt is not None
            and rt is not None
            and lt in _SCALAR_TYPES
            and rt in _SCALAR_TYPES
            and lt != rt
        ):
            raise TranspileError(
                f"Type mismatch in {self._kfn.__name__!r}: "
                f"'{op}' applied to {lt!r} (left) and {rt!r} (right).\n"
                f"Cast one operand explicitly: np.<type>(value).\n"
                f"Open an issue: {ISSUES_URL}"
            )

        left = self.visit(node.left)
        right = self.visit(node.right)
        return f"({left} {op} {right})"

    def visit_UnaryOp(self, node: ast.UnaryOp) -> str:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.Not):
            return f"!({operand})"
        if isinstance(node.op, ast.USub):
            return f"-{operand}"
        if isinstance(node.op, ast.Invert):
            return f"~({operand})"
        if isinstance(node.op, ast.UAdd):
            return operand
        raise TranspileError(
            f"Unsupported unary operator {type(node.op).__name__!r}.\n"
            f"Open an issue: {ISSUES_URL}"
        )

    def visit_Compare(self, node: ast.Compare) -> str:
        if len(node.ops) != 1:
            raise TranspileError(
                f"Chained comparisons (e.g. a < b < c) are not supported.\n"
                f"Split into separate comparisons joined with 'and'.\n"
                f"Open an issue: {ISSUES_URL}"
            )
        op = _CMPOP_MAP.get(type(node.ops[0]))
        if op is None:
            raise TranspileError(
                f"Unsupported comparison operator {type(node.ops[0]).__name__!r}.\n"
                f"Open an issue: {ISSUES_URL}"
            )

        lt = self._infer_type(node.left)
        rt = self._infer_type(node.comparators[0])
        if (
            lt is not None
            and rt is not None
            and lt in _SCALAR_TYPES
            and rt in _SCALAR_TYPES
            and lt != rt
        ):
            raise TranspileError(
                f"Type mismatch in {self._kfn.__name__!r}: "
                f"'{op}' compares {lt!r} (left) with {rt!r} (right).\n"
                f"Cast one operand explicitly: np.<type>(value).\n"
                f"Open an issue: {ISSUES_URL}"
            )

        left = self.visit(node.left)
        right = self.visit(node.comparators[0])
        return f"({left} {op} {right})"

    def visit_BoolOp(self, node: ast.BoolOp) -> str:
        op = "&&" if isinstance(node.op, ast.And) else "||"
        parts = [self.visit(v) for v in node.values]
        return f"({f' {op} '.join(parts)})"

    def visit_Subscript(self, node: ast.Subscript) -> str:
        val = self.visit(node.value)
        idx = self.visit(node.slice)
        return f"{val}[{idx}]"

    def visit_Attribute(self, node: ast.Attribute) -> str:
        obj = self.visit(node.value)
        return f"{obj}.{node.attr}"

    def visit_Call(self, node: ast.Call) -> str:
        if node.keywords:
            raise TranspileError(
                f"Keyword arguments in function calls are not supported in @gpu_kernel.\n"
                f"Open an issue: {ISSUES_URL}"
            )
        args_str = ", ".join(self.visit(a) for a in node.args)

        # np.uint32(x), np.float32(x), etc.
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "np"
        ):
            cast = _NP_TYPE_MAP.get(node.func.attr)
            if cast:
                return f"{cast}({args_str})"
            raise TranspileError(
                f"Unsupported numpy call np.{node.func.attr} in @gpu_kernel.\n"
                f"Supported numpy casts: {', '.join(f'np.{k}' for k in _NP_TYPE_MAP)}.\n"
                f"Open an issue: {ISSUES_URL}"
            )

        # Simple name call
        if isinstance(node.func, ast.Name):
            fn_name = node.func.id
            if fn_name == "range":
                raise TranspileError(
                    f"range() is only valid as the iterator in a for loop.\n"
                    f"Open an issue: {ISSUES_URL}"
                )
            wgsl_fn = _BUILTIN_FN_MAP.get(fn_name, fn_name)
            return f"{wgsl_fn}({args_str})"

        # Attribute call: obj.method(...)
        if isinstance(node.func, ast.Attribute):
            obj_str = self.visit(node.func.value)
            return f"{obj_str}.{node.func.attr}({args_str})"

        raise TranspileError(
            f"Unsupported call form in {self._kfn.__name__!r}.\n"
            f"Open an issue: {ISSUES_URL}"
        )

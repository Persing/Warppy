# Copyright (c) 2026 Nick Persing
# Licensed under the MIT License. See LICENSE for details.

"""AST → WGSL transpiler for @gpu_kernel functions.

Converts the body of an annotated Python function into a complete WGSL compute
shader string. Supports a strict subset of Python — see TranspileError messages
for what is and is not supported.
"""

from __future__ import annotations

import ast
import inspect
from typing import TYPE_CHECKING, get_type_hints

import numpy as np

from .bindings import BindingKind
from .errors import ISSUES_URL, TranspileError
from .types import dataclass_to_wgsl_struct, dtype_to_wgsl

if TYPE_CHECKING:
    from .kernel import KernelFn


# ---------------------------------------------------------------------------
# Operator / type maps
# ---------------------------------------------------------------------------

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

# numpy attribute name → WGSL scalar type (for np.uint32(x) casts and annotations)
_NP_TYPE_MAP: dict[str, str] = {
    "uint32": "u32",
    "int32": "i32",
    "float32": "f32",
    "float16": "f16",
    "uint16": "u16",
    "int16": "i16",
    "bool_": "bool",
}

# WGSL scalar type → WGSL scalar type (passthrough for users who annotate with WGSL names)
_WGSL_SCALAR_NAMES: frozenset[str] = frozenset({"u32", "i32", "f32", "f16", "u16", "i16", "bool"})

# Python built-in / math names that map directly to WGSL built-ins
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

# numpy scalar types accepted as the thread-index parameter annotation
_IDX_NUMPY_TYPES: tuple[type, ...] = (np.uint32, np.int32)
_IDX_WGSL: dict[type, str] = {np.uint32: "u32", np.int32: "i32"}


# ---------------------------------------------------------------------------
# Transpiler
# ---------------------------------------------------------------------------

class WGSLTranspiler:
    """Converts a :class:`~warppy.KernelFn` to a complete WGSL shader string."""

    def __init__(self, kernel_fn: KernelFn, workgroup_size: int) -> None:
        self._kfn = kernel_fn
        self._workgroup_size = workgroup_size
        self._fn_node: ast.FunctionDef = self._extract_fn_node()
        self._type_env: dict[str, str] = {}
        self._build_type_env()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def transpile(self) -> str:
        """Return the complete WGSL shader source."""
        parts = [
            self._emit_structs(),
            self._emit_bindings(),
            self._emit_entry_point(),
        ]
        return "\n\n".join(p for p in parts if p)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _extract_fn_node(self) -> ast.FunctionDef:
        if self._kfn.ast_tree is None:
            raise TranspileError(
                f"Cannot transpile {self._kfn.__name__!r}: source code is unavailable.\n"
                f"Define @gpu_kernel functions in a source file, not a REPL.\n"
                f"Open an issue: {ISSUES_URL}"
            )
        for node in self._kfn.ast_tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == self._kfn.fn.__name__:
                return node
        raise TranspileError(
            f"Could not find function {self._kfn.__name__!r} in parsed AST.\n"
            f"Open an issue: {ISSUES_URL}"
        )

    def _build_type_env(self) -> None:
        """Populate type_env with parameter WGSL types."""
        hints = get_type_hints(self._kfn.fn)
        sig = inspect.signature(self._kfn.fn)
        param_names = list(sig.parameters.keys())

        # First param is always the thread index
        idx_ann = hints.get(param_names[0])
        self._type_env[param_names[0]] = _IDX_WGSL.get(idx_ann, "u32")

        # Remaining params come from inferred bindings
        for i, name in enumerate(param_names[1:]):
            spec = self._kfn.bindings[i]
            if spec.kind == BindingKind.UNIFORM:
                self._type_env[name] = spec.payload.__name__
            else:
                self._type_env[name] = f"array<{dtype_to_wgsl(spec.payload)}>"

    # ------------------------------------------------------------------
    # Emission helpers
    # ------------------------------------------------------------------

    def _emit_structs(self) -> str:
        parts = []
        for spec in self._kfn.bindings:
            if spec.kind == BindingKind.UNIFORM:
                parts.append(dataclass_to_wgsl_struct(spec.payload))
        return "\n\n".join(parts)

    def _emit_bindings(self) -> str:
        sig = inspect.signature(self._kfn.fn)
        param_names = list(sig.parameters.keys())[1:]  # skip idx
        lines = []
        for param_name, spec in zip(param_names, self._kfn.bindings):
            prefix = f"@group({spec.group}) @binding({spec.binding})"
            if spec.kind == BindingKind.UNIFORM:
                lines.append(f"{prefix} var<uniform> {param_name}: {spec.payload.__name__};")
            else:
                wgsl_type = dtype_to_wgsl(spec.payload)
                lines.append(f"{prefix} var<storage, read_write> {param_name}: array<{wgsl_type}>;")
        return "\n".join(lines)

    def _emit_entry_point(self) -> str:
        idx_name = self._kfn.idx_param
        idx_type = self._type_env[idx_name]

        body_lines: list[str] = [f"    let {idx_name}: {idx_type} = gid.x;"]
        for stmt in self._fn_node.body:
            body_lines.extend(self._transpile_stmt(stmt, indent=1))

        body = "\n".join(body_lines)
        return (
            f"@compute @workgroup_size({self._workgroup_size})\n"
            f"fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{\n"
            f"{body}\n"
            f"}}"
        )

    # ------------------------------------------------------------------
    # Statement transpilation
    # ------------------------------------------------------------------

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
            # elif: single If node in orelse
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

    # ------------------------------------------------------------------
    # Expression transpilation
    # ------------------------------------------------------------------

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

        # np.uint32(x), np.float32(x), etc.
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

        # Simple name call
        if isinstance(expr.func, ast.Name):
            fn_name = expr.func.id
            if fn_name == "range":
                raise TranspileError(
                    f"range() is only valid as the iterator in a for loop.\n"
                    f"Open an issue: {ISSUES_URL}"
                )
            wgsl_fn = _BUILTIN_FN_MAP.get(fn_name, fn_name)
            return f"{wgsl_fn}({args_str})"

        # Attribute call: obj.method(...)
        if isinstance(expr.func, ast.Attribute):
            obj_str = self._transpile_expr(expr.func.value)
            return f"{obj_str}.{expr.func.attr}({args_str})"

        raise TranspileError(
            f"Unsupported call form in {self._kfn.__name__!r}.\n"
            f"Open an issue: {ISSUES_URL}"
        )

    # ------------------------------------------------------------------
    # Annotation resolution
    # ------------------------------------------------------------------

    def _resolve_annotation(self, ann: ast.expr) -> str:
        """Convert a local-variable annotation AST node to a WGSL type string."""
        # np.uint32, np.float32, etc.
        if isinstance(ann, ast.Attribute) and isinstance(ann.value, ast.Name):
            if ann.value.id == "np":
                wgsl = _NP_TYPE_MAP.get(ann.attr)
                if wgsl:
                    return wgsl
                raise TranspileError(
                    f"Unsupported numpy type annotation np.{ann.attr!r} in {self._kfn.__name__!r}.\n"
                    f"Supported types: {', '.join(f'np.{k}' for k in _NP_TYPE_MAP)}.\n"
                    f"Open an issue: {ISSUES_URL}"
                )

        # Bare WGSL scalar name: u32, i32, f32, ...
        if isinstance(ann, ast.Name):
            if ann.id in _WGSL_SCALAR_NAMES:
                return ann.id
            raise TranspileError(
                f"Unknown type annotation {ann.id!r} in {self._kfn.__name__!r}.\n"
                f"Use numpy scalar types (np.uint32, np.float32, ...) "
                f"or bare WGSL names (u32, f32, ...).\n"
                f"Open an issue: {ISSUES_URL}"
            )

        raise TranspileError(
            f"Cannot resolve type annotation {ast.unparse(ann)!r} to a WGSL type "
            f"in {self._kfn.__name__!r}.\n"
            f"Use numpy scalar types (e.g. np.uint32) for local variable annotations.\n"
            f"Open an issue: {ISSUES_URL}"
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

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

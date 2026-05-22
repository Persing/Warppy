# Copyright (c) 2026 Nick Persing
# Licensed under the MIT License. See LICENSE for details.

"""Type resolution and inference helpers for the WGSL transpiler."""

from __future__ import annotations

import ast
import inspect
from typing import TYPE_CHECKING, get_type_hints

import numpy as np

from ..bindings import BindingKind
from ..errors import ISSUES_URL, TranspileError
from ..types import dtype_to_wgsl

if TYPE_CHECKING:
    from ..gpu_kernel import KernelFn


_NP_TYPE_MAP: dict[str, str] = {
    "uint32": "u32",
    "int32": "i32",
    "float32": "f32",
    "float16": "f16",
    "uint16": "u16",
    "int16": "i16",
    "bool_": "bool",
}

_WGSL_SCALAR_NAMES: frozenset[str] = frozenset({"u32", "i32", "f32", "f16", "u16", "i16", "bool"})

# Alias used by type-checking call sites for clarity
_SCALAR_TYPES: frozenset[str] = _WGSL_SCALAR_NAMES

_IDX_NUMPY_TYPES: tuple[type, ...] = (np.uint32, np.int32)
_IDX_WGSL: dict[type, str] = {np.uint32: "u32", np.int32: "i32"}


# ---------------------------------------------------------------------------
# Expression type inferer (visitor pattern)
# ---------------------------------------------------------------------------

class ExprTypeInferer(ast.NodeVisitor):
    """Infers the WGSL scalar type of an expression node, or ``None`` if abstract/unknown.

    Holds a live reference to the transpiler's ``_type_env`` dict so it always
    sees the current set of declared local variables.
    """

    def __init__(self, type_env: dict[str, str]) -> None:
        self._type_env = type_env

    def infer(self, expr: ast.expr) -> str | None:
        """Return the WGSL scalar type of *expr*, or ``None`` if indeterminate."""
        return self.visit(expr)

    # ------------------------------------------------------------------
    # Leaf nodes
    # ------------------------------------------------------------------

    def visit_Constant(self, node: ast.Constant) -> str | None:
        val = node.value
        if isinstance(val, bool):
            return "bool"
        # Integer and float literals are abstract — compatible with any
        # same-category concrete type (u32, i32, f32, …).
        return None

    def visit_Name(self, node: ast.Name) -> str | None:
        t = self._type_env.get(node.id)
        return t if t in _SCALAR_TYPES else None

    def visit_Subscript(self, node: ast.Subscript) -> str | None:
        """``array<u32>[i]`` → ``"u32"``."""
        if isinstance(node.value, ast.Name):
            t = self._type_env.get(node.value.id)
            if t and t.startswith("array<") and t.endswith(">"):
                return t[6:-1]
        return None

    def visit_Call(self, node: ast.Call) -> str | None:
        """``np.uint32(x)`` → ``"u32"``, etc."""
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "np"
        ):
            return _NP_TYPE_MAP.get(node.func.attr)
        return None

    # ------------------------------------------------------------------
    # Compound expressions
    # ------------------------------------------------------------------

    def visit_BinOp(self, node: ast.BinOp) -> str | None:
        lt = self.visit(node.left)
        rt = self.visit(node.right)
        if lt is not None and rt is not None:
            # Mismatched concrete types — ASTWalker.visit_BinOp will raise;
            # return None here so callers don't see a spurious type.
            return lt if lt == rt else None
        return lt or rt  # propagate whichever side is known

    def visit_UnaryOp(self, node: ast.UnaryOp) -> str | None:
        if isinstance(node.op, ast.Not):
            return "bool"
        return self.visit(node.operand)

    def visit_Compare(self, node: ast.Compare) -> str:
        return "bool"

    def visit_BoolOp(self, node: ast.BoolOp) -> str:
        return "bool"

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Struct field access — field types not tracked; treat as unknown.
        return None

    def generic_visit(self, node: ast.AST) -> None:  # type: ignore[override]
        return None


# ---------------------------------------------------------------------------
# TypeChecker mixin
# ---------------------------------------------------------------------------

class TypeChecker:
    """Mixin: builds and queries the WGSL type environment."""

    _kfn: KernelFn
    _type_env: dict[str, str]
    _type_inferer: ExprTypeInferer

    def _build_type_env(self) -> None:
        hints = get_type_hints(self._kfn.fn)
        sig = inspect.signature(self._kfn.fn)
        param_names = list(sig.parameters.keys())

        idx_ann = hints.get(param_names[0])
        self._type_env[param_names[0]] = _IDX_WGSL.get(idx_ann, "u32")

        for i, name in enumerate(param_names[1:]):
            spec = self._kfn.bindings[i]
            if spec.kind == BindingKind.UNIFORM:
                self._type_env[name] = spec.payload.__name__
            else:
                self._type_env[name] = f"array<{dtype_to_wgsl(spec.payload)}>"

        # Create the inferer once; it holds a live reference to _type_env
        # so mutations (local var declarations, loop vars) are visible automatically.
        self._type_inferer = ExprTypeInferer(self._type_env)

    def _infer_type(self, expr: ast.expr) -> str | None:
        """Return the WGSL scalar type of *expr*, or ``None`` if indeterminate."""
        return self._type_inferer.infer(expr)

    def _resolve_annotation(self, ann: ast.expr) -> str:
        """Convert a local-variable annotation AST node to a WGSL type string."""
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

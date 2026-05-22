# Copyright (c) 2026 Nick Persing
# Licensed under the MIT License. See LICENSE for details.

"""Type resolution helpers for the WGSL transpiler."""

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

_IDX_NUMPY_TYPES: tuple[type, ...] = (np.uint32, np.int32)
_IDX_WGSL: dict[type, str] = {np.uint32: "u32", np.int32: "i32"}


class TypeChecker:
    """Mixin: builds and queries the WGSL type environment."""

    _kfn: KernelFn
    _type_env: dict[str, str]

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

# Copyright (c) 2026 Nick Persing
# Licensed under the MIT License. See LICENSE for details.

"""AST → WGSL transpiler for @gpu_kernel functions.

Converts the body of an annotated Python function into a complete WGSL compute
shader string. Supports a strict subset of Python — see TranspileError messages
for what is and is not supported.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from ..errors import ISSUES_URL, TranspileError
from .ast_walker import ASTWalker
from .codegen import CodeGen
from .type_checker import TypeChecker

if TYPE_CHECKING:
    from ..gpu_kernel import KernelFn


class WGSLTranspiler(TypeChecker, ASTWalker, CodeGen):
    """Converts a :class:`~warppy.KernelFn` to a complete WGSL shader string."""

    def __init__(self, kernel_fn: KernelFn, workgroup_size: int) -> None:
        self._kfn = kernel_fn
        self._workgroup_size = workgroup_size
        self._fn_node: ast.FunctionDef = self._extract_fn_node()
        self._type_env: dict[str, str] = {}
        self._build_type_env()

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

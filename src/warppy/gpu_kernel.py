# Copyright (c) 2026 Nick Persing
# Licensed under the MIT License. See LICENSE for details.

"""@gpu_kernel decorator and supporting types for V2 annotated kernel functions."""

from __future__ import annotations

import ast
import dataclasses
import inspect
import textwrap
from collections.abc import Callable
from typing import Any, get_type_hints

import numpy as np

from .bindings import BindingKind, BindingSpec
from .errors import ISSUES_URL, TranspileError


@dataclasses.dataclass(frozen=True)
class ArraySpec:
    """Runtime representation of Array[dtype] or Array[dtype, N] annotations."""

    dtype: np.dtype
    size: int | None = None


class Array:
    """Type hint for GPU storage buffer parameters.

    Usage::

        Array[np.float32]        # runtime-sized storage buffer
        Array[np.float32, 52]    # fixed-size (used for local arrays in the kernel)

    The subscript returns an :class:`ArraySpec` that the :func:`gpu_kernel`
    decorator inspects to infer storage bindings automatically.
    """

    def __class_getitem__(cls, params: Any) -> ArraySpec:
        if isinstance(params, tuple):
            if len(params) != 2:
                raise TranspileError(
                    f"Array[] expects one or two arguments (dtype) or (dtype, size), "
                    f"got {len(params)}.\n"
                    f"Open an issue: {ISSUES_URL}"
                )
            dtype_arg, size = params
            if not isinstance(size, int) or size <= 0:
                raise TranspileError(
                    f"Array size must be a positive integer, got {size!r}.\n"
                    f"Open an issue: {ISSUES_URL}"
                )
        else:
            dtype_arg, size = params, None
        return ArraySpec(dtype=np.dtype(dtype_arg), size=size)


# numpy scalar types accepted as the thread-index (first) parameter
_VALID_IDX_TYPES: tuple[type, ...] = (np.uint32, np.int32)


@dataclasses.dataclass
class KernelFn:
    """A GPU kernel function produced by :func:`gpu_kernel`.

    Stores the original function, the binding specs inferred from its type
    annotations, and the parsed AST for later transpilation.

    Pass directly to :meth:`~warppy.ShaderBuilder.kernel` instead of a WGSL string.
    """

    fn: Callable[..., None]
    idx_param: str
    bindings: list[BindingSpec]
    ast_tree: ast.Module | None

    def to_wgsl(self, workgroup_size: int) -> str:
        """Transpile this kernel to a complete WGSL shader string.

        Args:
            workgroup_size: Threads per workgroup — must match the value passed
                to :meth:`~warppy.ShaderBuilder.workgroup_size`.

        Returns:
            A complete WGSL shader string ready for GPU compilation.

        Raises:
            TranspileError: if the function body contains unsupported Python patterns.
        """
        from .transpiler import WGSLTranspiler
        return WGSLTranspiler(self, workgroup_size).transpile()

    @property
    def __name__(self) -> str:
        return self.fn.__name__


def gpu_kernel(fn: Callable[..., None]) -> KernelFn:
    """Decorator that marks an annotated Python function as a GPU compute kernel.

    Validates annotations at decoration time and infers GPU bindings from parameter
    types. The first parameter is always the thread index; subsequent parameters
    become uniform (dataclass) or storage (Array) bindings automatically.

    All parameters must have type annotations. The return type must be ``None``.

    Args:
        fn: A Python function with full type annotations. Supported parameter types:

            - First param: ``np.uint32`` or ``np.int32`` (thread index)
            - ``@dataclass`` type → uniform buffer (read-only params struct)
            - ``Array[dtype]`` or ``Array[dtype, N]`` → storage buffer

    Returns:
        A :class:`KernelFn` wrapper storing the function, inferred bindings, and AST.

    Raises:
        TranspileError: if any annotation is missing, the return type is not ``None``,
            or a parameter type is not supported.

    Example::

        from dataclasses import dataclass
        import numpy as np
        from warppy import gpu_kernel, Array

        @dataclass
        class Params:
            num_trials: np.uint32

        @gpu_kernel
        def my_kernel(idx: np.uint32, params: Params, output: Array[np.float32]) -> None:
            ...
    """
    if not callable(fn):
        raise TranspileError(
            f"@gpu_kernel must decorate a callable, got {fn!r}.\n"
            f"Open an issue: {ISSUES_URL}"
        )

    try:
        hints = get_type_hints(fn)
    except Exception as exc:
        raise TranspileError(
            f"Could not resolve type hints for {fn.__name__!r}: {exc}.\n"
            f"Ensure all annotations reference names available at decoration time.\n"
            f"Open an issue: {ISSUES_URL}"
        ) from exc

    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())

    # All parameters must have annotations
    for name in params:
        if name not in hints:
            raise TranspileError(
                f"Parameter {name!r} in {fn.__name__!r} is missing a type annotation.\n"
                f"@gpu_kernel requires all parameters to be annotated.\n"
                f"Example: def my_kernel(idx: np.uint32, data: Array[np.float32]) -> None\n"
                f"Open an issue: {ISSUES_URL}"
            )

    # Return annotation must be None
    ret = hints.get("return")
    if ret is not type(None):
        raise TranspileError(
            f"@gpu_kernel function {fn.__name__!r} must return None, got {ret!r}.\n"
            f"GPU kernels do not return values; write results into an Array parameter.\n"
            f"Open an issue: {ISSUES_URL}"
        )

    if not params:
        raise TranspileError(
            f"@gpu_kernel function {fn.__name__!r} has no parameters.\n"
            f"The first parameter must be the thread index (np.uint32 or np.int32).\n"
            f"Open an issue: {ISSUES_URL}"
        )

    # First parameter must be a valid integer index type
    idx_param = params[0]
    idx_type = hints[idx_param]
    if idx_type not in _VALID_IDX_TYPES:
        raise TranspileError(
            f"First parameter {idx_param!r} must be np.uint32 or np.int32 (the thread index), "
            f"got {idx_type!r}.\n"
            f"The first parameter receives @builtin(global_invocation_id) from the GPU.\n"
            f"Open an issue: {ISSUES_URL}"
        )

    # Infer bindings from remaining parameters
    bindings = _infer_bindings(fn.__name__, params[1:], hints)

    # Parse AST for later transpilation (best-effort; None if source unavailable)
    ast_tree: ast.Module | None = None
    try:
        source = textwrap.dedent(inspect.getsource(fn))
        ast_tree = ast.parse(source)
    except (OSError, TypeError):
        pass

    return KernelFn(fn=fn, idx_param=idx_param, bindings=bindings, ast_tree=ast_tree)


def _infer_bindings(fn_name: str, param_names: list[str], hints: dict[str, Any]) -> list[BindingSpec]:
    """Infer BindingSpec list from non-index parameter annotations."""
    bindings: list[BindingSpec] = []
    for binding_idx, name in enumerate(param_names):
        ann = hints[name]
        if dataclasses.is_dataclass(ann) and isinstance(ann, type):
            bindings.append(
                BindingSpec(group=0, binding=binding_idx, kind=BindingKind.UNIFORM, payload=ann)
            )
        elif isinstance(ann, ArraySpec):
            bindings.append(
                BindingSpec(group=0, binding=binding_idx, kind=BindingKind.STORAGE, payload=ann.dtype)
            )
        else:
            raise TranspileError(
                f"Parameter {name!r} in {fn_name!r} has unsupported type {ann!r}.\n"
                f"@gpu_kernel parameters (after the thread index) must be:\n"
                f"  - a @dataclass type   → uniform buffer (read-only params)\n"
                f"  - Array[dtype]        → storage buffer (read/write)\n"
                f"  - Array[dtype, N]     → fixed-size storage buffer\n"
                f"Open an issue: {ISSUES_URL}"
            )
    return bindings

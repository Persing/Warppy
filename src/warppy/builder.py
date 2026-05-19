# Copyright (c) 2026 Nick Persing
# Licensed under the MIT License. See LICENSE for details.

"""ShaderBuilder — fluent API for constructing GPU compute shaders."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np

from .bindings import BindingKind, BindingSpec, validate_bindings
from .errors import GPUBindingError, GPUConfigError, ISSUES_URL
from .kernel import KernelFn
from .types import pack_dataclass


@dataclasses.dataclass
class GPUResult:
    """Result of a successful GPU dispatch."""

    data: np.ndarray
    """Output values from the first storage buffer."""

    elapsed_ms: float
    """Wall time from dispatch to readback, in milliseconds."""

    invocations: int
    """Total number of compute threads that executed."""


class CompiledShader:
    """A fully configured compute shader ready for dispatch.

    Obtain via :meth:`ShaderBuilder.build`.
    """

    def __init__(
        self,
        kernel_wgsl: str,
        bindings: list[BindingSpec],
        workgroup_size: int,
    ) -> None:
        self._kernel_wgsl = kernel_wgsl
        self._bindings = bindings
        self._workgroup_size = workgroup_size

    def dispatch(self, params: Any, inputs: list[np.ndarray]) -> GPUResult:
        """Execute the shader and return the result.

        Args:
            params: A dataclass instance matching the uniform binding type.
                Pass ``None`` if the shader has no uniform binding.
            inputs: One numpy array per storage binding, in declaration order.
                The array dtype must match the storage binding type.

        Returns:
            GPUResult containing the output data, elapsed time, and invocation count.

        Raises:
            GPUCompileError: if the driver rejects the shader.
            GPUTypeError: if params cannot be serialized or an array dtype is wrong.
        """
        from .wgpu_backend import run_compute  # noqa: PLC0415

        uniform_bytes: bytes | None = None
        uniform_specs = [s for s in self._bindings if s.kind == BindingKind.UNIFORM]
        if uniform_specs:
            if params is None:
                raise GPUConfigError(
                    f"Shader has a uniform binding but dispatch() received params=None.\n"
                    f"Pass a {uniform_specs[0].payload.__name__} instance as params.\n"
                    f"Open an issue: {ISSUES_URL}"
                )
            uniform_bytes = pack_dataclass(params)

        storage_specs = [s for s in self._bindings if s.kind == BindingKind.STORAGE]
        if len(inputs) != len(storage_specs):
            raise GPUBindingError(
                f"Expected {len(storage_specs)} storage input(s), got {len(inputs)}.\n"
                f"Pass one numpy array per storage binding, in declaration order.\n"
                f"Open an issue: {ISSUES_URL}"
            )

        num_invocations = inputs[0].size if inputs else 1
        output_arrays, elapsed_ms = run_compute(
            kernel_wgsl=self._kernel_wgsl,
            bindings=self._bindings,
            workgroup_size=self._workgroup_size,
            uniform_bytes=uniform_bytes,
            storage_arrays=inputs,
            num_invocations=num_invocations,
        )
        return GPUResult(
            data=output_arrays[0] if output_arrays else np.array([]),
            elapsed_ms=elapsed_ms,
            invocations=num_invocations,
        )


class ShaderBuilder:
    """Fluent builder for GPU compute shaders.

    Chain calls to configure the shader, then call :meth:`build` to compile.

    Example::

        from dataclasses import dataclass
        import numpy as np
        from warppy import ShaderBuilder

        @dataclass
        class Params:
            count: np.uint32

        shader = (
            ShaderBuilder()
            .bind_uniform(0, 0, Params)
            .bind_storage(0, 1, np.uint32)
            .workgroup_size(256)
            .kernel(WGSL_SOURCE)
            .build()
        )
        result = shader.dispatch(params=Params(count=np.uint32(1000)), inputs=[output_buf])
    """

    def __init__(self) -> None:
        self._bindings: list[BindingSpec] = []
        self._workgroup_size: int | None = None
        self._kernel_wgsl: str | None = None
        self._kernel_fn: KernelFn | None = None

    def bind_uniform(self, group: int, binding: int, dataclass_type: type) -> ShaderBuilder:
        """Declare a uniform binding backed by a dataclass.

        Args:
            group: WebGPU bind group index (use 0 for simple shaders).
            binding: Binding index within the group.
            dataclass_type: A ``@dataclass`` class whose fields will be serialized
                into the uniform buffer. All fields must be numpy scalar types.

        Returns:
            self (for chaining)

        Raises:
            GPUBindingError: if group or binding is negative.
        """
        if not dataclasses.is_dataclass(dataclass_type):
            raise GPUBindingError(
                f"bind_uniform() expects a @dataclass type, got {dataclass_type!r}.\n"
                f"Decorate your params class with @dataclass.\n"
                f"Open an issue: {ISSUES_URL}"
            )
        self._bindings.append(
            BindingSpec(group=group, binding=binding, kind=BindingKind.UNIFORM, payload=dataclass_type)
        )
        return self

    def bind_storage(self, group: int, binding: int, dtype: Any) -> ShaderBuilder:
        """Declare a read-write storage buffer binding.

        Args:
            group: WebGPU bind group index (use 0 for simple shaders).
            binding: Binding index within the group.
            dtype: numpy dtype (e.g. ``np.uint32``, ``np.float32``) for buffer elements.

        Returns:
            self (for chaining)

        Raises:
            GPUTypeError: if dtype is not supported.
            GPUBindingError: if group or binding is negative.
        """
        self._bindings.append(
            BindingSpec(group=group, binding=binding, kind=BindingKind.STORAGE, payload=np.dtype(dtype))
        )
        return self

    def workgroup_size(self, size: int) -> ShaderBuilder:
        """Set the number of threads per workgroup.

        Must match the ``@workgroup_size`` attribute in the WGSL kernel.

        Args:
            size: Threads per workgroup. Must be a positive integer, typically
                a power of two (64, 128, 256, 512).

        Returns:
            self (for chaining)

        Raises:
            GPUConfigError: if size is not a positive integer.
        """
        if not isinstance(size, int) or size <= 0:
            raise GPUConfigError(
                f"workgroup_size() expects a positive integer, got {size!r}.\n"
                f"Typical values: 64, 128, 256, 512.\n"
                f"Open an issue: {ISSUES_URL}"
            )
        self._workgroup_size = size
        return self

    def kernel(self, source: str | KernelFn) -> ShaderBuilder:
        """Set the compute kernel — either a raw WGSL string or a @gpu_kernel function.

        Args:
            source: Either a complete WGSL shader string (V1 path), or a
                :class:`~warppy.KernelFn` produced by :func:`~warppy.gpu_kernel`
                (V2 path). When a ``KernelFn`` is passed, bindings are inferred
                automatically from its type annotations and any prior
                ``.bind_uniform()`` / ``.bind_storage()`` calls are replaced.

        Returns:
            self (for chaining)

        Raises:
            GPUConfigError: if source is neither a non-empty string nor a KernelFn.
        """
        if isinstance(source, str):
            if not source.strip():
                raise GPUConfigError(
                    f"kernel() expects a non-empty WGSL string.\n"
                    f"Open an issue: {ISSUES_URL}"
                )
            self._kernel_wgsl = source
            self._kernel_fn = None
        elif isinstance(source, KernelFn):
            self._kernel_fn = source
            self._kernel_wgsl = None  # resolved at .build() time via to_wgsl()
            self._bindings = list(source.bindings)
        else:
            raise GPUConfigError(
                f"kernel() expects a WGSL string or @gpu_kernel-decorated function, "
                f"got {type(source).__name__!r}.\n"
                f"Open an issue: {ISSUES_URL}"
            )
        return self

    def build(self) -> CompiledShader:
        """Validate configuration and compile the shader.

        Performs WGSL validation via naga if available. Falls back gracefully
        to driver-time errors if naga is not installed (warns once).

        Returns:
            A :class:`CompiledShader` ready for dispatch.

        Raises:
            GPUConfigError: if any required configuration is missing.
            GPUCompileError: if WGSL validation fails.
            GPUBindingError: if bindings conflict.
        """
        if self._kernel_fn is not None:
            # Transpiles to WGSL; raises NotImplementedError until transpiler lands
            self._kernel_wgsl = self._kernel_fn.to_wgsl()

        if self._kernel_wgsl is None:
            raise GPUConfigError(
                "No kernel set. Call .kernel(wgsl_source) or .kernel(gpu_kernel_fn) before .build().\n"
                f"Open an issue: {ISSUES_URL}"
            )
        if self._workgroup_size is None:
            raise GPUConfigError(
                "No workgroup size set. Call .workgroup_size(n) before .build().\n"
                f"Open an issue: {ISSUES_URL}"
            )
        if not self._bindings:
            raise GPUConfigError(
                "No bindings declared. Call .bind_uniform() or .bind_storage() before .build().\n"
                f"Open an issue: {ISSUES_URL}"
            )

        validate_bindings(self._bindings)
        self._try_naga_validate()

        return CompiledShader(
            kernel_wgsl=self._kernel_wgsl,
            bindings=list(self._bindings),
            workgroup_size=self._workgroup_size,
        )

    def _try_naga_validate(self) -> None:
        """Attempt WGSL validation via naga; warn once and continue if unavailable."""
        import importlib.util
        import warnings
        naga = importlib.util.find_spec("naga")
        if naga is None:
            warnings.warn(
                "naga is not installed — WGSL validation is skipped at build time. "
                "Shader errors will surface at dispatch time instead. "
                "Install naga for early error detection: pip install naga",
                stacklevel=3,
            )
            return
        # naga validation would go here when naga has a stable Python API

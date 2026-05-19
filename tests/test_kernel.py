# Copyright (c) 2026 Nick Persing
# Licensed under the MIT License. See LICENSE for details.

"""Tests for @gpu_kernel decorator and Array type — no GPU required."""

from __future__ import annotations

import ast
from dataclasses import dataclass

import numpy as np
import pytest

from warppy import Array, ArraySpec, KernelFn, ShaderBuilder, TranspileError, gpu_kernel
from warppy.bindings import BindingKind


@dataclass
class Params:
    num_trials: np.uint32
    seed: np.uint32


# ---------------------------------------------------------------------------
# Array type
# ---------------------------------------------------------------------------


class TestArray:
    def test_unbounded_returns_arrayspec(self):
        spec = Array[np.float32]
        assert isinstance(spec, ArraySpec)
        assert spec.dtype == np.dtype(np.float32)
        assert spec.size is None

    def test_fixed_size_returns_arrayspec(self):
        spec = Array[np.uint32, 52]
        assert isinstance(spec, ArraySpec)
        assert spec.dtype == np.dtype(np.uint32)
        assert spec.size == 52

    def test_invalid_size_raises(self):
        with pytest.raises(TranspileError, match="positive integer"):
            Array[np.float32, 0]

    def test_negative_size_raises(self):
        with pytest.raises(TranspileError, match="positive integer"):
            Array[np.float32, -1]

    def test_wrong_arg_count_raises(self):
        with pytest.raises(TranspileError, match="one or two"):
            Array[np.float32, 52, 99]  # type: ignore[misc]

    def test_different_dtypes(self):
        assert Array[np.int32].dtype == np.dtype(np.int32)
        assert Array[np.float32].dtype == np.dtype(np.float32)
        assert Array[np.uint32].dtype == np.dtype(np.uint32)


# ---------------------------------------------------------------------------
# @gpu_kernel decorator — happy path
# ---------------------------------------------------------------------------


class TestGpuKernelDecorator:
    def test_returns_kernelfn(self):
        @gpu_kernel
        def k(idx: np.uint32, params: Params, out: Array[np.float32]) -> None:
            pass

        assert isinstance(k, KernelFn)

    def test_stores_original_function(self):
        def raw(idx: np.uint32, out: Array[np.uint32]) -> None:
            pass

        wrapped = gpu_kernel(raw)
        assert wrapped.fn is raw

    def test_stores_idx_param_name(self):
        @gpu_kernel
        def k(thread_id: np.uint32, out: Array[np.float32]) -> None:
            pass

        assert k.idx_param == "thread_id"

    def test_name_property(self):
        @gpu_kernel
        def my_kernel(idx: np.uint32, out: Array[np.float32]) -> None:
            pass

        assert my_kernel.__name__ == "my_kernel"

    def test_parses_ast(self):
        @gpu_kernel
        def k(idx: np.uint32, out: Array[np.float32]) -> None:
            pass

        assert isinstance(k.ast_tree, ast.Module)

    def test_int32_idx_accepted(self):
        @gpu_kernel
        def k(idx: np.int32, out: Array[np.float32]) -> None:
            pass

        assert isinstance(k, KernelFn)

    def test_no_extra_params_zero_bindings(self):
        @gpu_kernel
        def k(idx: np.uint32) -> None:
            pass

        assert k.bindings == []

    def test_to_wgsl_raises_not_implemented(self):
        @gpu_kernel
        def k(idx: np.uint32, out: Array[np.float32]) -> None:
            pass

        with pytest.raises(NotImplementedError):
            k.to_wgsl()


# ---------------------------------------------------------------------------
# @gpu_kernel — binding inference
# ---------------------------------------------------------------------------


class TestBindingInference:
    def test_storage_binding_inferred_from_array(self):
        @gpu_kernel
        def k(idx: np.uint32, out: Array[np.float32]) -> None:
            pass

        assert len(k.bindings) == 1
        b = k.bindings[0]
        assert b.kind == BindingKind.STORAGE
        assert b.group == 0
        assert b.binding == 0
        assert b.payload == np.dtype(np.float32)

    def test_uniform_binding_inferred_from_dataclass(self):
        @gpu_kernel
        def k(idx: np.uint32, params: Params, out: Array[np.uint32]) -> None:
            pass

        assert len(k.bindings) == 2
        uniform = k.bindings[0]
        assert uniform.kind == BindingKind.UNIFORM
        assert uniform.group == 0
        assert uniform.binding == 0
        assert uniform.payload is Params

    def test_storage_binding_after_uniform(self):
        @gpu_kernel
        def k(idx: np.uint32, params: Params, out: Array[np.uint32]) -> None:
            pass

        storage = k.bindings[1]
        assert storage.kind == BindingKind.STORAGE
        assert storage.binding == 1

    def test_multiple_storage_bindings(self):
        @gpu_kernel
        def k(idx: np.uint32, a: Array[np.float32], b: Array[np.uint32]) -> None:
            pass

        assert len(k.bindings) == 2
        assert k.bindings[0].binding == 0
        assert k.bindings[1].binding == 1
        assert k.bindings[0].payload == np.dtype(np.float32)
        assert k.bindings[1].payload == np.dtype(np.uint32)

    def test_fixed_size_array_infers_storage(self):
        @gpu_kernel
        def k(idx: np.uint32, deck: Array[np.uint32, 52]) -> None:
            pass

        assert k.bindings[0].kind == BindingKind.STORAGE
        assert k.bindings[0].payload == np.dtype(np.uint32)

    def test_bindings_assigned_in_order(self):
        @gpu_kernel
        def k(idx: np.uint32, params: Params, a: Array[np.float32], b: Array[np.uint32]) -> None:
            pass

        indices = [b.binding for b in k.bindings]
        assert indices == [0, 1, 2]


# ---------------------------------------------------------------------------
# @gpu_kernel — validation errors
# ---------------------------------------------------------------------------


class TestGpuKernelValidation:
    def test_missing_param_annotation_raises(self):
        with pytest.raises(TranspileError, match="missing a type annotation"):
            @gpu_kernel
            def k(idx, out: Array[np.float32]) -> None:  # type: ignore[misc]
                pass

    def test_missing_return_annotation_raises(self):
        with pytest.raises(TranspileError, match="must return None"):
            @gpu_kernel
            def k(idx: np.uint32, out: Array[np.float32]):  # type: ignore[return]
                pass

    def test_non_none_return_raises(self):
        with pytest.raises(TranspileError, match="must return None"):
            @gpu_kernel
            def k(idx: np.uint32, out: Array[np.float32]) -> np.uint32:  # type: ignore[return-value]
                pass

    def test_wrong_idx_type_raises(self):
        with pytest.raises(TranspileError, match="np.uint32 or np.int32"):
            @gpu_kernel
            def k(idx: int, out: Array[np.float32]) -> None:
                pass

    def test_float_idx_type_raises(self):
        with pytest.raises(TranspileError, match="np.uint32 or np.int32"):
            @gpu_kernel
            def k(idx: np.float32, out: Array[np.float32]) -> None:
                pass

    def test_unsupported_param_type_raises(self):
        with pytest.raises(TranspileError, match="unsupported type"):
            @gpu_kernel
            def k(idx: np.uint32, data: list) -> None:  # type: ignore[type-arg]
                pass

    def test_dict_param_type_raises(self):
        with pytest.raises(TranspileError, match="unsupported type"):
            @gpu_kernel
            def k(idx: np.uint32, data: dict) -> None:  # type: ignore[type-arg]
                pass

    def test_error_messages_include_issues_link(self):
        with pytest.raises(TranspileError) as exc_info:
            @gpu_kernel
            def k(idx: np.uint32, data: list) -> None:  # type: ignore[type-arg]
                pass

        assert "github.com" in str(exc_info.value)


# ---------------------------------------------------------------------------
# ShaderBuilder integration with KernelFn
# ---------------------------------------------------------------------------


class TestBuilderKernelFnIntegration:
    def test_builder_accepts_kernelfn(self):
        @gpu_kernel
        def k(idx: np.uint32, out: Array[np.float32]) -> None:
            pass

        builder = ShaderBuilder().workgroup_size(256).kernel(k)
        assert builder._kernel_fn is k

    def test_builder_populates_bindings_from_kernelfn(self):
        @gpu_kernel
        def k(idx: np.uint32, params: Params, out: Array[np.uint32]) -> None:
            pass

        builder = ShaderBuilder().workgroup_size(256).kernel(k)
        assert len(builder._bindings) == 2
        assert builder._bindings[0].kind == BindingKind.UNIFORM
        assert builder._bindings[1].kind == BindingKind.STORAGE

    def test_builder_kernel_wgsl_none_after_kernelfn(self):
        @gpu_kernel
        def k(idx: np.uint32, out: Array[np.float32]) -> None:
            pass

        builder = ShaderBuilder().workgroup_size(256).kernel(k)
        assert builder._kernel_wgsl is None

    def test_builder_build_raises_not_implemented_for_kernelfn(self):
        @gpu_kernel
        def k(idx: np.uint32, out: Array[np.float32]) -> None:
            pass

        with pytest.raises(NotImplementedError):
            ShaderBuilder().workgroup_size(256).kernel(k).build()

    def test_builder_invalid_source_type_raises(self):
        from warppy.errors import GPUConfigError

        with pytest.raises(GPUConfigError, match="WGSL string or @gpu_kernel"):
            ShaderBuilder().kernel(42)  # type: ignore[arg-type]

    def test_v1_string_path_unaffected(self):
        """Passing a raw WGSL string still works exactly as in V1."""
        wgsl = """
        @group(0) @binding(0) var<storage, read_write> out: array<u32>;
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) { out[gid.x] = 0u; }
        """
        builder = (
            ShaderBuilder()
            .bind_storage(0, 0, np.uint32)
            .workgroup_size(64)
            .kernel(wgsl)
        )
        assert builder._kernel_wgsl == wgsl
        assert builder._kernel_fn is None

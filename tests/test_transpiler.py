# Copyright (c) 2026 Nick Persing
# Licensed under the MIT License. See LICENSE for details.

"""Tests for the AST → WGSL transpiler — no GPU required."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from warppy import Array, CompiledShader, ShaderBuilder, TranspileError, gpu_kernel


@dataclass
class Params:
    num_trials: np.uint32
    seed: np.uint32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def wgsl(kernel_fn, workgroup_size: int = 256) -> str:
    return kernel_fn.to_wgsl(workgroup_size=workgroup_size)


# ---------------------------------------------------------------------------
# Overall structure
# ---------------------------------------------------------------------------


class TestTranspilerStructure:
    def test_produces_string(self):
        @gpu_kernel
        def k(idx: np.uint32, out: Array[np.float32]) -> None:
            pass

        result = wgsl(k)
        assert isinstance(result, str)

    def test_contains_compute_attribute(self):
        @gpu_kernel
        def k(idx: np.uint32, out: Array[np.float32]) -> None:
            pass

        assert "@compute" in wgsl(k)

    def test_workgroup_size_embedded(self):
        @gpu_kernel
        def k(idx: np.uint32, out: Array[np.float32]) -> None:
            pass

        assert "@workgroup_size(128)" in wgsl(k, workgroup_size=128)
        assert "@workgroup_size(64)" in wgsl(k, workgroup_size=64)

    def test_fn_main_present(self):
        @gpu_kernel
        def k(idx: np.uint32, out: Array[np.float32]) -> None:
            pass

        assert "fn main(" in wgsl(k)

    def test_global_invocation_id_binding(self):
        @gpu_kernel
        def k(idx: np.uint32, out: Array[np.float32]) -> None:
            pass

        assert "@builtin(global_invocation_id)" in wgsl(k)

    def test_idx_param_bound_to_gid_x(self):
        @gpu_kernel
        def k(idx: np.uint32, out: Array[np.float32]) -> None:
            pass

        assert "let idx: u32 = gid.x;" in wgsl(k)

    def test_custom_idx_name(self):
        @gpu_kernel
        def k(thread_id: np.uint32, out: Array[np.float32]) -> None:
            pass

        assert "let thread_id: u32 = gid.x;" in wgsl(k)

    def test_int32_idx_emits_i32(self):
        @gpu_kernel
        def k(idx: np.int32, out: Array[np.float32]) -> None:
            pass

        assert "let idx: i32 = gid.x;" in wgsl(k)


# ---------------------------------------------------------------------------
# Binding declarations
# ---------------------------------------------------------------------------


class TestBindingDeclarations:
    def test_storage_binding_declaration(self):
        @gpu_kernel
        def k(idx: np.uint32, out: Array[np.float32]) -> None:
            pass

        result = wgsl(k)
        assert "@group(0) @binding(0)" in result
        assert "var<storage, read_write>" in result
        assert "array<f32>" in result

    def test_storage_binding_param_name(self):
        @gpu_kernel
        def k(idx: np.uint32, my_output: Array[np.uint32]) -> None:
            pass

        assert "my_output: array<u32>" in wgsl(k)

    def test_uniform_binding_declaration(self):
        @gpu_kernel
        def k(idx: np.uint32, params: Params, out: Array[np.uint32]) -> None:
            pass

        result = wgsl(k)
        assert "var<uniform> params: Params;" in result

    def test_struct_definition_emitted(self):
        @gpu_kernel
        def k(idx: np.uint32, params: Params, out: Array[np.uint32]) -> None:
            pass

        result = wgsl(k)
        assert "struct Params" in result
        assert "num_trials: u32" in result
        assert "seed: u32" in result

    def test_multiple_storage_bindings(self):
        @gpu_kernel
        def k(idx: np.uint32, a: Array[np.float32], b: Array[np.uint32]) -> None:
            pass

        result = wgsl(k)
        assert "@group(0) @binding(0)" in result
        assert "@group(0) @binding(1)" in result

    def test_uint32_storage_dtype(self):
        @gpu_kernel
        def k(idx: np.uint32, out: Array[np.uint32]) -> None:
            pass

        assert "array<u32>" in wgsl(k)

    def test_int32_storage_dtype(self):
        @gpu_kernel
        def k(idx: np.uint32, out: Array[np.int32]) -> None:
            pass

        assert "array<i32>" in wgsl(k)


# ---------------------------------------------------------------------------
# Statement transpilation
# ---------------------------------------------------------------------------


class TestAnnotatedAssignment:
    def test_var_declaration_with_value(self):
        @gpu_kernel
        def k(idx: np.uint32, out: Array[np.uint32]) -> None:
            x: np.uint32 = np.uint32(42)
            out[idx] = x

        result = wgsl(k)
        assert "var x: u32 = u32(42);" in result

    def test_var_declaration_without_value(self):
        @gpu_kernel
        def k(idx: np.uint32, out: Array[np.uint32]) -> None:
            x: np.uint32
            x = np.uint32(0)
            out[idx] = x

        result = wgsl(k)
        assert "var x: u32;" in result

    def test_float32_local_variable(self):
        @gpu_kernel
        def k(idx: np.uint32, out: Array[np.float32]) -> None:
            acc: np.float32 = np.float32(0.0)
            out[idx] = acc

        assert "var acc: f32" in wgsl(k)

    def test_int32_local_variable(self):
        @gpu_kernel
        def k(idx: np.uint32, out: Array[np.int32]) -> None:
            n: np.int32 = np.int32(10)
            out[idx] = n

        assert "var n: i32" in wgsl(k)


class TestAssignment:
    def test_simple_reassignment(self):
        @gpu_kernel
        def k(idx: np.uint32, out: Array[np.uint32]) -> None:
            x: np.uint32 = np.uint32(0)
            x = np.uint32(1)
            out[idx] = x

        result = wgsl(k)
        assert "x = u32(1);" in result

    def test_array_element_assignment(self):
        @gpu_kernel
        def k(idx: np.uint32, out: Array[np.float32]) -> None:
            out[idx] = np.float32(1.0)

        assert "out[idx] = f32(1.0);" in wgsl(k)


class TestAugmentedAssignment:
    def test_add_assign(self):
        @gpu_kernel
        def k(idx: np.uint32, out: Array[np.uint32]) -> None:
            x: np.uint32 = np.uint32(0)
            x += np.uint32(1)
            out[idx] = x

        assert "x += u32(1);" in wgsl(k)

    def test_bitxor_assign(self):
        @gpu_kernel
        def k(idx: np.uint32, out: Array[np.uint32]) -> None:
            x: np.uint32 = np.uint32(1)
            x ^= np.uint32(0xDEAD)
            out[idx] = x

        assert "x ^= u32(57005);" in wgsl(k)


class TestReturnStatement:
    def test_bare_return(self):
        @gpu_kernel
        def k(idx: np.uint32, params: Params, out: Array[np.uint32]) -> None:
            if idx >= params.num_trials:
                return
            out[idx] = np.uint32(1)

        assert "return;" in wgsl(k)


class TestIfStatement:
    def test_simple_if(self):
        @gpu_kernel
        def k(idx: np.uint32, params: Params, out: Array[np.uint32]) -> None:
            if idx >= params.num_trials:
                return

        result = wgsl(k)
        assert "if (" in result
        assert "return;" in result

    def test_if_else(self):
        @gpu_kernel
        def k(idx: np.uint32, params: Params, out: Array[np.uint32]) -> None:
            if idx >= params.num_trials:
                return
            else:
                out[idx] = np.uint32(1)

        result = wgsl(k)
        assert "} else {" in result

    def test_elif(self):
        @gpu_kernel
        def k(idx: np.uint32, out: Array[np.uint32]) -> None:
            x: np.uint32 = out[idx]
            if x == np.uint32(0):
                out[idx] = np.uint32(1)
            elif x == np.uint32(1):
                out[idx] = np.uint32(2)
            else:
                out[idx] = np.uint32(3)

        result = wgsl(k)
        assert "} else if (" in result

    def test_condition_uses_comparison_operator(self):
        @gpu_kernel
        def k(idx: np.uint32, params: Params, out: Array[np.uint32]) -> None:
            if idx >= params.num_trials:
                return

        assert ">=" in wgsl(k)


class TestForLoop:
    def test_range_one_arg(self):
        @gpu_kernel
        def k(idx: np.uint32, out: Array[np.float32]) -> None:
            acc: np.float32 = np.float32(0.0)
            for i in range(10):
                acc += np.float32(1.0)
            out[idx] = acc

        result = wgsl(k)
        assert "for (var i: i32 = 0; i < 10; i++)" in result

    def test_range_two_args(self):
        @gpu_kernel
        def k(idx: np.uint32, out: Array[np.float32]) -> None:
            acc: np.float32 = np.float32(0.0)
            for i in range(2, 10):
                acc += np.float32(1.0)
            out[idx] = acc

        result = wgsl(k)
        assert "for (var i: i32 = 2; i < 10; i++)" in result

    def test_range_three_args(self):
        @gpu_kernel
        def k(idx: np.uint32, out: Array[np.float32]) -> None:
            acc: np.float32 = np.float32(0.0)
            for i in range(0, 20, 2):
                acc += np.float32(1.0)
            out[idx] = acc

        result = wgsl(k)
        assert "for (var i: i32 = 0; i < 20; i += 2)" in result

    def test_loop_body_indented(self):
        @gpu_kernel
        def k(idx: np.uint32, out: Array[np.uint32]) -> None:
            for i in range(5):
                out[i] = np.uint32(i)

        result = wgsl(k)
        assert "out[i] = u32(i);" in result


# ---------------------------------------------------------------------------
# Expression transpilation
# ---------------------------------------------------------------------------


class TestExpressions:
    def test_numpy_cast_uint32(self):
        @gpu_kernel
        def k(idx: np.uint32, out: Array[np.uint32]) -> None:
            out[idx] = np.uint32(99)

        assert "u32(99)" in wgsl(k)

    def test_numpy_cast_float32(self):
        @gpu_kernel
        def k(idx: np.uint32, out: Array[np.float32]) -> None:
            out[idx] = np.float32(3.14)

        assert "f32(" in wgsl(k)

    def test_numpy_cast_int32(self):
        @gpu_kernel
        def k(idx: np.uint32, out: Array[np.int32]) -> None:
            out[idx] = np.int32(-1)

        assert "i32(-1)" in wgsl(k)

    def test_binary_add(self):
        @gpu_kernel
        def k(idx: np.uint32, out: Array[np.uint32]) -> None:
            x: np.uint32 = np.uint32(1) + np.uint32(2)
            out[idx] = x

        assert "(u32(1) + u32(2))" in wgsl(k)

    def test_binary_xor(self):
        @gpu_kernel
        def k(idx: np.uint32, out: Array[np.uint32]) -> None:
            x: np.uint32 = np.uint32(0xFF) ^ np.uint32(0x0F)
            out[idx] = x

        assert "^" in wgsl(k)

    def test_bitshift_left(self):
        @gpu_kernel
        def k(idx: np.uint32, out: Array[np.uint32]) -> None:
            x: np.uint32 = np.uint32(1) << np.uint32(3)
            out[idx] = x

        assert "<<" in wgsl(k)

    def test_comparison_ge(self):
        @gpu_kernel
        def k(idx: np.uint32, params: Params, out: Array[np.uint32]) -> None:
            if idx >= params.num_trials:
                return

        assert ">=" in wgsl(k)

    def test_attribute_access(self):
        @gpu_kernel
        def k(idx: np.uint32, params: Params, out: Array[np.uint32]) -> None:
            out[idx] = params.num_trials

        assert "params.num_trials" in wgsl(k)

    def test_subscript_with_variable_index(self):
        @gpu_kernel
        def k(idx: np.uint32, out: Array[np.uint32]) -> None:
            out[idx] = np.uint32(0)

        assert "out[idx]" in wgsl(k)

    def test_integer_literal(self):
        @gpu_kernel
        def k(idx: np.uint32, out: Array[np.uint32]) -> None:
            x: np.uint32 = np.uint32(42)
            out[idx] = x

        assert "42" in wgsl(k)

    def test_float_literal_has_decimal(self):
        @gpu_kernel
        def k(idx: np.uint32, out: Array[np.float32]) -> None:
            out[idx] = np.float32(1.0)

        result = wgsl(k)
        assert "1.0" in result

    def test_bool_literal_true(self):
        @gpu_kernel
        def k(idx: np.uint32, out: Array[np.uint32]) -> None:
            x: bool = True  # noqa: F841 — just testing bool literal
            out[idx] = np.uint32(0)

        assert "true" in wgsl(k)

    def test_unary_negation(self):
        @gpu_kernel
        def k(idx: np.uint32, out: Array[np.int32]) -> None:
            out[idx] = np.int32(-1)

        result = wgsl(k)
        assert "-" in result

    def test_bool_and(self):
        @gpu_kernel
        def k(idx: np.uint32, params: Params, out: Array[np.uint32]) -> None:
            if idx < params.num_trials and idx > np.uint32(0):
                out[idx] = np.uint32(1)

        assert "&&" in wgsl(k)

    def test_bool_or(self):
        @gpu_kernel
        def k(idx: np.uint32, out: Array[np.uint32]) -> None:
            if idx == np.uint32(0) or idx == np.uint32(1):
                out[idx] = np.uint32(99)

        assert "||" in wgsl(k)


# ---------------------------------------------------------------------------
# Unsupported constructs → TranspileError
# ---------------------------------------------------------------------------


class TestUnsupportedConstructs:
    def test_while_loop_raises(self):
        with pytest.raises(TranspileError, match="while"):
            @gpu_kernel
            def k(idx: np.uint32, out: Array[np.uint32]) -> None:
                while True:
                    out[idx] = np.uint32(0)

            wgsl(k)

    def test_try_except_raises(self):
        with pytest.raises(TranspileError, match="try"):
            @gpu_kernel
            def k(idx: np.uint32, out: Array[np.uint32]) -> None:
                try:
                    out[idx] = np.uint32(0)
                except Exception:
                    pass

            wgsl(k)

    def test_for_non_range_raises(self):
        with pytest.raises(TranspileError, match="range"):
            @gpu_kernel
            def k(idx: np.uint32, out: Array[np.uint32]) -> None:
                for x in [1, 2, 3]:
                    out[idx] = np.uint32(x)

            wgsl(k)

    def test_chained_comparison_raises(self):
        with pytest.raises(TranspileError, match="Chained"):
            @gpu_kernel
            def k(idx: np.uint32, out: Array[np.uint32]) -> None:
                if np.uint32(0) < idx < np.uint32(10):
                    out[idx] = np.uint32(1)

            wgsl(k)

    def test_unsupported_numpy_fn_raises(self):
        with pytest.raises(TranspileError, match="np.sin"):
            @gpu_kernel
            def k(idx: np.uint32, out: Array[np.float32]) -> None:
                out[idx] = np.sin(np.float32(1.0))  # type: ignore[attr-defined]

            wgsl(k)

    def test_keyword_arg_in_call_raises(self):
        with pytest.raises(TranspileError, match="Keyword"):
            @gpu_kernel
            def k(idx: np.uint32, out: Array[np.float32]) -> None:
                out[idx] = abs(x=np.float32(1.0))  # type: ignore[call-arg]

            wgsl(k)

    def test_error_messages_include_issues_link(self):
        with pytest.raises(TranspileError) as exc_info:
            @gpu_kernel
            def k(idx: np.uint32, out: Array[np.uint32]) -> None:
                while True:
                    out[idx] = np.uint32(0)

            wgsl(k)
        assert "github.com" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Builder integration: end-to-end .build() with @gpu_kernel
# ---------------------------------------------------------------------------


class TestBuilderIntegration:
    def test_build_returns_compiled_shader(self):
        @gpu_kernel
        def k(idx: np.uint32, out: Array[np.float32]) -> None:
            out[idx] = np.float32(1.0)

        shader = ShaderBuilder().workgroup_size(256).kernel(k).build()
        assert isinstance(shader, CompiledShader)

    def test_build_with_uniform_and_storage(self):
        @gpu_kernel
        def k(idx: np.uint32, params: Params, out: Array[np.uint32]) -> None:
            if idx >= params.num_trials:
                return
            out[idx] = np.uint32(1)

        shader = ShaderBuilder().workgroup_size(256).kernel(k).build()
        assert isinstance(shader, CompiledShader)

    def test_compiled_shader_wgsl_contains_compute(self):
        @gpu_kernel
        def k(idx: np.uint32, out: Array[np.float32]) -> None:
            out[idx] = np.float32(0.0)

        shader = ShaderBuilder().workgroup_size(64).kernel(k).build()
        assert "@compute" in shader._kernel_wgsl

    def test_v1_and_v2_both_produce_compiled_shader(self):
        """Regression: passing a WGSL string still works after V2 changes."""
        wgsl_str = """
        @group(0) @binding(0) var<storage, read_write> out: array<f32>;
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) { out[gid.x] = 1.0; }
        """
        v1_shader = (
            ShaderBuilder()
            .bind_storage(0, 0, np.float32)
            .workgroup_size(64)
            .kernel(wgsl_str)
            .build()
        )

        @gpu_kernel
        def k(idx: np.uint32, out: Array[np.float32]) -> None:
            out[idx] = np.float32(1.0)

        v2_shader = ShaderBuilder().workgroup_size(64).kernel(k).build()

        assert isinstance(v1_shader, CompiledShader)
        assert isinstance(v2_shader, CompiledShader)

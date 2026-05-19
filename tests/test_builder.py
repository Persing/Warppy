# Copyright (c) 2026 Nick Persing
# Licensed under the MIT License. See LICENSE for details.

"""Tests for ShaderBuilder state management — no GPU required."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from warppy import ShaderBuilder
from warppy.bindings import BindingKind
from warppy.errors import GPUBindingError, GPUConfigError


@dataclass
class Params:
    count: np.uint32


MINIMAL_WGSL = """
@group(0) @binding(0) var<storage, read_write> output: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) { output[gid.x] = 0u; }
"""


class TestShaderBuilderState:
    def test_bind_uniform_stores_spec(self):
        builder = ShaderBuilder().bind_uniform(0, 0, Params)
        assert len(builder._bindings) == 1
        assert builder._bindings[0].kind == BindingKind.UNIFORM
        assert builder._bindings[0].group == 0
        assert builder._bindings[0].binding == 0
        assert builder._bindings[0].payload is Params

    def test_bind_storage_stores_spec(self):
        builder = ShaderBuilder().bind_storage(0, 1, np.uint32)
        assert len(builder._bindings) == 1
        assert builder._bindings[0].kind == BindingKind.STORAGE

    def test_workgroup_size_stored(self):
        builder = ShaderBuilder().workgroup_size(256)
        assert builder._workgroup_size == 256

    def test_kernel_stored(self):
        builder = ShaderBuilder().kernel(MINIMAL_WGSL)
        assert builder._kernel_wgsl == MINIMAL_WGSL

    def test_method_chaining_returns_builder(self):
        builder = ShaderBuilder()
        result = builder.bind_storage(0, 0, np.uint32)
        assert result is builder

    def test_multiple_bindings_stored_in_order(self):
        builder = (
            ShaderBuilder()
            .bind_uniform(0, 0, Params)
            .bind_storage(0, 1, np.uint32)
            .bind_storage(0, 2, np.float32)
        )
        assert len(builder._bindings) == 3
        assert builder._bindings[0].binding == 0
        assert builder._bindings[1].binding == 1
        assert builder._bindings[2].binding == 2


class TestShaderBuilderValidation:
    def test_build_without_kernel_raises(self):
        with pytest.raises(GPUConfigError, match="No kernel set"):
            ShaderBuilder().bind_storage(0, 0, np.uint32).workgroup_size(64).build()

    def test_build_without_workgroup_size_raises(self):
        with pytest.raises(GPUConfigError, match="No workgroup size"):
            ShaderBuilder().bind_storage(0, 0, np.uint32).kernel(MINIMAL_WGSL).build()

    def test_build_without_bindings_raises(self):
        with pytest.raises(GPUConfigError, match="No bindings"):
            ShaderBuilder().workgroup_size(64).kernel(MINIMAL_WGSL).build()

    def test_build_with_duplicate_bindings_raises(self):
        with pytest.raises(GPUBindingError, match="Duplicate binding"):
            (
                ShaderBuilder()
                .bind_storage(0, 0, np.uint32)
                .bind_storage(0, 0, np.float32)
                .workgroup_size(64)
                .kernel(MINIMAL_WGSL)
                .build()
            )

    def test_bind_uniform_non_dataclass_raises(self):
        with pytest.raises(GPUBindingError, match="dataclass"):
            ShaderBuilder().bind_uniform(0, 0, dict)

    def test_invalid_workgroup_size_raises(self):
        with pytest.raises(GPUConfigError, match="positive integer"):
            ShaderBuilder().workgroup_size(0)

    def test_negative_workgroup_size_raises(self):
        with pytest.raises(GPUConfigError):
            ShaderBuilder().workgroup_size(-1)

    def test_empty_kernel_raises(self):
        with pytest.raises(GPUConfigError):
            ShaderBuilder().kernel("   ")

    def test_all_error_messages_include_issues_link(self):
        with pytest.raises(GPUConfigError) as exc_info:
            ShaderBuilder().bind_storage(0, 0, np.uint32).workgroup_size(64).build()
        assert "github.com" in str(exc_info.value)

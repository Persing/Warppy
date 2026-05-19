# Copyright (c) 2026 Nick Persing
# Licensed under the MIT License. See LICENSE for details.

"""Tests for binding specification and validation — no GPU required."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from warppy.bindings import BindingKind, BindingSpec, validate_bindings
from warppy.errors import GPUBindingError


@dataclass
class Params:
    count: np.uint32


class TestBindingSpec:
    def test_uniform_wgsl_type_is_class_name(self):
        spec = BindingSpec(group=0, binding=0, kind=BindingKind.UNIFORM, payload=Params)
        assert spec.wgsl_type == "Params"

    def test_storage_wgsl_type_for_uint32(self):
        spec = BindingSpec(group=0, binding=1, kind=BindingKind.STORAGE, payload=np.dtype(np.uint32))
        assert spec.wgsl_type == "u32"

    def test_storage_wgsl_type_for_float32(self):
        spec = BindingSpec(group=0, binding=1, kind=BindingKind.STORAGE, payload=np.dtype(np.float32))
        assert spec.wgsl_type == "f32"


class TestValidateBindings:
    def test_no_conflict_passes(self):
        specs = [
            BindingSpec(group=0, binding=0, kind=BindingKind.UNIFORM, payload=Params),
            BindingSpec(group=0, binding=1, kind=BindingKind.STORAGE, payload=np.dtype(np.uint32)),
        ]
        validate_bindings(specs)  # should not raise

    def test_duplicate_group_binding_raises(self):
        specs = [
            BindingSpec(group=0, binding=0, kind=BindingKind.UNIFORM, payload=Params),
            BindingSpec(group=0, binding=0, kind=BindingKind.STORAGE, payload=np.dtype(np.uint32)),
        ]
        with pytest.raises(GPUBindingError, match="Duplicate binding"):
            validate_bindings(specs)

    def test_different_groups_do_not_conflict(self):
        specs = [
            BindingSpec(group=0, binding=0, kind=BindingKind.UNIFORM, payload=Params),
            BindingSpec(group=1, binding=0, kind=BindingKind.STORAGE, payload=np.dtype(np.uint32)),
        ]
        validate_bindings(specs)  # should not raise

    def test_negative_binding_raises(self):
        specs = [
            BindingSpec(group=0, binding=-1, kind=BindingKind.STORAGE, payload=np.dtype(np.uint32)),
        ]
        with pytest.raises(GPUBindingError, match="non-negative"):
            validate_bindings(specs)

    def test_error_message_includes_group_and_binding(self):
        specs = [
            BindingSpec(group=2, binding=3, kind=BindingKind.UNIFORM, payload=Params),
            BindingSpec(group=2, binding=3, kind=BindingKind.STORAGE, payload=np.dtype(np.uint32)),
        ]
        with pytest.raises(GPUBindingError, match="group=2") as exc_info:
            validate_bindings(specs)
        assert "binding=3" in str(exc_info.value)

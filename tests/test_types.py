# Copyright (c) 2026 Nick Persing
# Licensed under the MIT License. See LICENSE for details.

"""Tests for dtype mapping and struct serialization — no GPU required."""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np
import pytest

from warppy.errors import GPUTypeError
from warppy.types import dtype_to_wgsl, pack_dataclass, dtype_byte_size


class TestDtypeToWgsl:
    def test_uint32(self):
        assert dtype_to_wgsl(np.dtype(np.uint32)) == "u32"

    def test_int32(self):
        assert dtype_to_wgsl(np.dtype(np.int32)) == "i32"

    def test_float32(self):
        assert dtype_to_wgsl(np.dtype(np.float32)) == "f32"

    def test_float16(self):
        assert dtype_to_wgsl(np.dtype(np.float16)) == "f16"

    def test_unsupported_dtype_raises(self):
        with pytest.raises(GPUTypeError, match="Unsupported numpy dtype"):
            dtype_to_wgsl(np.dtype(np.float64))

    def test_error_includes_issues_link(self):
        with pytest.raises(GPUTypeError, match="github.com"):
            dtype_to_wgsl(np.dtype(np.complex64))


class TestDtypeByteSize:
    def test_uint32_is_4_bytes(self):
        assert dtype_byte_size(np.dtype(np.uint32)) == 4

    def test_float16_is_2_bytes(self):
        assert dtype_byte_size(np.dtype(np.float16)) == 2


class TestPackDataclass:
    def test_four_uint32_fields(self):
        @dataclass
        class Params:
            a: np.uint32
            b: np.uint32
            c: np.uint32
            d: np.uint32

        p = Params(a=np.uint32(1), b=np.uint32(2), c=np.uint32(3), d=np.uint32(4))
        result = pack_dataclass(p)
        assert len(result) == 16
        assert struct.unpack("<4I", result) == (1, 2, 3, 4)

    def test_output_is_multiple_of_16(self):
        @dataclass
        class Single:
            x: np.uint32

        result = pack_dataclass(Single(x=np.uint32(42)))
        assert len(result) % 16 == 0

    def test_non_dataclass_raises(self):
        with pytest.raises(GPUTypeError, match="dataclass"):
            pack_dataclass({"a": 1})

    def test_float32_field(self):
        @dataclass
        class F:
            x: np.float32

        result = pack_dataclass(F(x=np.float32(3.14)))
        (val,) = struct.unpack("<f", result[:4])
        assert abs(val - 3.14) < 0.001

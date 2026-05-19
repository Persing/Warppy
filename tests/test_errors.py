# Copyright (c) 2026 Nick Persing
# Licensed under the MIT License. See LICENSE for details.

"""Tests for error classes — no GPU required."""

import pytest

from warppy.errors import (
    GPUBindingError,
    GPUCompileError,
    GPUConfigError,
    GPUTypeError,
    WarpyError,
)


def test_all_errors_are_warppy_errors():
    for cls in (GPUCompileError, GPUTypeError, GPUBindingError, GPUConfigError):
        assert issubclass(cls, WarpyError)
        assert issubclass(cls, Exception)


def test_error_messages_are_preserved():
    exc = GPUCompileError("bad wgsl on line 5")
    assert "bad wgsl on line 5" in str(exc)


def test_errors_can_be_caught_as_warpyerror():
    with pytest.raises(WarpyError):
        raise GPUTypeError("unsupported dtype")

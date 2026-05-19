# Copyright (c) 2026 Nick Persing
# Licensed under the MIT License. See LICENSE for details.

"""Warppy error classes.

Every error answers three questions: what went wrong, where, and what to do.
All errors include a GitHub issues link — no telemetry, no phone-home.
"""

ISSUES_URL = "https://github.com/warppy/warppy/issues"


class WarpyError(Exception):
    """Base class for all warppy errors."""


class GPUCompileError(WarpyError):
    """WGSL shader failed to compile.

    Raised at `.build()` time when naga validation detects invalid WGSL,
    or at dispatch time if naga was not available and the driver rejects the shader.
    """


class GPUTypeError(WarpyError):
    """Unsupported or mismatched numpy dtype.

    Raised when a dtype has no WGSL equivalent or when a dataclass field
    uses a type that cannot be serialized into a WGSL uniform struct.
    """


class GPUBindingError(WarpyError):
    """Invalid binding configuration.

    Raised when bindings conflict (duplicate group+binding index),
    when a required binding is missing, or when binding arguments are invalid.
    """


class GPUConfigError(WarpyError):
    """Invalid shader configuration.

    Raised by `.build()` when required configuration is missing (no kernel,
    no bindings, invalid workgroup size, etc.).
    """

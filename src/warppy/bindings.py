# Copyright (c) 2026 Nick Persing
# Licensed under the MIT License. See LICENSE for details.

"""Binding specification types for ShaderBuilder."""

from __future__ import annotations

import dataclasses
from enum import Enum, auto
from typing import Any

import numpy as np

from .errors import GPUBindingError, GPUTypeError, ISSUES_URL
from .types import dtype_to_wgsl, dataclass_to_wgsl_struct


class BindingKind(Enum):
    UNIFORM = auto()
    STORAGE = auto()


@dataclasses.dataclass(frozen=True)
class BindingSpec:
    """Describes a single bind group entry."""

    group: int
    binding: int
    kind: BindingKind
    # UNIFORM: a dataclass type; STORAGE: a numpy dtype
    payload: Any

    @property
    def wgsl_type(self) -> str:
        """WGSL type name for this binding's element type."""
        if self.kind == BindingKind.UNIFORM:
            if not dataclasses.is_dataclass(self.payload):
                raise GPUTypeError(
                    f"Uniform binding ({self.group}, {self.binding}) expects a dataclass type, "
                    f"got {self.payload!r}.\n"
                    f"Open an issue: {ISSUES_URL}"
                )
            return self.payload.__name__
        else:
            return dtype_to_wgsl(np.dtype(self.payload))


def validate_bindings(specs: list[BindingSpec]) -> None:
    """Check that no two specs share the same (group, binding) index.

    Raises:
        GPUBindingError: on duplicate or invalid binding.
    """
    seen: set[tuple[int, int]] = set()
    for spec in specs:
        if spec.group < 0 or spec.binding < 0:
            raise GPUBindingError(
                f"Binding indices must be non-negative integers. "
                f"Got group={spec.group}, binding={spec.binding}.\n"
                f"Open an issue: {ISSUES_URL}"
            )
        key = (spec.group, spec.binding)
        if key in seen:
            raise GPUBindingError(
                f"Duplicate binding at group={spec.group}, binding={spec.binding}.\n"
                f"Each (group, binding) pair must be unique.\n"
                f"Open an issue: {ISSUES_URL}"
            )
        seen.add(key)

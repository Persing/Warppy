# Copyright (c) 2026 Nick Persing
# Licensed under the MIT License. See LICENSE for details.

"""WGSL shader emission helpers for the transpiler."""

from __future__ import annotations

import inspect

from ..bindings import BindingKind
from ..types import dataclass_to_wgsl_struct, dtype_to_wgsl


class CodeGen:
    """Mixin: emits complete WGSL shader sections."""

    def transpile(self) -> str:
        """Return the complete WGSL shader source."""
        parts = [
            self._emit_structs(),
            self._emit_bindings(),
            self._emit_entry_point(),
        ]
        return "\n\n".join(p for p in parts if p)

    def _emit_structs(self) -> str:
        parts = []
        for spec in self._kfn.bindings:
            if spec.kind == BindingKind.UNIFORM:
                parts.append(dataclass_to_wgsl_struct(spec.payload))
        return "\n\n".join(parts)

    def _emit_bindings(self) -> str:
        sig = inspect.signature(self._kfn.fn)
        param_names = list(sig.parameters.keys())[1:]  # skip idx
        lines = []
        for param_name, spec in zip(param_names, self._kfn.bindings):
            prefix = f"@group({spec.group}) @binding({spec.binding})"
            if spec.kind == BindingKind.UNIFORM:
                lines.append(f"{prefix} var<uniform> {param_name}: {spec.payload.__name__};")
            else:
                wgsl_type = dtype_to_wgsl(spec.payload)
                lines.append(f"{prefix} var<storage, read_write> {param_name}: array<{wgsl_type}>;")
        return "\n".join(lines)

    def _emit_entry_point(self) -> str:
        idx_name = self._kfn.idx_param
        idx_type = self._type_env[idx_name]

        body_lines: list[str] = [f"    let {idx_name}: {idx_type} = gid.x;"]
        for stmt in self._fn_node.body:
            body_lines.extend(self._transpile_stmt(stmt, indent=1))

        body = "\n".join(body_lines)
        return (
            f"@compute @workgroup_size({self._workgroup_size})\n"
            f"fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{\n"
            f"{body}\n"
            f"}}"
        )

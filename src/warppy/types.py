"""numpy dtype <-> WGSL scalar type mapping and struct serialization."""

from __future__ import annotations

import dataclasses
import struct
from typing import Any

import numpy as np

from .errors import GPUTypeError, ISSUES_URL


# Scalar dtype → (wgsl_type, byte_size, struct_format)
_DTYPE_MAP: dict[np.dtype, tuple[str, int, str]] = {
    np.dtype(np.uint32): ("u32", 4, "I"),
    np.dtype(np.int32): ("i32", 4, "i"),
    np.dtype(np.float32): ("f32", 4, "f"),
    np.dtype(np.float16): ("f16", 2, "e"),
    np.dtype(np.uint16): ("u16", 2, "H"),  # note: limited WGSL support
    np.dtype(np.int16): ("i16", 2, "h"),
}


def dtype_to_wgsl(dtype: np.dtype) -> str:
    """Return the WGSL scalar type name for a numpy dtype.

    Raises:
        GPUTypeError: if the dtype has no WGSL equivalent.
    """
    dtype = np.dtype(dtype)
    entry = _DTYPE_MAP.get(dtype)
    if entry is None:
        supported = ", ".join(str(d) for d in _DTYPE_MAP)
        raise GPUTypeError(
            f"Unsupported numpy dtype: {dtype!r}.\n"
            f"Supported dtypes: {supported}\n"
            f"Open an issue if you need additional types: {ISSUES_URL}"
        )
    return entry[0]


def dtype_byte_size(dtype: np.dtype) -> int:
    """Return the byte size of a WGSL-compatible numpy dtype."""
    dtype = np.dtype(dtype)
    entry = _DTYPE_MAP.get(dtype)
    if entry is None:
        raise GPUTypeError(
            f"Unsupported numpy dtype: {dtype!r}.\n"
            f"Open an issue: {ISSUES_URL}"
        )
    return entry[1]


# WGSL uniform buffer alignment rules:
# Scalar u32/i32/f32 → align 4, size 4
# vec2<T>            → align 8, size 8
# vec3<T>            → align 16, size 12
# vec4<T>            → align 16, size 16
# struct             → align = max(member aligns), size = rounded up to align
_SCALAR_ALIGN = 4


def pack_dataclass(obj: Any) -> bytes:
    """Serialize a dataclass instance to bytes matching WGSL uniform buffer layout.

    Only scalar numeric fields (u32, i32, f32, f16) are supported.
    Fields are packed with WGSL std140/WebGPU uniform alignment rules.

    Raises:
        GPUTypeError: if the dataclass contains unsupported field types.
    """
    if not dataclasses.is_dataclass(obj):
        raise GPUTypeError(
            f"Expected a dataclass instance, got {type(obj).__name__!r}.\n"
            f"Uniform bindings must use a @dataclass. Open an issue: {ISSUES_URL}"
        )

    buf = bytearray()
    for field in dataclasses.fields(obj):
        value = getattr(obj, field.name)
        dtype = _resolve_field_dtype(field, value)
        entry = _DTYPE_MAP.get(np.dtype(dtype))
        if entry is None:
            raise GPUTypeError(
                f"Field {field.name!r} has unsupported type {field.type!r}.\n"
                f"Supported numpy scalar types: {', '.join(str(d) for d in _DTYPE_MAP)}\n"
                f"Open an issue: {ISSUES_URL}"
            )
        _, size, fmt = entry
        # Align to scalar boundary (all scalars are 4-byte aligned except f16/u16/i16)
        align = size if size >= 4 else 4  # WebGPU scalars align to at least 4 bytes
        padding = (align - len(buf) % align) % align
        buf.extend(b"\x00" * padding)
        buf.extend(struct.pack(f"<{fmt}", value))

    # Pad struct to 16-byte boundary (WebGPU uniform buffer min binding size)
    remainder = len(buf) % 16
    if remainder:
        buf.extend(b"\x00" * (16 - remainder))

    return bytes(buf)


def _resolve_field_dtype(field: dataclasses.Field, value: Any) -> type:
    """Resolve a dataclass field to a numpy scalar dtype."""
    annotation = field.type
    # Handle string annotations (from __future__ annotations or forward refs)
    if isinstance(annotation, str):
        import builtins
        annotation = getattr(builtins, annotation, None) or annotation

    # Direct numpy scalar types
    numpy_scalars = (
        np.uint32, np.int32, np.float32, np.float16, np.uint16, np.int16,
    )
    if annotation in numpy_scalars:
        return annotation
    if isinstance(value, numpy_scalars):
        return type(value)

    raise GPUTypeError(
        f"Field {field.name!r}: cannot determine WGSL type from annotation "
        f"{field.type!r} and value {value!r}.\n"
        f"Annotate fields with numpy scalar types (e.g. np.uint32, np.float32).\n"
        f"Open an issue: {ISSUES_URL}"
    )


def dataclass_to_wgsl_struct(cls: type, struct_name: str | None = None) -> str:
    """Generate a WGSL struct definition from a dataclass type.

    Raises:
        GPUTypeError: if any field uses an unsupported dtype.
    """
    if not dataclasses.is_dataclass(cls):
        raise GPUTypeError(
            f"Expected a dataclass type, got {cls!r}.\n"
            f"Open an issue: {ISSUES_URL}"
        )
    name = struct_name or cls.__name__
    lines = [f"struct {name} {{"]
    for field in dataclasses.fields(cls):
        annotation = field.type
        numpy_scalars = (np.uint32, np.int32, np.float32, np.float16, np.uint16, np.int16)
        if annotation not in numpy_scalars:
            raise GPUTypeError(
                f"Field {field.name!r} has annotation {annotation!r} which cannot "
                f"be converted to a WGSL type.\n"
                f"Use numpy scalar type annotations (e.g. np.uint32).\n"
                f"Open an issue: {ISSUES_URL}"
            )
        wgsl_type = dtype_to_wgsl(np.dtype(annotation))
        lines.append(f"    {field.name}: {wgsl_type},")
    lines.append("}")
    return "\n".join(lines)

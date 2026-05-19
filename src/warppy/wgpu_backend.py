"""wgpu-py GPU backend — ALL wgpu-py imports are isolated to this file.

No other warppy module imports wgpu directly. This isolation ensures that:
  - Future backend additions (SPIR-V, alternative APIs) are a one-file change.
  - wgpu-py API changes only require updates here.
  - Unit tests for the rest of the library never require a GPU.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .bindings import BindingSpec


def run_compute(
    kernel_wgsl: str,
    bindings: list[BindingSpec],
    workgroup_size: int,
    uniform_bytes: bytes | None,
    storage_arrays: list[np.ndarray],
    num_invocations: int,
) -> tuple[list[np.ndarray], float]:
    """Execute a WGSL compute shader and return output arrays + elapsed time.

    This is the only function in warppy that calls wgpu-py.

    Args:
        kernel_wgsl: Complete WGSL shader source.
        bindings: Ordered list of BindingSpec describing each bind group entry.
        workgroup_size: Number of threads per workgroup (must match shader).
        uniform_bytes: Serialized uniform struct bytes, or None if no uniform binding.
        storage_arrays: One numpy array per storage binding (read-write).
        num_invocations: Total number of compute threads to launch.

    Returns:
        (output_arrays, elapsed_ms): output_arrays mirrors storage_arrays structure,
        elapsed_ms is the wall time from queue.submit to readback complete.

    Raises:
        GPUCompileError: if the driver rejects the WGSL shader.
    """
    import wgpu  # noqa: PLC0415 — intentional: this is the only wgpu import in the library

    from .errors import GPUCompileError, ISSUES_URL
    from .bindings import BindingKind

    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()

    # Upload uniform buffer
    gpu_uniform_buf = None
    if uniform_bytes is not None:
        gpu_uniform_buf = device.create_buffer(
            size=len(uniform_bytes),
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        device.queue.write_buffer(gpu_uniform_buf, 0, uniform_bytes)

    # Allocate storage buffers + readback buffers
    gpu_storage_bufs: list = []
    gpu_readback_bufs: list = []
    for arr in storage_arrays:
        nbytes = arr.nbytes
        gpu_buf = device.create_buffer(
            size=nbytes,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        rb_buf = device.create_buffer(
            size=nbytes,
            usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST,
        )
        gpu_storage_bufs.append(gpu_buf)
        gpu_readback_bufs.append(rb_buf)

    # Build bind group layout entries
    bgl_entries = []
    for spec in bindings:
        if spec.kind == BindingKind.UNIFORM:
            bgl_entries.append({
                "binding": spec.binding,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            })
        else:
            bgl_entries.append({
                "binding": spec.binding,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.storage},
            })

    bgl = device.create_bind_group_layout(entries=bgl_entries)

    # Build bind group entries, pairing specs with GPU buffers
    bg_entries = []
    storage_idx = 0
    for spec in bindings:
        if spec.kind == BindingKind.UNIFORM:
            bg_entries.append({
                "binding": spec.binding,
                "resource": {"buffer": gpu_uniform_buf},
            })
        else:
            bg_entries.append({
                "binding": spec.binding,
                "resource": {"buffer": gpu_storage_bufs[storage_idx]},
            })
            storage_idx += 1

    bg = device.create_bind_group(layout=bgl, entries=bg_entries)
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bgl])

    try:
        shader_module = device.create_shader_module(code=kernel_wgsl)
        pipeline = device.create_compute_pipeline(
            layout=pipeline_layout,
            compute={"module": shader_module, "entry_point": "main"},
        )
    except Exception as exc:
        raise GPUCompileError(
            f"Shader compilation failed: {exc}\n"
            f"Hint: validate your WGSL at https://webgpufundamentals.org/webgpu/lessons/webgpu-wgsl.html\n"
            f"Open an issue: {ISSUES_URL}"
        ) from exc

    workgroups = math.ceil(num_invocations / workgroup_size)

    t_start = time.perf_counter()
    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bg)
    compute_pass.dispatch_workgroups(workgroups)
    compute_pass.end()

    for gpu_buf, rb_buf, arr in zip(gpu_storage_bufs, gpu_readback_bufs, storage_arrays):
        encoder.copy_buffer_to_buffer(gpu_buf, 0, rb_buf, 0, arr.nbytes)

    device.queue.submit([encoder.finish()])

    output_arrays: list[np.ndarray] = []
    for rb_buf, arr in zip(gpu_readback_bufs, storage_arrays):
        rb_buf.map_sync(wgpu.MapMode.READ)
        raw = rb_buf.read_mapped()
        output_arrays.append(np.frombuffer(raw, dtype=arr.dtype).copy())
        rb_buf.unmap()

    elapsed_ms = (time.perf_counter() - t_start) * 1000
    return output_arrays, elapsed_ms

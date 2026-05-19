"""Warppy — Python GPU compute via a fluent builder API.

Abstracts wgpu-py boilerplate so you can focus on your WGSL kernel,
not device initialization, buffer allocation, and pipeline setup.

Typical usage::

    from dataclasses import dataclass
    import numpy as np
    from warppy import ShaderBuilder

    @dataclass
    class Params:
        count: np.uint32

    shader = (
        ShaderBuilder()
        .bind_uniform(0, 0, Params)
        .bind_storage(0, 1, np.uint32)
        .workgroup_size(256)
        .kernel(WGSL_SOURCE)
        .build()
    )
    result = shader.dispatch(params=Params(count=np.uint32(1_000_000)), inputs=[buf])
    print(result.data, result.elapsed_ms)
"""

from .builder import CompiledShader, GPUResult, ShaderBuilder
from .errors import (
    GPUBindingError,
    GPUCompileError,
    GPUConfigError,
    GPUTypeError,
    WarpyError,
)

__version__ = "0.1.0"

__all__ = [
    "ShaderBuilder",
    "CompiledShader",
    "GPUResult",
    "WarpyError",
    "GPUCompileError",
    "GPUTypeError",
    "GPUBindingError",
    "GPUConfigError",
]

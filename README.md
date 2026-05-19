# Warppy

**Python GPU compute without the boilerplate.**

Warppy wraps [wgpu-py](https://github.com/pygfx/wgpu-py) in a fluent builder API so you can run compute shaders on macOS, Linux, and Windows with a fraction of the setup code.

You write the WGSL kernel. Warppy handles device initialization, buffer allocation, bind group layout, pipeline setup, dispatch, and readback.

---

## Why Warppy

Running a GPU compute shader in raw wgpu-py takes **36 lines of boilerplate** before you get to your actual computation. With Warppy, the same setup is 10 lines:

```python
shader = (
    ShaderBuilder()
    .bind_uniform(0, 0, Params)
    .bind_storage(0, 1, np.uint32)
    .workgroup_size(256)
    .kernel(WGSL)
    .build()
)
result = shader.dispatch(params=Params(...), inputs=[output_buf])
```

**Benchmark — card draw Monte Carlo, Apple M1 Max (GPU vs. best-effort vectorized NumPy):**

| Trials | CPU (NumPy) | GPU (Warppy) | Speedup |
|--------|-------------|--------------|---------|
| 1M     | 2,028 ms    | 77 ms        | 26×     |
| 10M    | ~17,500 ms  | 88 ms        | **~200×** |

The GPU time barely increases from 1M to 10M — the parallelism is saturated, so more work costs almost nothing. The CPU scales linearly. The gap widens with trial count and kernel complexity.

> Pi estimation (simple random sampling): NumPy's SIMD is already near-optimal for that shape of problem. GPU advantage shows on compute-intensive, hard-to-vectorize kernels like the shuffle simulation above.

---

## Install

```bash
pip install warppy
```

Requires Python 3.12+. GPU backend via wgpu-py (Metal on macOS, Vulkan/DX12 on Linux/Windows).

---

## Quick Start

```python
from dataclasses import dataclass
import numpy as np
from warppy import ShaderBuilder

@dataclass
class Params:
    count: np.uint32

WGSL = """
struct Params { count: u32 }

@group(0) @binding(0) var<uniform>            params: Params;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x < params.count { output[gid.x] = gid.x * 2u; }
}
"""

N = 1_000_000
output_buf = np.zeros(N, dtype=np.uint32)

shader = (
    ShaderBuilder()
    .bind_uniform(0, 0, Params)
    .bind_storage(0, 1, np.uint32)
    .workgroup_size(256)
    .kernel(WGSL)
    .build()
)

result = shader.dispatch(params=Params(count=np.uint32(N)), inputs=[output_buf])
print(result.data[:5])        # [0 2 4 6 8]
print(f"{result.elapsed_ms:.1f} ms")
```

---

## API

### `ShaderBuilder`

| Method | Description |
|--------|-------------|
| `.bind_uniform(group, binding, dataclass_type)` | Uniform buffer backed by a `@dataclass` |
| `.bind_storage(group, binding, dtype)` | Read-write storage buffer for numpy arrays |
| `.workgroup_size(n)` | Threads per workgroup — must match `@workgroup_size` in WGSL |
| `.kernel(wgsl)` | WGSL compute shader source |
| `.build()` | Validate and compile — returns `CompiledShader` |

### `CompiledShader.dispatch(params, inputs)`

Returns a `GPUResult` with:
- `data` — numpy array (first storage buffer output)
- `elapsed_ms` — dispatch wall time in milliseconds
- `invocations` — total threads launched

### Error Classes

| Class | When raised |
|-------|-------------|
| `GPUCompileError` | Invalid WGSL at build or dispatch time |
| `GPUTypeError` | Unsupported numpy dtype or bad dataclass field |
| `GPUBindingError` | Duplicate or invalid binding configuration |
| `GPUConfigError` | Missing kernel, workgroup size, or bindings |

All errors include an actionable message and a link to GitHub issues.

---

## Supported Platforms

| Platform | Backend |
|----------|---------|
| macOS | Metal |
| Linux | Vulkan |
| Windows | DX12 / Vulkan |

---

## Development

```bash
git clone https://github.com/warppy/warppy
cd warppy
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/            # unit tests (no GPU required)
python examples/spike_card_draw_raw.py   # Phase 0 validation spike
```

---

## Roadmap

- **V1 (current):** Fluent builder API with raw WGSL kernels
- **V2:** `@gpu_kernel` decorator — write GPU kernels as annotated Python functions
- **V3:** Learned translation layer for patterns outside the V2 supported subset

See [warppy_roadmap.md](warppy_roadmap.md) for full details.

---

## License

MIT

# Changelog

All notable changes to warppy will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [0.1.0] — 2026-05-19

First public release.

### Added

**Core API**
- `ShaderBuilder` — fluent builder for GPU compute shaders with method chaining
- `.bind_uniform(binding, dataclass_type)` — uniform buffer backed by a `@dataclass`; group defaults to 0
- `.bind_uniform(group, binding, dataclass_type)` — same with explicit bind group (advanced use)
- `.bind_storage(binding, dtype)` — read-write storage buffer for numpy arrays; group defaults to 0
- `.bind_storage(group, binding, dtype)` — same with explicit bind group (advanced use)
- `.workgroup_size(n)` — threads per workgroup (must match WGSL `@workgroup_size`)
- `.kernel(wgsl)` — raw WGSL compute shader source
- `.build()` — validates configuration and returns a `CompiledShader`
- `CompiledShader.dispatch(params, inputs)` — executes the shader and returns a `GPUResult`
- `GPUResult` dataclass with `data` (numpy array), `elapsed_ms` (float), `invocations` (int)

**Type System**
- numpy dtype ↔ WGSL scalar type mapping: `np.uint32` ↔ `u32`, `np.int32` ↔ `i32`, `np.float32` ↔ `f32`, `np.float16` ↔ `f16`
- Dataclass serialization to WGSL-compatible uniform buffer bytes with correct alignment and 16-byte struct padding

**Error Classes**
- `GPUCompileError` — invalid WGSL at build or dispatch time
- `GPUTypeError` — unsupported numpy dtype or unserializable dataclass field
- `GPUBindingError` — duplicate or invalid (group, binding) pair
- `GPUConfigError` — missing kernel, workgroup size, or bindings at `.build()` time
- All errors include: what went wrong, where, what to do, and a link to GitHub issues

**Backend**
- `wgpu_backend.py` isolation — all wgpu-py calls in a single file; no other module imports wgpu directly
- Graceful naga fallback: warns once at `.build()` time if naga is not installed, continues without early validation

**Examples**
- `examples/spike_card_draw_raw.py` — raw wgpu-py card draw (Phase 0 baseline, ~36 lines of boilerplate)
- `examples/card_draw.py` — same simulation via warppy API (10 lines of GPU setup)
- `examples/pi_estimation.py` — Monte Carlo π estimation at 10M trials

**Infrastructure**
- `pyproject.toml` with hatchling build backend, PyPI classifiers, and `[dev]` extras
- GitHub Actions CI: unit tests on Python 3.12 on every push (no GPU required)
- MIT license

### Benchmarks (Apple M1 Max)

Card draw Monte Carlo (Fisher-Yates shuffle, compute-intensive):

| Trials | CPU (NumPy) | GPU (Warppy) | Speedup |
|--------|-------------|--------------|---------|
| 1M     | 2,028 ms    | 77 ms        | 26×     |
| 10M    | ~17,500 ms  | 88 ms        | ~200×   |

Pi estimation (simple random sampling): NumPy SIMD is near-optimal; GPU advantage is minimal for this kernel shape.

### Known Limitations

- Single bind group only: all bindings must be in group 0. Multi-group support is planned for a future release.
- Maximum dispatch: 65,535 workgroups per dimension (~16.7M threads at workgroup size 256). Larger dispatches require multiple calls.
- Synchronous dispatch only. Async support is V2+.
- WGSL validation at `.build()` time requires naga (not yet available as a stable Python package).

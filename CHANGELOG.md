# Changelog

All notable changes to warppy will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added
- Project skeleton: `ShaderBuilder`, `CompiledShader`, `GPUResult`
- Binding system: `bind_uniform()` and `bind_storage()` with (group, binding) addressing
- Type system: numpy dtype ↔ WGSL scalar mapping, dataclass serialization
- Error classes: `GPUCompileError`, `GPUTypeError`, `GPUBindingError`, `GPUConfigError`
- `wgpu_backend.py` isolation — all wgpu-py calls in a single file
- Phase 0 validation spike: raw wgpu-py card draw Monte Carlo on Metal (Apple M1 Max)

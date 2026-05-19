# Contributing to Warppy

Thanks for your interest in contributing. Warppy is a young library with tight scope — read this before opening a PR.

---

## Before You Start

**Check the scope.** Warppy V1 deliberately does not include:
- Annotated Python function compilation (that is V2)
- Async dispatch
- Custom WGSL type definitions
- Shipped standard shader libraries
- Telemetry of any kind

If your idea touches one of these, open an issue first to discuss whether it belongs in V1, V2, or not at all.

**Open an issue before large PRs.** Especially for new API surface. A PR that changes the public API without prior discussion is likely to be declined.

---

## Setup

```bash
git clone https://github.com/warppy/warppy
cd warppy
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Run unit tests (no GPU required):
```bash
pytest tests/
```

Run the GPU examples (requires a WebGPU-capable device):
```bash
python examples/spike_card_draw_raw.py   # Phase 0 baseline
python examples/card_draw.py             # warppy API
python examples/pi_estimation.py         # correctness check
```

---

## What Good Contributions Look Like

**Bug fixes:** Include a test that fails before the fix and passes after. Describe the platform and wgpu-py version in the PR.

**New dtype support:** Add an entry to `_DTYPE_MAP` in `types.py`, add unit tests in `tests/test_types.py`, and verify the WGSL type is actually usable in a compute shader on at least one platform.

**Error message improvements:** Every error must answer: what went wrong, where, what to do. Keep the GitHub issues link.

**Documentation:** README and docstrings only. Don't create additional docs files without discussion.

---

## Code Style

- Ruff enforces formatting and linting: `ruff check src/ tests/`
- Type hints on all public functions
- No comments unless the WHY is non-obvious
- No emojis in code

---

## The `wgpu_backend.py` Rule

**All wgpu-py imports must stay inside `wgpu_backend.py`.** No exceptions. If you need a wgpu-py type in another module for type checking, use `TYPE_CHECKING` and a string annotation.

This isolation is intentional and non-negotiable. It protects the rest of the library from wgpu-py API changes and keeps unit tests GPU-free.

---

## Issue Reports

Include in every bug report:
- Platform (macOS / Linux / Windows) and GPU model
- Python version
- wgpu version (`python -c "import wgpu; print(wgpu.__version__)"`)
- Full error traceback
- Minimal reproducer

---

## Platform Coverage

Warppy targets macOS (Metal), Linux (Vulkan), and Windows (DX12/Vulkan). If you only have access to one platform, say so in the PR and we'll try to get it reviewed on others.

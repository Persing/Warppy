# Warppy Roadmap

*Python GPU shader compilation made accessible*

---

## V1 — Builder + String Kernel

**Goal:** Ship a real, usable tool. Solve the wgpu-py boilerplate problem. Get it in people's hands.

### Core Components

**ShaderBuilder**
- Fluent API with method chaining, terminated by `.build()`
- Internal state management across chained calls
- `.build()` triggers validation and returns a `CompiledShader`

**Binding System**
- `bind_uniform(group, binding, dataclass_type)`
- `bind_storage(group, binding, numpy_dtype)`
- Validation that groups and bindings do not conflict

**Kernel Dispatch**
- `.dispatch(params)` method on compiled shader
- Buffer allocation and GPU memory transfer
- Workgroup sizing calculation
- Timing instrumentation (elapsed_ms)

**Result Object**
- `GPUResult` dataclass with `data`, `elapsed_ms`, `invocations`
- Clean numpy array unwrapping

**Error Handling**
- WGSL validation via naga at `.build()` time
- Graceful fallback if naga is not installed — warns once, continues
- Four error classes: `GPUCompileError`, `GPUTypeError`, `GPUBindingError`, `GPUConfigError`
- Every error answers: what went wrong, where, and what to do about it
- All errors point to GitHub issues — no telemetry, no phone-home

**Type System**
- numpy dtype to WGSL type mapping (`np.uint32` ↔ `u32`, etc.)
- Dataclass to WGSL struct serialization
- WGSL uniform buffer alignment and padding handling (critical — silent corruption if wrong)

**WGPU Backend**
- `wgpu_backend.py` isolation — no other module imports wgpu-py directly
- Buffer allocation, shader compilation pipeline, dispatch, readback, synchronization

### Build Phases

| Phase | Name | Duration | Go/No-Go Gate |
|-------|------|----------|---------------|
| 0 | Validation Spike | 1-2 days | wgpu-py runs and produces correct output on your machine |
| 1 | Project Skeleton | 0.5 days | `import warppy` succeeds, CI runs |
| 2 | Type System | 2-3 days | Structs serialize with correct WGSL alignment |
| 3 | Builder Core | 2-3 days | Builder stores state correctly across chained calls |
| 4 | WGPU Backend | 3-4 days | Card draw dispatch produces correct output |
| 5 | Result + Errors | 1-2 days | All four error classes raise with correct message format |
| 6 | Validation + Docs | 2-3 days | Someone unfamiliar can install and run the example |

**Total estimated time:** 3-4 weeks at a reasonable pace.

### Test Strategy

Three levels required — do not skip the third.

- **Unit tests** — builder state, type mapping, error messages, struct serialization. No GPU required. Run on every commit.
- **Integration tests** — real GPU dispatch. Card draw, pi estimation. Run before merge.
- **Correctness tests** — known mathematical results compared to GPU output. This is the proof the GPU is computing correctly, not just running without errors.

### V1 Success Criteria

- [ ] Card draw Monte Carlo produces output within 0.5% of expected probability (~39%)
- [ ] Pi estimation is within 0.01 of `math.pi` at 10M iterations
- [ ] GPU dispatch is measurably faster than CPU numpy at 100M+ iterations
- [ ] All four error classes raise with clear, actionable messages
- [ ] naga validates at `.build()` time and falls back cleanly when not installed
- [ ] `pip install warppy` completes in under 60 seconds
- [ ] README example runs successfully for someone who has never seen warppy before
- [ ] Works on macOS, Linux, and Windows

---

## V1 → V2 Decision Gate

After V1 ships, answer these questions before starting V2:

- **Are people using it?** Check GitHub stars, issues, and community mentions.
- **What patterns are people writing?** The errors and issues tell you what the builder cannot handle and what V2 needs to support.
- **Is the builder itself stable?** V2 builds on V1. Fix architectural problems before layering a transpiler on top.

If nobody is using V1, V2 saves nobody any time. Ship V1 first. Let real usage inform V2 scope.

---

## V2 — Annotated Python Function Compilation

**Goal:** Let users write GPU kernels as typed Python functions instead of raw WGSL strings. No new language to learn — just annotated Python.

### Core Components

**Python AST Walker**
- Extract function signature (parameters, return type)
- Validate all parameters and local variables have type annotations — enforced, not optional
- Walk function body AST nodes
- Type checking and propagation at decoration time

**Code Generation**
- Map Python AST nodes to WGSL statements
- Handle control flow: `if/else`, `for` loops with `range()`, `break`, `return`
- Handle array indexing
- Handle function calls — verify the called function is registered in the shader's function registry

**Type System Expansion**
- Full numpy scalar type support (`u32`, `i32`, `f32`, `f16`, etc.)
- `Array[T, N]` type hint support
- Literal context inference (integer literal `13` in a `u32` context becomes `u32(13)`)
- Clear type mismatch errors caught at decoration time

**Decorator System**
- `@gpu_kernel` decorator validates annotations at decoration time
- Returns a callable that can be passed directly to `.kernel()` on the builder
- Caches compiled WGSL — no recompilation on repeated dispatch

**Integration with Builder**
- `ShaderBuilder.kernel(my_function)` accepts annotated functions or raw strings
- Function registry for resolving calls inside kernels
- Automatic WGSL struct generation from dataclasses used in function signatures

### Supported Python Subset

```
Supported:
  if / else
  for i in range(n)
  array indexing and assignment
  arithmetic operators: + - * / % **
  bitwise operators: ^ | & << >>
  type casting: np.uint32(x), np.float32(x)
  function calls to registered functions
  early return

Not supported (raises TranspileError with clear message):
  list comprehensions
  recursion
  dynamic dispatch (calling functions via variables)
  classes and methods
  generators
  exceptions
  any Python stdlib import
```

### Example

```python
@gpu_kernel
def card_sim(idx: np.uint32, params: Params, deck: Array[np.uint32, 52]) -> None:
    if idx >= params.num_trials:
        return

    rng: np.uint32 = params.rng_seed ^ (idx * np.uint32(2654435761))

    for i in range(52):
        deck[i] = np.uint32(i % 13)

    for i in range(params.hand_size):
        j: np.uint32 = i + rand_range(rng, np.uint32(52) - i)
        tmp: np.uint32 = deck[i]
        deck[i] = deck[j]
        deck[j] = tmp

shader = (ShaderBuilder()
    .bind_uniform(0, Params)
    .bind_storage(1, np.uint32)
    .workgroup_size(256)
    .kernel(card_sim))       # annotated function passed here
    .build())
```

### Build Phases

| Phase | Name | Duration | Go/No-Go Gate |
|-------|------|----------|---------------|
| 0 | AST Walker Spike | 1-2 days | Can parse and print a simple typed function |
| 1 | Type Checker | 2-3 days | Type mismatches caught at decoration time |
| 2 | Control Flow Codegen | 2-3 days | if/else and for loops emit correct WGSL |
| 3 | Full Function Compilation | 2-3 days | Card draw function compiles and dispatches correctly |
| 4 | Error Layer | 1-2 days | Unsupported patterns raise TranspileError with clear message |
| 5 | Docs + Examples | 1-2 days | V2 card draw example runs from README |

**Total estimated time:** 6-8 weeks.

### V2 Success Criteria

- [ ] Card draw example works using `@gpu_kernel` decorated function
- [ ] `for` loops, `if/else`, array ops, and arithmetic compile correctly
- [ ] Type errors caught at decoration time, not dispatch time
- [ ] Unsupported patterns raise `TranspileError` with a message pointing to the supported subset
- [ ] Raw string kernel still works — V2 is additive, not a replacement
- [ ] 80% of common scientific compute patterns supported

---

## V2 → V3 Decision Gate

After V2 ships, evaluate before considering a learned translation layer:

- **What patterns are failing TranspileError?** The GitHub issues will show you what the supported subset is missing.
- **Is the subset good enough for most users?** If 90% of use cases work, V3 may not be worth the investment.
- **Is there appetite for a research component?** V3 is a different kind of project — part compiler, part ML. Be honest about the scope before starting.

---

## V3 — Learned Translation Layer (Exploratory)

**Goal:** Use a trained model to handle Python patterns that fall outside the deterministic transpiler's supported subset. Not a replacement for V2 — a fallback and extension layer.

**Status:** Do not start until V2 ships and V2 usage patterns are well understood.

### Concept

The V2 transpiler handles common patterns deterministically. V3 adds a learned model as a second pass for patterns the transpiler cannot handle:

```
Python function
     ↓
V2 transpiler (deterministic, fast, covers 80% of cases)
     ↓ (on TranspileError)
V3 model (learned, slower, handles edge cases)
     ↓
WGSL output (validated by naga before use)
```

### Data Strategy

V1 and V2 are the data generators. Every successful V2 compilation is a training pair:

```
Python function (typed, annotated)  →  WGSL output (correct, validated)
```

The quality of the deterministic transpiler caps the quality of the initial training data. Manual curation of 100-500 carefully translated examples is required to push the model beyond what V2 can already do.

### Architecture Candidates

- **Seq2seq transformer** with cross-attention over source Python tokens and partially-generated WGSL — standard approach, well understood
- **EBM as reranker** — generate N WGSL candidates from a simpler model, score each for validity and semantic equivalence, return the lowest-energy valid output. More tractable than pure EBM generation.

### Hard Requirements

- naga validation runs on every model output before it is returned to the user — no unvalidated WGSL ever reaches the GPU
- `TranspileError` still raised with a clear message when both V2 and V3 fail — no silent fallback to broken output
- Annotations still required — V3 does not remove the type annotation requirement, it extends what annotated code can be compiled

### Key Risk

Training data quality is the ceiling. If the training data only covers patterns V2 already handles deterministically, V3 adds no value. The manual curation phase is the investment that determines whether V3 is worth building.

**Defer all V3 decisions until V2 is shipping and generating real training data.**

---

## Project-Wide Quality Bar

Applies to every version.

- Every public function has a docstring
- Type hints throughout the codebase
- 100% of public API covered by tests
- CI passes on every push (GitHub Actions)
- Changelog maintained from day one
- Semantic versioning from first release
- Issue templates capture platform, Python version, wgpu-py version
- Contributing guide present from V1
- README leads with the benchmark number — GPU vs CPU speedup is the headline

---

## OSS Adoption Strategy

- PyPI package with correct classifiers (`Scientific/Engineering`, `GPU`)
- README benchmark is the hook — show the speedup number in the first screen
- Keywords: `Python GPU compute`, `WGSL Python`, `Monte Carlo GPU`, `wgpu Python`
- Answer Stack Overflow and Reddit questions about GPU simulation in Python with warppy examples
- One well-written blog post demonstrating the card draw benchmark outperforms any SEO work
- Let GitHub issues be the feedback channel — loud descriptive errors drive issue creation, issue creation drives roadmap

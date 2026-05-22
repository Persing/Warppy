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

**Goal:** Let users write GPU kernels as typed Python functions instead of raw WGSL strings. No new language to learn — just annotated Python. Each sub-version is independently shippable and additive.

### Foundational Design Decisions (apply to all V2.x)

**Strict type annotations are required and enforced.**
All function parameters and local variables must have explicit type annotations. The transpiler does not infer types — it validates them. This is a feature, not a limitation. Silent type coercion causes data corruption that is extremely hard to debug after the fact.

**Mixed-type operations are always errors.**
```python
x: u32 = a + b_f32                  # GPUTypeError — must cast explicitly
x: u32 = a + np.uint32(b_f32)       # correct
```

**Buffer parameters in helper functions are declarative, not passed.**
Buffer parameters (Array[T, N], storage buffers) in helper functions document what data the function accesses. The transpiler strips them from the WGSL function signature. The function accesses the buffer directly via its binding. Scalar parameters are passed normally. This eliminates WGSL pointer restriction issues and works consistently in all contexts including loops with unknown iteration counts.

```python
# What you write
@gpu_kernel
def helper(data: Array[f32, 256], idx: u32) -> f32:
    return data[idx] * 2.0

# What WGSL receives
# fn helper(idx: u32) -> f32 { return data[idx] * 2.0; }
```

**All types must be fixed-size.**
Variable-length strings, dynamic arrays, sets, dicts, and any object that can change size at runtime are not supported. GPU memory is pre-allocated and fixed.

**Supported Python subset (all V2.x):**
```
Supported:
  if / else
  for i in range(n)
  array indexing and assignment
  arithmetic operators:  + - * / % **
  bitwise operators:     ^ | & << >>
  type casting:          np.uint32(x), np.float32(x)
  early return
  local variable declarations with type annotations

Not supported (raises TranspileError with clear message):
  list comprehensions
  recursion (also illegal in WGSL)
  dynamic dispatch (calling functions via variables)
  classes and methods
  generators
  exceptions / try-except
  variable-length containers (list, dict, set)
  any dynamic memory allocation
```

---

### V2.1 — Single Annotated Function

**Goal:** Let users write a single GPU kernel as an annotated Python function. No external calls. No helper functions. Just clean annotated Python that compiles to WGSL.

**Components:**
- `@gpu_kernel` decorator validates annotations at decoration time
- AST walker using visitor pattern
- Strict type checker — no inference, no coercion
- Codegen for supported Python subset
- Compiled WGSL cached — no recompilation on repeated dispatch
- `ShaderBuilder.kernel()` accepts annotated function or raw string
- Raw string kernel still works — V2.1 is additive

**Native types supported:**
```
u32, i32, f32
Array[T, N]  — fixed-size only, N must be a compile-time constant
Dataclass structs — for uniform buffer params
```

**Example:**
```python
@gpu_kernel
def card_sim(idx: u32, params: Params, deck: Array[u32, 52]) -> None:
    if idx >= params.num_trials:
        return

    rng: u32 = params.rng_seed ^ (idx * u32(2654435761))

    for i in range(52):
        deck[i] = u32(i % 13)

    for i in range(params.hand_size):
        j: u32 = i + rand_range(rng, u32(52) - i)
        tmp: u32 = deck[i]
        deck[i] = deck[j]
        deck[j] = tmp

shader = (ShaderBuilder()
    .bind_uniform(0, Params)
    .bind_storage(1, np.uint32)
    .workgroup_size(256)
    .kernel(card_sim)
    .build())
```

**V2.1 Success Criteria:**
- [ ] Card draw example works using `@gpu_kernel` decorated function
- [ ] `for` loops, `if/else`, array ops, arithmetic compile correctly
- [ ] Type errors caught at decoration time, not dispatch time
- [ ] Unsupported patterns raise `TranspileError` with clear message
- [ ] Raw string kernel still works unchanged

---

### V2.2 — WGSL Escape Hatch + Standard Library

**Goal:** Give users an escape valve for patterns the transpiler does not yet support, and provide a standard library of common GPU functions so users stay in Python without needing to know WGSL.

**WGSL Escape Hatch:**
Allows hand-written WGSL functions to be registered with the builder and called from annotated Python kernels. Useful when warppy does not yet support a pattern or when hand-optimization of a specific function is needed.

This is not a solution for using external Python libraries. External libraries cannot be transpiled regardless of how they are wrapped.

```python
shader = (ShaderBuilder()
    .bind_uniform(0, Params)
    .bind_storage(1, np.float32)
    .register_wgsl('fast_inverse_sqrt', """
        fn fast_inverse_sqrt(x: f32) -> f32 {
            // hand-optimized WGSL implementation
        }
    """)
    .workgroup_size(256)
    .kernel(my_kernel)
    .build())
```

**Standard Library:**
A curated set of GPU functions implemented internally in WGSL, exposed as Python-callable names. Users write familiar Python and numpy-style calls. The transpiler maps them to stdlib implementations automatically. Users never see or write WGSL.

```python
# User writes
result: f32 = np.sqrt(x)
clamped: f32 = np.clip(value, 0.0, 1.0)

# Transpiler maps to stdlib WGSL implementations automatically
```

**`.use_stdlib()` builder method:**
```python
shader = (ShaderBuilder()
    .use_stdlib('math')
    .bind_uniform(0, Params)
    .kernel(my_kernel)
    .build())
```

**Stdlib rollout is a separate planning phase.** See Stdlib Planning section below.

**V2.2 Success Criteria:**
- [ ] WGSL escape hatch registers and calls correctly from annotated kernels
- [ ] numpy math builtins recognized and mapped to stdlib automatically
- [ ] `.use_stdlib()` loads correct functions into shader
- [ ] Missing stdlib functions raise clear error with feature request link

---

### V2.3 — Extended Type System

**Goal:** Support scientific computing precision requirements. Float64, Int64, Float128, Int128 via software emulation using multi-word arithmetic. Users declare the type they need — warppy handles the GPU representation transparently.

**The problem:**
Python supports arbitrary precision natively. Most GPUs do not support f64 or i64 as hardware types. Warppy solves this using multi-word arithmetic emulation. The user declares `Float64` and warppy emits correct two-f32 WGSL operations underneath.

**Emulated types:**
```
Float32   — native f32 (1 word)
Float64   — f32x2, double-double arithmetic (2 words)
Float128  — f32x4, quad-double arithmetic (4 words)
Int64     — u32x2, two-word integer arithmetic (2 words)
Int128    — u32x4, four-word integer arithmetic (4 words)
```

**Usage:**
```python
@gpu_kernel
def high_precision_sim(x: Float64, y: Float64) -> Float64:
    result: Float64 = x * y    # emits f32x2_mul(x, y) in WGSL
    return result
```

**Operation dispatch:**
Every arithmetic operation on emulated types automatically selects the correct multi-word WGSL implementation. Stdlib functions are injected automatically when emulated types are used.

**Precision note (documented clearly in README):**
Float64 emulation uses double-double arithmetic and provides approximately equivalent precision to hardware f64 but may differ in edge cases (overflow, NaN, denormals). Not a guaranteed IEEE 754 f64 replacement.

**V2.3 Success Criteria:**
- [ ] Float64, Int64, Float128, Int128 usable in annotated kernels
- [ ] Arithmetic on emulated types produces correct results
- [ ] Precision within documented bounds vs hardware types
- [ ] Alignment handled automatically in uniform structs
- [ ] Clear error when emulated type used in unsupported context

---

### V2.4 — Multi-Function Annotation

**Goal:** Let users structure GPU code across multiple annotated Python functions. Clean code organization without hitting WGSL pointer restrictions.

**How it works:**
The buffer parameter rule (see foundational design decisions) handles WGSL pointer restrictions entirely. Helper functions strip buffer params from their WGSL signatures. Scalars pass normally. This works in all contexts including unbounded loops.

**Function registry:**
All helper functions are registered at decoration time. The transpiler validates call signatures and type compatibility across function boundaries at decoration time.

**Cycle detection:**
Mutual recursion detected at decoration time and raises `GPUConfigError` before any WGSL is generated.

**Call depth:**
Soft warning and hard limit based on transpiler performance profiling. Exact values determined by profiling spike before implementation.

**Example:**
```python
@gpu_kernel
def normalize(data: Array[f32, 256], idx: u32) -> f32:
    val: f32 = data[idx]
    return val / length(val)

@gpu_kernel
def process(idx: u32, params: Params, data: Array[f32, 256]) -> None:
    data[idx] = normalize(data, idx)   # buffer param stripped in WGSL
```

**V2.4 Success Criteria:**
- [ ] Helper functions called correctly from main kernel
- [ ] Buffer params stripped from WGSL signatures automatically
- [ ] Works correctly inside loops with unknown bounds
- [ ] Type mismatches across function boundaries caught at decoration time
- [ ] Cycle detection raises clear error
- [ ] Call depth warning and hard limit enforced

---

### V2.5 — Safety Layer

**Goal:** Detect GPU-unsafe patterns at decoration time and raise clear, actionable errors before users waste time debugging mysterious GPU failures.

**Detected patterns:**
```
Network:          requests, socket, urllib, httpx
File I/O:         open(), os, pathlib, sys
Time-dependent:   time, datetime
Concurrency:      threading, multiprocessing
Dynamic code:     eval, exec, compile, __import__
Reflection:       getattr, setattr
Python I/O:       print, input
Async:            async def, await, yield
Context managers: with statements
Exceptions:       try, except, raise
Recursion:        direct or via cycle
Variable-length:  list, dict, set, str as param or local type
```

**Detection is best-effort, not exhaustive.**
Dynamic imports and reflection-based dispatch cannot always be detected statically. The safety layer catches common cases. Naga validation at `.build()` time provides a second safety layer.

**Error format:**
```
GPUUnsafeError: Line 8: unsafe pattern detected — network call (requests.get).
GPUs cannot make network requests.
Move data fetching outside the kernel and pass results as a parameter.
Open an issue: github.com/you/warppy/issues
```

**V2.5 Success Criteria:**
- [ ] All listed patterns detected at decoration time
- [ ] Clear error messages with actionable fix for each pattern
- [ ] No false positives on safe code
- [ ] Detection runs in under 10ms for typical kernel functions

---

## V2 → V3 Decision Gate

After V2.x ships, answer these questions before starting V3:

- **Are workgroup memory patterns showing up in GitHub issues?** If users are not asking for thread coordination, V3 may not be necessary yet.
- **Is multi-pass kernel chaining a real bottleneck?** If users are working around it with CPU round-trips, that is the signal.
- **Is V2 itself stable?** V3 builds on V2's function registry and type system. Fix V2 issues first.

---

## V3 — Workgroup Memory + Atomics + Multi-Pass

**Goal:** Enable thread coordination patterns required for advanced parallel algorithms. Reductions, histograms, sorting, convolution. These require shared memory, atomic operations, and synchronization primitives.

### Why V3 is a major version

V1 and V2 kernels are embarrassingly parallel — each thread is independent. V3 allows threads to communicate and coordinate. This changes the programming model significantly:
- New memory scope (workgroup)
- Synchronization requirements (barriers)
- New class of algorithms (reductions, prefix sums, histograms)

### Workgroup Shared Memory

Declared via the builder. Memory layout remains a builder concern — consistent with V1 design philosophy.

```python
shader = (ShaderBuilder()
    .bind_uniform(0, Params)
    .bind_storage(1, np.float32)
    .workgroup_memory(np.float32, 256)
    .workgroup_size(256)
    .kernel(my_reduction)
    .build())
```

**Structured two-phase API:**
Workgroup memory access is structured into explicit fill and read phases. Barriers are inserted automatically between phases. Users never place barriers manually — correct synchronization is guaranteed by construction.

```python
@gpu_kernel
def reduction(data: Array[f32, 1024], result: Array[f32, 1]) -> None:
    # Phase 1 — fill shared memory
    shared: Workgroup[f32, 256]
    shared[local_id()] = data[global_id()]
    # barrier auto-inserted here

    # Phase 2 — read shared memory (safe by construction)
    if local_id() == 0:
        total: f32 = 0.0
        for i in range(256):
            total += shared[i]
        result[workgroup_id()] = total
```

### Atomics

Thread-safe operations on shared values. Required for any pattern where multiple threads write to the same memory location.

```python
# Without atomics — race condition, wrong result
result[0] += 1

# With atomics — correct
atomic_add(result, 0, u32(1))
```

**Supported atomic operations:**
```
atomic_add, atomic_sub
atomic_min, atomic_max
atomic_and, atomic_or, atomic_xor
atomic_exchange, atomic_compare_exchange
```

### Multi-Pass Kernel Chaining

Output of one kernel dispatch feeds directly into the next without round-tripping through CPU memory.

```python
pipeline = (KernelPipeline()
    .stage(normalize_kernel, inputs=[raw_data], outputs=[normalized])
    .stage(reduction_kernel, inputs=[normalized], outputs=[result])
    .dispatch())
```

### V3 Success Criteria

- [ ] Workgroup memory declared via builder and accessible in kernel
- [ ] Two-phase API enforces correct barrier placement by construction
- [ ] Atomic operations produce correct results under parallel execution
- [ ] Multi-pass pipeline chains correctly without CPU round-trip
- [ ] Parallel reduction example produces mathematically correct result
- [ ] Histogram example produces correct output

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

## Known Constraints and Scope Boundaries

Intentional design decisions, not gaps to fill later.

**Out of scope for all versions:**
- Texture and sampler types (graphics, not compute)
- Vertex and fragment shaders (compute only)
- Async dispatch
- External Python library transpilation
- Inference mode / unannotated function transpilation
- Telemetry of any kind

**GPU type system constraints:**
- All types must be fixed-size and known at compile time
- Variable-length containers are not GPU types
- Overflow behavior on native types follows GPU hardware
- Float64/Int64 emulation approximates but does not guarantee exact IEEE 754

**WGSL constraints handled transparently by warppy:**
- Uniform buffer alignment and padding
- Buffer parameter restrictions in helper functions (buffer param rule)
- Workgroup size limits (warned at build time)
- No recursion (detected at decoration time)

---

## Stdlib Planning

Standard library rollout is a separate planning phase. V2.2 infrastructure supports incremental addition. Rollout order driven by user demand via GitHub issues.

```
Phase 1 (ships with V2.2):
  Math:     sqrt, abs, sin, cos, tan, floor, ceil, round
  Utility:  min, max, clamp, mix (lerp)

Phase 2 (after V2.2, demand-driven):
  Vector:   length, normalize, dot, cross
  Matrix:   mat2, mat3, mat4 operations

Phase 3 (demand-driven):
  Interpolation: smoothstep, bezier
  Noise:         perlin, simplex (feasibility TBD)
```

---

## OSS Adoption Strategy

- PyPI package with correct classifiers (`Scientific/Engineering`, `GPU`)
- README benchmark is the hook — show the speedup number in the first screen
- Keywords: `Python GPU compute`, `WGSL Python`, `Monte Carlo GPU`, `wgpu Python`
- Answer Stack Overflow and Reddit questions about GPU simulation in Python with warppy examples
- One well-written blog post demonstrating the card draw benchmark outperforms any SEO work
- Let GitHub issues be the feedback channel — loud descriptive errors drive issue creation, issue creation drives roadmap

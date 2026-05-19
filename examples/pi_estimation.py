"""Monte Carlo pi estimation — warppy API.

Each GPU thread samples one random point (x, y) in [0,1)² and checks whether
it falls inside the unit circle (x² + y² < 1). The fraction inside × 4 ≈ π.

Expected accuracy at 10M trials: within 0.01 of math.pi (~3.14159).
Standard error at 10M: ≈ 4 × sqrt(π/4 × (1 - π/4) / 10M) ≈ 0.0005
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from warppy import ShaderBuilder


@dataclass
class Params:
    num_trials: np.uint32
    rng_seed:   np.uint32


WGSL = """
struct Params {
    num_trials: u32,
    rng_seed  : u32,
}

@group(0) @binding(0) var<uniform>            params : Params;
@group(0) @binding(1) var<storage, read_write> results: array<u32>;

fn wang_hash(seed: u32) -> u32 {
    var s = seed;
    s = (s ^ 61u) ^ (s >> 16u);
    s = s * 9u;
    s = s ^ (s >> 4u);
    s = s * 0x27d4eb2du;
    s = s ^ (s >> 15u);
    return s;
}

fn xorshift32(state: ptr<function, u32>) -> u32 {
    var s = *state;
    s ^= s << 13u;
    s ^= s >> 17u;
    s ^= s << 5u;
    *state = s;
    return s;
}

// Convert a u32 to a float in [0, 1)
fn to_unit_float(v: u32) -> f32 {
    return f32(v) / 4294967296.0;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.num_trials { return; }

    var rng: u32 = wang_hash(params.rng_seed ^ (idx * 2654435761u));
    if rng == 0u { rng = 1u; }

    let x = to_unit_float(xorshift32(&rng));
    let y = to_unit_float(xorshift32(&rng));

    results[idx] = select(0u, 1u, x * x + y * y < 1.0);
}
"""


def run(num_trials: int = 10_000_000, rng_seed: int = 0xC0FFEE) -> None:
    results_buf = np.zeros(num_trials, dtype=np.uint32)

    shader = (
        ShaderBuilder()
        .bind_uniform(0, 0, Params)
        .bind_storage(0, 1, np.uint32)
        .workgroup_size(256)
        .kernel(WGSL)
        .build()
    )

    result = shader.dispatch(
        params=Params(
            num_trials=np.uint32(num_trials),
            rng_seed=np.uint32(rng_seed),
        ),
        inputs=[results_buf],
    )

    pi_estimate = 4.0 * result.data.sum() / num_trials
    delta = abs(pi_estimate - math.pi)

    print(f"Trials:      {num_trials:,}")
    print(f"π estimate:  {pi_estimate:.6f}")
    print(f"math.pi:     {math.pi:.6f}")
    print(f"Delta:       {delta:.6f}  {'✓ PASS' if delta < 0.01 else '✗ FAIL'}")
    print(f"Elapsed:     {result.elapsed_ms:.1f} ms")

    if delta >= 0.01:
        raise AssertionError(
            f"π estimate {pi_estimate:.6f} differs from math.pi by {delta:.6f} "
            f"(threshold 0.01 at {num_trials:,} trials)"
        )


if __name__ == "__main__":
    run()

# Copyright (c) 2026 Nick Persing
# Licensed under the MIT License. See LICENSE for details.

"""Monte Carlo pi estimation — V2 @gpu_kernel API.

Same algorithm as pi_estimation.py but the shader is written as an annotated
Python function and transpiled to WGSL automatically. No raw WGSL string needed.

Because the V2 transpiler does not yet support custom function definitions, the
wang_hash and xorshift32 helpers are inlined directly into the kernel body.
`select` is a WGSL built-in that passes through the transpiler unchanged.

Expected accuracy at 10M trials: within 0.01 of math.pi (~3.14159).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from warppy import Array, ShaderBuilder, gpu_kernel


@dataclass
class Params:
    num_trials: np.uint32
    rng_seed:   np.uint32


@gpu_kernel
def pi_kernel(idx: np.uint32, params: Params, results: Array[np.uint32]) -> None:
    if idx >= params.num_trials:
        return

    # Wang hash — initialize RNG state from seed and thread index
    rng: np.uint32 = params.rng_seed ^ (idx * np.uint32(2654435761))
    rng = (rng ^ np.uint32(61)) ^ (rng >> np.uint32(16))
    rng = rng * np.uint32(9)
    rng = rng ^ (rng >> np.uint32(4))
    rng = rng * np.uint32(0x27D4EB2D)
    rng = rng ^ (rng >> np.uint32(15))
    if rng == np.uint32(0):
        rng = np.uint32(1)

    # xorshift32 — sample x
    rng ^= rng << np.uint32(13)
    rng ^= rng >> np.uint32(17)
    rng ^= rng << np.uint32(5)
    x: np.float32 = np.float32(rng) / np.float32(4294967296.0)

    # xorshift32 — sample y
    rng ^= rng << np.uint32(13)
    rng ^= rng >> np.uint32(17)
    rng ^= rng << np.uint32(5)
    y: np.float32 = np.float32(rng) / np.float32(4294967296.0)

    # select(false_val, true_val, condition) is a WGSL built-in
    results[idx] = select(np.uint32(0), np.uint32(1), x * x + y * y < np.float32(1.0))


def run(num_trials: int = 10_000_000, rng_seed: int = 0xC0FFEE) -> None:
    results_buf = np.zeros(num_trials, dtype=np.uint32)

    shader = (
        ShaderBuilder()
        .workgroup_size(256)
        .kernel(pi_kernel)
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

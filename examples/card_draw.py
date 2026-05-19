"""Card draw Monte Carlo — warppy API version.

Estimates the probability that a 5-card hand dealt from a shuffled 52-card deck
contains at least one card of a target rank (e.g. Ace).

Analytical answer: 1 - C(48,5) / C(52,5) ≈ 0.3412

This example is the Phase 4 gate for warppy V1. It demonstrates the full
ShaderBuilder API replacing the ~36 lines of boilerplate in the raw wgpu-py
spike (examples/spike_card_draw_raw.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from warppy import ShaderBuilder

# ── Simulation parameters ─────────────────────────────────────────────────────
NUM_TRIALS = 1_000_000
HAND_SIZE = 5
TARGET_RANK = 0       # Aces
RNG_SEED = 0xDEADBEEF
WORKGROUP_SIZE = 256

EXPECTED = 1.0 - math.comb(48, 5) / math.comb(52, 5)


# ── Params dataclass — fields map directly to the WGSL struct ─────────────────
@dataclass
class Params:
    num_trials:  np.uint32
    hand_size:   np.uint32
    target_rank: np.uint32
    rng_seed:    np.uint32


# ── WGSL kernel ───────────────────────────────────────────────────────────────
WGSL = """
struct Params {
    num_trials : u32,
    hand_size  : u32,
    target_rank: u32,
    rng_seed   : u32,
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

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.num_trials { return; }

    var rng: u32 = wang_hash(params.rng_seed ^ (idx * 2654435761u));
    if rng == 0u { rng = 1u; }

    var deck: array<u32, 52>;
    for (var i: u32 = 0u; i < 52u; i++) { deck[i] = i; }

    for (var i: u32 = 51u; i > 0u; i--) {
        let j = xorshift32(&rng) % (i + 1u);
        let tmp = deck[i];
        deck[i] = deck[j];
        deck[j] = tmp;
    }

    var hit: u32 = 0u;
    for (var i: u32 = 0u; i < params.hand_size; i++) {
        if (deck[i] % 13u) == params.target_rank { hit = 1u; }
    }
    results[idx] = hit;
}
"""


def run() -> None:
    results_buf = np.zeros(NUM_TRIALS, dtype=np.uint32)

    shader = (
        ShaderBuilder()
        .bind_uniform(0, 0, Params)
        .bind_storage(0, 1, np.uint32)
        .workgroup_size(WORKGROUP_SIZE)
        .kernel(WGSL)
        .build()
    )

    result = shader.dispatch(
        params=Params(
            num_trials=np.uint32(NUM_TRIALS),
            hand_size=np.uint32(HAND_SIZE),
            target_rank=np.uint32(TARGET_RANK),
            rng_seed=np.uint32(RNG_SEED),
        ),
        inputs=[results_buf],
    )

    probability = result.data.sum() / NUM_TRIALS
    delta = abs(probability - EXPECTED)

    print(f"Trials:      {NUM_TRIALS:,}")
    print(f"Hand size:   {HAND_SIZE}")
    print(f"GPU result:  {probability:.4f}")
    print(f"Expected:    {EXPECTED:.4f}")
    print(f"Delta:       {delta:.4f}  {'✓ PASS' if delta < 0.005 else '✗ FAIL'}")
    print(f"Elapsed:     {result.elapsed_ms:.1f} ms")
    print(f"Invocations: {result.invocations:,}")

    if delta >= 0.005:
        raise AssertionError(
            f"GPU result {probability:.4f} differs from expected "
            f"{EXPECTED:.4f} by {delta:.4f} (threshold 0.005)"
        )


if __name__ == "__main__":
    run()

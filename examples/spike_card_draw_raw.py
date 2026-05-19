# Copyright (c) 2026 Nick Persing
# Licensed under the MIT License. See LICENSE for details.

"""
Phase 0 validation spike — raw wgpu-py card draw Monte Carlo.

This is NOT a warppy file. It is a raw wgpu-py implementation of the card draw
simulation used to:
  1. Verify wgpu-py works on this machine
  2. Document every line of boilerplate warppy must eliminate
  3. Establish the baseline line count for the README benchmark

Simulation: deal `hand_size` cards from a shuffled 52-card deck per trial.
Count how many trials contain at least one card of `target_rank`.
Expected probability (hand_size=5, target_rank=Ace, 4 aces in 52 cards):
  P = 1 - C(48,5) / C(52,5) = 1 - 1712304 / 2598960 ≈ 0.3412

BOILERPLATE LINE COUNT (lines warppy will eliminate):
  Device initialization:           2 lines
  Struct packing:                  1 line
  Uniform buffer creation:         3 lines
  Storage buffer creation:         1 line
  Readback buffer creation:        1 line
  Bind group layout:               7 lines
  Bind group:                      6 lines
  Pipeline layout:                 1 line
  Shader module:                   1 line
  Compute pipeline:                1 line
  Command encoder + dispatch:      6 lines
  Buffer copy + submit:            2 lines
  Readback + numpy conversion:     4 lines
  -------------------------------------------
  Total boilerplate:              ~36 lines

With warppy:
  ShaderBuilder chain + dispatch:  8 lines
"""

import math
import struct
import time

import numpy as np
import wgpu

# ── Simulation parameters ────────────────────────────────────────────────────
NUM_TRIALS = 1_000_000
HAND_SIZE = 5
TARGET_RANK = 0         # Aces
RNG_SEED = 0xDEADBEEF
WORKGROUP_SIZE = 256

# Analytical expected probability
# P(at least one Ace in hand of 5 from 52 cards with 4 Aces)
_c48_5 = math.comb(48, 5)
_c52_5 = math.comb(52, 5)
EXPECTED_PROBABILITY = 1.0 - _c48_5 / _c52_5

# ── WGSL kernel ───────────────────────────────────────────────────────────────
WGSL_KERNEL = """
struct Params {
    num_trials : u32,
    hand_size  : u32,
    target_rank: u32,
    rng_seed   : u32,
}

@group(0) @binding(0) var<uniform>            params  : Params;
@group(0) @binding(1) var<storage, read_write> results: array<u32>;

// Wang hash for seeding — breaks correlation between consecutive thread IDs
fn wang_hash(seed: u32) -> u32 {
    var s = seed;
    s = (s ^ 61u) ^ (s >> 16u);
    s = s * 9u;
    s = s ^ (s >> 4u);
    s = s * 0x27d4eb2du;
    s = s ^ (s >> 15u);
    return s;
}

// Xorshift32 — fast, high-quality 32-bit PRNG; LCG produces correlated shuffles
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
    if idx >= params.num_trials {
        return;
    }

    // Per-thread seed: hash breaks correlation between consecutive invocations
    var rng: u32 = wang_hash(params.rng_seed ^ (idx * 2654435761u));
    if rng == 0u { rng = 1u; }  // xorshift requires non-zero state

    // Build deck: cards 0..51, rank = card % 13
    var deck: array<u32, 52>;
    for (var i: u32 = 0u; i < 52u; i++) {
        deck[i] = i;
    }

    // Fisher-Yates shuffle
    for (var i: u32 = 51u; i > 0u; i--) {
        let j = xorshift32(&rng) % (i + 1u);
        let tmp = deck[i];
        deck[i] = deck[j];
        deck[j] = tmp;
    }

    // Check if target rank appears in first hand_size cards
    var hit: u32 = 0u;
    for (var i: u32 = 0u; i < params.hand_size; i++) {
        if (deck[i] % 13u) == params.target_rank {
            hit = 1u;
        }
    }

    results[idx] = hit;
}
"""


def run() -> None:
    # ── BOILERPLATE START ─────────────────────────────────────────────────────

    # Device initialization (2 lines)
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()

    # Struct packing — Params: num_trials, hand_size, target_rank, rng_seed (1 line)
    params_bytes = struct.pack("<4I", NUM_TRIALS, HAND_SIZE, TARGET_RANK, RNG_SEED)

    # Uniform buffer — upload params (3 lines)
    uniform_buf = device.create_buffer(
        size=len(params_bytes),
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )
    device.queue.write_buffer(uniform_buf, 0, params_bytes)

    # Storage buffer — output results array (1 line)
    results_buf = device.create_buffer(
        size=NUM_TRIALS * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )

    # Readback buffer (1 line)
    readback_buf = device.create_buffer(
        size=NUM_TRIALS * 4,
        usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST,
    )

    # Bind group layout (7 lines)
    bgl = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.storage},
            },
        ]
    )

    # Bind group (6 lines)
    bg = device.create_bind_group(
        layout=bgl,
        entries=[
            {"binding": 0, "resource": {"buffer": uniform_buf}},
            {"binding": 1, "resource": {"buffer": results_buf}},
        ],
    )

    # Pipeline layout (1 line)
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bgl])

    # Shader module (1 line)
    shader_module = device.create_shader_module(code=WGSL_KERNEL)

    # Compute pipeline (1 line)
    pipeline = device.create_compute_pipeline(
        layout=pipeline_layout,
        compute={"module": shader_module, "entry_point": "main"},
    )

    # Command encoder + dispatch (6 lines)
    workgroups = math.ceil(NUM_TRIALS / WORKGROUP_SIZE)
    t_start = time.perf_counter()
    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bg)
    compute_pass.dispatch_workgroups(workgroups)
    compute_pass.end()

    # Buffer copy + submit (2 lines)
    encoder.copy_buffer_to_buffer(results_buf, 0, readback_buf, 0, NUM_TRIALS * 4)
    device.queue.submit([encoder.finish()])

    # Readback + numpy conversion (4 lines)
    readback_buf.map_sync(wgpu.MapMode.READ)
    raw = readback_buf.read_mapped()
    results = np.frombuffer(raw, dtype=np.uint32).copy()
    readback_buf.unmap()
    elapsed_ms = (time.perf_counter() - t_start) * 1000

    # ── BOILERPLATE END ───────────────────────────────────────────────────────

    probability = results.sum() / NUM_TRIALS
    delta = abs(probability - EXPECTED_PROBABILITY)

    print(f"Backend:      {adapter.info['backend_type']}")
    print(f"Device:       {adapter.info['device']}")
    print(f"Trials:       {NUM_TRIALS:,}")
    print(f"Hand size:    {HAND_SIZE}")
    print(f"GPU result:   {probability:.4f}")
    print(f"Expected:     {EXPECTED_PROBABILITY:.4f}")
    print(f"Delta:        {delta:.4f}  {'✓ PASS' if delta < 0.005 else '✗ FAIL'}")
    print(f"Elapsed:      {elapsed_ms:.1f} ms")

    if delta >= 0.005:
        raise AssertionError(
            f"GPU result {probability:.4f} differs from expected "
            f"{EXPECTED_PROBABILITY:.4f} by {delta:.4f} (threshold 0.005)"
        )


if __name__ == "__main__":
    run()

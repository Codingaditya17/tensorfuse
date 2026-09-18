"""
test_correctness_v2.py — unfused / fused-untuned / fused-autotuned
must all agree, on the full 2-layer demo graph (mixed AVX-512-eligible
+ scalar-only chains), across several shapes including non-power-of-2.
"""

import numpy as np
from demo_graph import build_demo_graph
from fusion2 import fuse_elementwise
from executor_v2 import run_unfused, run_fused_untuned, run_fused_autotuned


def make_values(x_name, params, batch, d_in, d_hidden, d_out, seed):
    rng = np.random.default_rng(seed)
    values = {x_name: rng.standard_normal((batch, d_in)).astype(np.float32)}
    values[params["W1"]] = rng.standard_normal((d_in, d_hidden)).astype(np.float32)
    values[params["b1"]] = rng.standard_normal((d_hidden,)).astype(np.float32)
    values[params["W2"]] = rng.standard_normal((d_hidden, d_out)).astype(np.float32)
    values[params["b2"]] = rng.standard_normal((d_out,)).astype(np.float32)
    values[params["scale"]] = rng.standard_normal((d_out,)).astype(np.float32)
    return values


def check(batch, d_in, d_hidden, d_out, seed):
    g, x_name, params, out_name = build_demo_graph()
    fg, n_fused, n_absorbed = fuse_elementwise(g)
    values = make_values(x_name, params, batch, d_in, d_hidden, d_out, seed)

    out_unfused = run_unfused(g, values)[out_name]
    out_untuned = run_fused_untuned(fg, {k: v.copy() for k, v in values.items()})[out_name]
    out_auto = run_fused_autotuned(fg, {k: v.copy() for k, v in values.items()}, verbose=False)[out_name]

    d1 = np.abs(out_unfused - out_untuned).max()
    d2 = np.abs(out_unfused - out_auto).max()
    assert np.allclose(out_unfused, out_untuned, atol=1e-4), f"untuned mismatch: {d1}"
    assert np.allclose(out_unfused, out_auto, atol=1e-4), f"autotuned mismatch: {d2}"
    return n_fused, d1, d2


if __name__ == "__main__":
    cases = [
        (1, 8, 16, 4),
        (8, 32, 64, 16),
        (256, 512, 1024, 256),
        (37, 100, 50, 13),
    ]
    for i, (batch, d_in, d_hidden, d_out) in enumerate(cases):
        n_fused, d1, d2 = check(batch, d_in, d_hidden, d_out, seed=i)
        print(f"case {i}: batch={batch:4d} hidden={d_hidden:4d}  "
              f"fused_nodes={n_fused}  untuned_diff={d1:.1e}  autotuned_diff={d2:.1e}  OK")
    print("\nAll three execution modes agree across all shapes. Fusion + autotuning are semantics-preserving.")

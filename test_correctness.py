"""
test_correctness.py — Fusion must be semantics-preserving.

Runs the same MLP graph through the unfused interpreter and the
fused interpreter (after the fusion pass rewrites it) on random
inputs, at several sizes, and asserts the outputs match to float32
precision. This is the check that matters most in a real compiler:
an optimization that changes results is a bug, not a speedup.
"""

import numpy as np
from ir import build_mlp_graph
from fusion import fuse_bias_relu
from executor import run_unfused, run_fused
from benchmark import make_values


def check(batch, d_in, d_hidden, d_out, seed):
    g, x_name, weights, out_name = build_mlp_graph(2)
    fg, _ = fuse_bias_relu(g)
    values = make_values(g, x_name, weights, batch, d_in, d_hidden, d_out, seed=seed)

    out_unfused = run_unfused(g, values)[out_name]
    out_fused = run_fused(fg, values)[out_name]

    diff = np.abs(out_unfused - out_fused).max()
    assert np.allclose(out_unfused, out_fused, atol=1e-5), (
        f"MISMATCH at batch={batch}, hidden={d_hidden}: max abs diff {diff}"
    )
    return diff


if __name__ == "__main__":
    cases = [
        (1, 4, 8, 2),
        (8, 16, 32, 10),
        (256, 512, 1024, 256),
        (37, 100, 50, 13),   # odd, non-power-of-2 sizes on purpose
    ]
    for i, (batch, d_in, d_hidden, d_out) in enumerate(cases):
        diff = check(batch, d_in, d_hidden, d_out, seed=i)
        print(f"case {i}: batch={batch:4d} d_in={d_in:4d} "
              f"hidden={d_hidden:4d} d_out={d_out:4d}  max_abs_diff={diff:.2e}  OK")
    print("\nAll correctness checks passed: fusion pass is semantics-preserving.")

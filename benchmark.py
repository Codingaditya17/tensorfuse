"""
benchmark.py — Fused vs unfused MLP forward pass, across sizes.

For each (batch, hidden_dim) size we time:
  - run_unfused: MatMul, then a separate Add, then a separate ReLU
    (each of the last two materializes a brand-new array)
  - run_fused:   MatMul, then one compiled pass doing bias+relu in place

Both paths use the exact same MatMul call (numpy/BLAS) — only the
post-matmul elementwise chain differs. This isolates the effect of
fusion instead of conflating it with matmul performance.
"""

import time
import numpy as np
import csv

from ir import build_mlp_graph
from fusion import fuse_bias_relu
from executor import run_unfused, run_fused


def make_values(g, x_name, weights, batch, d_in, d_hidden, d_out, seed=0):
    rng = np.random.default_rng(seed)
    values = {x_name: rng.standard_normal((batch, d_in), dtype=np.float32)}
    dims_in = [d_in, d_hidden]
    dims_out = [d_hidden, d_out]
    for (w_name, b_name), din, dout in zip(weights, dims_in, dims_out):
        values[w_name] = rng.standard_normal((din, dout), dtype=np.float32)
        values[b_name] = rng.standard_normal((dout,), dtype=np.float32)
    return values


def time_fn(fn, graph, values, reps):
    # warmup (also triggers gcc compile / cache warmup, outside timing)
    fn(graph, {k: v.copy() for k, v in values.items()})
    best = float("inf")
    for _ in range(reps):
        vals = {k: v.copy() for k, v in values.items()}
        t0 = time.perf_counter()
        fn(graph, vals)
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return best


def main():
    g, x_name, weights, out_name = build_mlp_graph(2)
    fg, n_fused = fuse_bias_relu(g)
    print(f"Fused {n_fused} Add+ReLU pairs into FusedBiasReLU nodes.\n")

    d_in, d_out = 512, 256
    batch_sizes = [64, 256, 1024, 4096]
    hidden_sizes = [512, 1024, 2048, 4096]

    rows = []
    print(f"{'batch':>7} {'hidden':>7} {'unfused_ms':>12} {'fused_ms':>10} {'speedup':>9}")
    for batch in batch_sizes:
        for hidden in hidden_sizes:
            values = make_values(g, x_name, weights, batch, d_in, hidden, d_out)
            t_unfused = time_fn(run_unfused, g, values, reps=7)
            t_fused = time_fn(run_fused, fg, values, reps=7)
            speedup = t_unfused / t_fused
            rows.append((batch, hidden, t_unfused * 1000, t_fused * 1000, speedup))
            print(f"{batch:7d} {hidden:7d} {t_unfused*1000:12.3f} {t_fused*1000:10.3f} {speedup:8.2f}x")

    with open("results/benchmark.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["batch", "hidden", "unfused_ms", "fused_ms", "speedup"])
        w.writerows(rows)

    return rows


if __name__ == "__main__":
    main()

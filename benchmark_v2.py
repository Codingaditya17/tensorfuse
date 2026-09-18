import time
import csv
import numpy as np

from demo_graph import build_demo_graph
from fusion2 import fuse_elementwise
from executor_v2 import run_unfused, run_fused_untuned, run_fused_autotuned
from test_correctness_v2 import make_values


def time_fn(fn, graph, values, reps):
    fn(graph, {k: v.copy() for k, v in values.items()})  # warmup / compile
    best = float("inf")
    for _ in range(reps):
        vals = {k: v.copy() for k, v in values.items()}
        t0 = time.perf_counter()
        fn(graph, vals)
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    g, x_name, params, out_name = build_demo_graph()
    fg, n_fused, n_absorbed = fuse_elementwise(g)
    print(f"Fused {n_fused} nodes ({n_absorbed} ops absorbed) into the graph.\n")

    d_in, d_out = 512, 256
    configs = [
        (64, 512), (256, 512), (1024, 512), (4096, 512),
        (64, 4096), (256, 4096), (1024, 4096), (4096, 4096),
    ]

    rows = []
    print(f"{'batch':>6} {'hidden':>7} {'unfused_ms':>11} {'untuned_ms':>11} "
          f"{'auto_ms':>9} {'fuse_speedup':>13} {'tune_speedup':>13}")
    for batch, hidden in configs:
        values = make_values(x_name, params, batch, d_in, hidden, d_out, seed=0)
        t_unfused = time_fn(run_unfused, g, values, reps=5)
        t_untuned = time_fn(run_fused_untuned, fg, values, reps=5)
        t_auto = time_fn(lambda gr, v: run_fused_autotuned(gr, v, verbose=False), fg, values, reps=5)

        fuse_speedup = t_unfused / t_untuned
        tune_speedup = t_untuned / t_auto
        rows.append((batch, hidden, t_unfused * 1000, t_untuned * 1000,
                     t_auto * 1000, fuse_speedup, tune_speedup))
        print(f"{batch:6d} {hidden:7d} {t_unfused*1000:11.3f} {t_untuned*1000:11.3f} "
              f"{t_auto*1000:9.3f} {fuse_speedup:12.2f}x {tune_speedup:12.2f}x")

    with open("results/benchmark_v2.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["batch", "hidden", "unfused_ms", "untuned_ms", "auto_ms",
                     "fuse_speedup", "tune_speedup"])
        w.writerows(rows)

    return rows


if __name__ == "__main__":
    main()

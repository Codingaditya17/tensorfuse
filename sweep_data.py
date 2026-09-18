"""
sweep_data.py — generates a broad training set for the learned cost
model: many DIFFERENT op sequences (not just the 2 from before) across
many shapes, measuring whether fusion actually wins at each point.

This is deliberately wider than benchmark_isolated.py's 2 sequences —
the point of the learned model is to generalize to sequences it has
never seen timed directly, so the training set needs sequence
diversity, not just shape diversity.
"""

import csv
import time
import numpy as np

from codegen_v2 import CompiledKernel

SEQUENCES = {
    "Add-ReLU": [("Add", "bias"), ("ReLU", None)],
    "Add-Sigmoid": [("Add", "bias"), ("Sigmoid", None)],
    "Add-Tanh": [("Add", "bias"), ("Tanh", None)],
    "Add-Mul-ReLU": [("Add", "bias"), ("Mul", "scale"), ("ReLU", None)],
    "Mul-Add-ReLU": [("Mul", "scale"), ("Add", "bias"), ("ReLU", None)],
    "Add-Sigmoid-Mul-ReLU": [("Add", "bias"), ("Sigmoid", None), ("Mul", "scale"), ("ReLU", None)],
    "Sub-ReLU": [("Sub", "bias"), ("ReLU", None)],
    "Add-Neg-ReLU": [("Add", "bias"), ("Neg", None), ("ReLU", None)],
    "Add-ReLU-Mul": [("Add", "bias"), ("ReLU", None), ("Mul", "scale")],
}

SHAPES = [(64, 64), (256, 256), (1024, 256), (1024, 1024),
           (4096, 512), (4096, 2048), (4096, 4096), (4096, 8192)]


def numpy_apply(seq, c, operands):
    v = c.copy()
    for kind, name in seq:
        if kind == "Add":
            v = v + operands[name]
        elif kind == "Mul":
            v = v * operands[name]
        elif kind == "Sub":
            v = v - operands[name]
        elif kind == "Neg":
            v = -v
        elif kind == "ReLU":
            v = np.maximum(v, 0)
        elif kind == "Sigmoid":
            v = 1.0 / (1.0 + np.exp(-v))
        elif kind == "Tanh":
            v = np.tanh(v)
    return v


def time_ms(fn, reps=5):
    fn()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1000


def main():
    rows_out = []
    for seq_name, seq in SEQUENCES.items():
        operand_names = sorted({op for _, op in seq if op is not None})
        for rows, cols in SHAPES:
            rng = np.random.default_rng(0)
            c0 = rng.standard_normal((rows, cols)).astype(np.float32)
            operands = {n: rng.standard_normal((cols,)).astype(np.float32) for n in operand_names}
            operand_list = [operands[n] for n in operand_names]

            t_unfused = time_ms(lambda: numpy_apply(seq, c0, operands))
            kernel = CompiledKernel("scalar", 1, seq, f"sweep_{abs(hash((seq_name, cols)))}")
            t_fused = time_ms(lambda: kernel(c0.copy(), operand_list))

            n_ops = len(seq)
            has_transcendental = int(any(k in ("Sigmoid", "Tanh") for k, _ in seq))
            n_elements = rows * cols
            speedup = t_unfused / t_fused
            won = int(speedup >= 1.0)

            rows_out.append((seq_name, rows, cols, n_elements, n_ops,
                             has_transcendental, t_unfused, t_fused, speedup, won))
            print(f"{seq_name:22s} {rows:5d}x{cols:<5d} n_ops={n_ops} "
                  f"trans={has_transcendental}  speedup={speedup:6.2f}x  won={won}")

    with open("results/cost_sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sequence", "rows", "cols", "n_elements", "n_ops",
                     "has_transcendental", "unfused_ms", "fused_ms", "speedup", "won"])
        w.writerows(rows_out)
    print(f"\nSaved {len(rows_out)} rows to results/cost_sweep.csv")


if __name__ == "__main__":
    main()

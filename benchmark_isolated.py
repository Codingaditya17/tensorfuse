"""
benchmark_isolated.py — times ONLY the elementwise post-processing
(bias+relu, or bias+sigmoid+mul+relu), decoupled from the matmul that
precedes it. This is the "clean" comparison: the full-pipeline
benchmark (benchmark_v2.py) shows fusion's contribution getting
diluted by BLAS-dominated matmul time at small hidden sizes; this
isolates exactly what fusion + autotuning buy on the part of the
computation they actually touch.
"""

import csv
import time
import numpy as np

from codegen_v2 import CompiledKernel
from autotune import autotune, _kernel_memo

SIMPLE_SEQ = [("Add", "bias"), ("ReLU", None)]
COMPLEX_SEQ = [("Add", "bias"), ("Sigmoid", None), ("Mul", "scale"), ("ReLU", None)]


def numpy_unfused(c, ops_and_operands):
    v = c.copy()
    for kind, arr in ops_and_operands:
        if kind == "Add":
            v = v + arr
        elif kind == "Mul":
            v = v * arr
        elif kind == "ReLU":
            v = np.maximum(v, 0)
        elif kind == "Sigmoid":
            v = 1.0 / (1.0 + np.exp(-v))
    return v


def time_ms(fn, reps=7):
    fn()  # warmup
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1000


def run_case(seq, seq_label, rows, cols, operand_names):
    rng = np.random.default_rng(0)
    c0 = rng.standard_normal((rows, cols)).astype(np.float32)
    operands = {n: rng.standard_normal((cols,)).astype(np.float32) for n in operand_names}

    ops_and_arrays = [(kind, (operands[op] if op else None)) for kind, op in seq]
    operand_list = [operands[name] for name in operand_names]

    t_unfused = time_ms(lambda: numpy_unfused(c0, ops_and_arrays))

    untuned = CompiledKernel("scalar", 1, seq, f"iso_untuned_{abs(hash((seq_label, cols)))}")
    t_untuned = time_ms(lambda: untuned(c0.copy(), operand_list))

    kernel, info = autotune(seq, rows, cols, operand_names, verbose=False)
    t_auto = time_ms(lambda: kernel(c0.copy(), operand_list))

    return t_unfused, t_untuned, t_auto, kernel.label()


def main():
    _kernel_memo.clear()
    sizes = [(256, 256), (1024, 1024), (4096, 1024), (4096, 4096), (4096, 8192)]
    rows_out = []

    for label, seq, operand_names in [
        ("simple (Add+ReLU)", SIMPLE_SEQ, ["bias"]),
        ("complex (Add+Sigmoid+Mul+ReLU)", COMPLEX_SEQ, ["bias", "scale"]),
    ]:
        print(f"\n=== {label} ===")
        print(f"{'rows':>6} {'cols':>6} {'unfused_ms':>11} {'untuned_ms':>11} "
              f"{'auto_ms':>9} {'winner':>16} {'fuse_speedup':>13} {'tune_speedup':>13}")
        for rows, cols in sizes:
            t_unf, t_unt, t_auto, winner = run_case(seq, label, rows, cols, operand_names)
            fuse_sp = t_unf / t_unt
            tune_sp = t_unt / t_auto
            rows_out.append((label, rows, cols, t_unf, t_unt, t_auto, winner, fuse_sp, tune_sp))
            print(f"{rows:6d} {cols:6d} {t_unf:11.4f} {t_unt:11.4f} {t_auto:9.4f} "
                  f"{winner:>16} {fuse_sp:12.2f}x {tune_sp:12.2f}x")

    with open("results/benchmark_isolated.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sequence", "rows", "cols", "unfused_ms", "untuned_ms", "auto_ms",
                     "winner", "fuse_speedup", "tune_speedup"])
        w.writerows(rows_out)

    return rows_out


if __name__ == "__main__":
    main()

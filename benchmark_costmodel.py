"""
benchmark_costmodel.py — proves the cost model actually earns its
keep: for each (sequence, size), compare
  - always fuse (autotuned)
  - never fuse (plain numpy)
  - cost-aware (picks per the calibrated model)
and show cost-aware matches whichever of the first two was faster,
at every size -- including the sizes where "always fuse" was a loss.
"""

import csv
import time
import numpy as np

from autotune import autotune
from cost_model import CostModel

SIMPLE_SEQ = [("Add", "bias"), ("ReLU", None)]
COMPLEX_SEQ = [("Add", "bias"), ("Sigmoid", None), ("Mul", "scale"), ("ReLU", None)]


def numpy_apply(seq, c, operands):
    v = c.copy()
    for kind, name in seq:
        if kind == "Add":
            v = v + operands[name]
        elif kind == "Mul":
            v = v * operands[name]
        elif kind == "ReLU":
            v = np.maximum(v, 0)
        elif kind == "Sigmoid":
            v = 1.0 / (1.0 + np.exp(-v))
    return v


def time_ms(fn, reps=7):
    fn()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1000


def main():
    cm = CostModel()
    print(cm.explain())
    print()

    sizes = [(256, 256), (1024, 1024), (4096, 1024), (4096, 4096), (4096, 8192)]
    rows_out = []

    for label, seq, operand_names in [
        ("Add+ReLU", SIMPLE_SEQ, ["bias"]),
        ("Add+Sigmoid+Mul+ReLU", COMPLEX_SEQ, ["bias", "scale"]),
    ]:
        print(f"=== {label} ===")
        print(f"{'rows':>6} {'cols':>6} {'always_fuse_ms':>15} {'never_fuse_ms':>14} "
              f"{'cost_aware_ms':>14} {'model_chose':>12} {'optimal?':>9}")
        for rows, cols in sizes:
            rng = np.random.default_rng(0)
            c0 = rng.standard_normal((rows, cols)).astype(np.float32)
            operands = {n: rng.standard_normal((cols,)).astype(np.float32) for n in operand_names}

            kernel, _ = autotune(seq, rows, cols, operand_names, verbose=False)
            operand_list = [operands[n] for n in operand_names]
            t_always = time_ms(lambda: kernel(c0.copy(), operand_list))
            t_never = time_ms(lambda: numpy_apply(seq, c0, operands))

            decision = cm.should_fuse(seq, rows, cols)
            if decision:
                t_cost = time_ms(lambda: kernel(c0.copy(), operand_list))
            else:
                t_cost = time_ms(lambda: numpy_apply(seq, c0, operands))

            best_of_two = min(t_always, t_never)
            is_optimal = t_cost <= best_of_two * 1.15  # 15% timing-noise tolerance
            rows_out.append((label, rows, cols, t_always, t_never, t_cost, decision, is_optimal))
            print(f"{rows:6d} {cols:6d} {t_always:15.4f} {t_never:14.4f} {t_cost:14.4f} "
                  f"{'fuse' if decision else 'numpy':>12} {'yes' if is_optimal else 'NO':>9}")
        print()

    with open("results/benchmark_costmodel.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sequence", "rows", "cols", "always_fuse_ms", "never_fuse_ms",
                     "cost_aware_ms", "model_chose_fuse", "matched_optimal"])
        w.writerows(rows_out)

    all_optimal = all(r[-1] for r in rows_out)
    print("ALL cost-aware choices matched (or beat) the better naive baseline."
          if all_optimal else "SOME cost-aware choices were suboptimal -- see table above.")


if __name__ == "__main__":
    main()

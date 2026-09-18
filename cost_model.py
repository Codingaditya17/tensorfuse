"""
cost_model.py — decides, at runtime, whether to actually use the fused
kernel or fall back to plain numpy for a given (op sequence, shape).

Why this exists: benchmark_isolated.py showed fusion is NOT always a
win. Algebraic chains (Add/Mul/Sub/ReLU/Neg only) won at every size
tested. Chains containing a transcendental (Sigmoid/Tanh) LOST to
numpy at small/medium sizes, because numpy's own vectorized exp is
already well-optimized, and only pulled ahead once array size was
large enough to make the extra memory passes numpy pays for outweigh
that. Rather than hardcode a guessed threshold, this reads the actual
measured crossover from results/benchmark_isolated.csv.

The graph-rewrite pass (fusion2.py) stays purely structural and
shape-agnostic -- it doesn't know concrete shapes at trace time. This
model is applied at the EXECUTION/dispatch layer instead, exactly like
how real JIT / shape-specialized runtimes decide whether to use a fast
specialized path or a generic fallback for a given input shape.
"""

import csv
import os

_HAS_TRANSCENDENTAL = {"Sigmoid", "Tanh"}
_DEFAULT_THRESHOLD_ELEMENTS = 8_000_000  # used if no calibration data found


def _signature(seq):
    return "-".join(kind for kind, _ in seq)


def calibrate_from_csv(path="results/benchmark_isolated.csv"):
    """
    Returns {sequence_signature: crossover_n_elements or None}.
    None means "always fuse" (no size at which it lost, in the data we have).
    Crossover is the smallest measured size where fuse_speedup >= 1.0,
    taken from the largest size that was still < 1.0 (conservative:
    round up to the next measured size, not the exact unmeasured point).
    """
    if not os.path.exists(path):
        return {}

    by_seq = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            seq_label = row["sequence"]
            n = int(row["rows"]) * int(row["cols"])
            speedup = float(row["fuse_speedup"])
            by_seq.setdefault(seq_label, []).append((n, speedup))

    thresholds = {}
    for seq_label, points in by_seq.items():
        points.sort()
        crossover = None
        for n, speedup in points:
            if speedup < 1.0:
                crossover = n  # keep updating -- we want the LARGEST losing size
        if crossover is not None:
            # smallest WINNING size strictly after the largest losing size
            winning_sizes = [n for n, sp in points if sp >= 1.0 and n > crossover]
            thresholds[seq_label] = min(winning_sizes) if winning_sizes else None
        else:
            thresholds[seq_label] = 0  # won everywhere measured -> always fuse
    return thresholds


class CostModel:
    def __init__(self, csv_path="results/benchmark_isolated.csv"):
        self._label_thresholds = calibrate_from_csv(csv_path)
        # benchmark_isolated.py labels sequences descriptively, not by
        # signature -- map known signatures to those labels here so
        # should_fuse can key off the actual op-kind signature.
        self._sig_thresholds = {}
        for label, threshold in self._label_thresholds.items():
            if "Add+ReLU" in label and "Sigmoid" not in label:
                self._sig_thresholds["Add-ReLU"] = threshold
            elif "Sigmoid" in label:
                self._sig_thresholds["Add-Sigmoid-Mul-ReLU"] = threshold

    def should_fuse(self, seq, rows, cols) -> bool:
        sig = _signature(seq)
        n = rows * cols
        has_transcendental = any(k in _HAS_TRANSCENDENTAL for k, _ in seq)

        if sig in self._sig_thresholds:
            threshold = self._sig_thresholds[sig]
            if threshold is None:
                return False  # measured: never wins in the data we have
            return n >= threshold

        # Unseen signature: fall back to the general rule the data
        # supports -- algebraic-only chains fuse, transcendental chains
        # need the default conservative size threshold.
        if not has_transcendental:
            return True
        return n >= _DEFAULT_THRESHOLD_ELEMENTS

    def explain(self):
        lines = ["Cost model, calibrated from results/benchmark_isolated.csv:"]
        for sig, threshold in self._sig_thresholds.items():
            if threshold is None:
                lines.append(f"  {sig:30s} -> NEVER fuse (lost at every measured size)")
            elif threshold == 0:
                lines.append(f"  {sig:30s} -> ALWAYS fuse (won at every measured size)")
            else:
                lines.append(f"  {sig:30s} -> fuse only if rows*cols >= {threshold:,}")
        return "\n".join(lines)


if __name__ == "__main__":
    cm = CostModel()
    print(cm.explain())
    print()
    tests = [
        (["Add", "ReLU"], 64, 64),
        (["Add", "ReLU"], 4096, 8192),
        (["Add", "Sigmoid", "Mul", "ReLU"], 256, 256),
        (["Add", "Sigmoid", "Mul", "ReLU"], 4096, 8192),
    ]
    for kinds, rows, cols in tests:
        seq = [(k, "x" if k in ("Add", "Mul", "Sub") else None) for k in kinds]
        decision = cm.should_fuse(seq, rows, cols)
        print(f"{'-'.join(kinds):30s} {rows}x{cols:<6d} -> fuse={decision}")

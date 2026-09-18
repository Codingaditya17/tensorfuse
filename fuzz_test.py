"""
fuzz_test.py — compiler correctness fuzzing: the same technique real
compiler teams use to find bugs (Csmith for C compilers, NNSmith for
ML compilers). Every correctness check elsewhere in this project is a
handful of hand-picked cases; this generates hundreds of RANDOM graphs
-- random op sequences, random shapes, random fan-out/fan-in patterns,
random batch sizes -- and differentially tests unfused vs fused
execution on every single one.

This project's whole bug-fixing narrative (7 real bugs, all in fusion
correctness around fan-out/fan-in and broadcast operands) makes this
the natural next level of rigor: the fuzzer's binary_tensor op
generates EXACTLY the class of pattern (two live values of the same
shape combined by a binary op) that caused bugs #1 and #4 by hand.
Running it at scale is the real test of whether those fixes actually
generalize, not just patch the one case that was found.
"""

import sys

import numpy as np

from ir import Graph
import ir2  # noqa: F401 -- registers mul/sub/sigmoid/tanh/neg/gelu on Graph
from fusion2 import fuse_elementwise
from executor_v2 import run_unfused, run_fused_untuned

UNARY_KINDS = ["relu", "sigmoid", "tanh", "neg", "gelu"]
BINARY_KINDS = ["add", "mul", "sub"]


def build_random_graph(rng, num_ops):
    g = Graph()
    x = g.input("x")
    shapes = {x: int(rng.choice([8, 16, 32, 64]))}
    live = [x]

    for _ in range(num_ops):
        choice = rng.choice(["matmul", "binary_param", "binary_tensor", "unary"],
                             p=[0.25, 0.35, 0.15, 0.25])

        if choice == "matmul":
            src = rng.choice(live)
            in_cols = shapes[src]
            out_cols = int(rng.choice([8, 16, 32, 64]))
            w = g.param(f"W{len(g.ops)}")
            shapes[w] = ("matmul_weight", in_cols, out_cols)
            new = g.matmul(src, w)
            shapes[new] = out_cols
            live.append(new)

        elif choice == "binary_param":
            src = rng.choice(live)
            cols = shapes[src]
            if not isinstance(cols, int):
                continue
            operand = g.param(f"p{len(g.ops)}")
            shapes[operand] = ("bias_vector", cols)
            kind = rng.choice(BINARY_KINDS)
            new = getattr(g, kind)(src, operand)
            shapes[new] = cols
            live.append(new)

        elif choice == "binary_tensor":
            by_cols = {}
            for v in live:
                c = shapes[v]
                if isinstance(c, int):
                    by_cols.setdefault(c, []).append(v)
            eligible = [c for c, vs in by_cols.items() if len(vs) >= 1]
            if not eligible:
                continue
            cols = rng.choice(eligible)
            candidates = by_cols[cols]
            a = rng.choice(candidates)
            b = rng.choice(candidates)
            kind = rng.choice(BINARY_KINDS)
            new = getattr(g, kind)(a, b)
            shapes[new] = cols
            live.append(new)

        elif choice == "unary":
            src = rng.choice(live)
            cols = shapes[src]
            if not isinstance(cols, int):
                continue
            kind = rng.choice(UNARY_KINDS)
            new = getattr(g, kind)(src)
            shapes[new] = cols
            live.append(new)

    output = live[-1]
    return g, x, shapes, output


def materialize_values(g, x, shapes, rng, batch):
    values = {x: rng.standard_normal((batch, shapes[x])).astype(np.float32)}
    for op in g.ops:
        if op.kind != "Param":
            continue
        spec = shapes[op.name]
        if spec[0] == "matmul_weight":
            _, in_c, out_c = spec
            values[op.name] = (rng.standard_normal((in_c, out_c)) * 0.2).astype(np.float32)
        elif spec[0] == "bias_vector":
            _, c = spec
            values[op.name] = rng.standard_normal((c,)).astype(np.float32)
    return values


def run_one_case(seed, num_ops, batch):
    rng = np.random.default_rng(seed)
    g, x, shapes, out_name = build_random_graph(rng, num_ops)
    values = materialize_values(g, x, shapes, rng, batch)

    out_unfused = run_unfused(g, dict(values))[out_name]
    fg, n_fused, n_absorbed = fuse_elementwise(g)
    out_fused = run_fused_untuned(fg, {k: v.copy() for k, v in values.items()})[out_name]

    if not np.allclose(out_unfused, out_fused, atol=1e-3, rtol=1e-3):
        diff = np.abs(out_unfused - out_fused).max()
        return False, n_fused, diff, g, fg
    return True, n_fused, 0.0, g, fg


def main(n_cases=500, max_ops=12, seed0=0):
    total_fused_nodes = 0
    failures = []

    for i in range(n_cases):
        num_ops = 3 + (i % max_ops)
        batch = 1 + (i * 7) % 32
        try:
            ok, n_fused, diff, g, fg = run_one_case(seed0 + i, num_ops, batch)
        except Exception as e:
            failures.append((i, "exception", str(e)))
            continue
        total_fused_nodes += n_fused
        if not ok:
            failures.append((i, "mismatch", diff))

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_cases} cases run, {total_fused_nodes} total fused "
                  f"nodes exercised, {len(failures)} failure(s) so far")

    print(f"\n=== Fuzz run complete: {n_cases} random graphs, "
          f"{total_fused_nodes} total fused nodes exercised across all graphs ===")

    if failures:
        print(f"\n{len(failures)} FAILURE(S) FOUND:")
        for idx, kind, info in failures[:10]:
            print(f"  case {idx}: {kind} -- {info}")
        i, kind, info = failures[0]
        rng = np.random.default_rng(seed0 + i)
        num_ops = 3 + (i % max_ops)
        g, x, shapes, out_name = build_random_graph(rng, num_ops)
        print(f"\nFirst failing graph (case {i}):")
        print(g.pretty())
        return 1
    else:
        print("NO FAILURES. Fusion + execution agree with the unfused baseline "
              f"across all {n_cases} random graphs, spanning random shapes, "
              "random fan-out/fan-in patterns, and random batch sizes.")
        return 0


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    sys.exit(main(n_cases=n))

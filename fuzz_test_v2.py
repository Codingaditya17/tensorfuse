"""
fuzz_test_v2.py — grows fuzz_test.py's grammar to cover the two newer
fusion mechanisms this project added: diamond patterns (fusion_subgraph.py)
and LayerNorm/AddLayerNorm (ir_transformer.py) -- stress-testing them
with the same random-graph rigor already applied to elementwise fusion.

All three fusion passes (fuse_elementwise, fuse_diamonds,
fuse_add_layernorm) are composed on every generated graph and checked
against a fully unfused baseline.

Conv2D/MaxPool are NOT yet included here -- they need 4D spatial shape
tracking the fuzzer's current shape model (a single scalar "cols"
feature dimension) doesn't support. Left as documented, honest future
scope, same as this project's other scoping decisions.
"""

import sys
import numpy as np

from ir import Graph
import ir2  # noqa: F401
import ir_transformer  # noqa: F401 -- registers g.layernorm
from fusion2 import fuse_elementwise
from fusion_subgraph import fuse_diamonds, DiamondKernel
from ir_transformer import LayerNormKernel, layernorm_forward_numpy, fuse_add_layernorm
from executor_v2 import _numpy_apply, _apply_structural_op

UNARY_KINDS = ["relu", "sigmoid", "tanh", "neg", "gelu"]
BINARY_KINDS = ["add", "mul", "sub"]


def build_random_graph(rng, num_ops):
    g = Graph()
    x = g.input("x")
    shapes = {x: int(rng.choice([8, 16, 32, 64]))}
    live = [x]

    for _ in range(num_ops):
        choice = rng.choice(
            ["matmul", "binary_param", "binary_tensor", "unary", "diamond",
             "layernorm", "residual_layernorm"],
            p=[0.18, 0.22, 0.10, 0.18, 0.12, 0.08, 0.12],
        )

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

        elif choice == "diamond":
            src = rng.choice(live)
            cols = shapes[src]
            if not isinstance(cols, int):
                continue
            k1, k2 = rng.choice(UNARY_KINDS, size=2, replace=True)
            a = getattr(g, k1)(src)
            b = getattr(g, k2)(src)
            shapes[a] = cols
            shapes[b] = cols
            merge_kind = rng.choice(BINARY_KINDS)
            new = getattr(g, merge_kind)(a, b)
            shapes[new] = cols
            live.append(new)

        elif choice == "layernorm":
            src = rng.choice(live)
            cols = shapes[src]
            if not isinstance(cols, int):
                continue
            gamma = g.param(f"gamma{len(g.ops)}")
            beta = g.param(f"beta{len(g.ops)}")
            shapes[gamma] = ("bias_vector", cols)
            shapes[beta] = ("bias_vector", cols)
            new = g.layernorm(src, gamma, beta, eps=1e-5)
            shapes[new] = cols
            live.append(new)

        elif choice == "residual_layernorm":
            # Deliberately builds the pattern fuse_add_layernorm targets:
            # a genuine residual (Add of two full-tensor values of the
            # SAME shape) immediately followed by LayerNorm, guaranteeing
            # the AddLayerNorm fusion path itself gets exercised, not
            # just the standalone LayerNorm fallback. Reuses an existing
            # same-cols live value as the "residual" partner when one
            # exists (matching a real skip-connection); otherwise builds
            # one via a fresh unary branch.
            src = rng.choice(live)
            cols = shapes[src]
            if not isinstance(cols, int):
                continue
            same_cols = [v for v in live if isinstance(shapes.get(v), int)
                          and shapes[v] == cols and v != src]
            if same_cols:
                residual = rng.choice(same_cols)
            else:
                residual = getattr(g, rng.choice(UNARY_KINDS))(src)
                shapes[residual] = cols
            summed = g.add(src, residual)
            shapes[summed] = cols
            gamma = g.param(f"gamma{len(g.ops)}")
            beta = g.param(f"beta{len(g.ops)}")
            shapes[gamma] = ("bias_vector", cols)
            shapes[beta] = ("bias_vector", cols)
            new = g.layernorm(summed, gamma, beta, eps=1e-5)
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


def run_unfused(graph, values):
    env = dict(values)
    for op in graph.ops:
        if op.kind in ("Input", "Param"):
            continue
        elif op.kind in ("Add", "Mul", "Sub"):
            x, y = op.inputs
            fn = {"Add": np.add, "Mul": np.multiply, "Sub": np.subtract}[op.kind]
            env[op.name] = fn(env[x], env[y])
        elif op.kind == "ReLU":
            (v,) = op.inputs; env[op.name] = np.maximum(env[v], 0)
        elif op.kind == "Sigmoid":
            (v,) = op.inputs; env[op.name] = 1.0 / (1.0 + np.exp(-env[v]))
        elif op.kind == "Tanh":
            (v,) = op.inputs; env[op.name] = np.tanh(env[v])
        elif op.kind == "Neg":
            (v,) = op.inputs; env[op.name] = -env[v]
        elif op.kind == "GELU":
            from scipy.special import erf
            (v,) = op.inputs
            env[op.name] = (0.5 * env[v] * (1.0 + erf(env[v] / np.sqrt(2.0)))).astype(np.float32)
        elif op.kind == "LayerNorm":
            xn, gn, bn = op.inputs
            env[op.name] = layernorm_forward_numpy(env[xn], env[gn], env[bn], op.meta["eps"])
        else:
            env[op.name] = _apply_structural_op(op, env)
    return env


def run_fused(fg, values, diamond_cache, ln_kernel):
    env = dict(values)
    for op in fg.ops:
        if op.kind in ("Input", "Param"):
            continue
        elif op.kind == "FusedElementwise":
            root, *_ = op.inputs
            env[op.name] = _numpy_apply(op.meta["seq"], env[root], lambda n: env[n])
        elif op.kind == "FusedDiamond":
            (root,) = op.inputs
            key = (op.meta["branch_a"], op.meta["branch_b"], op.meta["merge"])
            if key not in diamond_cache:
                diamond_cache[key] = DiamondKernel(*key)
            env[op.name] = diamond_cache[key](env[root].copy())
        elif op.kind == "AddLayerNorm":
            a, b, gn, bn = op.inputs
            env[op.name] = ln_kernel(env[a], env[b], env[gn], env[bn], op.meta["eps"])
        elif op.kind == "LayerNorm":
            xn, gn, bn = op.inputs
            env[op.name] = ln_kernel(env[xn], None, env[gn], env[bn], op.meta["eps"])
        elif op.kind in ("Add", "Mul", "Sub"):
            x, y = op.inputs
            fn = {"Add": np.add, "Mul": np.multiply, "Sub": np.subtract}[op.kind]
            env[op.name] = fn(env[x], env[y])
        elif op.kind == "ReLU":
            (v,) = op.inputs; env[op.name] = np.maximum(env[v], 0)
        elif op.kind == "Sigmoid":
            (v,) = op.inputs; env[op.name] = 1.0 / (1.0 + np.exp(-env[v]))
        elif op.kind == "Tanh":
            (v,) = op.inputs; env[op.name] = np.tanh(env[v])
        elif op.kind == "Neg":
            (v,) = op.inputs; env[op.name] = -env[v]
        elif op.kind == "GELU":
            from scipy.special import erf
            (v,) = op.inputs
            env[op.name] = (0.5 * env[v] * (1.0 + erf(env[v] / np.sqrt(2.0)))).astype(np.float32)
        else:
            env[op.name] = _apply_structural_op(op, env)
    return env


def run_one_case(seed, num_ops, batch, ln_kernel, diamond_cache):
    rng = np.random.default_rng(seed)
    g, x, shapes, out_name = build_random_graph(rng, num_ops)
    values = materialize_values(g, x, shapes, rng, batch)

    out_unfused = run_unfused(g, dict(values))[out_name]

    fg, n_fused, n_absorbed = fuse_elementwise(g)
    fg, n_diamonds = fuse_diamonds(fg)
    fg, n_ln = fuse_add_layernorm(fg)

    out_fused = run_fused(fg, {k: v.copy() for k, v in values.items()}, diamond_cache, ln_kernel)[out_name]

    stats = {"fused": n_fused, "diamonds": n_diamonds, "ln": n_ln}
    if not np.allclose(out_unfused, out_fused, atol=1e-3, rtol=1e-3):
        diff = np.abs(out_unfused - out_fused).max()
        return False, stats, diff, g
    return True, stats, 0.0, g


def main(n_cases=300, max_ops=10, seed0=0):
    ln_kernel = LayerNormKernel(with_residual=True)
    diamond_cache = {}
    totals = {"fused": 0, "diamonds": 0, "ln": 0}
    failures = []

    for i in range(n_cases):
        num_ops = 3 + (i % max_ops)
        batch = 1 + (i * 7) % 32
        try:
            ok, stats, diff, g = run_one_case(seed0 + i, num_ops, batch, ln_kernel, diamond_cache)
        except Exception as e:
            failures.append((i, "exception", str(e)))
            continue
        for k in totals:
            totals[k] += stats[k]
        if not ok:
            failures.append((i, "mismatch", diff))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_cases} cases, totals={totals}, {len(failures)} failure(s) so far")

    print(f"\n=== Fuzz v2 complete: {n_cases} random graphs ===")
    print(f"Fused elementwise nodes: {totals['fused']}  "
          f"diamonds: {totals['diamonds']}  AddLayerNorm/LayerNorm: {totals['ln']}")

    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for idx, kind, info in failures[:10]:
            print(f"  case {idx}: {kind} -- {info}")
        return 1
    else:
        print("NO FAILURES. Elementwise fusion, diamond fusion, and AddLayerNorm")
        print("fusion all agree with the unfused baseline, composed together,")
        print("across all random graphs.")
        return 0


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    sys.exit(main(n_cases=n))

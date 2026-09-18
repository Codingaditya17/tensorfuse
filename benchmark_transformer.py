"""
benchmark_transformer.py — times the full FFN sublayer three ways:
plain numpy (every op separate), our compiled pipeline (elementwise
fusion for the GELU chain + the single-pass AddLayerNorm kernel), and
real PyTorch eager mode on the identical computation.
"""

import time
import numpy as np
import torch
from scipy.special import erf

from transformer_block import build_ffn_graph, make_params, TorchFFN
from fusion2 import fuse_elementwise
from ir_transformer import fuse_add_layernorm, LayerNormKernel, layernorm_forward_numpy
from executor_v2 import _numpy_apply

torch.set_num_threads(1)


def run_numpy_plain(x, p):
    h = x @ p["W1"] + p["b1"]
    h = 0.5 * h * (1.0 + erf(h / np.float32(np.sqrt(2.0))))
    ff = h @ p["W2"] + p["b2"]
    summed = ff + x
    return layernorm_forward_numpy(summed, p["gamma"], p["beta"], 1e-5)


def run_ours(fg2, values, ln_plain, ln_add):
    env = dict(values)
    for op in fg2.ops:
        if op.kind in ("Input", "Param"):
            continue
        elif op.kind == "MatMul":
            a, b = op.inputs
            env[op.name] = env[a] @ env[b]
        elif op.kind == "Add":
            a, b = op.inputs
            env[op.name] = env[a] + env[b]
        elif op.kind == "FusedElementwise":
            root, *_ = op.inputs
            env[op.name] = _numpy_apply(op.meta["seq"], env[root], lambda n: env[n])
        elif op.kind == "AddLayerNorm":
            a, b, gn, bn = op.inputs
            env[op.name] = ln_add(env[a], env[b], env[gn], env[bn], op.meta["eps"])
        elif op.kind == "LayerNorm":
            a, gn, bn = op.inputs
            env[op.name] = ln_plain(env[a], None, env[gn], env[bn], op.meta["eps"])
        else:
            raise ValueError(op.kind)
    return env


def time_ms(fn, reps=7):
    fn()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1000


def main():
    d_model, d_ff = 512, 2048
    g, x_name, params, out_name = build_ffn_graph(d_model, d_ff)
    fg, n_fused, n_absorbed = fuse_elementwise(g)
    fg2, n_ln_fused = fuse_add_layernorm(fg)
    print(f"Fused {n_fused} elementwise node(s) + {n_ln_fused} AddLayerNorm node(s)\n")

    ln_plain = LayerNormKernel(with_residual=False)
    ln_add = LayerNormKernel(with_residual=True)

    p = make_params(d_model, d_ff)
    values = {x_name: None}
    for pname, pval in p.items():
        values[params[pname]] = pval

    torch_model = TorchFFN(d_model, d_ff, p["W1"], p["b1"], p["W2"], p["b2"], p["gamma"], p["beta"])
    torch_model.eval()

    print(f"{'batch':>6} {'numpy_ms':>10} {'ours_ms':>9} {'torch_ms':>10} {'speedup_vs_numpy':>18}")
    for batch in (32, 128, 512, 2048):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((batch, d_model)).astype(np.float32)
        values[x_name] = x
        x_t = torch.from_numpy(x)

        t_numpy = time_ms(lambda: run_numpy_plain(x, p))
        t_ours = time_ms(lambda: run_ours(fg2, values, ln_plain, ln_add))
        with torch.no_grad():
            t_torch = time_ms(lambda: torch_model(x_t))

        print(f"{batch:6d} {t_numpy:10.3f} {t_ours:9.3f} {t_torch:10.3f} {t_numpy/t_ours:17.2f}x")

    with torch.no_grad():
        torch_out = torch_model(x_t).numpy()
    our_out = run_ours(fg2, values, ln_plain, ln_add)[out_name]
    diff = np.abs(our_out - torch_out).max()
    print(f"\nmax diff vs PyTorch (last batch): {diff:.2e}")
    assert np.allclose(our_out, torch_out, atol=1e-3)
    print("Correctness re-verified.")


if __name__ == "__main__":
    main()

"""
transformer_block.py — a real transformer FFN sublayer:

    ff  = Linear2(GELU(Linear1(x)))
    out = LayerNorm(x + ff)              <- residual add, then norm

Two different fusion mechanisms are exercised in one graph:
  - MatMul + Add(bias) + GELU  -> existing generalized elementwise fusion
  - Add(residual) + LayerNorm  -> the NEW AddLayerNorm fusion

Validated against an actual torch.nn implementation, not just our own
numpy reference -- an independent ground truth for GELU's erf and
LayerNorm's exact normalization formula, both easy to get subtly
wrong (approximate vs exact GELU, biased vs unbiased variance).
"""

import numpy as np
import torch
import torch.nn as nn

from ir import Graph
import ir2  # noqa: F401
from ir_transformer import layernorm, fuse_add_layernorm, LayerNormKernel
from fusion2 import fuse_elementwise


def build_ffn_graph(d_model, d_ff):
    g = Graph()
    x = g.input("x")
    W1 = g.param("W1")
    b1 = g.param("b1")
    W2 = g.param("W2")
    b2 = g.param("b2")
    gamma = g.param("gamma")
    beta = g.param("beta")

    h = g.matmul(x, W1)
    h = g.add(h, b1)
    h = g.gelu(h)
    h = g.matmul(h, W2)
    h = g.add(h, b2)
    residual_sum = g.add(h, x)
    out = g.layernorm(residual_sum, gamma, beta, eps=1e-5)

    params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "gamma": gamma, "beta": beta}
    return g, x, params, out


class TorchFFN(nn.Module):
    def __init__(self, d_model, d_ff, W1, b1, W2, b2, gamma, beta):
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_ff)
        self.lin2 = nn.Linear(d_ff, d_model)
        with torch.no_grad():
            self.lin1.weight.copy_(torch.from_numpy(W1.T.copy()))
            self.lin1.bias.copy_(torch.from_numpy(b1))
            self.lin2.weight.copy_(torch.from_numpy(W2.T.copy()))
            self.lin2.bias.copy_(torch.from_numpy(b2))
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(d_model, eps=1e-5)
        with torch.no_grad():
            self.ln.weight.copy_(torch.from_numpy(gamma))
            self.ln.bias.copy_(torch.from_numpy(beta))

    def forward(self, x):
        ff = self.lin2(self.gelu(self.lin1(x)))
        return self.ln(x + ff)


def make_params(d_model, d_ff, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "W1": (rng.standard_normal((d_model, d_ff)) * 0.02).astype(np.float32),
        "b1": np.zeros((d_ff,), dtype=np.float32),
        "W2": (rng.standard_normal((d_ff, d_model)) * 0.02).astype(np.float32),
        "b2": np.zeros((d_model,), dtype=np.float32),
        "gamma": np.ones((d_model,), dtype=np.float32),
        "beta": np.zeros((d_model,), dtype=np.float32),
    }


if __name__ == "__main__":
    d_model, d_ff, batch = 256, 1024, 64
    g, x_name, params, out_name = build_ffn_graph(d_model, d_ff)
    fg, n_fused, n_absorbed = fuse_elementwise(g)
    fg2, n_ln_fused = fuse_add_layernorm(fg)

    print("=== Graph after both fusion passes ===")
    print(fg2.pretty())
    print(f"\n{n_fused} elementwise node(s) fused ({n_absorbed} ops absorbed), "
          f"{n_ln_fused} AddLayerNorm node(s) fused")

    p = make_params(d_model, d_ff)
    rng = np.random.default_rng(42)
    x = rng.standard_normal((batch, d_model)).astype(np.float32)

    torch_model = TorchFFN(d_model, d_ff, p["W1"], p["b1"], p["W2"], p["b2"], p["gamma"], p["beta"])
    torch_model.eval()
    with torch.no_grad():
        torch_out = torch_model(torch.from_numpy(x)).numpy()

    values = {x_name: x}
    for pname, pval in p.items():
        values[params[pname]] = pval

    from executor_v2 import _numpy_apply
    env = dict(values)
    ln_kernel_plain = LayerNormKernel(with_residual=False)
    ln_kernel_add = LayerNormKernel(with_residual=True)
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
            root, *operands = op.inputs
            env[op.name] = _numpy_apply(op.meta["seq"], env[root], lambda n: env[n])
        elif op.kind == "AddLayerNorm":
            a, b, gamma_n, beta_n = op.inputs
            env[op.name] = ln_kernel_add(env[a], env[b], env[gamma_n], env[beta_n], op.meta["eps"])
        elif op.kind == "LayerNorm":
            a, gamma_n, beta_n = op.inputs
            env[op.name] = ln_kernel_plain(env[a], None, env[gamma_n], env[beta_n], op.meta["eps"])
        else:
            raise ValueError(op.kind)

    our_out = env[out_name]
    diff = np.abs(our_out - torch_out).max()
    print(f"\nmax diff vs PyTorch (Linear+GELU+Linear+residual+LayerNorm): {diff:.2e}")
    assert np.allclose(our_out, torch_out, atol=1e-3), f"MISMATCH: {diff}"
    print("Verified correct against real PyTorch nn.Linear/nn.GELU/nn.LayerNorm.")

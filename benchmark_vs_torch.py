"""
benchmark_vs_torch.py — the real external credibility check: how does
this project's fused+autotuned kernel compare to PyTorch eager mode
and to torch.compile (Inductor) on the EXACT SAME computation?

Same graph as demo_graph.py: h1=ReLU(x@W1+b1); h2=ReLU(Sigmoid(h1@W2+b2)*scale)
"""

import time
import numpy as np
import torch

from demo_graph import build_demo_graph
from fusion2 import fuse_elementwise
from executor_v2 import run_unfused, run_fused_untuned, run_fused_autotuned
from test_correctness_v2 import make_values

torch.set_num_threads(1)  # fair comparison: this sandbox has 1 physical core


class TorchMLP(torch.nn.Module):
    def __init__(self, W1, b1, W2, b2, scale):
        super().__init__()
        self.W1 = torch.nn.Parameter(torch.from_numpy(W1), requires_grad=False)
        self.b1 = torch.nn.Parameter(torch.from_numpy(b1), requires_grad=False)
        self.W2 = torch.nn.Parameter(torch.from_numpy(W2), requires_grad=False)
        self.b2 = torch.nn.Parameter(torch.from_numpy(b2), requires_grad=False)
        self.scale = torch.nn.Parameter(torch.from_numpy(scale), requires_grad=False)

    def forward(self, x):
        h1 = torch.relu(x @ self.W1 + self.b1)
        h2 = torch.relu(torch.sigmoid(h1 @ self.W2 + self.b2) * self.scale)
        return h2


def time_ms(fn, reps=7):
    fn()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1000


def main():
    g, x_name, params, out_name = build_demo_graph()
    fg, n_fused, n_absorbed = fuse_elementwise(g)

    d_in, d_out = 512, 256
    configs = [(64, 512), (1024, 512), (4096, 512), (1024, 4096), (4096, 4096)]

    print(f"{'batch':>6} {'hidden':>7} {'torch_eager':>12} {'torch_compile':>14} "
          f"{'ours_unfused':>13} {'ours_untuned':>13} {'ours_auto':>10}")
    for batch, hidden in configs:
        values = make_values(x_name, params, batch, d_in, hidden, d_out, seed=0)
        W1, b1 = values[params["W1"]], values[params["b1"]]
        W2, b2 = values[params["W2"]], values[params["b2"]]
        scale = values[params["scale"]]
        x_np = values[x_name]

        model = TorchMLP(W1, b1, W2, b2, scale)
        x_t = torch.from_numpy(x_np)

        t_eager = time_ms(lambda: model(x_t))

        compiled = torch.compile(model)
        compiled(x_t)  # trigger compilation, untimed
        t_compiled = time_ms(lambda: compiled(x_t))

        t_unfused = time_ms(lambda: run_unfused(g, {k: v.copy() for k, v in values.items()}))
        t_untuned = time_ms(lambda: run_fused_untuned(fg, {k: v.copy() for k, v in values.items()}))
        t_auto = time_ms(lambda: run_fused_autotuned(fg, {k: v.copy() for k, v in values.items()}, verbose=False))

        print(f"{batch:6d} {hidden:7d} {t_eager:12.3f} {t_compiled:14.3f} "
              f"{t_unfused:13.3f} {t_untuned:13.3f} {t_auto:10.3f}")

    out_ref = model(x_t).detach().numpy()
    out_ours = run_fused_autotuned(fg, {k: v.copy() for k, v in values.items()}, verbose=False)[out_name]
    diff = np.abs(out_ref - out_ours).max()
    print(f"\nmax diff vs torch (last config): {diff:.2e}")
    assert np.allclose(out_ref, out_ours, atol=1e-3)
    print("Correctness verified against PyTorch.")


if __name__ == "__main__":
    main()

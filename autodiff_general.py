"""
autodiff_general.py — generalizes backward_fusion.py's single hardcoded
Add+ReLU pattern to arbitrary sequences of {Add, Mul, Sub, Neg, ReLU,
Sigmoid, Tanh, GELU}, the same way fusion2.py generalized forward
fusion beyond one fixed pattern.

Forward pass caches every intermediate value (same as any eager
autograd system); backward walks the sequence in reverse, applying
each op's local derivative via the chain rule, and for binary ops,
accumulating gradients for the broadcast operand via a column-sum
reduction. The backward computation for the WHOLE chain is then
compiled into ONE kernel per sequence, the same fusion discipline used
everywhere else in this project.
"""

import ctypes
import os
import subprocess
import tempfile

import numpy as np
import torch

_TMPDIR = tempfile.mkdtemp(prefix="tensorfuse_autodiff_")


def forward_with_cache(seq, x0, operand_arrays):
    v = x0
    cache = []
    for kind, operand in seq:
        cache.append(v)
        if kind == "Add":
            v = v + operand_arrays[operand]
        elif kind == "Mul":
            v = v * operand_arrays[operand]
        elif kind == "Sub":
            v = v - operand_arrays[operand]
        elif kind == "Neg":
            v = -v
        elif kind == "ReLU":
            v = np.maximum(v, 0)
        elif kind == "Sigmoid":
            v = 1.0 / (1.0 + np.exp(-v))
        elif kind == "Tanh":
            v = np.tanh(v)
        elif kind == "GELU":
            from scipy.special import erf
            v = 0.5 * v * (1.0 + erf(v / np.sqrt(2.0)))
    return v, cache


def backward(seq, cache, grad_output, operand_arrays):
    from scipy.special import erf
    g = grad_output
    operand_grads = {}
    for i in reversed(range(len(seq))):
        kind, operand = seq[i]
        x_in = cache[i]
        if kind == "Add":
            operand_grads[operand] = operand_grads.get(operand, 0) + g.sum(axis=0)
        elif kind == "Mul":
            operand_grads[operand] = operand_grads.get(operand, 0) + (g * x_in).sum(axis=0)
            g = g * operand_arrays[operand]
        elif kind == "Sub":
            operand_grads[operand] = operand_grads.get(operand, 0) - g.sum(axis=0)
        elif kind == "Neg":
            g = -g
        elif kind == "ReLU":
            g = g * (x_in > 0)
        elif kind == "Sigmoid":
            s = 1.0 / (1.0 + np.exp(-x_in))
            g = g * s * (1 - s)
        elif kind == "Tanh":
            t = np.tanh(x_in)
            g = g * (1 - t * t)
        elif kind == "GELU":
            cdf = 0.5 * (1.0 + erf(x_in / np.sqrt(2.0)))
            pdf = np.exp(-0.5 * x_in * x_in) / np.sqrt(2 * np.pi)
            g = g * (cdf + x_in * pdf)
    return g, operand_grads


_LOCAL_GRAD_EXPR = {
    "ReLU": "(({x}) > 0.0f ? 1.0f : 0.0f)",
    "Sigmoid": "({s}) * (1.0f - ({s}))",
    "Tanh": "(1.0f - ({t}) * ({t}))",
    "Neg": "-1.0f",
    "GELU": "0.5f * (1.0f + erff(({x}) * 0.70710678118654752440f)) + "
            "({x}) * expf(-0.5f * ({x}) * ({x})) * 0.3989422804014327f",
}


def _build_backward_source(seq, fn_name):
    operand_names = [op for _, op in seq if op is not None]
    n_cached = len(seq)

    cached_params = "".join(f", const float * restrict cache{i}" for i in range(n_cached))
    operand_params = "".join(f", const float * restrict operand{k}" for k in range(len(operand_names)))
    grad_out_params = "".join(f", float * restrict grad_operand{k}" for k in range(len(operand_names)))

    # Each operand's gradient is a (cols,) array: contributions from
    # EVERY row accumulate into grad_operand{k}[jj] directly, because
    # the operand was broadcast across all rows in the forward pass.
    # (There is no per-row "local" scalar here -- that was the bug:
    # it wrote into a rows-shaped position on a cols-shaped array.)
    lines = ["    float g = grad_out[jj];"]
    operand_idx = len(operand_names) - 1
    for i in reversed(range(len(seq))):
        kind, operand = seq[i]
        x = f"cache{i}_row[jj]"
        if kind == "Add":
            lines.append(f"    grad_operand{operand_idx}[jj] += g;")
            operand_idx -= 1
        elif kind == "Mul":
            lines.append(f"    grad_operand{operand_idx}[jj] += g * {x};")
            lines.append(f"    g = g * operand{operand_idx}[jj];")
            operand_idx -= 1
        elif kind == "Sub":
            lines.append(f"    grad_operand{operand_idx}[jj] -= g;")
            operand_idx -= 1
        elif kind == "Neg":
            lines.append("    g = -g;")
        elif kind == "ReLU":
            lines.append(f"    g = g * {_LOCAL_GRAD_EXPR['ReLU'].format(x=x)};")
        elif kind == "Sigmoid":
            lines.append(f"    {{ float s = 1.0f/(1.0f+expf(-({x}))); "
                          f"g = g * {_LOCAL_GRAD_EXPR['Sigmoid'].format(s='s')}; }}")
        elif kind == "Tanh":
            lines.append(f"    {{ float t = tanhf({x}); "
                          f"g = g * {_LOCAL_GRAD_EXPR['Tanh'].format(t='t')}; }}")
        elif kind == "GELU":
            lines.append(f"    g = g * ({_LOCAL_GRAD_EXPR['GELU'].format(x=x)});")
    body = "\n".join(lines)

    src = f"""
#include <math.h>

void {fn_name}(const float * restrict grad_out_full{cached_params}{operand_params},
               float * restrict dx{grad_out_params},
               int rows, int cols) {{
    for (int i = 0; i < rows; i++) {{
        const float * restrict grad_out = grad_out_full + (long)i * cols;
        float * restrict dx_row = dx + (long)i * cols;
{"".join(f"        const float * restrict cache{i}_row = cache{i} + (long)i * cols;" + chr(10) for i in range(n_cached))}
        for (int j = 0; j < cols; j++) {{
            int jj = j;
{body}
            dx_row[jj] = g;
        }}
    }}
}}
"""
    return src, operand_names


class FusedBackwardKernel:
    def __init__(self, seq):
        self.seq = seq
        fn_name = f"bwd_{abs(hash(tuple(seq)))}"
        src, self.operand_names = _build_backward_source(seq, fn_name)
        src_path = os.path.join(_TMPDIR, f"{fn_name}.c")
        so_path = os.path.join(_TMPDIR, f"{fn_name}.so")
        with open(src_path, "w") as f:
            f.write(src)
        subprocess.run(
            ["gcc", "-O3", "-march=native", "-ffast-math", "-shared", "-fPIC",
             src_path, "-o", so_path, "-lm", "-lmvec"],
            check=True, capture_output=True,
        )
        lib = ctypes.CDLL(so_path)
        self._fn = getattr(lib, fn_name)
        FP = ctypes.POINTER(ctypes.c_float)
        n_cache = len(seq)
        n_op = len(self.operand_names)
        self._fn.argtypes = (
            [FP] * (1 + n_cache + n_op)
            + [FP]
            + [FP] * n_op
            + [ctypes.c_int, ctypes.c_int]
        )
        self._fn.restype = None

    def __call__(self, grad_out, cache, operand_arrays):
        rows, cols = grad_out.shape
        grad_out = np.ascontiguousarray(grad_out, dtype=np.float32)
        cache = [np.ascontiguousarray(c, dtype=np.float32) for c in cache]
        operand_list = [np.ascontiguousarray(operand_arrays[n], dtype=np.float32)
                          for n in self.operand_names]
        dx = np.empty((rows, cols), dtype=np.float32)
        grad_operands = [np.zeros((cols,), dtype=np.float32) for _ in self.operand_names]

        ptr = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        args = [ptr(grad_out)] + [ptr(c) for c in cache] + [ptr(o) for o in operand_list]
        args += [ptr(dx)] + [ptr(g) for g in grad_operands]
        args += [rows, cols]
        self._fn(*args)

        return dx, {name: grad for name, grad in zip(self.operand_names, grad_operands)}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    rows, cols = 128, 64

    test_sequences = [
        [("Add", "b"), ("ReLU", None)],
        [("Add", "b"), ("Sigmoid", None), ("Mul", "s"), ("ReLU", None)],
        [("Mul", "s"), ("Add", "b"), ("Tanh", None)],
        [("Add", "b"), ("GELU", None)],
        [("Sub", "b"), ("Neg", None), ("ReLU", None)],
    ]

    for seq in test_sequences:
        x0 = rng.standard_normal((rows, cols)).astype(np.float32)
        operand_names = sorted({op for _, op in seq if op is not None})
        operands = {n: rng.standard_normal((cols,)).astype(np.float32) for n in operand_names}
        dout = rng.standard_normal((rows, cols)).astype(np.float32)

        x0_t = torch.tensor(x0, requires_grad=True)
        operand_ts = {n: torch.tensor(v, requires_grad=True) for n, v in operands.items()}
        v = x0_t
        for kind, operand in seq:
            if kind == "Add": v = v + operand_ts[operand]
            elif kind == "Mul": v = v * operand_ts[operand]
            elif kind == "Sub": v = v - operand_ts[operand]
            elif kind == "Neg": v = -v
            elif kind == "ReLU": v = torch.relu(v)
            elif kind == "Sigmoid": v = torch.sigmoid(v)
            elif kind == "Tanh": v = torch.tanh(v)
            elif kind == "GELU": v = torch.nn.functional.gelu(v)
        v.backward(torch.from_numpy(dout))
        expected_dx = x0_t.grad.numpy()
        expected_doperands = {n: t.grad.numpy() for n, t in operand_ts.items()}

        _, cache = forward_with_cache(seq, x0, operands)
        our_dx_np, our_dop_np = backward(seq, cache, dout, operands)

        kernel = FusedBackwardKernel(seq)
        our_dx_c, our_dop_c = kernel(dout, cache, operands)

        sig = "-".join(k for k, _ in seq)
        diff_dx_np = np.abs(our_dx_np - expected_dx).max()
        diff_dx_c = np.abs(our_dx_c - expected_dx).max()
        assert np.allclose(our_dx_np, expected_dx, atol=1e-3), f"{sig}: numpy dx mismatch {diff_dx_np}"
        assert np.allclose(our_dx_c, expected_dx, atol=1e-3), f"{sig}: compiled dx mismatch {diff_dx_c}"
        for n in operand_names:
            d_np = np.abs(our_dop_np[n] - expected_doperands[n]).max()
            d_c = np.abs(our_dop_c[n] - expected_doperands[n]).max()
            assert np.allclose(our_dop_np[n], expected_doperands[n], atol=1e-3), f"{sig}: numpy d{n} mismatch {d_np}"
            assert np.allclose(our_dop_c[n], expected_doperands[n], atol=1e-3), f"{sig}: compiled d{n} mismatch {d_c}"
        print(f"{sig:30s} numpy dx diff={diff_dx_np:.2e}  compiled dx diff={diff_dx_c:.2e}  OK")

    print("\nAll 5 random sequences: general numpy backward AND the compiled")
    print("fused backward kernel both verified correct against real PyTorch autograd.")

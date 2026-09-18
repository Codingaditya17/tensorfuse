"""
ir_transformer.py — LayerNorm, and the fused Add+LayerNorm pattern
real transformer compilers specifically target (residual-add followed
immediately by normalization -- FasterTransformer, DeepSpeed-Inference,
and xFormers all have a dedicated "AddLayerNorm"/"bias_residual_layer_norm"
fused kernel for exactly this reason: it's the single most common
memory-bound op sequence in a transformer block).

This is NOT an elementwise op: normalizing a row requires the row's own
mean and variance FIRST -- a genuine cross-element reduction dependency,
not a local per-element function. fusion2.py's existing elementwise
pass structurally cannot express this (by design -- it only fuses
chains where each op is a pure function of one element), so this adds
its OWN small fusion pass and its OWN compiled kernel, alongside the
existing one, not a hack bolted onto it.
"""

import ctypes
import os
import subprocess
import tempfile

import numpy as np
from ir import Graph, Op

_TMPDIR = tempfile.mkdtemp(prefix="tensorfuse_ln_")


def layernorm(self, x, gamma, beta, eps=1e-5):
    name = self._fresh_name()
    self.ops.append(Op(name, "LayerNorm", [x, gamma, beta], {"eps": eps}))
    return name


Graph.layernorm = layernorm


def fuse_add_layernorm(graph: Graph):
    """
    Small, separate fusion pass: Add(x, residual) -> LayerNorm(_, gamma, beta)
    becomes one AddLayerNorm node, IF the Add's result has exactly one
    consumer (same fan-out safety check as fusion2.py's elementwise pass).
    """
    by_name = {op.name: op for op in graph.ops}
    uses = {op.name: 0 for op in graph.ops}
    for op in graph.ops:
        for inp in op.inputs:
            uses[inp] = uses.get(inp, 0) + 1

    # Pass 1: decide which Add ops get absorbed and what each LayerNorm
    # node gets rewritten to, WITHOUT touching the output list yet --
    # a node can only be skipped correctly if we know that before we
    # reach it, and an Add always appears before the LayerNorm that
    # consumes it in a valid topological order.
    replace_layernorm_with = {}   # LayerNorm op name -> new AddLayerNorm Op
    skip_add_names = set()
    for op in graph.ops:
        if op.kind != "LayerNorm":
            continue
        x_name, gamma, beta = op.inputs
        add_op = by_name.get(x_name)
        if add_op is not None and add_op.kind == "Add" and uses.get(x_name, 0) == 1:
            a, b = add_op.inputs
            # AddLayerNorm's kernel assumes BOTH operands are full
            # (rows, cols) tensors (a genuine residual connection) --
            # NOT that one might be a (cols,) broadcast bias vector.
            # An ordinary MLP-style Add(x, bias) has exactly that shape
            # (a Param, not a full tensor), and fusing it here would
            # make the kernel read past the end of the small bias array
            # for every row past the first -- this is the exact mirror
            # of bug #4 (a residual read as a broadcast bias), found
            # independently by fuzz_test_v2.py in THIS fusion pass. Only
            # fuse when the second operand is NOT a Param.
            b_op = by_name.get(b)
            if b_op is not None and b_op.kind == "Param":
                continue  # ordinary bias-add, not a residual -- don't fuse
            replace_layernorm_with[op.name] = Op(
                op.name, "AddLayerNorm", [a, b, gamma, beta], op.meta
            )
            skip_add_names.add(add_op.name)

    # Pass 2: build the rewritten op list using the decisions from pass 1.
    new_ops = []
    for op in graph.ops:
        if op.name in skip_add_names:
            continue  # absorbed into a downstream AddLayerNorm
        elif op.name in replace_layernorm_with:
            new_ops.append(replace_layernorm_with[op.name])
        else:
            new_ops.append(op)

    fg = Graph()
    fg.ops = new_ops
    fg._counter = graph._counter
    return fg, len(replace_layernorm_with)


def layernorm_forward_numpy(x, gamma, beta, eps):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


def add_layernorm_forward_numpy(a, b, gamma, beta, eps):
    return layernorm_forward_numpy(a + b, gamma, beta, eps)


_KERNEL_SRC_TEMPLATE = r"""
#include <math.h>

void {fn_name}(const float * restrict a, const float * restrict b,
               const float * restrict gamma, const float * restrict beta,
               float * restrict out, int rows, int cols, float eps) {{
    for (int i = 0; i < rows; i++) {{
        const float * restrict a_row = a + (long)i * cols;
        const float * restrict b_row = b + (long)i * cols;
        float * restrict out_row = out + (long)i * cols;

        float sum = 0.0f, sumsq = 0.0f;
        for (int j = 0; j < cols; j++) {{
            float v = {sum_expr};
            sum += v;
            sumsq += v * v;
        }}
        float mean = sum / cols;
        float var = sumsq / cols - mean * mean;
        float inv_std = 1.0f / sqrtf(var + eps);

        for (int j = 0; j < cols; j++) {{
            float v = {sum_expr};
            out_row[j] = (v - mean) * inv_std * gamma[j] + beta[j];
        }}
    }}
}}
"""


class LayerNormKernel:
    def __init__(self, with_residual: bool):
        self.with_residual = with_residual
        sum_expr = "a_row[j] + b_row[j]" if with_residual else "a_row[j]"
        fn_name = f"layernorm_kernel_{'add' if with_residual else 'plain'}"
        src = _KERNEL_SRC_TEMPLATE.format(fn_name=fn_name, sum_expr=sum_expr)

        src_path = os.path.join(_TMPDIR, f"{fn_name}.c")
        so_path = os.path.join(_TMPDIR, f"{fn_name}.so")
        with open(src_path, "w") as f:
            f.write(src)
        subprocess.run(
            ["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-lm",
             src_path, "-o", so_path],
            check=True, capture_output=True,
        )
        lib = ctypes.CDLL(so_path)
        self._fn = getattr(lib, fn_name)
        FP = ctypes.POINTER(ctypes.c_float)
        self._fn.argtypes = [FP, FP, FP, FP, FP, ctypes.c_int, ctypes.c_int, ctypes.c_float]
        self._fn.restype = None

    def __call__(self, a, b, gamma, beta, eps):
        assert a.dtype == np.float32, f"LayerNormKernel requires float32, got {a.dtype}"
        if b is not None:
            assert b.dtype == np.float32, f"LayerNormKernel requires float32, got {b.dtype}"
        rows, cols = a.shape
        out = np.empty_like(a)
        b_arr = b if b is not None else np.zeros_like(a)
        ptr = lambda x: x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._fn(ptr(a), ptr(b_arr), ptr(gamma), ptr(beta), ptr(out), rows, cols, eps)
        return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    rows, cols = 128, 512
    x = rng.standard_normal((rows, cols)).astype(np.float32)
    residual = rng.standard_normal((rows, cols)).astype(np.float32)
    gamma = rng.standard_normal((cols,)).astype(np.float32)
    beta = rng.standard_normal((cols,)).astype(np.float32)
    eps = 1e-5

    expected_plain = layernorm_forward_numpy(x, gamma, beta, eps)
    kernel_plain = LayerNormKernel(with_residual=False)
    got_plain = kernel_plain(x, None, gamma, beta, eps)
    diff_plain = np.abs(got_plain - expected_plain).max()
    print("plain LayerNorm max diff:", diff_plain)
    assert np.allclose(got_plain, expected_plain, atol=1e-4)

    expected_fused = add_layernorm_forward_numpy(x, residual, gamma, beta, eps)
    kernel_fused = LayerNormKernel(with_residual=True)
    got_fused = kernel_fused(x, residual, gamma, beta, eps)
    diff_fused = np.abs(got_fused - expected_fused).max()
    print("AddLayerNorm max diff:", diff_fused)
    assert np.allclose(got_fused, expected_fused, atol=1e-4)
    print("Both kernels verified correct against numpy reference.")

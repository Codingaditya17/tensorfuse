"""
conv_fusion.py — the row-broadcast fusion documented as a gap
elsewhere in this project: Conv2D's bias is per-CHANNEL, constant
across spatial positions -- the OPPOSITE broadcast axis from every
other fused kernel here (an MLP's bias is per-COLUMN, constant across
rows/batch). Reusing the existing column-broadcast codegen would
silently misread the tensor, exactly like bug #4 (a residual read as
a broadcast bias). So this gets its own small, explicit codegen path
instead of overloading the existing one.

Layout: a conv output (N, C, H, W) is viewed as (N*C, H*W) -- each
"row" is one (batch, channel) pair, constant across all H*W columns
in that row. The bias value for row i is bias[i % C] (tiled across
batches). This is genuinely a different fusion primitive, not a
generalization of the column-broadcast one.
"""

import ctypes
import os
import subprocess
import tempfile

import numpy as np
from ir_cnn import _im2col

_TMPDIR = tempfile.mkdtemp(prefix="tensorfuse_conv_")

_KERNEL_SRC = r"""
#include <math.h>

void conv_bias_relu_row_broadcast(float * restrict c, const float * restrict bias,
                                    int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float * restrict row = c + (long)i * cols;
        float b = bias[i];
        for (int j = 0; j < cols; j++) {
            float v = row[j] + b;
            row[j] = v > 0.0f ? v : 0.0f;
        }
    }
}
"""


class ConvBiasReLURowBroadcastKernel:
    def __init__(self):
        src_path = os.path.join(_TMPDIR, "conv_fused.c")
        so_path = os.path.join(_TMPDIR, "conv_fused.so")
        with open(src_path, "w") as f:
            f.write(_KERNEL_SRC)
        subprocess.run(
            ["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-lm",
             src_path, "-o", so_path],
            check=True, capture_output=True,
        )
        lib = ctypes.CDLL(so_path)
        self._fn = lib.conv_bias_relu_row_broadcast
        FP = ctypes.POINTER(ctypes.c_float)
        self._fn.argtypes = [FP, FP, ctypes.c_int, ctypes.c_int]
        self._fn.restype = None

    def __call__(self, conv_out_2d, bias_tiled_per_row):
        assert conv_out_2d.dtype == np.float32
        assert bias_tiled_per_row.dtype == np.float32
        assert conv_out_2d.flags["C_CONTIGUOUS"]
        rows, cols = conv_out_2d.shape
        ptr = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._fn(ptr(conv_out_2d), ptr(bias_tiled_per_row), rows, cols)
        return conv_out_2d


def conv2d_no_bias_forward(x, weight, stride, padding):
    n, c_in, h, w = x.shape
    c_out, c_in_w, kh, kw = weight.shape
    cols, out_h, out_w = _im2col(x, kh, kw, stride, padding)
    W_flat = weight.reshape(c_out, -1)
    out = np.einsum("oc,ncp->nop", W_flat, cols)
    return out.reshape(n, c_out, out_h, out_w)


def fused_conv_bias_relu_forward(x, weight, bias, stride, padding, kernel):
    conv_out = conv2d_no_bias_forward(x, weight, stride, padding)
    n, c_out, out_h, out_w = conv_out.shape
    flat = np.ascontiguousarray(conv_out.reshape(n * c_out, out_h * out_w))
    bias_tiled = np.tile(bias, n).astype(np.float32)
    kernel(flat, bias_tiled)
    return flat.reshape(n, c_out, out_h, out_w)


def numpy_reference(x, weight, bias, stride, padding):
    conv_out = conv2d_no_bias_forward(x, weight, stride, padding)
    biased = conv_out + bias.reshape(1, -1, 1, 1)
    return np.maximum(biased, 0)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, c_in, h, w = 4, 3, 16, 16
    c_out, kh, kw = 8, 3, 3
    x = rng.standard_normal((n, c_in, h, w)).astype(np.float32)
    weight = (rng.standard_normal((c_out, c_in, kh, kw)) * 0.2).astype(np.float32)
    bias = rng.standard_normal((c_out,)).astype(np.float32)

    expected = numpy_reference(x, weight, bias, stride=1, padding=1)

    kernel = ConvBiasReLURowBroadcastKernel()
    got = fused_conv_bias_relu_forward(x, weight, bias, stride=1, padding=1, kernel=kernel)

    diff = np.abs(got - expected).max()
    print(f"max diff vs numpy (separate conv, broadcast-add, maximum): {diff:.2e}")
    assert np.allclose(got, expected, atol=1e-4)
    print("Row-broadcast Conv+bias+ReLU fusion verified correct.")

    import torch
    import torch.nn.functional as F
    x_t = torch.from_numpy(x)
    w_t = torch.from_numpy(weight)
    b_t = torch.from_numpy(bias)
    torch_out = F.relu(F.conv2d(x_t, w_t, b_t, stride=1, padding=1)).numpy()
    diff_torch = np.abs(got - torch_out).max()
    print(f"max diff vs real torch.nn.functional.conv2d+relu: {diff_torch:.2e}")
    assert np.allclose(got, torch_out, atol=1e-4)
    print("Also verified correct against real PyTorch Conv2d+ReLU.")

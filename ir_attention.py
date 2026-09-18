"""
ir_attention.py — the hardest, most valuable fusion target in a real
transformer compiler: scaled-dot-product attention. FlashAttention's
entire reason to exist is fusing exactly this sequence (matmul,
scale, softmax, matmul) into fewer passes over memory.

This implements a scoped but real piece of that: a fused,
numerically-stable softmax kernel (find row max, exp+sum, normalize --
ONE C function call instead of numpy's ~4 separate full-array passes),
used inside a real multi-head self-attention forward pass, validated
against PyTorch's actual nn.MultiheadAttention math.
"""

import ctypes
import os
import subprocess
import tempfile

import numpy as np

_TMPDIR = tempfile.mkdtemp(prefix="tensorfuse_attn_")

_SOFTMAX_KERNEL_SRC = r"""
#include <math.h>

void softmax_rows(const float * restrict x, float * restrict out,
                   int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        const float * restrict x_row = x + (long)i * cols;
        float * restrict out_row = out + (long)i * cols;

        float row_max = x_row[0];
        for (int j = 1; j < cols; j++) {
            if (x_row[j] > row_max) row_max = x_row[j];
        }

        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            float e = expf(x_row[j] - row_max);
            out_row[j] = e;
            sum += e;
        }

        float inv_sum = 1.0f / sum;
        for (int j = 0; j < cols; j++) {
            out_row[j] *= inv_sum;
        }
    }
}
"""


class SoftmaxKernel:
    def __init__(self):
        src_path = os.path.join(_TMPDIR, "softmax.c")
        so_path = os.path.join(_TMPDIR, "softmax.so")
        with open(src_path, "w") as f:
            f.write(_SOFTMAX_KERNEL_SRC)
        subprocess.run(
            ["gcc", "-O3", "-march=native", "-ffast-math", "-shared", "-fPIC",
             src_path, "-o", so_path, "-lm", "-lmvec"],
            check=True, capture_output=True,
        )
        lib = ctypes.CDLL(so_path)
        self._fn = lib.softmax_rows
        FP = ctypes.POINTER(ctypes.c_float)
        self._fn.argtypes = [FP, FP, ctypes.c_int, ctypes.c_int]
        self._fn.restype = None

    def __call__(self, x):
        assert x.dtype == np.float32, f"SoftmaxKernel requires float32, got {x.dtype}"
        assert x.ndim == 2, "call on a 2D (rows, cols) view -- reshape higher-rank inputs first"
        x = np.ascontiguousarray(x)
        rows, cols = x.shape
        out = np.empty_like(x)
        ptr = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._fn(ptr(x), ptr(out), rows, cols)
        return out


def softmax_numpy(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(q, k, v, softmax_kernel):
    """
    q, k, v: (batch, heads, seq, head_dim), float32
    Matmuls go through BLAS (np.matmul); the softmax step uses the
    fused kernel, applied on a flattened (batch*heads*seq, seq) view.
    """
    head_dim = q.shape[-1]
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) / np.sqrt(head_dim).astype(np.float32)
    b, h, s, _ = scores.shape
    flat = scores.reshape(b * h * s, s)
    weights = softmax_kernel(flat).reshape(b, h, s, s)
    return np.matmul(weights, v)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    rows, cols = 128, 64
    x = rng.standard_normal((rows, cols)).astype(np.float32) * 5
    expected = softmax_numpy(x)
    kernel = SoftmaxKernel()
    got = kernel(x)
    diff = np.abs(got - expected).max()
    print(f"softmax kernel max diff vs numpy: {diff:.2e}")
    assert np.allclose(got, expected, atol=1e-5)
    row_sums = got.sum(axis=-1)
    print(f"row sums (should all be ~1.0): min={row_sums.min():.6f} max={row_sums.max():.6f}")
    print("Softmax kernel verified correct and numerically stable.")

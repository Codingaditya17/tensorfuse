"""
backward_fusion.py — a real (if scoped) step toward a TRAINING
compiler, not just an inference one: reverse-mode gradient computation
for the Add(bias)+ReLU fused chain, with the backward pass itself
fused into ONE compiled kernel, verified against PyTorch's autograd.

Forward:  out = ReLU(c + bias)
Backward, given dL/dout:
    dc     = dL/dout * (out > 0)              -- elementwise, ReLU's local derivative
    dbias  = column-sum of dc                  -- REDUCTION over the batch dimension,
                                                   because bias was broadcast forward

This dbias reduction is the genuinely new piece: forward fusion in
this project never needed a cross-row reduction (LayerNorm was the
first, and that reduces across COLUMNS per row; this reduces across
ROWS per column). The backward kernel below computes dc AND
accumulates the column-sum for dbias in the SAME pass, instead of
numpy's separate mask-multiply then separate .sum(axis=0) call.
"""

import ctypes
import os
import subprocess
import tempfile

import numpy as np
import torch

_TMPDIR = tempfile.mkdtemp(prefix="tensorfuse_bwd_")

_KERNEL_SRC = r"""
#include <string.h>

void bias_relu_backward(const float * restrict dout, const float * restrict out,
                          float * restrict dc, float * restrict dbias,
                          int rows, int cols) {
    memset(dbias, 0, sizeof(float) * cols);
    for (int i = 0; i < rows; i++) {
        const float * restrict dout_row = dout + (long)i * cols;
        const float * restrict out_row = out + (long)i * cols;
        float * restrict dc_row = dc + (long)i * cols;
        for (int j = 0; j < cols; j++) {
            float g = out_row[j] > 0.0f ? dout_row[j] : 0.0f;
            dc_row[j] = g;
            dbias[j] += g;
        }
    }
}
"""


class BiasReLUBackwardKernel:
    def __init__(self):
        src_path = os.path.join(_TMPDIR, "bwd.c")
        so_path = os.path.join(_TMPDIR, "bwd.so")
        with open(src_path, "w") as f:
            f.write(_KERNEL_SRC)
        subprocess.run(
            ["gcc", "-O3", "-march=native", "-shared", "-fPIC", src_path, "-o", so_path],
            check=True, capture_output=True,
        )
        lib = ctypes.CDLL(so_path)
        self._fn = lib.bias_relu_backward
        FP = ctypes.POINTER(ctypes.c_float)
        self._fn.argtypes = [FP, FP, FP, FP, ctypes.c_int, ctypes.c_int]
        self._fn.restype = None

    def __call__(self, dout, out):
        assert dout.dtype == np.float32 and out.dtype == np.float32
        rows, cols = dout.shape
        dc = np.empty_like(dout)
        dbias = np.empty((cols,), dtype=np.float32)
        ptr = lambda x: x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._fn(ptr(dout), ptr(out), ptr(dc), ptr(dbias), rows, cols)
        return dc, dbias


def forward_bias_relu_numpy(c, bias):
    return np.maximum(c + bias, 0)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    rows, cols = 256, 512
    c_np = rng.standard_normal((rows, cols)).astype(np.float32)
    bias_np = rng.standard_normal((cols,)).astype(np.float32)
    dout_np = rng.standard_normal((rows, cols)).astype(np.float32)

    c_t = torch.tensor(c_np, requires_grad=True)
    bias_t = torch.tensor(bias_np, requires_grad=True)
    out_t = torch.relu(c_t + bias_t)
    out_t.backward(torch.from_numpy(dout_np))
    expected_dc = c_t.grad.numpy()
    expected_dbias = bias_t.grad.numpy()

    out_np = forward_bias_relu_numpy(c_np, bias_np)
    kernel = BiasReLUBackwardKernel()
    our_dc, our_dbias = kernel(dout_np, out_np)

    diff_dc = np.abs(our_dc - expected_dc).max()
    diff_dbias = np.abs(our_dbias - expected_dbias).max()
    print(f"max diff dc    vs torch autograd: {diff_dc:.2e}")
    print(f"max diff dbias vs torch autograd: {diff_dbias:.2e}")
    assert np.allclose(our_dc, expected_dc, atol=1e-5)
    assert np.allclose(our_dbias, expected_dbias, atol=1e-4)
    print("Fused backward kernel verified correct against real PyTorch autograd.")

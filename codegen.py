"""
codegen.py — Hardware backend (CPU) for the FusedBiasReLU op.

Generates a small C kernel, compiles it with gcc -O3 -march=native,
and loads it via ctypes. This is the "codegen" stage: turning an IR
node into actual machine code for the target.

out[i][j] = max(c[i][j] + bias[j], 0)

done in ONE pass over the (rows x cols) buffer, in place, with no
Python-level looping and no extra numpy temporaries.
"""

import ctypes
import os
import subprocess
import tempfile

KERNEL_SRC = r"""
#include <string.h>

// out[i*cols+j] = max(c[i*cols+j] + bias[j], 0)
// Written in-place into `c`. restrict pointers let gcc vectorize freely.
void fused_bias_relu(float * restrict c, const float * restrict bias,
                      int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float * restrict row = c + (long)i * cols;
        for (int j = 0; j < cols; j++) {
            float v = row[j] + bias[j];
            row[j] = v > 0.0f ? v : 0.0f;
        }
    }
}
"""


class FusedBiasReLUKernel:
    """Compiles once, callable many times."""

    def __init__(self):
        self._tmpdir = tempfile.mkdtemp(prefix="tensorfuse_")
        src_path = os.path.join(self._tmpdir, "kernel.c")
        so_path = os.path.join(self._tmpdir, "kernel.so")

        with open(src_path, "w") as f:
            f.write(KERNEL_SRC)

        subprocess.run(
            [
                "gcc", "-O3", "-march=native", "-shared", "-fPIC",
                src_path, "-o", so_path,
            ],
            check=True,
            capture_output=True,
        )

        self._lib = ctypes.CDLL(so_path)
        self._lib.fused_bias_relu.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._lib.fused_bias_relu.restype = None

    def __call__(self, c_array, bias_array):
        """
        c_array: numpy float32 array, shape (rows, cols), C-contiguous.
                 Modified IN PLACE.
        bias_array: numpy float32 array, shape (cols,), C-contiguous.
        """
        rows, cols = c_array.shape
        c_ptr = c_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        b_ptr = bias_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._lib.fused_bias_relu(c_ptr, b_ptr, rows, cols)
        return c_array


if __name__ == "__main__":
    import numpy as np

    kernel = FusedBiasReLUKernel()
    c = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    bias = np.array([0.5, 0.5], dtype=np.float32)
    expected = np.maximum(c + bias, 0)
    kernel(c, bias)
    assert np.allclose(c, expected), (c, expected)
    print("FusedBiasReLU kernel matches numpy reference. OK.")

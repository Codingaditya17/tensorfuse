"""
attention_orchestration.py — closes the documented gap in encoder_layer.py:
"the gap traces cleanly to Python-level orchestration overhead
(reshapes, multiple numpy calls for head-splitting), not kernel
quality."

Before: qkv split (view) -> 3x reshape -> 3x transpose(0,2,1,3) (creates
non-contiguous views that numpy/BLAS then has to copy internally when
used in matmul anyway). After: ONE compiled pass that gathers directly
from the (batch, seq, 3*d_model) QKV projection output into three
CONTIGUOUS (batch, nhead, seq, head_dim) arrays -- and the symmetric
merge kernel for undoing it after attention.
"""

import ctypes
import os
import subprocess
import tempfile

import numpy as np

_TMPDIR = tempfile.mkdtemp(prefix="tensorfuse_qkv_")

_SPLIT_SRC = r"""
void qkv_split_to_heads(const float * restrict qkv,
                          float * restrict q_out, float * restrict k_out, float * restrict v_out,
                          int batch, int seq, int nhead, int head_dim) {
    int d_model = nhead * head_dim;
    long qkv_stride_b = (long)seq * 3 * d_model;
    long out_stride_b = (long)nhead * seq * head_dim;
    long out_stride_h = (long)seq * head_dim;

    for (int b = 0; b < batch; b++) {
        const float * restrict qkv_b = qkv + (long)b * qkv_stride_b;
        for (int h = 0; h < nhead; h++) {
            float * restrict qo = q_out + (long)b * out_stride_b + (long)h * out_stride_h;
            float * restrict ko = k_out + (long)b * out_stride_b + (long)h * out_stride_h;
            float * restrict vo = v_out + (long)b * out_stride_b + (long)h * out_stride_h;
            for (int s = 0; s < seq; s++) {
                const float * restrict row = qkv_b + (long)s * 3 * d_model;
                const float * restrict q_src = row + h * head_dim;
                const float * restrict k_src = row + d_model + h * head_dim;
                const float * restrict v_src = row + 2 * d_model + h * head_dim;
                float * restrict qo_s = qo + (long)s * head_dim;
                float * restrict ko_s = ko + (long)s * head_dim;
                float * restrict vo_s = vo + (long)s * head_dim;
                for (int d = 0; d < head_dim; d++) {
                    qo_s[d] = q_src[d];
                    ko_s[d] = k_src[d];
                    vo_s[d] = v_src[d];
                }
            }
        }
    }
}
"""

_MERGE_SRC = r"""
void heads_merge(const float * restrict heads, float * restrict out,
                  int batch, int seq, int nhead, int head_dim) {
    int d_model = nhead * head_dim;
    long heads_stride_b = (long)nhead * seq * head_dim;
    long heads_stride_h = (long)seq * head_dim;
    long out_stride_b = (long)seq * d_model;

    for (int b = 0; b < batch; b++) {
        const float * restrict heads_b = heads + (long)b * heads_stride_b;
        float * restrict out_b = out + (long)b * out_stride_b;
        for (int h = 0; h < nhead; h++) {
            const float * restrict heads_h = heads_b + (long)h * heads_stride_h;
            for (int s = 0; s < seq; s++) {
                const float * restrict src = heads_h + (long)s * head_dim;
                float * restrict dst = out_b + (long)s * d_model + h * head_dim;
                for (int d = 0; d < head_dim; d++) {
                    dst[d] = src[d];
                }
            }
        }
    }
}
"""


def _compile(src, fn_name):
    src_path = os.path.join(_TMPDIR, f"{fn_name}.c")
    so_path = os.path.join(_TMPDIR, f"{fn_name}.so")
    with open(src_path, "w") as f:
        f.write(src)
    subprocess.run(
        ["gcc", "-O3", "-march=native", "-shared", "-fPIC", src_path, "-o", so_path],
        check=True, capture_output=True,
    )
    lib = ctypes.CDLL(so_path)
    return getattr(lib, fn_name)


class QKVSplitKernel:
    def __init__(self):
        self._fn = _compile(_SPLIT_SRC, "qkv_split_to_heads")
        FP = ctypes.POINTER(ctypes.c_float)
        self._fn.argtypes = [FP, FP, FP, FP, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self._fn.restype = None

    def __call__(self, qkv, nhead, head_dim):
        assert qkv.dtype == np.float32
        batch, seq, three_d_model = qkv.shape
        qkv = np.ascontiguousarray(qkv)
        q = np.empty((batch, nhead, seq, head_dim), dtype=np.float32)
        k = np.empty((batch, nhead, seq, head_dim), dtype=np.float32)
        v = np.empty((batch, nhead, seq, head_dim), dtype=np.float32)
        ptr = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._fn(ptr(qkv), ptr(q), ptr(k), ptr(v), batch, seq, nhead, head_dim)
        return q, k, v


class HeadsMergeKernel:
    def __init__(self):
        self._fn = _compile(_MERGE_SRC, "heads_merge")
        FP = ctypes.POINTER(ctypes.c_float)
        self._fn.argtypes = [FP, FP, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self._fn.restype = None

    def __call__(self, heads):
        assert heads.dtype == np.float32
        batch, nhead, seq, head_dim = heads.shape
        heads = np.ascontiguousarray(heads)
        out = np.empty((batch, seq, nhead * head_dim), dtype=np.float32)
        ptr = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._fn(ptr(heads), ptr(out), batch, seq, nhead, head_dim)
        return out


def numpy_split_to_heads(qkv, nhead, head_dim):
    batch, seq, three_d_model = qkv.shape
    q, k, v = np.split(qkv, 3, axis=-1)

    def to_heads(t):
        return t.reshape(batch, seq, nhead, head_dim).transpose(0, 2, 1, 3)

    return to_heads(q), to_heads(k), to_heads(v)


def numpy_heads_merge(heads):
    batch, nhead, seq, head_dim = heads.shape
    return heads.transpose(0, 2, 1, 3).reshape(batch, seq, nhead * head_dim)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    batch, seq, nhead, head_dim = 8, 32, 6, 16
    d_model = nhead * head_dim
    qkv = rng.standard_normal((batch, seq, 3 * d_model)).astype(np.float32)

    split_kernel = QKVSplitKernel()
    q_ours, k_ours, v_ours = split_kernel(qkv, nhead, head_dim)
    q_np, k_np, v_np = numpy_split_to_heads(qkv, nhead, head_dim)
    for name, ours, npv in [("q", q_ours, q_np), ("k", k_ours, k_np), ("v", v_ours, v_np)]:
        diff = np.abs(ours - npv).max()
        assert np.array_equal(ours, npv), f"{name} mismatch: {diff}"
        print(f"split {name}: bit-exact vs numpy")

    merge_kernel = HeadsMergeKernel()
    heads = rng.standard_normal((batch, nhead, seq, head_dim)).astype(np.float32)
    merged_ours = merge_kernel(heads)
    merged_np = numpy_heads_merge(heads)
    diff = np.abs(merged_ours - merged_np).max()
    assert np.array_equal(merged_ours, merged_np)
    print(f"merge: bit-exact vs numpy")

    print("\nQKV split/merge kernels verified correct.")

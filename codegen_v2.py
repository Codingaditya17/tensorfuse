"""
codegen_v2.py — JIT: compiles a FusedElementwise op *sequence* (not one
fixed kernel) into machine code, with two backends.

Backend A ("scalar"): generic, works for ANY sequence of
{Add, Mul, Sub, ReLU, Sigmoid, Tanh, Neg}, parameterized by an unroll
factor. Relies on gcc -O3 -march=native to auto-vectorize.

Backend B ("avx512"): hand-written AVX-512 intrinsics. Only applies to
the specific, common (Add, ReLU) sequence — there's no cheap vector
instruction for exp/tanh, so transcendental sequences fall back to
scalar. Parameterized by how many 16-wide vectors are unrolled per
outer-loop iteration.

Every candidate is verified against a numpy reference on random data
BEFORE it is allowed to be timed or selected — see autotune.py.
"""

import ctypes
import functools
import os
import re
import subprocess
import tempfile

import numpy as np

_TMPDIR = tempfile.mkdtemp(prefix="tensorfuse_jit_")
_COUNTER = [0]



def gen_scalar_source(seq, func_name, unroll):
    n_operands = sum(1 for _, operand in seq if operand is not None)
    params = "".join(f", const float* restrict operand{k}" for k in range(n_operands))
    body_lines = []
    k = 0
    for kind, operand in seq:
        if kind == "Add":
            body_lines.append(f"                v = v + operand{k}[jj];")
            k += 1
        elif kind == "Mul":
            body_lines.append(f"                v = v * operand{k}[jj];")
            k += 1
        elif kind == "Sub":
            body_lines.append(f"                v = v - operand{k}[jj];")
            k += 1
        elif kind == "ReLU":
            body_lines.append("                v = v > 0.0f ? v : 0.0f;")
        elif kind == "Sigmoid":
            body_lines.append("                v = 1.0f / (1.0f + expf(-v));")
        elif kind == "Tanh":
            body_lines.append("                v = tanhf(v);")
        elif kind == "Neg":
            body_lines.append("                v = -v;")
        elif kind == "GELU":
            # exact erf-based GELU, matching PyTorch's default nn.GELU()
            body_lines.append(
                "                v = 0.5f * v * (1.0f + erff(v * 0.70710678118654752440f));"
            )
        else:
            raise ValueError(f"unsupported op kind: {kind}")
    body = "\n".join(body_lines)

    src = f"""
#include <math.h>

void {func_name}(float * restrict c{params}, int rows, int cols) {{
    for (int i = 0; i < rows; i++) {{
        float * restrict row = c + (long)i * cols;
        int j = 0;
        for (; j + {unroll} <= cols; j += {unroll}) {{
            #pragma GCC unroll {unroll}
            for (int u = 0; u < {unroll}; u++) {{
                int jj = j + u;
                float v = row[jj];
{body}
                row[jj] = v;
            }}
        }}
        for (; j < cols; j++) {{
            int jj = j;
            float v = row[jj];
{body}
            row[jj] = v;
        }}
    }}
}}
"""
    return src, n_operands


@functools.lru_cache(maxsize=1)
def _host_supports_avx512() -> bool:
    """
    Real host-capability detection, not just try-and-catch: CI runners
    (GitHub Actions' standard ubuntu-latest included) typically do NOT
    have AVX-512 hardware. Attempting to compile the AVX-512 backend
    there would fail (undeclared intrinsics) on every single call.
    autotune.py's per-candidate try/except already catches that
    gracefully and falls back to the scalar backend -- but probing
    capability up front, once, and caching it, is the correct fix:
    don't attempt a codegen path the host can never support, rather
    than relying on failure handling to paper over it every time.
    """
    try:
        with open("/proc/cpuinfo") as f:
            return "avx512f" in f.read()
    except OSError:
        return False  # non-Linux or unreadable -- assume no AVX-512


def avx512_applicable(seq):
    return [k for k, _ in seq] == ["Add", "ReLU"] and _host_supports_avx512()


def gen_avx512_source(seq, func_name, vec_unroll):
    assert avx512_applicable(seq)
    bias_c = "operand0"
    step = 16 * vec_unroll

    load_lines, store_lines = [], []
    for k in range(vec_unroll):
        off = f"j + {16*k}" if k else "j"
        load_lines.append(
            f"            __m512 v{k} = _mm512_loadu_ps(row + {off});\n"
            f"            __m512 b{k} = _mm512_loadu_ps({bias_c} + {off});\n"
            f"            v{k} = _mm512_add_ps(v{k}, b{k});\n"
            f"            v{k} = _mm512_max_ps(v{k}, zero);\n"
            f"            _mm512_storeu_ps(row + {off}, v{k});"
        )
    unrolled_block = "\n".join(load_lines)

    src = f"""
#include <immintrin.h>

void {func_name}(float * restrict c, const float * restrict {bias_c},
                  int rows, int cols) {{
    const __m512 zero = _mm512_setzero_ps();
    for (int i = 0; i < rows; i++) {{
        float * restrict row = c + (long)i * cols;
        int j = 0;
        for (; j + {step} <= cols; j += {step}) {{
{unrolled_block}
        }}
        for (; j + 16 <= cols; j += 16) {{
            __m512 v = _mm512_loadu_ps(row + j);
            __m512 b = _mm512_loadu_ps({bias_c} + j);
            v = _mm512_add_ps(v, b);
            v = _mm512_max_ps(v, zero);
            _mm512_storeu_ps(row + j, v);
        }}
        for (; j < cols; j++) {{
            float v = row[j] + {bias_c}[j];
            row[j] = v > 0.0f ? v : 0.0f;
        }}
    }}
}}
"""
    return src, 1


def compile_kernel(src: str, func_name: str):
    _COUNTER[0] += 1
    src_path = os.path.join(_TMPDIR, f"{func_name}.c")
    so_path = os.path.join(_TMPDIR, f"{func_name}.so")
    with open(src_path, "w") as f:
        f.write(src)
    subprocess.run(
        ["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-lm",
         src_path, "-o", so_path],
        check=True, capture_output=True,
    )
    lib = ctypes.CDLL(so_path)
    fn = getattr(lib, func_name)
    return fn


class CompiledKernel:
    """Wraps a compiled function with a uniform call signature."""

    def __init__(self, backend, unroll, seq, func_name):
        self.backend = backend
        self.unroll = unroll
        self.seq = seq
        self.func_name = func_name

        if backend == "scalar":
            src, self.n_operands = gen_scalar_source(seq, func_name, unroll)
        elif backend == "avx512":
            src, self.n_operands = gen_avx512_source(seq, func_name, unroll)
        else:
            raise ValueError(backend)

        self._fn = compile_kernel(src, func_name)
        self._fn.argtypes = (
            [ctypes.POINTER(ctypes.c_float)] * (1 + self.n_operands)
            + [ctypes.c_int, ctypes.c_int]
        )
        self._fn.restype = None

    def __call__(self, c_array, operand_arrays):
        """
        operand_arrays: an ordered list (or any sequence) of numpy
        arrays, in the SAME ORDER as the binary ops appear in `seq`.
        Purely positional -- a compiled kernel doesn't know or care
        what these tensors were called in the IR, only how many there
        are and in what order, so the same compiled kernel object is
        safely reusable across any node with the same op-kind sequence.
        """
        assert c_array.dtype == np.float32, \
            f"CompiledKernel requires float32, got {c_array.dtype} " \
            f"(a numpy dtype-promotion bug upstream corrupted this once already -- see GELU fix)"
        for arr in operand_arrays:
            assert arr.dtype == np.float32, f"CompiledKernel requires float32, got {arr.dtype}"
        rows, cols = c_array.shape
        ptrs = [c_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))]
        for arr in operand_arrays:
            ptrs.append(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        self._fn(*ptrs, rows, cols)
        return c_array

    def label(self):
        return f"{self.backend}-unroll{self.unroll}"

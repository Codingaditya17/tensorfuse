"""
fusion_subgraph.py — the generalization fusion2.py explicitly declined
to do: fusing a genuine SUBGRAPH, not just a linear chain.

Pattern:  M = merge(f(H), g(H))     e.g.  Mul(Sigmoid(h), Tanh(h))

Both f(H) and g(H) are full tensors of the SAME shape as H -- neither
is a broadcast vector, so this can't be expressed with fusion2.py's
"one root + N broadcast operands" model at all. It needs its own
codegen: a kernel taking ONE input array and computing both branches
plus the merge in a single pass, instead of numpy's ~3 separate
full-array-materializing calls (branch A, branch B, merge).

Scoped deliberately to depth-1 diamonds (each branch a single unary op
directly on the shared input) -- deeper diamonds, or n-ary merges, are
a further generalization left for later, same as this project's other
honest scoping decisions.
"""

import ctypes
import os
import subprocess
import tempfile

import numpy as np
from ir import Graph, Op
from ir2 import ELEMENTWISE_UNARY

_TMPDIR = tempfile.mkdtemp(prefix="tensorfuse_diamond_")
_MERGE_KINDS = {"Add", "Mul", "Sub"}


def fuse_diamonds(graph: Graph):
    by_name = {op.name: op for op in graph.ops}
    uses = {op.name: 0 for op in graph.ops}
    for op in graph.ops:
        for inp in op.inputs:
            uses[inp] = uses.get(inp, 0) + 1

    replace_with = {}
    skip = set()

    for op in graph.ops:
        if op.kind not in _MERGE_KINDS or len(op.inputs) != 2:
            continue
        a_name, b_name = op.inputs
        a_op, b_op = by_name.get(a_name), by_name.get(b_name)
        if a_op is None or b_op is None:
            continue
        if a_op.kind not in ELEMENTWISE_UNARY or b_op.kind not in ELEMENTWISE_UNARY:
            continue
        if len(a_op.inputs) != 1 or len(b_op.inputs) != 1:
            continue
        if a_op.inputs[0] != b_op.inputs[0]:
            continue
        if uses.get(a_name, 0) != 1 or uses.get(b_name, 0) != 1:
            continue
        if a_name in skip or b_name in skip:
            continue

        shared_root = a_op.inputs[0]
        # No aliasing hazard here (unlike FusedElementwise): the diamond
        # kernel reads `in` as const and always writes to a freshly
        # allocated `out`, never mutating its input -- so there's no
        # root-exclusivity condition to track at all.
        new_op = Op(
            op.name, "FusedDiamond", [shared_root],
            {"branch_a": a_op.kind, "branch_b": b_op.kind, "merge": op.kind},
        )
        replace_with[op.name] = new_op
        skip.add(a_name)
        skip.add(b_name)

    new_ops = []
    for op in graph.ops:
        if op.name in skip:
            continue
        elif op.name in replace_with:
            new_ops.append(replace_with[op.name])
        else:
            new_ops.append(op)

    fg = Graph()
    fg.ops = new_ops
    fg._counter = graph._counter
    return fg, len(replace_with)


_UNARY_EXPR = {
    "ReLU": "{v} > 0.0f ? {v} : 0.0f",
    "Sigmoid": "1.0f / (1.0f + expf(-({v})))",
    "Tanh": "tanhf({v})",
    "Neg": "-({v})",
    "GELU": "0.5f * ({v}) * (1.0f + erff(({v}) * 0.70710678118654752440f))",
}
_MERGE_EXPR = {
    "Add": "({a}) + ({b})",
    "Mul": "({a}) * ({b})",
    "Sub": "({a}) - ({b})",
}


def _diamond_source(branch_a, branch_b, merge, fn_name):
    a_expr = _UNARY_EXPR[branch_a].format(v="h")
    b_expr = _UNARY_EXPR[branch_b].format(v="h")
    merge_expr = _MERGE_EXPR[merge].format(a="a", b="b")
    return f"""
#include <math.h>

void {fn_name}(const float * restrict in, float * restrict out, int n) {{
    for (int i = 0; i < n; i++) {{
        float h = in[i];
        float a = {a_expr};
        float b = {b_expr};
        out[i] = {merge_expr};
    }}
}}
"""


class DiamondKernel:
    def __init__(self, branch_a, branch_b, merge):
        self.branch_a, self.branch_b, self.merge = branch_a, branch_b, merge
        fn_name = f"diamond_{branch_a}_{branch_b}_{merge}".lower()
        src = _diamond_source(branch_a, branch_b, merge, fn_name)
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
        self._fn.argtypes = [FP, FP, ctypes.c_int]
        self._fn.restype = None

    def __call__(self, h_array, out_array=None):
        assert h_array.dtype == np.float32, f"DiamondKernel requires float32, got {h_array.dtype}"
        h_flat = np.ascontiguousarray(h_array).ravel()
        out = out_array if out_array is not None else np.empty_like(h_flat)
        ptr = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._fn(ptr(h_flat), ptr(out), h_flat.size)
        return out.reshape(h_array.shape)


def numpy_reference(branch_a, branch_b, merge, h):
    def apply_unary(kind, x):
        if kind == "ReLU":
            return np.maximum(x, 0)
        if kind == "Sigmoid":
            return 1.0 / (1.0 + np.exp(-x))
        if kind == "Tanh":
            return np.tanh(x)
        if kind == "Neg":
            return -x
        if kind == "GELU":
            from scipy.special import erf
            return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))
    a = apply_unary(branch_a, h)
    b = apply_unary(branch_b, h)
    if merge == "Add":
        return a + b
    if merge == "Mul":
        return a * b
    if merge == "Sub":
        return a - b


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    h = rng.standard_normal((256, 512)).astype(np.float32)

    for branch_a, branch_b, merge in [("Sigmoid", "Tanh", "Mul"),
                                        ("ReLU", "Neg", "Add"),
                                        ("GELU", "Sigmoid", "Sub")]:
        kernel = DiamondKernel(branch_a, branch_b, merge)
        got = kernel(h.copy())
        expected = numpy_reference(branch_a, branch_b, merge, h)
        diff = np.abs(got - expected).max()
        ok = np.allclose(got, expected, atol=1e-5)
        print(f"{merge}({branch_a}(h), {branch_b}(h)):  max_diff={diff:.2e}  {'OK' if ok else 'FAIL'}")
        assert ok
    print("\nAll diamond kernels verified correct against numpy reference.")

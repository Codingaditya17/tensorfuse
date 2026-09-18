"""
executor.py — Tiny interpreter that runs a Graph given concrete tensors.

Two backends:
  - run_unfused: every op (MatMul, Add, ReLU) is a separate numpy call.
                 Add and ReLU each read+write the full array -> 2 extra
                 full memory passes per layer, plus 2 extra Python/numpy
                 call overheads per layer.
  - run_fused:   MatMul still goes through numpy (BLAS) — we don't try
                 to beat BLAS — but FusedBiasReLU is a single compiled
                 pass that does bias-add + relu together, in place.
"""

import numpy as np
from ir import Graph
from codegen import FusedBiasReLUKernel

_fused_kernel = None


def _get_kernel():
    global _fused_kernel
    if _fused_kernel is None:
        _fused_kernel = FusedBiasReLUKernel()
    return _fused_kernel


def run_unfused(graph: Graph, values: dict) -> dict:
    env = dict(values)
    for op in graph.ops:
        if op.kind in ("Input", "Param"):
            continue
        elif op.kind == "MatMul":
            a, b = op.inputs
            env[op.name] = env[a] @ env[b]
        elif op.kind == "Add":
            x, y = op.inputs
            env[op.name] = env[x] + env[y]          # full pass #1 (new array)
        elif op.kind == "ReLU":
            (x,) = op.inputs
            env[op.name] = np.maximum(env[x], 0)     # full pass #2 (new array)
        else:
            raise ValueError(f"unknown op {op.kind}")
    return env


def run_fused(graph: Graph, values: dict) -> dict:
    kernel = _get_kernel()
    env = dict(values)
    for op in graph.ops:
        if op.kind in ("Input", "Param"):
            continue
        elif op.kind == "MatMul":
            a, b = op.inputs
            env[op.name] = env[a] @ env[b]           # still BLAS, unchanged
        elif op.kind == "FusedBiasReLU":
            mm, bias = op.inputs
            c = env[mm]
            if not c.flags["C_CONTIGUOUS"]:
                c = np.ascontiguousarray(c)
            kernel(c, env[bias])                     # ONE fused in-place pass
            env[op.name] = c
        else:
            raise ValueError(f"unknown op {op.kind}")
    return env

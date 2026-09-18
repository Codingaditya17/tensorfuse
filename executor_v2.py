"""
executor_v2.py — runs a graph three ways, for the isolation benchmark:

  run_unfused         every op is a separate numpy call
  run_fused_untuned   FusedElementwise nodes use the naive default
                       kernel (scalar, unroll=1) — "I fused but didn't
                       bother tuning it" baseline
  run_fused_autotuned FusedElementwise nodes use whatever autotune()
                       selects (cached after the first call per shape)

Comparing all three isolates: fusion's contribution (unfused vs
fused-untuned) from autotuning's contribution on top of that
(fused-untuned vs fused-autotuned).
"""

import numpy as np
from scipy.special import erf
from codegen_v2 import CompiledKernel
from autotune import autotune
from ir_cnn import conv2d_forward, maxpool2d_forward, flatten_forward

_untuned_cache = {}


def _apply_structural_op(op, env):
    """Ops that are neither Input/Param, elementwise, nor FusedElementwise.
    Shared by every executor variant so Conv2D/MaxPool2D/Flatten only
    needed to be implemented once."""
    if op.kind == "MatMul":
        a, b = op.inputs
        return env[a] @ env[b]
    elif op.kind == "Conv2D":
        x, w, b = op.inputs
        return conv2d_forward(env[x], env[w], env[b],
                               op.meta["stride"], op.meta["padding"])
    elif op.kind == "MaxPool2D":
        (x,) = op.inputs
        return maxpool2d_forward(env[x], op.meta["kernel"], op.meta["stride"])
    elif op.kind == "Flatten":
        (x,) = op.inputs
        return flatten_forward(env[x], op.meta["start_axis"])
    else:
        raise ValueError(op.kind)


def _numpy_apply(seq, v, env_lookup):
    for kind, operand in seq:
        if kind == "Add":
            v = v + env_lookup(operand)
        elif kind == "Mul":
            v = v * env_lookup(operand)
        elif kind == "Sub":
            v = v - env_lookup(operand)
        elif kind == "ReLU":
            v = np.maximum(v, 0)
        elif kind == "Sigmoid":
            v = 1.0 / (1.0 + np.exp(-v))
        elif kind == "Tanh":
            v = np.tanh(v)
        elif kind == "Neg":
            v = -v
        elif kind == "GELU":
            v = (0.5 * v * (1.0 + erf(v / np.sqrt(2.0)))).astype(v.dtype, copy=False)
    return v


def run_unfused(graph, values):
    env = dict(values)
    for op in graph.ops:
        if op.kind in ("Input", "Param"):
            continue
        elif op.kind == "FusedElementwise":
            root, *operands = op.inputs
            env[op.name] = _numpy_apply(op.meta["seq"], env[root], lambda n: env[n])
        elif op.kind in ("Add", "Mul", "Sub"):
            x, y = op.inputs
            fn = {"Add": np.add, "Mul": np.multiply, "Sub": np.subtract}[op.kind]
            env[op.name] = fn(env[x], env[y])
        elif op.kind == "ReLU":
            (x,) = op.inputs
            env[op.name] = np.maximum(env[x], 0)
        elif op.kind == "Sigmoid":
            (x,) = op.inputs
            env[op.name] = 1.0 / (1.0 + np.exp(-env[x]))
        elif op.kind == "Tanh":
            (x,) = op.inputs
            env[op.name] = np.tanh(env[x])
        elif op.kind == "Neg":
            (x,) = op.inputs
            env[op.name] = -env[x]
        elif op.kind == "GELU":
            (x,) = op.inputs
            env[op.name] = (0.5 * env[x] * (1.0 + erf(env[x] / np.sqrt(2.0)))).astype(env[x].dtype, copy=False)
        else:
            env[op.name] = _apply_structural_op(op, env)
    return env


def _get_mutation_safe_buffer(op, env, root):
    """
    The compiled kernels mutate their input buffer in place. That's
    only safe when fusion2.py determined this fused group is the ONLY
    consumer of `root` in the whole graph (op.meta["root_exclusive"]).
    Otherwise root may be read again elsewhere in the graph, and
    mutating it here would silently corrupt that other read -- exactly
    the bug fuzz_test.py found: a shared Input feeding both a fusible
    chain and an unrelated later MatMul. When it's not exclusive, copy
    first; when it is, only copy if needed for C-contiguity.
    """
    c = env[root]
    if op.meta.get("root_exclusive", False):
        if not c.flags["C_CONTIGUOUS"]:
            c = np.ascontiguousarray(c)
    else:
        c = np.array(c, copy=True, order="C")
    return c


def _get_untuned_kernel(seq):
    sig = "-".join(k for k, _ in seq)
    if sig not in _untuned_cache:
        func_name = f"untuned_{abs(hash(sig))}"
        _untuned_cache[sig] = CompiledKernel("scalar", 1, seq, func_name)
    return _untuned_cache[sig]


def _run_fused_graph(graph, values, get_kernel_fn):
    env = dict(values)
    for op in graph.ops:
        if op.kind in ("Input", "Param"):
            continue
        elif op.kind == "FusedElementwise":
            root, *operand_names = op.inputs
            c = _get_mutation_safe_buffer(op, env, root)
            rows, cols = c.shape
            kernel = get_kernel_fn(op.meta["seq"], rows, cols, operand_names)
            # positional, in the same order fusion2 recorded them in `seq`
            operand_list = [env[n] for n in operand_names]
            kernel(c, operand_list)
            env[op.name] = c
        elif op.kind in ("Add", "Mul", "Sub"):
            x, y = op.inputs
            fn = {"Add": np.add, "Mul": np.multiply, "Sub": np.subtract}[op.kind]
            env[op.name] = fn(env[x], env[y])
        elif op.kind == "ReLU":
            (x,) = op.inputs
            env[op.name] = np.maximum(env[x], 0)
        elif op.kind == "Sigmoid":
            (x,) = op.inputs
            env[op.name] = 1.0 / (1.0 + np.exp(-env[x]))
        elif op.kind == "Tanh":
            (x,) = op.inputs
            env[op.name] = np.tanh(env[x])
        elif op.kind == "Neg":
            (x,) = op.inputs
            env[op.name] = -env[x]
        elif op.kind == "GELU":
            (x,) = op.inputs
            env[op.name] = (0.5 * env[x] * (1.0 + erf(env[x] / np.sqrt(2.0)))).astype(env[x].dtype, copy=False)
        else:
            env[op.name] = _apply_structural_op(op, env)
    return env


def run_fused_untuned(graph, values):
    return _run_fused_graph(
        graph, values,
        lambda seq, rows, cols, operands: _get_untuned_kernel(seq),
    )


def run_fused_autotuned(graph, values, verbose=False):
    def get_kernel(seq, rows, cols, operand_names):
        kernel, _info = autotune(seq, rows, cols, operand_names, verbose=verbose)
        return kernel
    return _run_fused_graph(graph, values, get_kernel)


def run_fused_costaware(graph, values, cost_model, verbose=False):
    """
    Like run_fused_autotuned, but for each FusedElementwise node,
    consult `cost_model.should_fuse(seq, rows, cols)` first. If it
    says no, execute that node's op sequence via plain numpy instead
    of calling the compiled kernel at all -- this is what closes the
    loop on the honest finding that fusion sometimes loses.
    """
    env = dict(values)
    for op in graph.ops:
        if op.kind in ("Input", "Param"):
            continue
        elif op.kind == "FusedElementwise":
            root, *operand_names = op.inputs
            rows, cols = env[root].shape
            seq = op.meta["seq"]
            if cost_model.should_fuse(seq, rows, cols):
                c = _get_mutation_safe_buffer(op, env, root)
                kernel, _info = autotune(seq, rows, cols, operand_names, verbose=verbose)
                operand_list = [env[n] for n in operand_names]
                kernel(c, operand_list)
                env[op.name] = c
            else:
                env[op.name] = _numpy_apply(seq, env[root], lambda n: env[n])
        elif op.kind in ("Add", "Mul", "Sub"):
            x, y = op.inputs
            fn = {"Add": np.add, "Mul": np.multiply, "Sub": np.subtract}[op.kind]
            env[op.name] = fn(env[x], env[y])
        elif op.kind == "ReLU":
            (x,) = op.inputs
            env[op.name] = np.maximum(env[x], 0)
        elif op.kind == "Sigmoid":
            (x,) = op.inputs
            env[op.name] = 1.0 / (1.0 + np.exp(-env[x]))
        elif op.kind == "Tanh":
            (x,) = op.inputs
            env[op.name] = np.tanh(env[x])
        elif op.kind == "Neg":
            (x,) = op.inputs
            env[op.name] = -env[x]
        elif op.kind == "GELU":
            (x,) = op.inputs
            env[op.name] = (0.5 * env[x] * (1.0 + erf(env[x] / np.sqrt(2.0)))).astype(env[x].dtype, copy=False)
        else:
            env[op.name] = _apply_structural_op(op, env)
    return env


def run_fused_learned(graph, values, learned_model, verbose=False):
    """Same shape as run_fused_costaware, but the fuse/no-fuse gate is
    the learned logistic-regression model instead of the calibrated
    per-signature threshold table."""
    env = dict(values)
    for op in graph.ops:
        if op.kind in ("Input", "Param"):
            continue
        elif op.kind == "FusedElementwise":
            root, *operand_names = op.inputs
            rows, cols = env[root].shape
            seq = op.meta["seq"]
            if learned_model.should_fuse(seq, rows, cols):
                c = _get_mutation_safe_buffer(op, env, root)
                kernel, _info = autotune(seq, rows, cols, operand_names, verbose=verbose)
                operand_list = [env[n] for n in operand_names]
                kernel(c, operand_list)
                env[op.name] = c
            else:
                env[op.name] = _numpy_apply(seq, env[root], lambda n: env[n])
        elif op.kind in ("Add", "Mul", "Sub"):
            x, y = op.inputs
            fn = {"Add": np.add, "Mul": np.multiply, "Sub": np.subtract}[op.kind]
            env[op.name] = fn(env[x], env[y])
        elif op.kind == "ReLU":
            (x,) = op.inputs
            env[op.name] = np.maximum(env[x], 0)
        elif op.kind == "Sigmoid":
            (x,) = op.inputs
            env[op.name] = 1.0 / (1.0 + np.exp(-env[x]))
        elif op.kind == "Tanh":
            (x,) = op.inputs
            env[op.name] = np.tanh(env[x])
        elif op.kind == "Neg":
            (x,) = op.inputs
            env[op.name] = -env[x]
        elif op.kind == "GELU":
            (x,) = op.inputs
            env[op.name] = (0.5 * env[x] * (1.0 + erf(env[x] / np.sqrt(2.0)))).astype(env[x].dtype, copy=False)
        else:
            env[op.name] = _apply_structural_op(op, env)
    return env

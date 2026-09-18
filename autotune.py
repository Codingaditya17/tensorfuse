"""
autotune.py — the actual autotuning loop.

For a given fused op sequence and shape:
  1. Enumerate every candidate kernel (backend x unroll factor).
  2. Compile each one.
  3. VERIFY each one against a numpy reference on random data —
     a candidate that produces wrong output is discarded, never timed,
     never selectable. (This is what caught nothing here, but it's
     what WOULD catch a broken AVX-512 tail-handling bug, an off-by-one
     in unrolling, etc. — the fusion-pass bug earlier was caught by
     this same "verify before trust" discipline, just one layer up.)
  4. Benchmark the survivors, pick the fastest.
  5. Cache the winning (backend, unroll) choice keyed by
     (op signature, cols) so repeated calls with the same shape don't
     re-run the whole search — same idea as an AutoTVM/Ansor tuning log.
"""

import json
import os
import time
import numpy as np
from scipy.special import erf

from codegen_v2 import CompiledKernel, avx512_applicable
from triton_backend import is_triton_available, TritonBiasReLUKernel
import remote_cache

CACHE_PATH = "results/autotune_cache.json"

# In-process memoization: the JSON cache remembers WHICH config won
# (persists across process restarts, like an AutoTVM tuning log), but
# that alone doesn't help within a run if we recompile on every call.
# This dict remembers the actual compiled, loaded kernel object so a
# cache "hit" is a dict lookup, not a fresh gcc invocation.
_kernel_memo = {}


def _numpy_reference(seq, c0, operand_arrays):
    v = c0.copy()
    for kind, operand in seq:
        if kind == "Add":
            v = v + operand_arrays[operand]
        elif kind == "Mul":
            v = v * operand_arrays[operand]
        elif kind == "Sub":
            v = v - operand_arrays[operand]
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


def _signature(seq):
    return "-".join(kind for kind, _ in seq)


def _load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def _save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def _cache_lookup(sig, cols):
    """Checks the networked tuning_cache_service if TENSORFUSE_CACHE_URL
    is set, otherwise the local JSON file. Returns an entry dict or None."""
    url = remote_cache.cache_url()
    if url:
        return remote_cache.get_entry(url, sig, cols)
    cache = _load_cache()
    return cache.get(f"{sig}|cols={cols}")


def _cache_store(sig, cols, entry):
    url = remote_cache.cache_url()
    if url:
        ok = remote_cache.put_entry(url, sig, cols, entry)
        if not ok:
            # network hiccup -- still keep it locally so this process
            # isn't left without any record of the search it just ran
            cache = _load_cache()
            cache[f"{sig}|cols={cols}"] = entry
            _save_cache(cache)
        return
    cache = _load_cache()
    cache[f"{sig}|cols={cols}"] = entry
    _save_cache(cache)


def autotune(seq, rows, cols, operand_names, seed=0, reps=5, verbose=True, force=False):
    """
    Returns (kernel: CompiledKernel, info: dict).
    info includes whether the result came from cache and, on a fresh
    search, every candidate's verification/timing outcome.
    """
    sig = _signature(seq)
    cache_key = f"{sig}|cols={cols}"

    # 1. In-memory hit: already compiled and loaded in this process.
    if not force and cache_key in _kernel_memo:
        return _kernel_memo[cache_key], {"from_cache": True, "memory_hit": True}

    # 2. Cache hit -- networked service if configured, else local file.
    #    We know the winning config from a previous run (possibly on a
    #    different machine entirely, if using the networked cache), but
    #    must compile it once in THIS process.
    entry = None if force else _cache_lookup(sig, cols)
    if entry is not None:
        if entry["backend"] == "triton":
            kernel = TritonBiasReLUKernel(block_size=entry["unroll"])
        else:
            kernel = CompiledKernel(entry["backend"], entry["unroll"], seq,
                                     func_name=f"cached_{abs(hash(cache_key))}_{entry['backend']}_{entry['unroll']}")
        _kernel_memo[cache_key] = kernel
        if verbose:
            source = "remote" if remote_cache.cache_url() else "disk"
            print(f"[autotune] {source}-cache hit for '{sig}' cols={cols}: "
                  f"{entry['backend']} unroll={entry['unroll']} "
                  f"(logged best {entry['time_ms']:.4f} ms) — compiling once")
        return kernel, {"from_cache": True, **entry}

    rng = np.random.default_rng(seed)
    c0 = rng.standard_normal((rows, cols)).astype(np.float32)
    operand_arrays = {name: rng.standard_normal((cols,)).astype(np.float32)
                       for name in operand_names}
    reference = _numpy_reference(seq, c0, operand_arrays)

    candidates = []
    for u in (1, 2, 4, 8, 16):
        candidates.append(("scalar", u))
    if avx512_applicable(seq):
        for u in (1, 2, 4):
            candidates.append(("avx512", u))
    if avx512_applicable(seq) and is_triton_available():
        for block in (256, 1024, 4096):
            candidates.append(("triton", block))

    # positional list, in the same order seq's binary ops appear
    operand_list = [operand_arrays[operand] for _, operand in seq if operand is not None]

    results = []
    for backend, unroll in candidates:
        func_name = f"fused_{abs(hash((sig, backend, unroll, cols)))}"
        try:
            if backend == "triton":
                kernel = TritonBiasReLUKernel(block_size=unroll)
            else:
                kernel = CompiledKernel(backend, unroll, seq, func_name)
        except Exception as e:
            results.append({"backend": backend, "unroll": unroll,
                             "status": "compile_error", "error": str(e)[:200]})
            continue

        test_c = c0.copy()
        kernel(test_c, operand_list)
        max_diff = float(np.abs(test_c - reference).max())
        if not np.allclose(test_c, reference, atol=1e-4):
            results.append({"backend": backend, "unroll": unroll,
                             "status": "INCORRECT", "max_diff": max_diff})
            continue

        best_t = float("inf")
        for _ in range(reps):
            trial_c = c0.copy()
            t0 = time.perf_counter()
            kernel(trial_c, operand_list)
            best_t = min(best_t, time.perf_counter() - t0)
        results.append({"backend": backend, "unroll": unroll, "status": "ok",
                         "max_diff": max_diff, "time_ms": best_t * 1000,
                         "_kernel": kernel})

    if verbose:
        print(f"[autotune] searching '{sig}' cols={cols} rows={rows} "
              f"({len(candidates)} candidates):")
        for r in results:
            if r["status"] == "ok":
                print(f"    {r['backend']:8s} unroll={r['unroll']:<3d} "
                      f"{r['time_ms']:8.4f} ms   (max_diff={r['max_diff']:.1e})")
            else:
                print(f"    {r['backend']:8s} unroll={r['unroll']:<3d} "
                      f"{r['status']}")

    ok_results = [r for r in results if r["status"] == "ok"]
    if not ok_results:
        raise RuntimeError(f"No correct candidate found for sequence {seq}")

    best = min(ok_results, key=lambda r: r["time_ms"])
    if verbose:
        print(f"[autotune] winner: {best['backend']} unroll={best['unroll']} "
              f"({best['time_ms']:.4f} ms)\n")

    best_entry = {
        "backend": best["backend"], "unroll": best["unroll"],
        "time_ms": best["time_ms"],
        "all_candidates": [
            {k: v for k, v in r.items() if k != "_kernel"} for r in results
        ],
    }
    _cache_store(sig, cols, best_entry)

    # Reuse the exact object we already compiled and just benchmarked —
    # no reason to compile it a second time on the next call.
    _kernel_memo[cache_key] = best["_kernel"]

    return best["_kernel"], {"from_cache": False, **best_entry}


if __name__ == "__main__":
    kernel, info = autotune(
        seq=[("Add", "bias"), ("ReLU", None)],
        rows=4096, cols=4096, operand_names=["bias"],
    )
    print("Chosen:", kernel.label())

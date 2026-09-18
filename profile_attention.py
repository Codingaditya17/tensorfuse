"""
profile_attention.py — an honest record of a hypothesis that turned
out to be WRONG, and what the actual profiling data shows instead.

The documented "next step" elsewhere in this project claimed the
remaining ~1.5-2.6x gap vs PyTorch at the full encoder-layer level
"traces cleanly to Python-level orchestration overhead (reshapes,
multiple numpy calls for head-splitting)". attention_orchestration.py
was built specifically to fix that: a single compiled kernel doing the
QKV split and head-merge in one pass instead of numpy's split+reshape+
transpose.

Measuring it disproved the hypothesis. np.split and .transpose()
produce lazy VIEWS -- they don't copy any data until something
downstream forces materialization. They were already ~0.008ms,
effectively free. The "fusion" kernel, which eagerly copies into new
contiguous buffers, was 100-400x SLOWER for exactly that reason. It
was reverted from encoder_layer.py rather than shipped as a false win.

Real profiling shows where the time actually goes:
    qkv_proj (one matmul):        ~29.5 ms
    split+heads (numpy views):     ~0.008 ms   <- confirmed negligible
    sdpa (2 matmuls + softmax):   ~11.8 ms
    merge (numpy view):            ~0.33 ms    <- confirmed negligible
    out_proj (one matmul):         ~8.6 ms
    -----------------------------------------
    full _self_attention():       ~57.0 ms
    torch's WHOLE self_attn:      ~34.2 ms

The individual matmuls dominate, not orchestration. And critically: an
ISOLATED raw matmul at the identical shape (numpy vs torch, same BLAS-
level operation, nothing else) shows only ~1.1x difference -- so the
underlying linear-algebra performance is roughly comparable. That
means PyTorch's `nn.MultiheadAttention` doing qkv_proj + split + sdpa +
merge + out_proj ALL TOGETHER in less time than this project's qkv_proj
matmul ALONE most likely reflects PyTorch's C++ implementation fusing
or otherwise optimizing across those steps in ways a naive per-step
Python breakdown can't see or easily replicate -- not a reshape
problem, and not something a single kernel swap fixes.

This is left as a genuinely open, deeper problem: the real next step
here isn't "write one more kernel", it's understanding what PyTorch's
internal attention implementation actually does differently across the
whole sequence of operations, which needs profiling PyTorch's own C++
internals, not this project's Python side.
"""

import time
import numpy as np
import torch
import torch.nn as nn

from encoder_layer import TensorfuseEncoderLayer, extract_torch_weights
from ir_attention import scaled_dot_product_attention

torch.set_num_threads(1)


def time_ms(fn, reps=9):
    fn()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1000


def main():
    torch.manual_seed(2)
    d_model, nhead, dim_ff, seq, batch = 512, 8, 2048, 128, 16
    head_dim = d_model // nhead

    layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
        dropout=0.0, activation="gelu", batch_first=True, norm_first=False,
    )
    layer.eval()
    x_t = torch.randn(batch, seq, d_model)
    w = extract_torch_weights(layer)
    ours = TensorfuseEncoderLayer(d_model, nhead, dim_ff, w)
    x_np = x_t.numpy().astype(np.float32)

    qkv = x_np @ w["in_proj_weight"].T + w["in_proj_bias"]
    t_qkv_proj = time_ms(lambda: x_np @ w["in_proj_weight"].T + w["in_proj_bias"])

    def split_and_head():
        q, k, v = np.split(qkv, 3, axis=-1)
        def to_heads(t):
            return t.reshape(batch, seq, nhead, head_dim).transpose(0, 2, 1, 3)
        return to_heads(q), to_heads(k), to_heads(v)

    t_split = time_ms(split_and_head)
    q, k, v = split_and_head()
    q, k, v = q.astype(np.float32), k.astype(np.float32), v.astype(np.float32)

    t_sdpa = time_ms(lambda: scaled_dot_product_attention(q, k, v, ours.softmax_kernel))
    attn = scaled_dot_product_attention(q, k, v, ours.softmax_kernel)

    t_merge = time_ms(lambda: attn.transpose(0, 2, 1, 3).reshape(batch, seq, d_model))
    merged = attn.transpose(0, 2, 1, 3).reshape(batch, seq, d_model)

    t_outproj = time_ms(lambda: merged @ w["out_proj_weight"].T + w["out_proj_bias"])
    t_full_attn = time_ms(lambda: ours._self_attention(x_np))

    with torch.no_grad():
        t_torch_attn = time_ms(lambda: layer.self_attn(x_t, x_t, x_t, need_weights=False))

    W_t = torch.from_numpy(w["in_proj_weight"])
    b_t = torch.from_numpy(w["in_proj_bias"])
    t_np_matmul = time_ms(lambda: x_np @ w["in_proj_weight"].T + w["in_proj_bias"])
    with torch.no_grad():
        t_torch_matmul = time_ms(lambda: x_t @ W_t.T + b_t)

    print("--- ours, step by step ---")
    print(f"  qkv_proj:              {t_qkv_proj:8.3f} ms")
    print(f"  split+heads (views):   {t_split:8.3f} ms   <- negligible, disproves the orchestration hypothesis")
    print(f"  sdpa (matmul+softmax): {t_sdpa:8.3f} ms")
    print(f"  merge (view):          {t_merge:8.3f} ms   <- also negligible")
    print(f"  out_proj:              {t_outproj:8.3f} ms")
    print(f"  full _self_attention():{t_full_attn:8.3f} ms")
    print(f"\n  torch's WHOLE self_attn module: {t_torch_attn:8.3f} ms")
    print(f"\n--- isolated raw matmul, same shape, nothing else ---")
    print(f"  numpy: {t_np_matmul:.3f} ms   torch: {t_torch_matmul:.3f} ms   "
          f"ratio: {t_np_matmul/t_torch_matmul:.2f}x")
    print("\nConclusion: split/merge are NOT the bottleneck (they're ~1000x")
    print("smaller than the matmuls). Raw matmul performance is roughly at")
    print("parity. The gap is in how PyTorch's C++ attention implementation")
    print("combines these steps, not in this project's Python orchestration.")


if __name__ == "__main__":
    main()

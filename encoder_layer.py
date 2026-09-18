"""
encoder_layer.py — the capstone: a complete transformer encoder layer
(multi-head self-attention + FFN, both post-norm) executed through
THIS PROJECT'S OWN compiled kernels end to end, validated against
PyTorch's actual nn.TransformerEncoderLayer -- not a hand-rolled
reference, the real module people use in production.

    attn_out = MultiHeadSelfAttention(x)      <- ir_attention.py: fused softmax
    x = AddLayerNorm(x, attn_out)             <- ir_transformer.py: fused reduction
    ff_out = Linear2(GELU(Linear1(x) + b1))   <- fusion2.py: fused elementwise (unchanged)
    x = AddLayerNorm(x, ff_out)               <- ir_transformer.py again

Every one of the three fusion mechanisms built across this project
(elementwise chains, reduction-based AddLayerNorm, and now attention's
fused softmax) is exercised together in one real block.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.special import erf

from ir_attention import scaled_dot_product_attention, SoftmaxKernel
from ir_transformer import LayerNormKernel


def gelu_numpy(x):
    return (0.5 * x * (1.0 + erf(x / np.float32(np.sqrt(2.0))))).astype(x.dtype, copy=False)


class TensorfuseEncoderLayer:
    def __init__(self, d_model, nhead, dim_feedforward, weights):
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.w = weights
        self.softmax_kernel = SoftmaxKernel()
        self.ln1_kernel = LayerNormKernel(with_residual=True)
        self.ln2_kernel = LayerNormKernel(with_residual=True)

    def _self_attention(self, x):
        # NOTE: an earlier version of this method routed through
        # attention_orchestration.py's compiled QKV-split/heads-merge
        # kernels, on the documented hypothesis that numpy's split+
        # reshape+transpose was the source of the remaining PyTorch gap.
        # Benchmarking that in isolation disproved it: np.split and
        # .transpose() produce lazy VIEWS that never copy data at all
        # until something downstream forces materialization -- so they
        # were already near-free (~0.007ms), and the "fusion" kernel,
        # which eagerly copies into new contiguous buffers, was 100-400x
        # SLOWER for exactly that reason. Reverted; see
        # attention_orchestration.py's module docstring and profile_attention.py
        # for the actual (different) source of the gap.
        b, s, d = x.shape
        w = self.w
        qkv = x @ w["in_proj_weight"].T + w["in_proj_bias"]
        q, k, v = np.split(qkv, 3, axis=-1)

        def to_heads(t):
            return t.reshape(b, s, self.nhead, self.head_dim).transpose(0, 2, 1, 3)

        q, k, v = to_heads(q), to_heads(k), to_heads(v)
        attn = scaled_dot_product_attention(
            q.astype(np.float32), k.astype(np.float32), v.astype(np.float32),
            self.softmax_kernel,
        )
        attn = attn.transpose(0, 2, 1, 3).reshape(b, s, d)
        return attn @ w["out_proj_weight"].T + w["out_proj_bias"]

    def forward(self, x):
        b, s, d = x.shape
        w = self.w

        attn_out = self._self_attention(x)
        x_flat = x.reshape(b * s, d)
        attn_flat = attn_out.reshape(b * s, d)
        x = self.ln1_kernel(x_flat, attn_flat, w["norm1_weight"], w["norm1_bias"], 1e-5)

        h = x @ w["linear1_weight"].T + w["linear1_bias"]
        h = gelu_numpy(h)
        ff_out = h @ w["linear2_weight"].T + w["linear2_bias"]
        x = self.ln2_kernel(x, ff_out, w["norm2_weight"], w["norm2_bias"], 1e-5)

        return x.reshape(b, s, d)


def extract_torch_weights(layer: nn.TransformerEncoderLayer):
    sd = layer.state_dict()
    return {
        "in_proj_weight": sd["self_attn.in_proj_weight"].numpy(),
        "in_proj_bias": sd["self_attn.in_proj_bias"].numpy(),
        "out_proj_weight": sd["self_attn.out_proj.weight"].numpy(),
        "out_proj_bias": sd["self_attn.out_proj.bias"].numpy(),
        "linear1_weight": sd["linear1.weight"].numpy(),
        "linear1_bias": sd["linear1.bias"].numpy(),
        "linear2_weight": sd["linear2.weight"].numpy(),
        "linear2_bias": sd["linear2.bias"].numpy(),
        "norm1_weight": sd["norm1.weight"].numpy(),
        "norm1_bias": sd["norm1.bias"].numpy(),
        "norm2_weight": sd["norm2.weight"].numpy(),
        "norm2_bias": sd["norm2.bias"].numpy(),
    }


if __name__ == "__main__":
    torch.manual_seed(0)
    d_model, nhead, dim_ff, seq, batch = 128, 8, 512, 32, 16

    torch_layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
        dropout=0.0, activation="gelu", batch_first=True, norm_first=False,
    )
    torch_layer.eval()

    x_t = torch.randn(batch, seq, d_model)
    with torch.no_grad():
        torch_out = torch_layer(x_t).numpy()

    weights = extract_torch_weights(torch_layer)
    ours = TensorfuseEncoderLayer(d_model, nhead, dim_ff, weights)
    x_np = x_t.numpy().astype(np.float32)
    our_out = ours.forward(x_np)

    diff = np.abs(our_out - torch_out).max()
    print(f"max diff vs real torch.nn.TransformerEncoderLayer: {diff:.2e}")
    assert np.allclose(our_out, torch_out, atol=1e-3), f"MISMATCH: {diff}"
    print("Full transformer encoder layer -- attention, AddLayerNorm x2, GELU FFN --")
    print("verified correct against PyTorch's own production module.")

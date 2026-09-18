"""
ir_cnn.py — Conv2D, MaxPool2D, Flatten: structural (non-elementwise)
ops. fusion2.py already passes any op outside ELEMENTWISE_KINDS
through untouched, so no change to the fusion pass was needed to
support these -- only new op kinds and an executor for them.

Conv2D bakes in the bias-add (like Gemm does for matmul), so the op
immediately after it in a real CNN is a plain unary ReLU with no
operand -- which the EXISTING generic elementwise fusion/codegen
already handles with no changes, since a no-operand unary chain needs
no broadcast-axis handling at all.
"""

import numpy as np
from ir import Graph, Op

Graph_CONV_COUNTER = [0]


def conv2d(self, x, weight, bias, stride=1, padding=0):
    name = self._fresh_name()
    self.ops.append(Op(name, "Conv2D", [x, weight, bias],
                        {"stride": stride, "padding": padding}))
    return name


def maxpool2d(self, x, kernel=2, stride=2):
    name = self._fresh_name()
    self.ops.append(Op(name, "MaxPool2D", [x], {"kernel": kernel, "stride": stride}))
    return name


def flatten(self, x, start_axis=1):
    name = self._fresh_name()
    self.ops.append(Op(name, "Flatten", [x], {"start_axis": start_axis}))
    return name


Graph.conv2d = conv2d
Graph.maxpool2d = maxpool2d
Graph.flatten = flatten


def _im2col(x, kh, kw, stride, padding):
    n, c, h, w = x.shape
    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    h_pad, w_pad = x.shape[2], x.shape[3]
    out_h = (h_pad - kh) // stride + 1
    out_w = (w_pad - kw) // stride + 1

    cols = np.empty((n, c, kh, kw, out_h, out_w), dtype=x.dtype)
    for i in range(kh):
        i_max = i + stride * out_h
        for j in range(kw):
            j_max = j + stride * out_w
            cols[:, :, i, j, :, :] = x[:, :, i:i_max:stride, j:j_max:stride]
    cols = cols.reshape(n, c * kh * kw, out_h * out_w)
    return cols, out_h, out_w


def conv2d_forward(x, weight, bias, stride, padding):
    """
    x: (N, C_in, H, W)
    weight: (C_out, C_in, KH, KW)
    bias: (C_out,)
    Returns (N, C_out, out_H, out_W). Correct, im2col+matmul based --
    not autotuned/fused (documented as future work), just correct.
    """
    n, c_in, h, w = x.shape
    c_out, c_in_w, kh, kw = weight.shape
    assert c_in == c_in_w

    cols, out_h, out_w = _im2col(x, kh, kw, stride, padding)   # (N, C_in*KH*KW, out_H*out_W)
    W_flat = weight.reshape(c_out, -1)                          # (C_out, C_in*KH*KW)

    out = np.einsum("oc,ncp->nop", W_flat, cols)                # (N, C_out, out_H*out_W)
    out = out.reshape(n, c_out, out_h, out_w)
    out += bias.reshape(1, c_out, 1, 1)
    return out


def maxpool2d_forward(x, kernel, stride):
    n, c, h, w = x.shape
    out_h = (h - kernel) // stride + 1
    out_w = (w - kernel) // stride + 1
    # valid only for kernel == stride, no padding (matches this project's CNN) --
    # exactly what a real exported MaxPool with those params requires anyway.
    assert h % stride == 0 and w % stride == 0 and kernel == stride, \
        "maxpool2d_forward only supports kernel==stride, evenly-dividing input (documented limitation)"
    reshaped = x.reshape(n, c, out_h, stride, out_w, stride)
    return reshaped.max(axis=(3, 5))


def flatten_forward(x, start_axis):
    shape = x.shape[:start_axis] + (-1,)
    return x.reshape(shape)

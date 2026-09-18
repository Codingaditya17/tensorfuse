"""
onnx_frontend.py — imports a real ONNX model into this project's own
Graph IR, so the SAME fusion pass and autotuning pipeline built for
synthetic graphs runs, unmodified, on a real exported model.

ONNX's Gemm(A, B, C) computes alpha*A@B (+ B transposed if transB) + beta*C
in one node. Our IR doesn't have a fused Gemm primitive -- it has
MatMul and Add as separate ops -- so each Gemm is lowered to
MatMul(x, W) -> Add(_, bias) here at import time. This is itself a
tiny, honest piece of compiler work: translating one op set to
another, not just a format conversion.
"""

import numpy as np
import onnx
from onnx import numpy_helper

from ir import Graph
import ir2  # noqa: F401 -- registers g.sigmoid/tanh/mul/sub on Graph
import ir_cnn  # noqa: F401 -- registers g.conv2d/maxpool2d/flatten on Graph

_SUPPORTED = {"Gemm", "Relu", "Sigmoid", "Tanh", "Conv", "MaxPool", "Flatten"}


def load_onnx_graph(path):
    """
    Returns (graph, input_name, weights: dict[str, np.ndarray], output_name)
    weights maps our IR's Param names directly to concrete numpy arrays,
    ready to hand to the executor as part of `values`.
    """
    model = onnx.load(path)
    onnx.checker.check_model(model)
    onnx_graph = model.graph

    initializers = {init.name: numpy_helper.to_array(init).astype(np.float32)
                     for init in onnx_graph.initializer}

    g = Graph()
    name_map = {}   # ONNX tensor name -> our IR SSA name
    weights = {}    # our IR Param name -> concrete array

    onnx_input_name = onnx_graph.input[0].name
    x_name = g.input(onnx_input_name)
    name_map[onnx_input_name] = x_name

    for node in onnx_graph.node:
        if node.op_type not in _SUPPORTED:
            raise ValueError(
                f"onnx_frontend only supports {_SUPPORTED}, got '{node.op_type}'. "
                f"(Not a limitation of the fusion pass -- just this importer.)"
            )

        if node.op_type == "Gemm":
            attrs = {a.name: a for a in node.attribute}
            trans_b = attrs.get("transB")
            trans_b = bool(trans_b.i) if trans_b is not None else False
            alpha = attrs["alpha"].f if "alpha" in attrs else 1.0
            beta = attrs["beta"].f if "beta" in attrs else 1.0
            assert alpha == 1.0 and beta == 1.0, "only alpha=beta=1 supported"

            x_in, w_in, b_in = node.input
            W = initializers[w_in]
            if trans_b:
                W = W.T.copy()  # ONNX stores (out,in) with transB=1; we want (in,out)
            bias = initializers[b_in]

            w_param = g.param(w_in)
            weights[w_param] = W
            b_param = g.param(b_in)
            weights[b_param] = bias

            mm = g.matmul(name_map[x_in], w_param)
            added = g.add(mm, b_param)
            name_map[node.output[0]] = added

        elif node.op_type == "Relu":
            (x_in,) = node.input
            name_map[node.output[0]] = g.relu(name_map[x_in])

        elif node.op_type == "Sigmoid":
            (x_in,) = node.input
            name_map[node.output[0]] = g.sigmoid(name_map[x_in])

        elif node.op_type == "Tanh":
            (x_in,) = node.input
            name_map[node.output[0]] = g.tanh(name_map[x_in])

        elif node.op_type == "Conv":
            attrs = {a.name: a for a in node.attribute}
            strides = list(attrs["strides"].ints) if "strides" in attrs else [1, 1]
            pads = list(attrs["pads"].ints) if "pads" in attrs else [0, 0, 0, 0]
            assert strides[0] == strides[1], "only symmetric stride supported"
            assert len(set(pads)) == 1, "only symmetric padding supported"

            x_in, w_in, b_in = node.input
            weight = initializers[w_in]
            bias = initializers[b_in]
            w_param = g.param(w_in)
            weights[w_param] = weight
            b_param = g.param(b_in)
            weights[b_param] = bias

            out = g.conv2d(name_map[x_in], w_param, b_param,
                            stride=strides[0], padding=pads[0])
            name_map[node.output[0]] = out

        elif node.op_type == "MaxPool":
            attrs = {a.name: a for a in node.attribute}
            kernel = list(attrs["kernel_shape"].ints)
            strides = list(attrs["strides"].ints) if "strides" in attrs else kernel
            assert kernel[0] == kernel[1] and strides[0] == strides[1]
            (x_in,) = node.input
            out = g.maxpool2d(name_map[x_in], kernel=kernel[0], stride=strides[0])
            name_map[node.output[0]] = out

        elif node.op_type == "Flatten":
            attrs = {a.name: a for a in node.attribute}
            axis = attrs["axis"].i if "axis" in attrs else 1
            (x_in,) = node.input
            out = g.flatten(name_map[x_in], start_axis=axis)
            name_map[node.output[0]] = out

    onnx_output_name = onnx_graph.output[0].name
    out_name = name_map[onnx_output_name]

    return g, x_name, weights, out_name


if __name__ == "__main__":
    g, x_name, weights, out_name = load_onnx_graph("results/model.onnx")
    print(g.pretty())
    print("\nweights:", {k: v.shape for k, v in weights.items()})
    print("output:", out_name)

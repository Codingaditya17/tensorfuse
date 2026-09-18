"""
export_onnx.py — exports the trained MLP to a real .onnx file using
onnx.helper directly (no torch dependency needed). Produces a
standard graph any tool (Netron, onnxruntime, this project's own
importer) can load: Gemm -> Relu -> Gemm -> Relu -> Gemm -> Sigmoid.

Gemm's `transB=1` matches how frameworks typically store a Linear
layer's weight as (out_features, in_features) — so this also exercises
the transpose-handling our own importer needs to get right.
"""

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


def export(params, path="results/model.onnx"):
    # Gemm expects weight as (out, in) when transB=1 -- our numpy
    # weights are stored (in, out), so transpose for the ONNX file.
    W1t = params["W1"].T.copy()
    W2t = params["W2"].T.copy()
    W3t = params["W3"].T.copy()

    def init(name, arr):
        return numpy_helper.from_array(arr.astype(np.float32), name=name)

    initializers = [
        init("W1", W1t), init("b1", params["b1"]),
        init("W2", W2t), init("b2", params["b2"]),
        init("W3", W3t), init("b3", params["b3"]),
    ]

    nodes = [
        helper.make_node("Gemm", ["x", "W1", "b1"], ["z1"], transB=1, name="gemm1"),
        helper.make_node("Relu", ["z1"], ["h1"], name="relu1"),
        helper.make_node("Gemm", ["h1", "W2", "b2"], ["z2"], transB=1, name="gemm2"),
        helper.make_node("Relu", ["z2"], ["h2"], name="relu2"),
        helper.make_node("Gemm", ["h2", "W3", "b3"], ["z3"], transB=1, name="gemm3"),
        helper.make_node("Sigmoid", ["z3"], ["out"], name="sigmoid_out"),
    ]

    graph = helper.make_graph(
        nodes,
        "tensorfuse_demo_mlp",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 2])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, ["batch", 1])],
        initializer=initializers,
    )

    model = helper.make_model(graph, producer_name="tensorfuse",
                               opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"Saved {path}")
    return path


if __name__ == "__main__":
    data = np.load("results/trained_weights.npz")
    params = {k: data[k] for k in data.files}
    export(params)

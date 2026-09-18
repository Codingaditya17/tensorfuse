"""
demo_graph.py — the graph used for the end-to-end benchmark.

    h1  = ReLU(x @ W1 + b1)                     <- simple chain: Add, ReLU
                                                    (AVX-512 eligible)
    h2  = ReLU(Sigmoid(h1 @ W2 + b2) * scale)   <- complex chain: Add,
                                                    Sigmoid, Mul, ReLU
                                                    (scalar-only, has expf)
    out = h2

One graph, two different fused kernels, two different autotuning
outcomes — exercises both backends in one forward pass.
"""

from ir import Graph
import ir2  # noqa: F401  (registers g.sigmoid/mul/etc. on Graph)


def build_demo_graph():
    g = Graph()
    x = g.input("x")
    W1 = g.param("W1")
    b1 = g.param("b1")
    W2 = g.param("W2")
    b2 = g.param("b2")
    scale = g.param("scale")

    mm1 = g.matmul(x, W1)
    a1 = g.add(mm1, b1)
    h1 = g.relu(a1)

    mm2 = g.matmul(h1, W2)
    a2 = g.add(mm2, b2)
    sig = g.sigmoid(a2)
    gated = g.mul(sig, scale)
    h2 = g.relu(gated)

    params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "scale": scale}
    return g, x, params, h2


if __name__ == "__main__":
    g, x, params, out = build_demo_graph()
    print(g.pretty())

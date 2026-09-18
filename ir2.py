"""
ir2.py — extends ir.py with more elementwise ops, for generalized fusion.

Elementwise ops now include:
    ReLU(x), Sigmoid(x), Tanh(x), Neg(x)        -- unary
    Add(x, y), Mul(x, y), Sub(x, y)             -- binary, y broadcast over cols

These are exactly the primitives needed to express things like
GLU-ish blocks, gated activations, normalization-adjacent scaling,
etc. — richer than a single fixed bias+relu pattern.
"""

from ir import Graph, Op  # reuse Op/Graph, just add more builder methods

ELEMENTWISE_UNARY = {"ReLU", "Sigmoid", "Tanh", "Neg", "GELU"}
ELEMENTWISE_BINARY = {"Add", "Mul", "Sub"}
ELEMENTWISE_KINDS = ELEMENTWISE_UNARY | ELEMENTWISE_BINARY


def gelu(self, x: str) -> str:
    name = self._fresh_name()
    self.ops.append(Op(name, "GELU", [x]))
    return name


def mul(self, x: str, y: str) -> str:
    name = self._fresh_name()
    self.ops.append(Op(name, "Mul", [x, y]))
    return name


def sub(self, x: str, y: str) -> str:
    name = self._fresh_name()
    self.ops.append(Op(name, "Sub", [x, y]))
    return name


def sigmoid(self, x: str) -> str:
    name = self._fresh_name()
    self.ops.append(Op(name, "Sigmoid", [x]))
    return name


def tanh(self, x: str) -> str:
    name = self._fresh_name()
    self.ops.append(Op(name, "Tanh", [x]))
    return name


def neg(self, x: str) -> str:
    name = self._fresh_name()
    self.ops.append(Op(name, "Neg", [x]))
    return name


# monkey-patch onto Graph so build scripts can call g.mul(...), g.sigmoid(...), etc.
Graph.mul = mul
Graph.sub = sub
Graph.sigmoid = sigmoid
Graph.tanh = tanh
Graph.neg = neg
Graph.gelu = gelu


def build_gated_mlp_graph():
    """
    A graph that exercises a LONGER elementwise chain and a
    deliberate fan-out (residual-style reuse), to stress-test the
    generalized fusion pass beyond the simple bias+relu case:

        h1 = x @ W1 + b1
        g  = Sigmoid(h1)          \\
        t  = Tanh(h1)              >  gated activation, both read h1 (fan-out)
        gated = g * t              /
        out = gated + bias2       -- another elementwise op after the fan-in
        out = ReLU(out)
    """
    g = Graph()
    x = g.input("x")
    W1 = g.param("W1")
    b1 = g.param("b1")
    b2 = g.param("b2")

    h1_mm = g.matmul(x, W1)
    h1 = g.add(h1_mm, b1)          # Add fused with matmul-consumer chain start
    sig = g.sigmoid(h1)            # h1 has TWO consumers (sig, tan) -> fan-out
    tan = g.tanh(h1)
    gated = g.mul(sig, tan)        # fan-in: consumes two elementwise results
    biased = g.add(gated, b2)
    out = g.relu(biased)

    return g, x, {"W1": W1, "b1": b1, "b2": b2}, out


if __name__ == "__main__":
    g, x, params, out = build_gated_mlp_graph()
    print(g.pretty())

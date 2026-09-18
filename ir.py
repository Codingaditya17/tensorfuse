"""
ir.py — Minimal tensor-graph IR.

This mirrors (at toy scale) what a real ML compiler frontend does:
translate a computation graph into a small set of typed IR nodes that
later passes can pattern-match and rewrite.

Supported ops (deliberately small, like the graphs in the job specs
this project is based on):
    MatMul(a, b)      -> a @ b
    Add(x, y)         -> x + y            (y is broadcastable, e.g. bias)
    ReLU(x)           -> max(x, 0)

A Graph is a topologically-ordered list of Op nodes. Each Op has a
unique output name so later ops can reference it by name (SSA-style,
like real IRs).
"""

from dataclasses import dataclass, field
from typing import List, Union


@dataclass
class Op:
    name: str            # SSA-style output name, e.g. "%3"
    kind: str            # "MatMul" | "Add" | "ReLU" | "Input" | "Param"
    inputs: List[str]    # names of operands (empty for Input/Param)
    meta: dict = field(default_factory=dict)  # extra info (shapes, etc.)

    def __repr__(self):
        ins = ", ".join(self.inputs)
        return f"{self.name} = {self.kind}({ins})"


class Graph:
    """A simple topologically-ordered computation graph."""

    def __init__(self):
        self.ops: List[Op] = []
        self._counter = 0

    def _fresh_name(self) -> str:
        self._counter += 1
        return f"%{self._counter}"

    def input(self, label: str) -> str:
        name = self._fresh_name()
        self.ops.append(Op(name, "Input", [], {"label": label}))
        return name

    def param(self, label: str) -> str:
        name = self._fresh_name()
        self.ops.append(Op(name, "Param", [], {"label": label}))
        return name

    def matmul(self, a: str, b: str) -> str:
        name = self._fresh_name()
        self.ops.append(Op(name, "MatMul", [a, b]))
        return name

    def add(self, x: str, y: str) -> str:
        name = self._fresh_name()
        self.ops.append(Op(name, "Add", [x, y]))
        return name

    def relu(self, x: str) -> str:
        name = self._fresh_name()
        self.ops.append(Op(name, "ReLU", [x]))
        return name

    def pretty(self) -> str:
        return "\n".join(repr(op) for op in self.ops)


def build_mlp_graph(num_layers: int = 2):
    """
    Build the canonical example this project targets:
    an N-layer MLP forward pass expressed as a graph:

        h = x
        for each layer:
            h = ReLU(h @ W + b)
        return h

    Returns (graph, x_name, [(W_name, b_name), ...], output_name)
    """
    g = Graph()
    x = g.input("x")
    h = x
    weight_names = []
    for i in range(num_layers):
        w = g.param(f"W{i}")
        b = g.param(f"b{i}")
        mm = g.matmul(h, w)
        added = g.add(mm, b)
        h = g.relu(added)
        weight_names.append((w, b))
    return g, x, weight_names, h


if __name__ == "__main__":
    g, x, weights, out = build_mlp_graph(2)
    print("=== Unfused IR ===")
    print(g.pretty())

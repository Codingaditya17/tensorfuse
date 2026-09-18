"""
fusion.py — Operator fusion pass.

Pattern:  Add(MatMul(x, W), b)  ->  ReLU(...)

We rewrite that 3-op chain into a single FusedBiasReLU node whose
codegen (see codegen.py) does the bias-add and the ReLU in ONE pass
over memory instead of two. The MatMul itself is left alone — it's
already calling into BLAS, and no toy C loop should try to beat a
tuned GEMM.

This mirrors the real "fusion" pass stage described in the job specs
this project is based on: multiple math ops collapsed into a single
kernel launch to cut HBM round-trips.
"""

from ir import Graph, Op


def find_uses(graph: Graph):
    """Map: op_name -> list of ops that consume it as an input."""
    uses = {op.name: [] for op in graph.ops}
    for op in graph.ops:
        for inp in op.inputs:
            uses.setdefault(inp, []).append(op.name)
    return uses


def fuse_bias_relu(graph: Graph) -> Graph:
    by_name = {op.name: op for op in graph.ops}
    uses = find_uses(graph)

    new_ops = []
    fused_count = 0
    skip = set()

    for op in graph.ops:
        if op.name in skip:
            continue

        if op.kind == "Add":
            mm_name, bias_name = op.inputs
            mm_op = by_name.get(mm_name)
            is_matmul_input = mm_op is not None and mm_op.kind == "MatMul"
            # Add's result must be consumed by exactly one ReLU for a
            # safe fusion (no other consumer needs the un-activated value).
            consumers = uses.get(op.name, [])
            single_relu = (
                len(consumers) == 1 and by_name[consumers[0]].kind == "ReLU"
            )
            if is_matmul_input and single_relu:
                relu_op = by_name[consumers[0]]
                fused = Op(
                    relu_op.name,          # keep the ReLU's output name
                    "FusedBiasReLU",
                    [mm_name, bias_name],  # matmul result, bias
                )
                new_ops.append(fused)
                skip.add(relu_op.name)     # don't emit the old ReLU
                fused_count += 1
                continue

        new_ops.append(op)

    fused_graph = Graph()
    fused_graph.ops = new_ops
    fused_graph._counter = graph._counter
    return fused_graph, fused_count


if __name__ == "__main__":
    from ir import build_mlp_graph

    g, x, weights, out = build_mlp_graph(2)
    print("=== Before fusion ===")
    print(g.pretty())

    fg, n = fuse_bias_relu(g)
    print(f"\n=== After fusion ({n} Add+ReLU pairs fused) ===")
    print(fg.pretty())

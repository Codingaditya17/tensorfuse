"""
fusion2.py — generalized elementwise fusion (not one fixed pattern).

Rule for extending a fusion chain into `op`:
  1. op.kind is elementwise (unary or binary).
  2. op.inputs[0] (the "data" input) is the tail of an existing chain
     (or any single-use producer, to start a new chain).
  3. That producer has exactly ONE consumer (uses[name] == 1) — if a
     value is read by more than one downstream op (a residual/gated
     branch), we must NOT fuse across it, or we'd silently duplicate
     or drop work. This is the correctness-critical part.
  4. If op is binary (Add/Mul/Sub), the SECOND input must be a "leaf"
     tensor (Param/Input/MatMul result) — not another elementwise
     result. A binary op whose two inputs are both freshly-computed
     activations (e.g. Sigmoid(h) * Tanh(h)) is a real fan-in and is
     NOT something this pass fuses; it's left as a plain op, correctly
     executed by the interpreter instead of silently mishandled.

A chain only becomes a FusedElementwise node if it has >= 2 members —
fusing a single op buys nothing over calling it directly.
"""

from ir import Graph, Op
from ir2 import ELEMENTWISE_KINDS


def compute_uses(graph: Graph):
    uses = {op.name: 0 for op in graph.ops}
    for op in graph.ops:
        for inp in op.inputs:
            uses[inp] = uses.get(inp, 0) + 1
    return uses


def fuse_elementwise(graph: Graph):
    by_name = {op.name: op for op in graph.ops}
    uses = compute_uses(graph)

    # group_of[name] -> index into `groups`, for names that are the
    # current TAIL of an open chain (so a later op can extend it)
    tail_to_group = {}
    groups = []  # list of list[Op]

    for op in graph.ops:
        if op.kind not in ELEMENTWISE_KINDS:
            continue  # MatMul / Input / Param: never grouped

        # Eligibility: the codegen's operand model assumes a binary op's
        # second input is a 1D vector broadcast over columns (a bias or
        # scale parameter) -- NOT a full tensor of the same shape as the
        # primary input. The only IR node kind that is actually such a
        # broadcast vector in this project is Param. Anything else --
        # another elementwise result (a computed activation, real
        # fan-in) OR an Input/MatMul result (e.g. a residual connection,
        # which is full-shape, not a broadcast vector) -- must be
        # excluded from grouping entirely, or the compiled kernel would
        # silently misread a full 2D tensor as if it were a (cols,)
        # vector (this exact bug: Add(ff, x) in a residual connection
        # got fused as if x were a bias, reading only x's row 0 for
        # every row). Such an op falls through to the plain per-op
        # numpy interpreter instead, which handles arbitrary same-shape
        # tensor-tensor ops correctly regardless of origin.
        if len(op.inputs) == 2:
            operand_op = by_name.get(op.inputs[1])
            operand_is_broadcast_vector = (
                operand_op is not None and operand_op.kind == "Param"
            )
            if not operand_is_broadcast_vector:
                continue  # never registered -> stays standalone, never grouped

        root = op.inputs[0]
        can_extend = (
            root in tail_to_group
            and uses.get(root, 0) == 1
        )

        if can_extend:
            gidx = tail_to_group[root]
            groups[gidx].append(op)
            del tail_to_group[root]
            tail_to_group[op.name] = gidx
        else:
            groups.append([op])
            tail_to_group[op.name] = len(groups) - 1

    # Build the rewritten op list, replacing groups of len>=2 with
    # one FusedElementwise node.
    fused_result_name = {}   # last-op-name-in-group -> True if it was fused
    replace_with = {}        # last-op-name-in-group -> new Op (if fused)
    members_of = {}          # any member name in a fused group -> group members

    for group in groups:
        if len(group) >= 2:
            names_in_group = {m.name for m in group}
            seq = []           # [(kind, operand_name_or_None), ...]
            operand_names = []
            for m in group:
                if len(m.inputs) == 2:
                    seq.append((m.kind, m.inputs[1]))
                    operand_names.append(m.inputs[1])
                else:
                    seq.append((m.kind, None))
            root_input = group[0].inputs[0]
            # SAFETY: the compiled kernel mutates its root buffer in
            # place. That's only safe if this fused group is the ONLY
            # consumer of root_input in the whole graph -- if root_input
            # is read again elsewhere (e.g. a value that both feeds a
            # fusible chain AND is used unchanged by another branch,
            # like many residual/shared-feature patterns), in-place
            # mutation would silently corrupt that other read. `uses`
            # here is the ORIGINAL graph's global use-count, computed
            # once up front, so this is free to check.
            root_exclusive = uses.get(root_input, 0) == 1
            new_op = Op(
                group[-1].name,
                "FusedElementwise",
                [root_input] + operand_names,
                {"seq": seq, "root_exclusive": root_exclusive},
            )
            replace_with[group[-1].name] = new_op
            for m in group:
                members_of[m.name] = names_in_group

    new_ops = []
    skip_names = set()
    for grp_names in members_of.values():
        skip_names |= grp_names
    # the fused node itself keeps the LAST member's name, so un-skip that one
    for last_name in replace_with:
        skip_names.discard(last_name)

    for op in graph.ops:
        if op.name in replace_with:
            new_ops.append(replace_with[op.name])
        elif op.name in skip_names:
            continue  # absorbed into a fused node upstream
        else:
            new_ops.append(op)

    new_graph = Graph()
    new_graph.ops = new_ops
    new_graph._counter = graph._counter
    n_fused_nodes = len(replace_with)
    n_ops_absorbed = sum(len(g) for g in groups if len(g) >= 2)
    return new_graph, n_fused_nodes, n_ops_absorbed


if __name__ == "__main__":
    from ir2 import build_gated_mlp_graph

    g, x, params, out = build_gated_mlp_graph()
    print("=== Before fusion ===")
    print(g.pretty())

    fg, n_nodes, n_absorbed = fuse_elementwise(g)
    print(f"\n=== After fusion ({n_nodes} fused node(s), {n_absorbed} ops absorbed) ===")
    print(fg.pretty())
    print("\nExpectation: the Sigmoid/Tanh/Mul fan-out+fan-in region stays "
          "UNFUSED (correctness), but the trailing Add(b2)->ReLU chain fuses.")

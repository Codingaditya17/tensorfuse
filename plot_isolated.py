import csv
import matplotlib.pyplot as plt
from collections import defaultdict

rows = []
with open("results/benchmark_isolated.csv") as f:
    for r in csv.DictReader(f):
        rows.append({
            "sequence": r["sequence"],
            "rows": int(r["rows"]),
            "cols": int(r["cols"]),
            "unfused_ms": float(r["unfused_ms"]),
            "untuned_ms": float(r["untuned_ms"]),
            "auto_ms": float(r["auto_ms"]),
            "winner": r["winner"],
        })

by_seq = defaultdict(list)
for r in rows:
    by_seq[r["sequence"]].append(r)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

for ax, (seq_label, group) in zip(axes, by_seq.items()):
    group = sorted(group, key=lambda r: r["rows"] * r["cols"])
    x = list(range(len(group)))
    xlabels = [f"{r['rows']}x{r['cols']}" for r in group]

    ax.plot(x, [r["unfused_ms"] for r in group], marker="o", label="Unfused (numpy)")
    ax.plot(x, [r["untuned_ms"] for r in group], marker="s", label="Fused, untuned (scalar unroll=1)")
    ax.plot(x, [r["auto_ms"] for r in group], marker="^", label="Fused, autotuned")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=30, ha="right")
    ax.set_ylabel("Latency (ms, log scale)")
    ax.set_title(seq_label)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    for xi, r in zip(x, group):
        ax.annotate(r["winner"], (xi, r["auto_ms"]), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=7, color="green")

fig.suptitle("Fusion is not a free win: simple chains win big, transcendental-heavy\n"
             "chains lose at small sizes until array size outweighs libm cost",
             fontsize=12)
fig.tight_layout()
fig.savefig("results/isolated_chart.png", dpi=150)
print("Saved results/isolated_chart.png")

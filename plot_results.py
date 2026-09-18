import csv
import matplotlib.pyplot as plt
from collections import defaultdict

rows = []
with open("results/benchmark.csv") as f:
    for r in csv.DictReader(f):
        rows.append({
            "batch": int(r["batch"]),
            "hidden": int(r["hidden"]),
            "unfused_ms": float(r["unfused_ms"]),
            "fused_ms": float(r["fused_ms"]),
            "speedup": float(r["speedup"]),
        })

by_hidden = defaultdict(list)
for r in rows:
    by_hidden[r["hidden"]].append(r)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: latency, unfused vs fused, for the largest hidden size
hidden_pick = max(by_hidden.keys())
subset = sorted(by_hidden[hidden_pick], key=lambda r: r["batch"])
batches = [r["batch"] for r in subset]
ax = axes[0]
ax.plot(batches, [r["unfused_ms"] for r in subset], marker="o", label="Unfused (numpy Add + ReLU)")
ax.plot(batches, [r["fused_ms"] for r in subset], marker="o", label="Fused (compiled kernel)")
ax.set_xscale("log", base=2)
ax.set_xlabel("Batch size")
ax.set_ylabel("Latency (ms)")
ax.set_title(f"MLP forward latency (hidden={hidden_pick})")
ax.legend()
ax.grid(alpha=0.3)

# Right: speedup heatmap-ish, line per hidden size
ax = axes[1]
for hidden, group in sorted(by_hidden.items()):
    group = sorted(group, key=lambda r: r["batch"])
    ax.plot([r["batch"] for r in group], [r["speedup"] for r in group],
             marker="o", label=f"hidden={hidden}")
ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
ax.set_xscale("log", base=2)
ax.set_xlabel("Batch size")
ax.set_ylabel("Speedup (unfused_ms / fused_ms)")
ax.set_title("Fusion speedup vs. size")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.suptitle("Operator Fusion: bias-add + ReLU fused into one compiled pass", fontsize=13)
fig.tight_layout()
fig.savefig("results/benchmark_chart.png", dpi=150)
print("Saved results/benchmark_chart.png")

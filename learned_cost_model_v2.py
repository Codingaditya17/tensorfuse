"""
learned_cost_model_v2.py — fixes the exact misprediction documented in
learned_cost_model.py: a single binary `has_transcendental` feature
can't tell Tanh (much slower per-element in this project's naive
scalar kernel) from Sigmoid (less slow), so the v1 model over-predicted
fusion for Add-Tanh at its largest tested size.

Fix: one indicator feature PER op kind that appears in the sweep,
instead of one collapsed "has_transcendental" bit. The model can now
learn that Tanh's coefficient should be more negative than Sigmoid's,
because the training data actually shows that.
"""

import csv
import json
import numpy as np

ALL_KINDS = ["Add", "Mul", "Sub", "Neg", "ReLU", "Sigmoid", "Tanh"]
FEATURE_NAMES = ["bias", "log2_n_elements", "n_ops"] + [f"has_{k}" for k in ALL_KINDS]

HOLDOUT_SEQUENCES = {"Add-Tanh", "Mul-Add-ReLU"}


def load_sweep(path="results/cost_sweep.csv"):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "sequence": r["sequence"],
                "n_elements": int(r["n_elements"]),
                "n_ops": int(r["n_ops"]),
                "won": int(r["won"]),
            })
    return rows


def featurize(sequence_label, n_elements, n_ops):
    kinds_present = set(sequence_label.split("-"))
    x = [1.0, np.log2(n_elements), n_ops]
    x += [1.0 if k in kinds_present else 0.0 for k in ALL_KINDS]
    return np.array(x)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def train_logistic(X, y, lr=0.2, epochs=8000, l2=0.01):
    mean = X[:, 1:3].mean(axis=0)
    std = X[:, 1:3].std(axis=0) + 1e-8
    Xn = X.copy()
    Xn[:, 1:3] = (X[:, 1:3] - mean) / std

    w = np.zeros(X.shape[1])
    n = len(y)
    for epoch in range(epochs):
        z = Xn @ w
        p = sigmoid(z)
        grad = Xn.T @ (p - y) / n
        grad[1:] += l2 * w[1:] / n
        w -= lr * grad
        if epoch % 2000 == 0:
            loss = -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))
            acc = ((p > 0.5) == y).mean()
            print(f"  epoch {epoch:5d}  loss={loss:.4f}  train_acc={acc:.3f}")
    return w, mean, std


class LearnedCostModelV2:
    def __init__(self, weights_path="results/learned_cost_model_v2.json"):
        with open(weights_path) as f:
            d = json.load(f)
        self.w = np.array(d["weights"])
        self.mean = np.array(d["mean"])
        self.std = np.array(d["std"])

    def predict_proba(self, seq, rows, cols):
        n_elements = rows * cols
        n_ops = len(seq)
        kinds_present = {k for k, _ in seq}
        x = np.array([1.0, np.log2(n_elements), n_ops] +
                      [1.0 if k in kinds_present else 0.0 for k in ALL_KINDS])
        xn = x.copy()
        xn[1:3] = (x[1:3] - self.mean) / self.std
        return float(sigmoid(xn @ self.w))

    def should_fuse(self, seq, rows, cols) -> bool:
        return self.predict_proba(seq, rows, cols) >= 0.5


def main():
    data = load_sweep()
    train_rows = [r for r in data if r["sequence"] not in HOLDOUT_SEQUENCES]
    test_rows = [r for r in data if r["sequence"] in HOLDOUT_SEQUENCES]

    X_train = np.stack([featurize(r["sequence"], r["n_elements"], r["n_ops"])
                          for r in train_rows])
    y_train = np.array([r["won"] for r in train_rows], dtype=np.float64)

    print(f"Training on {len(train_rows)} points from "
          f"{len({r['sequence'] for r in train_rows})} sequences "
          f"(holding out {HOLDOUT_SEQUENCES} entirely):")
    w, mean, std = train_logistic(X_train, y_train)

    with open("results/learned_cost_model_v2.json", "w") as f:
        json.dump({"weights": w.tolist(), "mean": mean.tolist(), "std": std.tolist(),
                    "feature_names": FEATURE_NAMES}, f, indent=2)

    print(f"\nLearned per-op-kind weights:")
    for name, val in zip(FEATURE_NAMES, w):
        print(f"  {name:20s} {val:+.3f}")
    print("(Tanh's weight should be more negative than Sigmoid's, matching")
    print(" the measured fact that Tanh is more expensive in this project's kernel.)")

    model = LearnedCostModelV2()
    print(f"\n=== Generalization test: sequences NEVER seen during training ===")
    correct = 0
    for r in test_rows:
        seq_kinds = r["sequence"].split("-")
        seq = [(k, "x" if k in ("Add", "Mul", "Sub") else None) for k in seq_kinds]
        pred = model.should_fuse(seq, 1, r["n_elements"])
        actual = bool(r["won"])
        ok = pred == actual
        correct += ok
        print(f"  {r['sequence']:22s} n={r['n_elements']:>9,d}  "
              f"actual_won={actual}  predicted_fuse={pred}  {'OK' if ok else 'WRONG'}")
    print(f"\nHeld-out sequence accuracy: {correct}/{len(test_rows)} "
          f"({100*correct/len(test_rows):.0f}%)")


if __name__ == "__main__":
    main()

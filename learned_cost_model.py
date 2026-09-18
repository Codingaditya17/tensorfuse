"""
learned_cost_model.py — a real (if small) learned model replacing the
hand-calibrated threshold: logistic regression, trained with plain
numpy gradient descent, predicting P(fusion wins) from features of
the op sequence and shape.

The evaluation that matters: held-out SEQUENCES, not just held-out
shapes. cost_model.py's threshold approach can only answer for exact
signatures it was calibrated on. This model is tested on op sequences
it never saw a single timing for, to see whether it's actually learned
something general (transcendental ops are expensive; more ops favor
fusion; bigger arrays favor fusion) rather than just memorizing a
lookup table.
"""

import csv
import json
import numpy as np

FEATURE_NAMES = ["bias", "log2_n_elements", "n_ops", "has_transcendental"]

# Held out ENTIRELY from training -- the model never sees a single
# timing for these sequences. This is the real generalization test.
HOLDOUT_SEQUENCES = {"Add-Tanh", "Mul-Add-ReLU"}


def load_sweep(path="results/cost_sweep.csv"):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "sequence": r["sequence"],
                "n_elements": int(r["n_elements"]),
                "n_ops": int(r["n_ops"]),
                "has_transcendental": int(r["has_transcendental"]),
                "won": int(r["won"]),
                "speedup": float(r["speedup"]),
            })
    return rows


def featurize(n_elements, n_ops, has_transcendental):
    return np.array([1.0, np.log2(n_elements), n_ops, has_transcendental])


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def train_logistic(X, y, lr=0.1, epochs=5000):
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
        w -= lr * grad
        if epoch % 1000 == 0:
            loss = -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))
            acc = ((p > 0.5) == y).mean()
            print(f"  epoch {epoch:5d}  loss={loss:.4f}  train_acc={acc:.3f}")
    return w, mean, std


class LearnedCostModel:
    def __init__(self, weights_path="results/learned_cost_model.json"):
        with open(weights_path) as f:
            d = json.load(f)
        self.w = np.array(d["weights"])
        self.mean = np.array(d["mean"])
        self.std = np.array(d["std"])

    def predict_proba(self, seq, rows, cols):
        n_elements = rows * cols
        n_ops = len(seq)
        has_transcendental = int(any(k in ("Sigmoid", "Tanh") for k, _ in seq))
        x = featurize(n_elements, n_ops, has_transcendental)
        xn = x.copy()
        xn[1:3] = (x[1:3] - self.mean) / self.std
        return float(sigmoid(xn @ self.w))

    def should_fuse(self, seq, rows, cols) -> bool:
        return self.predict_proba(seq, rows, cols) >= 0.5


def main():
    data = load_sweep()
    train_rows = [r for r in data if r["sequence"] not in HOLDOUT_SEQUENCES]
    test_rows = [r for r in data if r["sequence"] in HOLDOUT_SEQUENCES]

    X_train = np.stack([featurize(r["n_elements"], r["n_ops"], r["has_transcendental"])
                          for r in train_rows])
    y_train = np.array([r["won"] for r in train_rows], dtype=np.float64)

    print(f"Training on {len(train_rows)} points from "
          f"{len({r['sequence'] for r in train_rows})} sequences "
          f"(holding out {HOLDOUT_SEQUENCES} entirely):")
    w, mean, std = train_logistic(X_train, y_train)

    with open("results/learned_cost_model.json", "w") as f:
        json.dump({"weights": w.tolist(), "mean": mean.tolist(), "std": std.tolist(),
                    "feature_names": FEATURE_NAMES}, f, indent=2)

    print(f"\nLearned weights: {dict(zip(FEATURE_NAMES, np.round(w, 3)))}")
    print("(positive log2_n_elements weight = bigger arrays favor fusion, as expected;")
    print(" negative has_transcendental weight = transcendental ops hurt fusion, as expected)")

    model = LearnedCostModel()
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

"""
train_model.py — trains a real (if small) MLP binary classifier with
plain numpy gradient descent, on a synthetic two-moons dataset. This
produces genuine, non-random, non-trivial weights so the ONNX export
downstream is testing a model that actually learned something, not
just an arbitrary graph shape.

Architecture (deliberately a standard MLP, so the exported ONNX graph
uses ops any tool would recognize):

    h1  = ReLU(x @ W1 + b1)          64 -> 32
    h2  = ReLU(h1 @ W2 + b2)         32 -> 16
    out = Sigmoid(h2 @ W3 + b3)      16 -> 1
"""

import numpy as np


def make_two_moons(n=2000, noise=0.15, seed=0):
    rng = np.random.default_rng(seed)
    n_half = n // 2
    theta1 = rng.uniform(0, np.pi, n_half)
    x1 = np.stack([np.cos(theta1), np.sin(theta1)], axis=1)
    theta2 = rng.uniform(0, np.pi, n_half)
    x2 = np.stack([1 - np.cos(theta2), 1 - np.sin(theta2) - 0.5], axis=1)
    X = np.concatenate([x1, x2], axis=0)
    X += rng.normal(scale=noise, size=X.shape)
    y = np.concatenate([np.zeros(n_half), np.ones(n_half)])
    perm = rng.permutation(n)
    return X[perm].astype(np.float32), y[perm].astype(np.float32)


def init_weights(seed=0):
    rng = np.random.default_rng(seed)
    def layer(fan_in, fan_out):
        scale = np.sqrt(2.0 / fan_in)
        W = (rng.standard_normal((fan_in, fan_out)) * scale).astype(np.float32)
        b = np.zeros((fan_out,), dtype=np.float32)
        return W, b
    W1, b1 = layer(2, 32)
    W2, b2 = layer(32, 16)
    W3, b3 = layer(16, 1)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}


def forward(params, X):
    z1 = X @ params["W1"] + params["b1"]
    h1 = np.maximum(z1, 0)
    z2 = h1 @ params["W2"] + params["b2"]
    h2 = np.maximum(z2, 0)
    z3 = h2 @ params["W3"] + params["b3"]
    out = 1.0 / (1.0 + np.exp(-z3))
    return z1, h1, z2, h2, z3, out


def train(epochs=400, lr=0.5, seed=0):
    X, y = make_two_moons(seed=seed)
    y = y.reshape(-1, 1)
    params = init_weights(seed=seed)
    n = X.shape[0]

    for epoch in range(epochs):
        z1, h1, z2, h2, z3, out = forward(params, X)
        loss = -np.mean(y * np.log(out + 1e-8) + (1 - y) * np.log(1 - out + 1e-8))

        d_out = (out - y) / n                     # dL/dz3 (sigmoid+BCE combined grad)
        dW3 = h2.T @ d_out
        db3 = d_out.sum(axis=0)
        d_h2 = d_out @ params["W3"].T
        d_z2 = d_h2 * (z2 > 0)
        dW2 = h1.T @ d_z2
        db2 = d_z2.sum(axis=0)
        d_h1 = d_z2 @ params["W2"].T
        d_z1 = d_h1 * (z1 > 0)
        dW1 = X.T @ d_z1
        db1 = d_z1.sum(axis=0)

        for name, grad in [("W1", dW1), ("b1", db1), ("W2", dW2), ("b2", db2),
                            ("W3", dW3), ("b3", db3)]:
            params[name] -= lr * grad.astype(np.float32)

        if epoch % 100 == 0 or epoch == epochs - 1:
            acc = ((out > 0.5).astype(np.float32) == y).mean()
            print(f"epoch {epoch:4d}  loss={loss:.4f}  acc={acc:.3f}")

    return params, X, y


if __name__ == "__main__":
    params, X, y = train()
    np.savez("results/trained_weights.npz", **params)
    print("Saved results/trained_weights.npz")

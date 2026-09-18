"""
train_cnn.py — trains a small real CNN with PyTorch (Conv2d -> ReLU ->
Conv2d -> ReLU -> Flatten -> Linear -> Sigmoid) on a synthetic image
classification task (no internet access to real MNIST in this sandbox,
so this generates its own: images with either a bright horizontal bar
or a bright vertical bar, label = which one).

This is what lets onnx_frontend.py be extended to handle Conv2d for
real -- a real trained model, real ONNX Conv nodes, not a hand-built
graph.
"""

import numpy as np
import torch
import torch.nn as nn


def make_bar_images(n=2000, size=16, seed=0):
    rng = np.random.default_rng(seed)
    imgs = np.zeros((n, 1, size, size), dtype=np.float32)
    labels = np.zeros((n,), dtype=np.float32)
    for i in range(n):
        imgs[i] += rng.normal(scale=0.1, size=(1, size, size))
        if i % 2 == 0:
            row = rng.integers(0, size)
            imgs[i, 0, row, :] += 1.0
            labels[i] = 0.0
        else:
            col = rng.integers(0, size)
            imgs[i, 0, :, col] += 1.0
            labels[i] = 1.0
    perm = rng.permutation(n)
    return imgs[perm], labels[perm]


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(8 * 4 * 4, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.flatten(1)
        x = torch.sigmoid(self.fc(x))
        return x


def train(epochs=15, lr=1e-2, seed=0):
    torch.manual_seed(seed)
    imgs, labels = make_bar_images(seed=seed)
    X = torch.from_numpy(imgs)
    y = torch.from_numpy(labels).unsqueeze(1)

    model = TinyCNN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    n = X.shape[0]
    batch_size = 64
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            out = model(X[idx])
            loss = loss_fn(out, y[idx])
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        if epoch % 3 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                preds = (model(X) > 0.5).float()
                acc = (preds == y).float().mean().item()
            print(f"epoch {epoch:3d}  loss={total_loss/n:.4f}  acc={acc:.3f}")

    return model, X, y


if __name__ == "__main__":
    model, X, y = train()
    torch.save(model.state_dict(), "results/cnn_weights.pt")
    print("Saved results/cnn_weights.pt")

    model.eval()
    torch.onnx.export(
        model, X[:1], "results/cnn_model.onnx",
        input_names=["x"], output_names=["out"],
        dynamic_axes={"x": {0: "batch"}, "out": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    print("Saved results/cnn_model.onnx")

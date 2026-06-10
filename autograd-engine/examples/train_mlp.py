import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autograd.tensor import Tensor
from autograd import nn, optim
from autograd import functional as F


def make_moons(n=400, noise=0.15, seed=0):
    rng = np.random.default_rng(seed)
    n_half = n // 2
    t = np.linspace(0, np.pi, n_half)
    outer = np.stack([np.cos(t), np.sin(t)], axis=1)
    inner = np.stack([1 - np.cos(t), 1 - np.sin(t) - 0.5], axis=1)
    X = np.concatenate([outer, inner], axis=0)
    X += rng.standard_normal(X.shape) * noise
    y = np.concatenate([np.zeros(n_half), np.ones(n_half)]).astype(np.int64)
    perm = rng.permutation(n)
    return X[perm], y[perm]


def main():
    X_np, y_np = make_moons()
    X = Tensor(X_np)

    model = nn.Sequential(
        nn.Linear(2, 32), nn.ReLU(),
        nn.Linear(32, 32), nn.ReLU(),
        nn.Linear(32, 2),
    )
    opt = optim.Adam(model.parameters(), lr=0.02)

    for epoch in range(1, 201):
        opt.zero_grad()
        logits = model(X)
        loss = F.cross_entropy(logits, y_np)
        loss.backward()
        opt.step()
        if epoch % 40 == 0 or epoch == 1:
            preds = logits.data.argmax(axis=1)
            acc = (preds == y_np).mean()
            print(f"epoch {epoch:3d}  loss {float(loss.data):.4f}  acc {acc:.3f}")

    preds = model(X).data.argmax(axis=1)
    print(f"\nFinal training accuracy: {(preds == y_np).mean():.3f}")


if __name__ == "__main__":
    main()
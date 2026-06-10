import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autograd.tensor import Tensor
from autograd import nn, optim
from autograd import functional as F


def test_overfit_single_batch():
    rng = np.random.default_rng(0)
    X = Tensor(rng.standard_normal((8, 4)))
    y = rng.integers(0, 3, size=(8,))

    model = nn.Sequential(
        nn.Linear(4, 16), nn.ReLU(),
        nn.Linear(16, 3),
    )
    opt = optim.Adam(model.parameters(), lr=0.05)

    losses = []
    for _ in range(300):
        opt.zero_grad()
        loss = F.cross_entropy(model(X), y)
        loss.backward()
        opt.step()
        losses.append(float(loss.data))

    assert losses[-1] < 0.05, f"failed to overfit; final loss {losses[-1]:.4f}"
    assert losses[-1] < losses[0], "loss did not decrease"


if __name__ == "__main__":
    test_overfit_single_batch()
    print("PASS  test_overfit_single_batch")
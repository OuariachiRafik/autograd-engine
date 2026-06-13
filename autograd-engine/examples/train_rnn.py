import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autograd.tensor import Tensor
from autograd import nn, optim


def main():
    rng = np.random.default_rng(0)
    T, n_in, n_hid, n_out = 8, 3, 32, 2

    xs = [Tensor(rng.standard_normal((n_in, 1))) for _ in range(T)]
    ys = [rng.standard_normal((n_out, 1)) for _ in range(T)]

    model = nn.RNN(n_in, n_hid, n_out)
    opt = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, 401):
        opt.zero_grad()
        preds = model(xs)
        loss = None
        for t in range(T):
            diff = preds[t] - Tensor(ys[t])
            term = (diff * diff).mean()
            loss = term if loss is None else loss + term
        loss.backward()
        opt.step()
        if epoch % 80 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  loss {float(loss.data):.5f}")

    print(f"\nFinal loss: {float(loss.data):.5f}  (BPTT handled automatically by the engine)")


if __name__ == "__main__":
    main()

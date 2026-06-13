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

    lstm = nn.LSTM(n_in, n_hid)
    W_hy = Tensor(rng.standard_normal((n_out, n_hid)) * 0.1, requires_grad=True)
    b_y = Tensor(np.zeros((n_out, 1)), requires_grad=True)
    params = lstm.parameters() + [W_hy, b_y]
    opt = optim.Adam(params, lr=0.01)

    for epoch in range(1, 401):
        opt.zero_grad()
        hidden_states, _, _ = lstm(xs)
        loss = None
        for t in range(T):
            pred = W_hy @ hidden_states[t] + b_y
            term = ((pred - Tensor(ys[t])) ** 2).mean()
            loss = term if loss is None else loss + term
        loss.backward()
        opt.step()
        if epoch % 80 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  loss {float(loss.data):.5f}")

    print(f"\nFinal loss: {float(loss.data):.5f}  (gates + BPTT handled by the engine)")


if __name__ == "__main__":
    main()
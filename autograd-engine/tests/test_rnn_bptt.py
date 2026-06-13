import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autograd.tensor import Tensor
from autograd import nn


class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))

    def forward(self, x):
        self.h_t = np.zeros((self.hidden_size, 1))
        self.h_states = [self.h_t]
        self.y_pred = []
        for t in range(len(x)):
            self.h_t = np.tanh(self.W_xh @ np.array(x[t]).reshape(-1, 1)
                               + self.W_hh @ self.h_t + self.b_h)
            y_t = self.W_hy @ self.h_t + self.b_y
            self.h_states.append(self.h_t)
            self.y_pred.append(y_t)
        return self.y_pred

    def backward(self, x, y, learning_rate):
        T = len(x)
        grad_W_xh = np.zeros_like(self.W_xh)
        grad_W_hh = np.zeros_like(self.W_hh)
        grad_W_hy = np.zeros_like(self.W_hy)
        grad_b_h = np.zeros_like(self.b_h)
        grad_b_y = np.zeros_like(self.b_y)
        grad_h_next = np.zeros_like(self.h_states[0])

        for t in reversed(range(T)):
            x_t = np.array(x[t]).reshape(-1, 1)
            y_t = np.array(y[t]).reshape(-1, 1)
            h_t = self.h_states[t + 1]
            h_t_1 = self.h_states[t]

            dL_dy = self.y_pred[t] - y_t          
            grad_W_hy += dL_dy @ h_t.T
            grad_b_y += dL_dy

            dh_t = self.W_hy.T @ dL_dy + grad_h_next
            dh_raw = dh_t * (1 - h_t ** 2)
            grad_h_next = self.W_hh.T @ dh_raw

            grad_W_xh += dh_raw @ x_t.T
            grad_W_hh += dh_raw @ h_t_1.T
            grad_b_h += dh_raw

        self.last_grads = {"W_xh": grad_W_xh, "W_hh": grad_W_hh,
                           "W_hy": grad_W_hy, "b_h": grad_b_h, "b_y": grad_b_y}

        self.W_xh -= learning_rate * grad_W_xh
        self.W_hh -= learning_rate * grad_W_hh
        self.W_hy -= learning_rate * grad_W_hy
        self.b_h -= learning_rate * grad_b_h
        self.b_y -= learning_rate * grad_b_y


def test_engine_matches_handwritten_bptt():
    rng = np.random.default_rng(42)
    T, n_in, n_hid, n_out = 5, 3, 4, 2
    x = [rng.standard_normal(n_in) for _ in range(T)]
    y = [rng.standard_normal(n_out) for _ in range(T)]

    net = SimpleRNN(n_in, n_hid, n_out)
    net.W_xh = rng.standard_normal((n_hid, n_in))
    net.W_hh = rng.standard_normal((n_hid, n_hid))
    net.W_hy = rng.standard_normal((n_out, n_hid))
    net.b_h = rng.standard_normal((n_hid, 1))
    net.b_y = rng.standard_normal((n_out, 1))
    net.forward(x)
    net.backward(x, y, learning_rate=0.0)

    er = nn.RNN(n_in, n_hid, n_out)
    er.W_xh.data = net.W_xh.copy()
    er.W_hh.data = net.W_hh.copy()
    er.W_hy.data = net.W_hy.copy()
    er.b_h.data = net.b_h.copy()
    er.b_y.data = net.b_y.copy()

    xs = [Tensor(xi.reshape(-1, 1)) for xi in x]
    preds = er(xs)
    
    loss = None
    for t in range(T):
        diff = preds[t] - Tensor(y[t].reshape(-1, 1))
        term = (diff * diff).sum() * 0.5
        loss = term if loss is None else loss + term
    loss.backward()

    for name in ["W_xh", "W_hh", "W_hy", "b_h", "b_y"]:
        engine_grad = getattr(er, name).grad
        manual_grad = net.last_grads[name]
        err = np.max(np.abs(engine_grad - manual_grad))
        assert err < 1e-8, f"{name}: engine vs manual BPTT differ by {err:.2e}"
        print(f"PASS  {name:5s}  max|engine - manual| = {err:.2e}")


if __name__ == "__main__":
    test_engine_matches_handwritten_bptt()
    print("\nEngine autodiff matches hand-written BPTT.")

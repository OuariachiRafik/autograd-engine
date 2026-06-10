import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autograd.tensor import Tensor
from autograd import functional as F
from autograd import nn


def _numeric_grad(f, x, eps=1e-6):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        fp = f()
        x[idx] = orig - eps
        fm = f()
        x[idx] = orig
        grad[idx] = (fp - fm) / (2 * eps)
        it.iternext()
    return grad


def _check(make_inputs, forward, names, tol=1e-5):
    tensors = make_inputs()
    datas = [t.data.copy() for t in tensors]

    out = forward(*tensors)
    out.backward()
    analytic = [t.grad.copy() for t in tensors]

    for i in range(len(tensors)):
        x = datas[i]  

        def f(i=i, x=x):
            ts = [Tensor(x if j == i else datas[j].copy(), requires_grad=True)
                  for j in range(len(datas))]
            return float(forward(*ts).data)

        numeric = _numeric_grad(f, x)
        err = np.max(np.abs(analytic[i] - numeric))
        assert err < tol, f"{names[i]}: grad mismatch, max err {err:.2e}"
    return True


rng = np.random.default_rng(0)


def test_add_mul_broadcast():
    def mk():
        return [Tensor(rng.standard_normal((3, 4)), requires_grad=True),
                Tensor(rng.standard_normal((4,)), requires_grad=True)]  
    _check(mk, lambda a, b: ((a + b) * a).sum(), ["a", "b"])


def test_matmul():
    def mk():
        return [Tensor(rng.standard_normal((3, 4)), requires_grad=True),
                Tensor(rng.standard_normal((4, 2)), requires_grad=True)]
    _check(mk, lambda a, b: (a @ b).sum(), ["a", "b"])


def test_pow_div():
    def mk():
        return [Tensor(rng.uniform(0.5, 2.0, (3, 3)), requires_grad=True)]
    _check(mk, lambda a: (a ** 3 / a).sum(), ["a"])


def test_relu():
    def mk():
        return [Tensor(rng.standard_normal((4, 5)), requires_grad=True)]
    _check(mk, lambda a: a.relu().sum(), ["a"])


def test_sigmoid_tanh():
    def mk():
        return [Tensor(rng.standard_normal((4, 5)), requires_grad=True)]
    _check(mk, lambda a: (a.sigmoid() + a.tanh()).sum(), ["a"])


def test_log_softmax():
    def mk():
        return [Tensor(rng.standard_normal((4, 6)), requires_grad=True)]
    _check(mk, lambda a: F.log_softmax(a).sum(), ["a"])


def test_cross_entropy():
    targets = rng.integers(0, 5, size=(4,))
    def mk():
        return [Tensor(rng.standard_normal((4, 5)), requires_grad=True)]
    _check(mk, lambda a: F.cross_entropy(a, targets), ["logits"])


def test_mean_reshape_transpose():
    def mk():
        return [Tensor(rng.standard_normal((2, 6)), requires_grad=True)]
    _check(mk, lambda a: a.reshape(3, 4).transpose().mean(), ["a"])


def test_linear_layer():
    layer = nn.Linear(4, 3)
    def mk():
        return [Tensor(rng.standard_normal((5, 4)), requires_grad=True),
                layer.W, layer.b]
    _check(mk, lambda x, W, b: (x @ W + b).sum(), ["x", "W", "b"])


def test_layernorm():
    ln = nn.LayerNorm(4)
    def mk():
        return [Tensor(rng.standard_normal((5, 4)), requires_grad=True),
                ln.gamma, ln.beta]
    def fwd(x, gamma, beta):
        ln.gamma, ln.beta = gamma, beta
        return ln(x).sum()
    _check(mk, fwd, ["x", "gamma", "beta"], tol=1e-4)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\nAll gradient checks passed.")
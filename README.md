# autograd-engine

A from-scratch **tensor** autograd engine in pure NumPy, built by taking the
ML primitives I'd been implementing as [Deep-ML](https://www.deep-ml.com/)
exercises and, inspired by Karpathy's [micrograd](https://github.com/karpathy/micrograd),
wiring them into a working engine, then **extending micrograd from scalars to
tensors** so real architectures (RNN, LSTM, Transformer, …) can be built on top.

> **Demonstration engine: clarity and correctness first, not speed.** It computes
> the *same* gradients PyTorch does — verified by numerical gradient checks — just
> slower, because it runs in Python/NumPy instead of dispatching to C++/CUDA
> kernels. Making the same computation *fast* is a separate concern (GPU kernels)
> and deliberately out of scope.

## Why tensors (the point of the project)

micrograd is scalar: every value is one number. That's perfect for *seeing* how
backprop works, but you can't naturally express a matmul, an attention head, or a
conv with it. So the core extension here is that every op carries a **tensor-aware
backward** — in particular the two load-bearing ones:

- **matmul backward** (`dA = dC @ Bᵀ`, `dB = Aᵀ @ dC`)
- **broadcasting backward** (summing a gradient back down the axes an op expanded, bias adds, reductions)

Once those ~12 primitives have correct backward rules, *whole architectures
compose on top for free*: the backward pass of an entire network assembles
itself. That's the whole idea, and it's what frameworks do underneath.

## What's in it

- Reverse-mode autodiff over a dynamic graph (`Tensor.backward()`).
- Ops with broadcasting-aware backward: add, mul, matmul, pow, exp, log, sum, mean, reshape, transpose.
- Activations: relu, sigmoid, tanh; stable softmax / log-softmax.
- Losses: MSE, cross-entropy (from logits).
- `nn`: `Module`, `Linear`, `LayerNorm`, `ReLU`, `Sequential`.
- `optim`: `SGD` (+momentum), `Adam`.

## Results

**Every op's analytic gradient matches central finite differences:**

```
PASS  test_add_mul_broadcast   PASS  test_matmul        PASS  test_pow_div
PASS  test_relu                PASS  test_sigmoid_tanh  PASS  test_log_softmax
PASS  test_cross_entropy       PASS  test_linear_layer  PASS  test_layernorm
PASS  test_mean_reshape_transpose
All gradient checks passed.
```

**An MLP trained only with this engine on a nonlinear (two-moons) dataset:**

```
epoch   1  loss 0.96  acc 0.50
epoch  80  loss 0.02  acc 0.99
epoch 200  loss 0.01  acc 0.995
Final training accuracy: 0.995
```

## Reproduce

```bash
pip install -r requirements.txt
python tests/test_gradcheck.py     # per-op gradient checks (the correctness proof)
python tests/test_train.py         # overfit-a-batch sanity check
python examples/train_mlp.py       # end-to-end training demo
```

## Layout

```
autograd/
  tensor.py      # Tensor, ops, _unbroadcast, backward()
  functional.py  # softmax, log_softmax, mse_loss, cross_entropy
  nn.py          # Module, Linear, LayerNorm, ReLU, Sequential
  optim.py       # SGD, Adam
tests/           # gradient checks + overfit sanity test
examples/        # end-to-end MLP demo
```

## Roadmap

The point of the tensor extension is to build real architectures *on this engine*:

- [x] Tensor autograd core + MLP
- [ ] RNN + backprop-through-time
- [ ] LSTM
- [ ] Transformer block (attention = matmul + softmax, reusing LayerNorm)
- [ ] Mixture-of-Experts layer (top-k gating on the Transformer block)

Each is built only from the primitives above — no new autodiff machinery
(except conv, which gets its own backward).

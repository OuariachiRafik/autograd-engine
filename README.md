# autograd-engine

A from-scratch **tensor** autograd engine in pure NumPy built by taking the
ML primitives I'd been implementing as [Deep-ML](https://www.deep-ml.com/)
exercises and, inspired by Karpathy's [micrograd](https://github.com/karpathy/micrograd),
wiring them into a working engine  then **extending micrograd from scalars to
tensors** so real architectures can be built and trained on top. **RNN, LSTM,
and a Transformer are now implemented and verified all running on the engine's
own autodiff.**

> **Demonstration engine: clarity and correctness first, not speed.** It computes
> the *same* gradients PyTorch does verified by numerical gradient checks just
> slower, because it runs in Python/NumPy instead of dispatching to C++/CUDA
> kernels. Making the same computation *fast* is a separate concern (GPU kernels)
> and deliberately out of scope.

## Why tensors (the point of the project)

micrograd is scalar: every value is one number. That's perfect for *seeing* how
backprop works, but you can't naturally express a matmul, an attention head, or a
conv with it. So the core extension here is that every op carries a **tensor-aware
backward** in particular the two load-bearing ones:

- **matmul backward** (`dA = dC @ Bᵀ`, `dB = Aᵀ @ dC`)
- **broadcasting backward** (summing a gradient back down the axes an op expanded bias adds, reductions)

Once those ~13 primitives have correct backward rules, *whole architectures
compose on top for free*: the backward pass of an entire network including
backprop-through-time and attention assembles itself. That's the whole idea,
and it's what frameworks do underneath.

## What's in it

- Reverse-mode autodiff over a dynamic graph (`Tensor.backward()`).
- Ops with broadcasting-aware backward: add, mul, matmul, pow, exp, log, sum, mean, reshape, transpose, concat.
- Activations: relu, sigmoid, tanh; stable softmax / log-softmax.
- Losses: MSE, cross-entropy (from logits).
- `nn`: `Module`, `Linear`, `LayerNorm`, `ReLU`, `Sequential`, `RNN`, `LSTM`,
  `MultiHeadAttention`, `FeedForward`, `TransformerBlock`, `positional_encoding`.
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

**Architectures: all trained on the engine, backprop automatic:**

| Model        | Result                                                        |
|--------------|---------------------------------------------------------------|
| MLP          | 99.5% on a nonlinear (two-moons) dataset                      |
| RNN          | trains to ~0 loss; **autodiff matches a hand-derived BPTT to ~1e-15** |
| LSTM         | trains to 0.00000 loss                                        |
| Transformer  | tiny GPT → **100% next-token accuracy**                       |

The RNN's gradients are checked against a from-scratch, hand-written
**backpropagation-through-time** implementation   they agree to *machine
precision* (`tests/test_rnn_bptt.py`). That's direct evidence the engine's
autodiff is correct, not just plausible: the automatic gradients reproduce the
calculus done by hand.

## Reproduce

```bash
pip install -r requirements.txt

# correctness
python tests/test_gradcheck.py     # per-op gradient checks
python tests/test_train.py         # overfit-a-batch sanity check
python tests/test_rnn_bptt.py      # engine autodiff vs hand-derived BPTT

# train the architectures
python examples/train_mlp.py
python examples/train_rnn.py
python examples/train_lstm.py
python examples/train_transformer.py
```

## How it works (the one idea)

Each operation, when it runs forward, also stores a closure describing how to
push gradients to its inputs. `backward()` builds a topological order of the
graph and calls those closures in reverse, seeding the output gradient with 1.
Because losses, layers, recurrences, and attention are all expressed in terms of
a handful of primitive ops, **the backward pass of an entire network is assembled
automatically**   you never hand-derive a gradient for a whole architecture, only
for the ~13 primitives. An RNN, for example, is just a forward loop in tensor ops;
unrolling it builds the graph through time, and `backward()` does BPTT for free.

The trickiest correctness detail is `_unbroadcast` in `autograd/tensor.py`: when
an op broadcasts (e.g. adding a `(out,)` bias to a `(batch, out)` activation),
the upstream gradient must be summed back down the expanded axes to match the
original shape. The gradient checks exist to catch exactly this class of bug.

## Layout

```
autograd/
  tensor.py      # Tensor, ops, _unbroadcast, cat, backward()
  functional.py  # softmax, log_softmax, mse_loss, cross_entropy
  nn.py          # Module, Linear, LayerNorm, RNN, LSTM, attention, Transformer
  optim.py       # SGD, Adam
tests/
  test_gradcheck.py  # numerical gradient checks (the correctness proof)
  test_train.py      # overfit-a-batch sanity check
  test_rnn_bptt.py   # engine autodiff vs hand-written BPTT (machine-precision match)
examples/            # MLP, RNN, LSTM, Transformer training demos
```

## Roadmap

- [x] Tensor autograd core + MLP
- [x] RNN + backprop-through-time
- [x] LSTM
- [x] Transformer block (multi-head causal attention, positional encoding)
- [ ] Mixture-of-Experts layer

Each architecture is built only from the primitives above   no new autodiff
machinery (except conv, when that lands).

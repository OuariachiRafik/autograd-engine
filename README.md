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

## Inference: KV-Cache

A frozen, forward-only inference path (`inference/kv_cache.py`) that reuses the
trained Transformer's weights to do autoregressive generation with a KV-cache.
It runs in plain NumPy with no autograd graph, so the benchmark measures the
inference computation, not backward-pass overhead.

**Why a cache:** in autoregressive decoding the keys/values of past tokens never
change, so recomputing them every step is wasted work. The cache stores them and
computes K,V for only the new token each step, trading recompute for memory,
the standard decode-phase optimization.

**Verified correct (not approximate):**
- frozen inference path matches the training-engine forward to ~1e-15
- cached decode produces token-identical output to full recompute
- last-token logits match to 0.0

**Measured (CPU/NumPy — the ratio and scaling shape are the result, not raw tok/s;
the no-cache baseline recomputes the full forward each step):**

| gen_len | no-cache tok/s | cache tok/s | speedup |
|--------:|---------------:|------------:|--------:|
| 32      | 106            | 451         | 4.2×    |
| 64      | 107            | 500         | 4.7×    |
| 128     | 57             | 490         | 8.6×    |
| 256     | 22             | 438         | 20.1×   |

Without the cache, throughput collapses as the sequence grows (recomputing K,V
over the whole sequence each step, ~quadratic). With the cache it stays roughly
flat (~linear). Prefill scales with prompt length; decode is ~2 ms/token
regardless of context, exactly what the memory-bound-decode analysis predicts.

## Layout

```
autograd-engine/
├── README.md                        
├── autograd/
│   ├── __init__.py
│   ├── tensor.py                    (Tensor, autodiff, ops, cat)
│   ├── functional.py                (softmax, log_softmax, mse, cross_entropy)
│   ├── nn.py                        (Linear, LayerNorm, RNN, LSTM, MHA, Transformer…)
│   └── optim.py                     (SGD, Adam)
├── inference/                       
│   ├── __init__.py                  
│   └── kv_cache.py                  
├── tests/
│   ├── test_gradcheck.py
│   ├── test_train.py
│   ├── test_rnn_bptt.py
│   └── test_kv_cache.py             
└── examples/
    ├── train_mlp.py
    ├── train_rnn.py
    ├── train_lstm.py
    ├── train_transformer.py
    └── benchmark_kv_cache.py        speedup benchmark
```

## Roadmap

- [x] Tensor autograd core + MLP
- [x] RNN + backprop-through-time
- [x] LSTM
- [x] Transformer block (multi-head causal attention, positional encoding)
- [ ] Mixture-of-Experts layer
- [x] KV-cache inference path (verified, benchmarked)

Each architecture is built only from the primitives above   no new autodiff
machinery (except conv, when that lands).

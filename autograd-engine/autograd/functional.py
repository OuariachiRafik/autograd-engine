from __future__ import annotations

import numpy as np

from .tensor import Tensor


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    shift = Tensor(x.data.max(axis=axis, keepdims=True))
    e = (x - shift).exp()
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    shift = Tensor(x.data.max(axis=axis, keepdims=True))
    z = x - shift
    return z - z.exp().sum(axis=axis, keepdims=True).log()


def mse_loss(pred: Tensor, target) -> Tensor:
    target = target if isinstance(target, Tensor) else Tensor(target)
    diff = pred - target
    return (diff * diff).mean()


def cross_entropy(logits: Tensor, targets) -> Tensor:
    targets = np.asarray(targets, dtype=np.int64)
    logp = log_softmax(logits, axis=-1)
    batch = logp.shape[0]
    mask = np.zeros_like(logp.data)
    mask[np.arange(batch), targets] = 1.0          
    picked = (logp * Tensor(mask)).sum(axis=-1)    
    return -picked.mean()
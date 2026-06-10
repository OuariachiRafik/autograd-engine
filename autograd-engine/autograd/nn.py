from __future__ import annotations

import numpy as np

from .tensor import Tensor


class Module:

    def parameters(self):
        params = []
        for v in self.__dict__.values():
            if isinstance(v, Tensor) and v.requires_grad:
                params.append(v)
            elif isinstance(v, Module):
                params.extend(v.parameters())
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Linear(Module):

    def __init__(self, in_features: int, out_features: int):
        limit = 1.0 / np.sqrt(in_features)
        self.W = Tensor(np.random.uniform(-limit, limit, (in_features, out_features)),
                        requires_grad=True)
        self.b = Tensor(np.zeros(out_features), requires_grad=True)   

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.W + self.b


class LayerNorm(Module):

    def __init__(self, dim: int, eps: float = 1e-5):
        self.gamma = Tensor(np.ones(dim), requires_grad=True)
        self.beta = Tensor(np.zeros(dim), requires_grad=True)
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mu = x.mean(axis=-1, keepdims=True)
        xc = x - mu
        var = (xc * xc).mean(axis=-1, keepdims=True)
        std = (var + self.eps) ** 0.5
        return (xc / std) * self.gamma + self.beta


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            if isinstance(layer, Module):
                params.extend(layer.parameters())
        return params
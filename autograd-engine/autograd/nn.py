from __future__ import annotations

import numpy as np

from .tensor import Tensor, cat
from . import functional

def _col_slice(t, sl):
    cols = range(t.shape[1])[sl]
    S = np.zeros((t.shape[1], len(cols)))
    for j, c in enumerate(cols):
        S[c, j] = 1.0
    return t @ Tensor(S)

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
 
 
class RNN(Module):
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.W_xh = Tensor(np.random.randn(hidden_size, input_size) * 0.01, requires_grad=True)
        self.W_hh = Tensor(np.random.randn(hidden_size, hidden_size) * 0.01, requires_grad=True)
        self.W_hy = Tensor(np.random.randn(output_size, hidden_size) * 0.01, requires_grad=True)
        self.b_h = Tensor(np.zeros((hidden_size, 1)), requires_grad=True)
        self.b_y = Tensor(np.zeros((output_size, 1)), requires_grad=True)
 
    def forward(self, xs):
        h = Tensor(np.zeros((self.hidden_size, 1)))
        ys = []
        for x_t in xs:
            h = (self.W_xh @ x_t + self.W_hh @ h + self.b_h).tanh()
            ys.append(self.W_hy @ h + self.b_y)
        return ys
 
 
class LSTM(Module):
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        D = input_size + hidden_size                      
        s = 1.0 / np.sqrt(D)                             
        self.Wf = Tensor(np.random.randn(hidden_size, D) * s, requires_grad=True)
        self.Wi = Tensor(np.random.randn(hidden_size, D) * s, requires_grad=True)
        self.Wc = Tensor(np.random.randn(hidden_size, D) * s, requires_grad=True)
        self.Wo = Tensor(np.random.randn(hidden_size, D) * s, requires_grad=True)
        self.bf = Tensor(np.zeros((hidden_size, 1)), requires_grad=True)
        self.bi = Tensor(np.zeros((hidden_size, 1)), requires_grad=True)
        self.bc = Tensor(np.zeros((hidden_size, 1)), requires_grad=True)
        self.bo = Tensor(np.zeros((hidden_size, 1)), requires_grad=True)
 
    def forward(self, xs, h0=None, c0=None):
        h = Tensor(np.zeros((self.hidden_size, 1))) if h0 is None else h0
        c = Tensor(np.zeros((self.hidden_size, 1))) if c0 is None else c0
        hidden_states = []
        for x_t in xs:                                    
            combined = cat([h, x_t], axis=0)              
            f = (self.Wf @ combined + self.bf).sigmoid()
            i = (self.Wi @ combined + self.bi).sigmoid()
            c_cand = (self.Wc @ combined + self.bc).tanh()
            o = (self.Wo @ combined + self.bo).sigmoid()
            c = f * c + i * c_cand
            h = o * c.tanh()
            hidden_states.append(h)
        return hidden_states, h, c
 
 
class MultiHeadAttention(Module):
    def __init__(self, d_model, n_heads, causal=False):
        assert d_model % n_heads == 0
        self.d_model, self.n_heads = d_model, n_heads
        self.d_k = d_model // n_heads
        self.causal = causal
        s = 1.0 / np.sqrt(d_model)
        self.W_q = Tensor(np.random.randn(d_model, d_model) * s, requires_grad=True)
        self.W_k = Tensor(np.random.randn(d_model, d_model) * s, requires_grad=True)
        self.W_v = Tensor(np.random.randn(d_model, d_model) * s, requires_grad=True)
        self.W_o = Tensor(np.random.randn(d_model, d_model) * s, requires_grad=True)
 
    def _attention(self, q, k, v, seq_len):
        scores = (q @ k.transpose()) * (1.0 / np.sqrt(self.d_k))
        if self.causal:
            mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1)  
            scores = scores + Tensor(mask)
        weights = functional.softmax(scores, axis=-1)               
        return weights @ v
 
    def forward(self, x):
        seq_len = x.shape[0]
        Q, K, V = x @ self.W_q, x @ self.W_k, x @ self.W_v
        heads = []
        for h in range(self.n_heads):                              
            sl = slice(h * self.d_k, (h + 1) * self.d_k)
            qh = Q[:, sl] if False else _col_slice(Q, sl)
            kh = _col_slice(K, sl)
            vh = _col_slice(V, sl)
            heads.append(self._attention(qh, kh, vh, seq_len))
        concat = cat(heads, axis=1)                                
        return concat @ self.W_o
 
 
class FeedForward(Module):
 
    def __init__(self, d_model, d_ff):
        self.l1 = Linear(d_model, d_ff)
        self.l2 = Linear(d_ff, d_model)
 
    def forward(self, x):
        return self.l2(self.l1(x).relu())
 
 
class TransformerBlock(Module):
 
    def __init__(self, d_model, n_heads, d_ff, causal=True):
        self.attn = MultiHeadAttention(d_model, n_heads, causal=causal)
        self.ff = FeedForward(d_model, d_ff)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
 
    def forward(self, x):
        x = x + self.attn(self.ln1(x))     
        x = x + self.ff(self.ln2(x))       
        return x
 
 
def positional_encoding(seq_len, d_model):
    pe = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(d_model // 2):
            denom = 10000 ** (2 * i / d_model)
            pe[pos, 2 * i] = np.sin(pos / denom)
            pe[pos, 2 * i + 1] = np.cos(pos / denom)
    return Tensor(pe)
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autograd.tensor import Tensor
from autograd import nn, optim
from autograd import functional as F


def main():
    rng = np.random.default_rng(0)
    V, T, d_model, n_heads, d_ff, n_layers = 6, 8, 32, 4, 64, 2

    seq = rng.integers(0, V, size=T + 1)          
    inp, target = seq[:-1], seq[1:]

    embed = Tensor(rng.standard_normal((V, d_model)) * 0.1, requires_grad=True)
    blocks = [nn.TransformerBlock(d_model, n_heads, d_ff, causal=True) for _ in range(n_layers)]
    W_out = Tensor(rng.standard_normal((d_model, V)) * 0.1, requires_grad=True)
    b_out = Tensor(np.zeros(V), requires_grad=True)

    params = [embed, W_out, b_out]
    for blk in blocks:
        params += blk.parameters()
    opt = optim.Adam(params, lr=0.01)

    pe = nn.positional_encoding(T, d_model)
    onehot = np.zeros((T, V)); onehot[np.arange(T), inp] = 1.0
    onehot = Tensor(onehot)

    for epoch in range(1, 401):
        opt.zero_grad()
        x = onehot @ embed + pe                   
        for blk in blocks:
            x = blk(x)
        logits = x @ W_out + b_out                
        loss = F.cross_entropy(logits, target)
        loss.backward()
        opt.step()
        if epoch % 80 == 0 or epoch == 1:
            preds = logits.data.argmax(axis=1)
            acc = (preds == target).mean()
            print(f"epoch {epoch:3d}  loss {float(loss.data):.5f}  next-token acc {acc:.3f}")

    preds = logits.data.argmax(axis=1)
    print(f"\nFinal next-token accuracy: {(preds == target).mean():.3f}  "
          f"(attention + LayerNorm + FFN all on the engine)")


if __name__ == "__main__":
    main()

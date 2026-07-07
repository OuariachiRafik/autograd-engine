import sys, os
sys.path.insert(0, '.')
import numpy as np
from autograd.tensor import Tensor
from autograd import nn
from inference.kv_cache import InferenceTransformer, _softmax

rng = np.random.default_rng(0)
V, T, d_model, n_heads, d_ff, n_layers = 6, 8, 32, 4, 64, 2

embed = Tensor(rng.standard_normal((V, d_model)) * 0.1, requires_grad=True)
blocks = [nn.TransformerBlock(d_model, n_heads, d_ff, causal=True) for _ in range(n_layers)]
W_out = Tensor(rng.standard_normal((d_model, V)) * 0.1, requires_grad=True)
b_out = Tensor(np.zeros(V), requires_grad=True)
pe = nn.positional_encoding(T + 8, d_model)   

def engine_logits(ids):
    onehot = np.zeros((len(ids), V)); onehot[np.arange(len(ids)), ids] = 1.0
    x = Tensor(onehot) @ embed + nn.positional_encoding(len(ids), d_model)
    for blk in blocks:
        x = blk(x)
    return (x @ W_out + b_out).data

infer = InferenceTransformer.from_trained(embed, pe, blocks, W_out, b_out)

ids = [rng.integers(0, V) for _ in range(5)]
eng = engine_logits(ids)
inf = infer.logits_full(ids)
err1 = np.max(np.abs(eng - inf))
print(f"[1] inference vs training-engine forward  max err: {err1:.2e}  ->", "PASS" if err1 < 1e-9 else "FAIL")

prompt = [rng.integers(0, V) for _ in range(4)]
g_nocache = infer.generate_no_cache(prompt, n_new=6, greedy=True)
g_cache   = infer.generate_with_cache(prompt, n_new=6, greedy=True)
match = g_nocache == g_cache
print(f"[2] cache vs no-cache tokens  {g_nocache} == {g_cache}  ->", "PASS" if match else "FAIL")

ids_a = list(prompt)
maxerr = 0.0
for _ in range(6):
    la = infer.logits_full(ids_a)[-1]
    ids_a.append(int(np.argmax(la)))
lb = infer.logits_full(g_cache[:-1])[-1]
lc_ref = infer.logits_full(g_nocache[:-1])[-1]
err3 = np.max(np.abs(lb - lc_ref))
print(f"[3] logit values consistent  max err: {err3:.2e}  ->", "PASS" if err3 < 1e-9 else "FAIL")

print("\nALL PASS" if (err1 < 1e-9 and match and err3 < 1e-9) else "\nSOMETHING FAILED")

import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from autograd.tensor import Tensor
from autograd import nn
from inference.kv_cache import InferenceTransformer


def build_model(V=32, d_model=128, n_heads=8, d_ff=256, n_layers=4, max_pos=600, seed=0):
    rng = np.random.default_rng(seed)
    embed = Tensor(rng.standard_normal((V, d_model)) * 0.02, requires_grad=True)
    blocks = [nn.TransformerBlock(d_model, n_heads, d_ff, causal=True) for _ in range(n_layers)]
    W_out = Tensor(rng.standard_normal((d_model, V)) * 0.02, requires_grad=True)
    b_out = Tensor(np.zeros(V), requires_grad=True)
    pe = nn.positional_encoding(max_pos, d_model)
    infer = InferenceTransformer.from_trained(embed, pe, blocks, W_out, b_out)
    return infer, V, dict(d_model=d_model, n_heads=n_heads, d_ff=d_ff, n_layers=n_layers)


def time_gen(fn, *a, **k):
    t0 = time.perf_counter(); fn(*a, **k); return time.perf_counter() - t0


def main():
    infer, V, cfg = build_model()
    rng = np.random.default_rng(1)
    prompt = [int(rng.integers(0, V)) for _ in range(16)]

    print(f"model: d_model={cfg['d_model']} heads={cfg['n_heads']} "
          f"layers={cfg['n_layers']} d_ff={cfg['d_ff']}  prompt_len={len(prompt)}\n")
    print(f"{'gen_len':>8} | {'no-cache (s)':>13} | {'cache (s)':>10} | "
          f"{'speedup':>8} | {'nocache tok/s':>13} | {'cache tok/s':>12}")
    print("-" * 80)

    for n_new in [32, 64, 128, 256]:
        t_no = time_gen(infer.generate_no_cache, prompt, n_new, greedy=True)
        t_ca = time_gen(infer.generate_with_cache, prompt, n_new, greedy=True)
        print(f"{n_new:>8} | {t_no:>13.4f} | {t_ca:>10.4f} | "
              f"{t_no / t_ca:>7.2f}x | {n_new / t_no:>13.1f} | {n_new / t_ca:>12.1f}")

    print("\nprefill vs decode (with cache):")
    prompt_lens = [16, 64, 128]
    for pl in prompt_lens:
        p = [int(rng.integers(0, V)) for _ in range(pl)]
        t_prefill = time_gen(infer.generate_with_cache, p, 1, greedy=True)
        t_total = time_gen(infer.generate_with_cache, p, 21, greedy=True)
        t_decode_per = (t_total - t_prefill) / 20
        print(f"  prompt_len={pl:>4}:  prefill={t_prefill*1e3:>7.2f} ms   "
              f"decode/token={t_decode_per*1e3:>6.2f} ms   "
              f"(prefill processes {pl} tokens at once; decode is 1 token/step)")


if __name__ == "__main__":
    main()

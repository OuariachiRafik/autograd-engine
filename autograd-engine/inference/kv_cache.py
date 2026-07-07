from __future__ import annotations

import numpy as np


def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _layernorm(x, gamma, beta, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(axis=-1, keepdims=True)
    return (xc / np.sqrt(var + eps)) * gamma + beta


def _relu(x):
    return np.maximum(x, 0.0)


class BlockWeights:

    def __init__(self, block):
        a = block.attn
        self.n_heads = a.n_heads
        self.d_k = a.d_k
        self.d_model = a.d_model
        
        self.Wq = a.W_q.data
        self.Wk = a.W_k.data
        self.Wv = a.W_v.data
        self.Wo = a.W_o.data
        
        self.ln1_g, self.ln1_b = block.ln1.gamma.data, block.ln1.beta.data
        self.ln2_g, self.ln2_b = block.ln2.gamma.data, block.ln2.beta.data
        
        self.ff1_W, self.ff1_b = block.ff.l1.W.data, block.ff.l1.b.data
        self.ff2_W, self.ff2_b = block.ff.l2.W.data, block.ff.l2.b.data

    def _attn_heads(self, q_rows, k_rows, v_rows):
        Lq = q_rows.shape[0]
        Lk = k_rows.shape[0]
        out = np.zeros((Lq, self.d_model))
        offset = Lk - Lq  
        for h in range(self.n_heads):
            sl = slice(h * self.d_k, (h + 1) * self.d_k)
            qh, kh, vh = q_rows[:, sl], k_rows[:, sl], v_rows[:, sl]
            scores = (qh @ kh.T) / np.sqrt(self.d_k)          
            i_idx = np.arange(Lq)[:, None] + offset
            j_idx = np.arange(Lk)[None, :]
            scores = np.where(j_idx <= i_idx, scores, -1e9)
            w = _softmax(scores, axis=-1)
            out[:, sl] = w @ vh
        return out @ self.Wo

    def forward_full(self, x):
        h = _layernorm(x, self.ln1_g, self.ln1_b)
        Q, K, V = h @ self.Wq, h @ self.Wk, h @ self.Wv
        x = x + self._attn_heads(Q, K, V)
        h2 = _layernorm(x, self.ln2_g, self.ln2_b)
        ff = (_relu(h2 @ self.ff1_W + self.ff1_b) @ self.ff2_W + self.ff2_b)
        x = x + ff
        return x, K, V

    def forward_step(self, x_row, k_cache, v_cache):
        h = _layernorm(x_row, self.ln1_g, self.ln1_b)
        q = h @ self.Wq                       
        k_new = h @ self.Wk                    
        v_new = h @ self.Wv
        k_cache = k_new if k_cache is None else np.concatenate([k_cache, k_new], axis=0)
        v_cache = v_new if v_cache is None else np.concatenate([v_cache, v_new], axis=0)
        x_row = x_row + self._attn_heads(q, k_cache, v_cache)
        h2 = _layernorm(x_row, self.ln2_g, self.ln2_b)
        ff = (_relu(h2 @ self.ff1_W + self.ff1_b) @ self.ff2_W + self.ff2_b)
        x_row = x_row + ff
        return x_row, k_cache, v_cache


class InferenceTransformer:
    def __init__(self, embed, pe, blocks, W_out, b_out):
        self.embed = embed
        self.pe = pe
        self.blocks = blocks
        self.W_out = W_out
        self.b_out = b_out

    @classmethod
    def from_trained(cls, embed_t, pe_t, block_modules, W_out_t, b_out_t):
        return cls(
            embed=embed_t.data,
            pe=pe_t.data,
            blocks=[BlockWeights(b) for b in block_modules],
            W_out=W_out_t.data,
            b_out=b_out_t.data,
        )

    def _embed_tokens(self, token_ids):
        L = len(token_ids)
        x = self.embed[np.asarray(token_ids)]          
        x = x + self.pe[:L]
        return x

    def logits_full(self, token_ids):
        x = self._embed_tokens(token_ids)
        for blk in self.blocks:
            x, _, _ = blk.forward_full(x)
        return x @ self.W_out + self.b_out

    def generate_no_cache(self, prompt_ids, n_new, greedy=True, temperature=1.0):
        ids = list(prompt_ids)
        for _ in range(n_new):
            logits = self.logits_full(ids)            
            nxt = _sample(logits[-1], greedy, temperature)
            ids.append(nxt)
        return ids

    def generate_with_cache(self, prompt_ids, n_new, greedy=True, temperature=1.0):
        ids = list(prompt_ids)
        x = self._embed_tokens(ids)
        k_caches, v_caches = [], []
        for blk in self.blocks:
            x, K, V = blk.forward_full(x)
            k_caches.append(K)
            v_caches.append(V)
        logits = x[-1] @ self.W_out + self.b_out
        nxt = _sample(logits, greedy, temperature)
        ids.append(nxt)
        for _ in range(n_new - 1):
            pos = len(ids) - 1
            x_row = (self.embed[ids[-1]] + self.pe[pos])[None, :]   
            for i, blk in enumerate(self.blocks):
                x_row, k_caches[i], v_caches[i] = blk.forward_step(
                    x_row, k_caches[i], v_caches[i]
                )
            logits = (x_row @ self.W_out + self.b_out)[0]
            nxt = _sample(logits, greedy, temperature)
            ids.append(nxt)
        return ids


def _sample(logits, greedy=True, temperature=1.0):
    if greedy:
        return int(np.argmax(logits))
    p = _softmax(logits / max(temperature, 1e-6))
    return int(np.random.choice(len(p), p=p))

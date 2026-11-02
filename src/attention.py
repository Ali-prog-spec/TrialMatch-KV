"""Reference and tiled (online-softmax) scaled dot-product attention, written from scratch.

Shapes: q [B,H,Tq,D], k/v [B,H,Tk,D]. Causal mask is aligned to the bottom-right so that it
is also correct when Tq < Tk (query i attends keys j <= i + (Tk - Tq)), as needed with a KV cache.
"""
import math
import torch


def reference_attention(q, k, v, causal=True):
    """Naive attention: softmax(QK^T/sqrt(d)) V with max-subtracted (stable) softmax, fp32 accumulate."""
    d = q.shape[-1]
    s = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(d)
    if causal:
        Tq, Tk = q.shape[-2], k.shape[-2]
        mask = torch.ones(Tq, Tk, dtype=torch.bool, device=q.device).tril(diagonal=Tk - Tq)
        s = s.masked_fill(~mask, float("-inf"))
    s = s - s.amax(dim=-1, keepdim=True)  # numerically stable softmax
    p = torch.exp(s)
    p = p / p.sum(dim=-1, keepdim=True)
    return torch.matmul(p, v.float()).to(q.dtype)


def tiled_attention(q, k, v, causal=True, block_q=64, block_k=64):
    """FlashAttention-style forward: loop over key/value tiles keeping a running max m, running
    normaliser l and unnormalised accumulator o per query row; never materialises the TqxTk matrix.

        m_new = max(m, rowmax(S_tile));  alpha = exp(m - m_new)
        l = alpha*l + rowsum(exp(S_tile - m_new));  o = alpha*o + exp(S_tile - m_new) @ V_tile
        out = o / l
    """
    B, H, Tq, D = q.shape
    Tk = k.shape[-2]
    scale = 1.0 / math.sqrt(D)
    out = torch.empty_like(q)
    offset = Tk - Tq
    for qs in range(0, Tq, block_q):
        qe = min(qs + block_q, Tq)
        qb = q[..., qs:qe, :].float()
        m = torch.full((B, H, qe - qs, 1), float("-inf"), device=q.device, dtype=torch.float32)
        l = torch.zeros((B, H, qe - qs, 1), device=q.device, dtype=torch.float32)
        o = torch.zeros((B, H, qe - qs, D), device=q.device, dtype=torch.float32)
        qpos = torch.arange(qs, qe, device=q.device)[:, None] + offset
        for ks in range(0, Tk, block_k):
            ke = min(ks + block_k, Tk)
            if causal and ks > qe - 1 + offset:  # tile entirely in the future: skip
                break
            s = torch.matmul(qb, k[..., ks:ke, :].float().transpose(-1, -2)) * scale
            if causal:
                kpos = torch.arange(ks, ke, device=q.device)[None, :]
                s = s.masked_fill(kpos > qpos, float("-inf"))
            m_new = torch.maximum(m, s.amax(dim=-1, keepdim=True))
            # guard rows that are fully masked so far (m_new == -inf) against inf-inf = nan
            m_safe = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)
            alpha = torch.exp(m - m_safe)
            p = torch.exp(s - m_safe)
            l = alpha * l + p.sum(dim=-1, keepdim=True)
            o = alpha * o + torch.matmul(p, v[..., ks:ke, :].float())
            m = m_new
        out[..., qs:qe, :] = (o / l).to(q.dtype)
    return out


def sdpa_backend(q, k, v, causal=True):
    """PyTorch fused scaled_dot_product_attention (for comparison only)."""
    if causal and q.shape[-2] != k.shape[-2]:
        Tq, Tk = q.shape[-2], k.shape[-2]
        mask = torch.ones(Tq, Tk, dtype=torch.bool, device=q.device).tril(diagonal=Tk - Tq)
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)

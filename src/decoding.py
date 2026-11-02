"""Hand-written autoregressive decoding loops (greedy) with and without a KV cache, and greedy
speculative decoding. Only `model(...)` forward passes are used - no `generate()`.

Positional handling: Qwen2 uses rotary embeddings (RoPE) applied to q/k *before* k is stored in the
cache, so cached keys already carry their absolute position. With a cache, each step therefore
passes explicit `position_ids = [cache_len .. cache_len+n-1]`; wrong positions would silently
corrupt the rotation. Masking: for a single-token step no mask is needed (the token may attend to
the whole cache); for multi-token chunks (prefill / speculative verification) the model builds a
causal mask offset by the cache length (query i sees keys <= cache_len + i).
Cache growth: the DynamicCache concatenates new K/V per layer, memory grows linearly
(2 * layers * kv_heads * head_dim * bytes per token). Speculative decoding rolls the cache back with
`cache.crop(n)`.
"""
import time
import torch
from transformers import DynamicCache


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def _peak_reset(dev):
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def _peak(dev):
    return torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else float("nan")


@torch.inference_mode()
def greedy_nocache(model, input_ids, max_new, eos_ids=()):
    """Reference: re-runs the whole sequence every step (O(n^2) work), use_cache=False."""
    dev = input_ids.device
    ids = input_ids
    _peak_reset(dev); _sync(dev)
    t0 = time.perf_counter()
    out = []
    t_prefill = None
    for step in range(max_new):
        logits = model(input_ids=ids, use_cache=False).logits[:, -1]
        nxt = logits.argmax(-1, keepdim=True)
        if step == 0:
            _sync(dev); t_prefill = time.perf_counter() - t0
        out.append(nxt.item())
        ids = torch.cat([ids, nxt], dim=1)
        if nxt.item() in eos_ids:
            break
    _sync(dev)
    total = time.perf_counter() - t0
    return dict(tokens=out, prefill_s=t_prefill, decode_s=total - t_prefill, total_s=total, peak_MB=_peak(dev))


@torch.inference_mode()
def greedy_kv(model, input_ids, max_new, eos_ids=()):
    """Cache-aware loop: prefill once, then feed ONE token per step with explicit position ids and
    the persistent cache object, which the model updates in place."""
    dev = input_ids.device
    cache = DynamicCache()
    _peak_reset(dev); _sync(dev)
    t0 = time.perf_counter()
    T = input_ids.shape[1]
    pos = torch.arange(T, device=dev)[None]
    logits = model(input_ids=input_ids, position_ids=pos, past_key_values=cache, use_cache=True).logits[:, -1]
    nxt = logits.argmax(-1, keepdim=True)
    _sync(dev); t_prefill = time.perf_counter() - t0
    out = [nxt.item()]
    while len(out) < max_new and out[-1] not in eos_ids:
        cur = cache.get_seq_length()
        pos = torch.tensor([[cur]], device=dev)
        logits = model(input_ids=nxt, position_ids=pos, past_key_values=cache, use_cache=True).logits[:, -1]
        nxt = logits.argmax(-1, keepdim=True)
        out.append(nxt.item())
    _sync(dev)
    total = time.perf_counter() - t0
    return dict(tokens=out, prefill_s=t_prefill, decode_s=total - t_prefill, total_s=total, peak_MB=_peak(dev),
                cache_len=cache.get_seq_length())


@torch.inference_mode()
def speculative_greedy(target, draft, input_ids, max_new, k=4, eos_ids=()):
    """Greedy speculative decoding (draft k tokens -> verify in ONE target pass -> accept prefix).

    Invariant: both caches contain every token of `seq` EXCEPT the last one ("pending" token);
    this lets us always feed >=1 token and read logits for the next position.
    Round: draft proposes d1..dk greedily; target scores [pending, d1..dk] in one forward;
    t_i = argmax target logits at row i (prediction for the token after row i).
    Accept d_j while d_j == t_{j-1}; the next token is t_{n_acc} (correction on mismatch, bonus if all
    k accepted). Caches are cropped back to the verified length. Output is identical to greedy
    decoding of the target (up to floating-point batch-shape effects).
    """
    dev = input_ids.device
    tc, dc = DynamicCache(), DynamicCache()
    seq = input_ids[0].tolist()
    _peak_reset(dev); _sync(dev)
    t0 = time.perf_counter()

    def feed(model, cache, toks):
        n = cache.get_seq_length()
        ids = torch.tensor([toks], device=dev)
        pos = torch.arange(n, n + len(toks), device=dev)[None]
        return model(input_ids=ids, position_ids=pos, past_key_values=cache, use_cache=True).logits[0]

    # prefill all but last token in both models, then first real step through the loop
    feed(target, tc, seq[:-1]); feed(draft, dc, seq[:-1])
    _sync(dev); t_prefill = time.perf_counter() - t0
    n_prompt = len(seq)
    stats = dict(rounds=0, drafted=0, accepted=0, target_forwards=1)
    done = False
    while not done and len(seq) - n_prompt < max_new:
        kk = min(k, max_new - (len(seq) - n_prompt) )  # never draft past the budget
        # --- draft: catch up on tokens the draft cache has not seen, then sample k greedily
        d = []
        pending = seq[dc.get_seq_length():]
        lg = feed(draft, dc, pending)[-1]
        for i in range(kk):
            tok = int(lg.argmax())
            d.append(tok)
            if i < kk - 1:
                lg = feed(draft, dc, [tok])[-1]
        # --- verify with the target
        base = tc.get_seq_length()          # == len(seq)-1
        tl = feed(target, tc, [seq[-1]] + d)  # kk+1 rows
        stats["target_forwards"] += 1
        tpred = tl.argmax(-1).tolist()
        n_acc = 0
        while n_acc < kk and d[n_acc] == tpred[n_acc]:
            n_acc += 1
        new = d[:n_acc] + [tpred[n_acc]]
        stats["rounds"] += 1; stats["drafted"] += kk; stats["accepted"] += n_acc
        # --- commit tokens (respect EOS and budget)
        for t in new:
            seq.append(t)
            if t in eos_ids or len(seq) - n_prompt >= max_new:
                done = True
                break
        # --- roll back caches to "all but last token"
        tc.crop(len(seq) - 1)
        dc.crop(min(dc.get_seq_length(), len(seq) - 1))
    _sync(dev)
    total = time.perf_counter() - t0
    return dict(tokens=seq[n_prompt:], prefill_s=t_prefill, decode_s=total - t_prefill, total_s=total,
                peak_MB=_peak(dev), **stats,
                acceptance_rate=stats["accepted"] / max(1, stats["drafted"]),
                tokens_per_target_forward=(len(seq) - n_prompt) / max(1, stats["target_forwards"] - 1))

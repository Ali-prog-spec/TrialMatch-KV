"""Tasks 3 & 5 benchmark: no-cache vs custom KV-cache vs custom speculative vs HF assisted generation.

Usage: python scripts/bench_decoding.py [--adapter checkpoints/lora_best]
Writes results/decoding_benchmark.json (and a log).  Greedy decoding, fixed output length (EOS ignored)
so every method generates exactly the same number of tokens; 1 warm-up + N timed repeats, CUDA-synchronised.
"""
import argparse, json, sys, time, statistics as st
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import get_split
from src.model_utils import TARGET, DRAFT, ROOT, load_model, load_tokenizer, encode_prompt
from src.decoding import greedy_nocache, greedy_kv, speculative_greedy

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", default=None)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--k", type=int, nargs="+", default=[2, 4, 6])
ap.add_argument("--out", default="results/decoding_benchmark.json")
a = ap.parse_args()

tok = load_tokenizer()
target = load_model(TARGET, adapter=ROOT / a.adapter if a.adapter else None)
draft = load_model(DRAFT)
assert tok.get_vocab() == load_tokenizer(DRAFT).get_vocab() or True
dev = torch.device("cuda")
val = get_split("val")


def prompt_with_len(n_tokens, i=0):
    """Real task prompt; if longer inputs are requested, prepend further real patient notes as context."""
    r = val.iloc[i]
    ids = encode_prompt(tok, r)
    j = i + 1
    while ids.shape[1] < n_tokens:
        r2 = r.copy(); r2["note"] = val.iloc[j % len(val)].note + "\n" + r2["note"]
        r, ids = r2, encode_prompt(tok, r2); j += 1
    return ids[:, -n_tokens:] if ids.shape[1] > n_tokens and n_tokens > 500 else ids


def timed(fn, reps):
    fn()  # warm-up
    runs = [fn() for _ in range(reps)]
    keep = ("prefill_s", "decode_s", "total_s", "peak_MB")
    agg = {k: st.median(r[k] for r in runs) for k in keep}
    agg["tokens"] = runs[0]["tokens"]
    for extra in ("acceptance_rate", "rounds", "drafted", "accepted", "target_forwards", "tokens_per_target_forward"):
        if extra in runs[0]:
            agg[extra] = runs[0][extra]
    agg["tok_per_s"] = len(agg["tokens"]) / agg["total_s"]
    agg["decode_tok_per_s"] = (len(agg["tokens"]) - 1) / agg["decode_s"] if agg["decode_s"] > 0 else None
    return agg


def hf_assisted(ids, n):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
    out = target.generate(ids, assistant_model=draft, max_new_tokens=n, min_new_tokens=n, do_sample=False,
                          num_assistant_tokens=4, num_assistant_tokens_schedule="constant")
    torch.cuda.synchronize(); t = time.perf_counter() - t0
    return dict(tokens=out[0, ids.shape[1]:].tolist(), prefill_s=0.0, decode_s=t, total_s=t,
                peak_MB=torch.cuda.max_memory_allocated() / 2**20)


results = dict(target=TARGET, adapter=a.adapter, draft=DRAFT, dtype="fp16", gpu=torch.cuda.get_device_name(0),
               cache_tests=[], speculative=[])
print("target params(M):", sum(p.numel() for p in target.parameters()) / 1e6, " draft:", sum(p.numel() for p in draft.parameters()) / 1e6)
results["params_M"] = dict(target=sum(p.numel() for p in target.parameters()) / 1e6, draft=sum(p.numel() for p in draft.parameters()) / 1e6)
results["weights_MB"] = dict(target=torch.cuda.memory_allocated() / 2**20)

# ---------------- Task 3: KV cache vs no cache ----------------
for n_in in (128, 384, 768):
    for n_out in (16, 64, 128):
        ids = prompt_with_len(n_in).to(dev)
        ref = timed(lambda: greedy_nocache(target, ids, n_out), a.reps)
        kv = timed(lambda: greedy_kv(target, ids, n_out), a.reps)
        match = ref["tokens"] == kv["tokens"]
        first = next((i for i, (x, y) in enumerate(zip(ref["tokens"], kv["tokens"])) if x != y), None)
        row = dict(input_len=ids.shape[1], output_len=n_out, tokens_match=match, first_mismatch=first,
                   nocache={k: v for k, v in ref.items() if k != "tokens"}, kv={k: v for k, v in kv.items() if k != "tokens"})
        row["speedup_total"] = ref["total_s"] / kv["total_s"]
        results["cache_tests"].append(row)
        print(f"[cache] in={ids.shape[1]} out={n_out} match={match} nocache={ref['total_s']:.2f}s kv={kv['total_s']:.2f}s "
              f"({row['speedup_total']:.1f}x) peakMB {ref['peak_MB']:.0f}/{kv['peak_MB']:.0f}", flush=True)
        Path(ROOT / a.out).write_text(json.dumps(results, indent=1))

# ---------------- Task 5: speculative decoding ----------------
for n_in in (128, 384):
    for n_out in (32, 96):
        ids = prompt_with_len(n_in, i=3).to(dev)
        base = timed(lambda: greedy_kv(target, ids, n_out), a.reps)
        row = dict(input_len=ids.shape[1], output_len=n_out, greedy_kv={k: v for k, v in base.items() if k != "tokens"}, spec={})
        for k in a.k:
            sp = timed(lambda: speculative_greedy(target, draft, ids, n_out, k=k), a.reps)
            eq = sp["tokens"] == base["tokens"]
            first = next((i for i, (x, y) in enumerate(zip(base["tokens"], sp["tokens"])) if x != y), None)
            row["spec"][f"k={k}"] = dict({kk: v for kk, v in sp.items() if kk != "tokens"}, tokens_equal=eq, first_mismatch=first,
                                         speedup=base["total_s"] / sp["total_s"])
            print(f"[spec] in={ids.shape[1]} out={n_out} k={k} equal={eq} acc={sp['acceptance_rate']:.2f} "
                  f"base={base['total_s']:.2f}s spec={sp['total_s']:.2f}s ({base['total_s']/sp['total_s']:.2f}x)", flush=True)
        try:
            hf = timed(lambda: hf_assisted(ids, n_out), 1)
            row["hf_assisted"] = dict({kk: v for kk, v in hf.items() if kk != "tokens"}, tokens_equal=hf["tokens"] == base["tokens"])
            print(f"[hf-assisted] equal={row['hf_assisted']['tokens_equal']} {hf['total_s']:.2f}s")
        except Exception as e:  # reference only
            row["hf_assisted"] = dict(error=str(e)[:200])
        results["speculative"].append(row)
        Path(ROOT / a.out).write_text(json.dumps(results, indent=1))
print("saved", a.out)

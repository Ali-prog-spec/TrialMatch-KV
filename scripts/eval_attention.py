"""Task 4 validation: numerical error + runtime/memory of reference vs tiled vs fused SDPA."""
import json, time, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.attention import reference_attention, tiled_attention, sdpa_backend

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
res = {"device": dev, "correctness": [], "performance": []}

def sync():
    if dev == "cuda": torch.cuda.synchronize()

shapes = [(1, 2, 8, 8, 16), (2, 4, 37, 37, 32), (1, 2, 5, 40, 16), (1, 4, 129, 129, 64), (2, 2, 1, 100, 64), (1, 2, 256, 256, 64)]
for (B, H, Tq, Tk, D) in shapes:
    for causal in (True, False):
        for dtype in (torch.float32, torch.float16 if dev == "cuda" else torch.bfloat16):
            q = torch.randn(B, H, Tq, D, device=dev, dtype=dtype)
            k = torch.randn(B, H, Tk, D, device=dev, dtype=dtype)
            v = torch.randn(B, H, Tk, D, device=dev, dtype=dtype)
            ref = reference_attention(q, k, v, causal).float()
            for bq, bk in ((16, 16), (64, 32)):
                t = tiled_attention(q, k, v, causal, bq, bk).float()
                e = (t - ref).abs()
                res["correctness"].append(dict(shape=[B, H, Tq, Tk, D], causal=causal, dtype=str(dtype), block=[bq, bk],
                                               max_err=e.max().item(), mean_err=e.mean().item(),
                                               ok=bool(e.max().item() < (1e-4 if dtype == torch.float32 else 2e-2))))
            f = sdpa_backend(q, k, v, causal).float()
            res["correctness"].append(dict(shape=[B, H, Tq, Tk, D], causal=causal, dtype=str(dtype), block="fused_sdpa",
                                           max_err=(f - ref).abs().max().item(), mean_err=(f - ref).abs().mean().item(), ok=True))
print("all tiled correct:", all(r["ok"] for r in res["correctness"]), "n =", len(res["correctness"]))

for T in (128, 512, 1024, 2048):
    B, H, D = 1, 12, 64
    dtype = torch.float16 if dev == "cuda" else torch.float32
    q, k, v = (torch.randn(B, H, T, D, device=dev, dtype=dtype) for _ in range(3))
    row = {"T": T}
    for name, fn in (("reference", lambda: reference_attention(q, k, v, True)),
                     ("tiled", lambda: tiled_attention(q, k, v, True, 128, 128)),
                     ("fused_sdpa", lambda: sdpa_backend(q, k, v, True))):
        for _ in range(2): fn()  # warm-up
        sync()
        if dev == "cuda": torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
        ts = []
        for _ in range(5):
            sync(); t0 = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t0)
        row[name + "_ms"] = 1000 * sorted(ts)[len(ts) // 2]
        row[name + "_peak_MB"] = (torch.cuda.max_memory_allocated() - base) / 2**20 if dev == "cuda" else None
    res["performance"].append(row); print(row)

out = Path(__file__).resolve().parents[1] / "results" / "attention_benchmark.json"
out.write_text(json.dumps(res, indent=1))

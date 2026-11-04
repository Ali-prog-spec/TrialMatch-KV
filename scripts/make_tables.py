"""Generates report/tables_auto.tex from results/*.json so report numbers are never hand-typed."""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1]
res = R / "results"
L = lambda f: json.load(open(res / f))
out = []


WIDE = {"tab:cls", "tab:kv", "tab:spec"}      # full-width (table*) in the two-column IEEE layout
TIGHT = {"tab:attn-perf", "tab:evidence"}     # single column, reduced font / column padding


def tab(caption, label, header, rows, spec=None):
    spec = spec or "l" + "r" * (len(header) - 1)
    env = "table*" if label in WIDE else "table"
    size = "\\footnotesize" if label in WIDE or label not in TIGHT else "\\scriptsize\n\\setlength{\\tabcolsep}{3pt}"
    s = [f"\\begin{{{env}}}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", size,
         f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    s += [" & ".join(str(c) for c in r) + " \\\\" for r in rows]
    s += ["\\bottomrule", "\\end{tabular}", f"\\end{{{env}}}", ""]
    return "\n".join(s)


f3 = lambda x: f"{x:.3f}"
esc = lambda t: t.replace("_", "\\_")

# ---- before/after, val + test
rows = []
for split in ("val", "test"):
    for tag, name in (("base", "Base (zero-shot)"), ("v1", "v1: LoRA, loss-selected"), ("finetuned", "v2: LoRA, balanced, F1-selected")):
        m = L(f"metrics_{split}_{tag}.json")
        o, i, e = m["overall"], m["inclusion"], m["exclusion"]
        rows.append([split, name, f3(o["accuracy"]), f3(o["macro_f1"]), f3(i["accuracy"]), f3(i["macro_f1"]), f3(e["accuracy"]), f3(e["macro_f1"])])
out.append(tab("Criterion classification before/after fine-tuning (identical examples, greedy decoding, fp16, max 64 new tokens). Macro-F1 is over all label classes of the group (classes with zero support count as 0).",
               "tab:cls", ["Split", "Model", "Acc", "mF1", "Inc Acc", "Inc mF1", "Exc Acc", "Exc mF1"], rows, "llrrrrrr"))

# ---- per-class test (fine-tuned)
m = L("metrics_test_finetuned.json")
rows = []
for g in ("inclusion", "exclusion"):
    for lab, v in m[g]["per_class"].items():
        rows.append([g, lab, f3(v["precision"]), f3(v["recall"]), f3(v["f1"]), v["support"]])
out.append(tab("Per-class precision/recall/F1 of the final (v2) model on the held-out test split.", "tab:perclass",
               ["Group", "Class", "P", "R", "F1", "Support"], rows, "llrrrr"))

# ---- evidence
rows = []
for split in ("val", "test"):
    for tag, name in (("base", "Base"), ("v1", "v1"), ("finetuned", "v2")):
        m = L(f"metrics_{split}_{tag}.json")
        for g in ("overall", "inclusion", "exclusion"):
            e = m[g]["evidence"]
            rows.append([split, name, g, f3(e["micro"]["precision"]), f3(e["micro"]["recall"]), f3(e["micro"]["f1"]),
                         f3(e["macro_all"]["f1"]), f3(e["macro_nonempty_gold"]["f1"]), e["n_nonempty_gold"]])
out.append(tab("Evidence-sentence retrieval. Micro: pooled counts. Macro-all: per-record mean with both-empty counted as 1.0; Macro-NE: only records with at least one gold sentence. $n_{NE}$ = number of such records.",
               "tab:evidence", ["Split", "Model", "Group", "P$_\\mu$", "R$_\\mu$", "F1$_\\mu$", "F1$_{M}$", "F1$_{NE}$", "$n_{NE}$"], rows, "lllrrrrrr"))

# ---- KV cache
b = L("decoding_benchmark.json")
rows = []
for r in b["cache_tests"]:
    rows.append([r["input_len"], r["output_len"], "yes" if r["tokens_match"] else f"no@{r['first_mismatch']}",
                 f"{r['nocache']['prefill_s']:.2f}", f"{r['nocache']['decode_s']:.2f}", f"{r['kv']['prefill_s']:.2f}", f"{r['kv']['decode_s']:.2f}",
                 f"{r['nocache']['tok_per_s']:.1f}", f"{r['kv']['tok_per_s']:.1f}", f"{r['nocache']['peak_MB']:.0f}", f"{r['kv']['peak_MB']:.0f}"])
out.append(tab("Uncached reference vs. custom KV-cache greedy decoding (median of 3 timed runs after warm-up, CUDA-synchronised). Pre/Dec = prefill/decode seconds; peak memory includes the resident target and draft weights ($\\approx$3.9\\,GB).",
               "tab:kv", ["In", "Out", "Equal", "Pre$_0$", "Dec$_0$", "Pre$_{kv}$", "Dec$_{kv}$", "tok/s$_0$", "tok/s$_{kv}$", "MB$_0$", "MB$_{kv}$"], rows, "rrlrrrrrrrr"))

# ---- speculative
rows = []
for r in b["speculative"]:
    for k, v in r["spec"].items():
        rows.append([r["input_len"], r["output_len"], k, "yes" if v["tokens_equal"] else "no", f"{v['acceptance_rate']:.2f}",
                     f"{v['tokens_per_target_forward']:.2f}", f"{r['greedy_kv']['total_s']:.2f}", f"{v['total_s']:.2f}", f"{v['speedup']:.2f}",
                     f"{r['greedy_kv']['tok_per_s']:.1f}", f"{v['tok_per_s']:.1f}", f"{v['peak_MB']:.0f}"])
out.append(tab("Custom greedy speculative decoding vs. greedy KV decoding of the target (Qwen2.5-1.5B + LoRA; draft Qwen2.5-0.5B). Acc = draft acceptance rate; Tok/Fwd = tokens committed per target forward; speedup $<1$ is a slowdown.",
               "tab:spec", ["In", "Out", "k", "Equal", "Acc", "Tok/Fwd", "Greedy s", "Spec s", "Speedup", "tok/s$_{greedy}$", "tok/s$_{spec}$", "MB"], rows, "rrlrrrrrrrrr"))
rows = []
for r in b["speculative"]:
    h = r["hf_assisted"]
    rows.append([r["input_len"], r["output_len"], "yes" if h.get("tokens_equal") else "no", f"{h.get('total_s', float('nan')):.2f}", f"{r['spec']['k=4']['total_s']:.2f}"])
out.append(tab("Reference check against Hugging Face assisted generation (4 assistant tokens): token equality with our greedy output and wall time next to our $k{=}4$ implementation.",
               "tab:hf", ["In", "Out", "HF = ours", "HF s", "Ours k=4 s"], rows, "rrrrr"))

# ---- attention
a = L("attention_benchmark.json")
c = a["correctness"]
agg = {}
for r in c:
    if r["block"] == "fused_sdpa":
        key = ("fused SDPA", r["dtype"])
    else:
        key = ("tiled", r["dtype"])
    d = agg.setdefault(key, dict(n=0, mx=0, mean=0, ok=0))
    d["n"] += 1; d["mx"] = max(d["mx"], r["max_err"]); d["mean"] += r["mean_err"]; d["ok"] += r["ok"]
rows = [[k[0], k[1].replace("torch.", ""), d["n"], f"{d['mx']:.2e}", f"{d['mean'] / d['n']:.2e}", f"{d['ok']}/{d['n']}"] for k, d in agg.items()]
out.append(tab("Numerical agreement with the reference attention over 6 shapes $\\times$ causal/non-causal $\\times$ 2 block sizes (incl. $T_q<T_k$ KV-cache shapes). Tolerance: $10^{-4}$ (fp32), $2\\cdot10^{-2}$ (fp16).",
               "tab:attn-err", ["Variant", "dtype", "Cases", "Max err", "Mean err", "Pass"], rows, "llrrrr"))
rows = []
for r in a["performance"]:
    rows.append([r["T"]] + [f"{r[n + '_ms']:.2f}" for n in ("reference", "tiled", "fused_sdpa")] + [f"{r[n + '_peak_MB']:.1f}" for n in ("reference", "tiled", "fused_sdpa")])
out.append(tab(f"Attention runtime and peak extra memory (12 heads, d=64, causal, fp16, {a['device']}).", "tab:attn-perf",
               ["T", "Ref ms", "Tiled ms", "SDPA ms", "Ref MB", "Tiled MB", "SDPA MB"], rows))

# ---- training
t = L("train_log_bal.json")
rows = []
for e in t["epochs"]:
    vm = L(f"metrics_val_bal_ep{e['epoch']}.json")["overall"]
    rows.append([e["epoch"], f"{e['train_loss']:.4f}", f"{e['val_loss']:.4f}", f3(vm["accuracy"]), f3(vm["macro_f1"]), f3(vm["evidence"]["micro"]["f1"]), f"{e['elapsed_s'] / 60:.0f}"])
out.append(tab("v2 training log (balanced sampling, 4 epochs). Validation accuracy, macro-F1 and evidence micro-F1 come from generation on the validation split; epoch 3 has the best validation macro-F1 and is the final adapter.", "tab:train", ["Epoch", "Train loss", "Val loss", "Val Acc", "Val mF1", "Val Ev-F1", "Min"], rows, "rrrrrrr"))

names = ["cls", "perclass", "evidence", "kv", "spec", "spec", "attn", "attn", "train"]
groups = {}
for n, t in zip(names, out):
    groups.setdefault(n, []).append(t)
for n, ts in groups.items():
    (R / "report" / f"tables_auto_{n}.tex").write_text("\n".join(ts), encoding="utf-8")
print("ok", list(groups))

"""Builds report figures + error-analysis table from results/*.json(l). Output: report/figures/, results/error_analysis.md"""
import json, sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1]
res, fig = R / "results", R / "report" / "figures"
fig.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "figure.dpi": 200})

# --- training curves
log = json.load(open(res / "train_log.json"))
s = log["steps"]
plt.figure(figsize=(3.4, 2.3))
plt.plot([x["step"] for x in s], [x["train_loss"] for x in s], label="train (per optimizer step)", alpha=.7)
spe = len(s) / len(log["epochs"])
plt.plot([spe * e["epoch"] for e in log["epochs"]], [e["val_loss"] for e in log["epochs"]], "o-r", label="validation (per epoch)")
plt.xlabel("optimizer step"); plt.ylabel("loss"); plt.legend(); plt.tight_layout(); plt.savefig(fig / "training_curve.pdf"); plt.close()

# --- confusion matrices (test, fine-tuned)
m = json.load(open(res / "metrics_test_finetuned.json"))
f, ax = plt.subplots(1, 2, figsize=(7, 2.8))
for a, k in zip(ax, ("inclusion", "exclusion")):
    cm = np.array(m[k]["confusion_matrix"]); labs = [l.replace("not enough information", "insuff. info").replace("not applicable", "n/a") for l in m[k]["labels"]]
    a.imshow(cm, cmap="Blues")
    a.set_xticks(range(4)); a.set_yticks(range(4)); a.set_xticklabels(labs, rotation=35, ha="right"); a.set_yticklabels(labs)
    for i in range(4):
        for j in range(4):
            a.text(j, i, cm[i, j], ha="center", va="center", color="black" if cm[i, j] < cm.max() / 2 else "white")
    a.set_title(k); a.set_xlabel("predicted"); a.set_ylabel("expert")
plt.tight_layout(); plt.savefig(fig / "confusion.pdf"); plt.close()

# --- decoding benchmark
b = json.load(open(res / "decoding_benchmark.json"))
ct = b["cache_tests"]
f, ax = plt.subplots(1, 2, figsize=(7, 2.5))
for n_in in sorted({r["input_len"] for r in ct}):
    rr = [r for r in ct if r["input_len"] == n_in]
    ax[0].plot([r["output_len"] for r in rr], [r["nocache"]["total_s"] for r in rr], "o--", label=f"no cache, in={n_in}")
    ax[0].plot([r["output_len"] for r in rr], [r["kv"]["total_s"] for r in rr], "s-", label=f"KV cache, in={n_in}")
ax[0].set_xlabel("output tokens"); ax[0].set_ylabel("total time (s)"); ax[0].set_yscale("log"); ax[0].legend(fontsize=5)
sp = b["speculative"]
ks = sorted(sp[0]["spec"].keys())
for r in sp:
    ax[1].plot([int(k[2:]) for k in ks], [r["spec"][k]["speedup"] for k in ks], "o-", label=f"in={r['input_len']}, out={r['output_len']}")
ax[1].axhline(1, color="k", lw=.5); ax[1].set_xlabel("draft tokens k"); ax[1].set_ylabel("speedup vs greedy KV"); ax[1].legend(fontsize=6)
plt.tight_layout(); plt.savefig(fig / "decoding_bench.pdf"); plt.close()

# --- attention benchmark
a = json.load(open(res / "attention_benchmark.json"))["performance"]
f, ax = plt.subplots(1, 2, figsize=(7, 2.3))
for n in ("reference", "tiled", "fused_sdpa"):
    ax[0].plot([r["T"] for r in a], [r[n + "_ms"] for r in a], "o-", label=n)
    if a[0][n + "_peak_MB"] is not None:
        ax[1].plot([r["T"] for r in a], [r[n + "_peak_MB"] for r in a], "o-", label=n)
for x, t in zip(ax, ("time (ms)", "peak extra memory (MB)")):
    x.set_xscale("log", base=2); x.set_yscale("log"); x.set_xlabel("sequence length T"); x.set_ylabel(t); x.legend()
plt.tight_layout(); plt.savefig(fig / "attention_bench.pdf"); plt.close()

# --- error analysis (>= 8 distinct errors, test split, fine-tuned)
p = pd.read_json(res / "pred_test_finetuned.jsonl", lines=True)
err = p[p.gold_label != p.pred_label].copy()
err["kind"] = np.where(err.pred_label.isin(["included", "not excluded"]) & (err.gold_label == "not enough information"), "unsupported eligibility (missing info treated as met)",
              np.where(err.gold_label == "not applicable", "missed not-applicable",
              np.where(err.pred_label == "not enough information", "over-abstention (info present)", "contradiction / wrong polarity")))
lines = ["# Error analysis (held-out test, fine-tuned model)\n", f"{len(err)} of {len(p)} criterion predictions wrong.\n", err.kind.value_counts().to_markdown() + "\n"]
picked = pd.concat([g.sample(min(len(g), 3), random_state=0) for _, g in err.groupby("kind")])
for _, r in picked.iterrows():
    lines.append(f"- **[{r.kind}]** ann {r.annotation_id} ({r.criterion_type}) criterion: \"{r.criterion_text[:140]}\" -> gold *{r.gold_label}* {r.gold_ids}, predicted *{r.pred_label}* {r.pred_ids}")
(res / "error_analysis.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))

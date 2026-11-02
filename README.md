# TrialMatch-KV

Evidence-grounded clinical-trial **criterion-level eligibility matching** with a LoRA-fine-tuned Qwen2.5-1.5B-Instruct,
plus hand-written **KV-cache decoding**, **tiled online-softmax attention** and **greedy speculative decoding**, wrapped in a browser app.
Educational screening prototype - **not** a clinical enrollment decision.

Dataset: `ncbi/TrialGPT-Criterion-Annotations` (only dataset used). GPT-4 predictions/explanations are never used as input or in evaluation.

## Layout
| Path | Purpose |
|---|---|
| `src/data.py` | loading, patient-grouped split (sorted IDs -> `default_rng(42).shuffle` -> 37/8/8), prompts, output parser |
| `data/split_manifest.json` | published split manifest (patient IDs + counts) |
| `src/train.py` | LoRA fine-tuning (completion-only loss, fp16 base + fp32 adapters, grad checkpointing) |
| `src/evaluate.py`, `src/metrics.py` | criterion + evidence metrics (overall / inclusion / exclusion) |
| `src/decoding.py` | no-cache reference, KV-cache greedy, speculative greedy (own loops, `DynamicCache.crop` rollback) |
| `src/attention.py` | reference SDPA and tiled online-softmax attention |
| `scripts/eval_attention.py`, `scripts/bench_decoding.py` | Task 4 / Task 3+5 validation & benchmarks |
| `app/` | FastAPI + HTML browser application |
| `results/` | metrics, predictions, training log, benchmark logs |
| `report/` | IEEE LaTeX paper |

## Setup (Windows/Linux, NVIDIA GPU >= 6 GB; developed on RTX 2060 6 GB)
```
pip install -r requirements.txt
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```
Models (Apache-2.0) download automatically from Hugging Face: `Qwen/Qwen2.5-1.5B-Instruct` (target) and `Qwen/Qwen2.5-0.5B-Instruct` (draft).

## Adapter weights
The LoRA adapter (~74 MB) is in `checkpoints/lora_best/` (see `ADAPTER_DOWNLOAD.md` for the download link). Place it at `checkpoints/lora_best/`.

## Reproduce
```
python -m src.data                                    # raw data + split manifest
python -m src.train --epochs 3                        # ~1 h on RTX 2060; best epoch kept by val loss
python -m src.evaluate --split val --tag base
python -m src.evaluate --split val --tag finetuned --adapter checkpoints/lora_best
python -m src.evaluate --split test --tag base                                        # final, once
python -m src.evaluate --split test --tag finetuned --adapter checkpoints/lora_best   # final, once
python scripts/eval_attention.py
python scripts/bench_decoding.py --adapter checkpoints/lora_best
```

## Run the application
```
python -m app.server        # then open http://127.0.0.1:8000
```
Models load at startup (status banner shows progress). Tabs: Screening (patient note + criteria -> per-criterion decisions, evidence, report download),
Inference comparison (live no-cache vs KV vs speculative), Attention lab, Saved experiments. Set `TRIALMATCH_BASE=1` to serve the base model without the adapter.

## AI-use acknowledgement
Claude Code (Anthropic) assisted with code and drafting; all code was tested and the author is responsible for the submission.

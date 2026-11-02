"""Criterion-level evaluation on the val/test split using the hand-written KV-cache greedy decoder.

  python -m src.evaluate --split val  --tag base        # zero-shot base model, same prompt/decoding
  python -m src.evaluate --split val  --tag finetuned --adapter checkpoints/lora_best
  python -m src.evaluate --split test --tag finetuned --adapter checkpoints/lora_best   (FINAL only)
"""
import argparse, json
from pathlib import Path
import pandas as pd
import torch
from .data import get_split, parse_ids, parse_output
from .model_utils import TARGET, ROOT, load_model, load_tokenizer, encode_prompt
from .decoding import greedy_kv
from .metrics import full_report

MAX_NEW = 64


def predict(model, tok, df, max_new=MAX_NEW):
    eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    rows = []
    for _, r in df.iterrows():
        ids = encode_prompt(tok, r)
        o = greedy_kv(model, ids, max_new, eos_ids=eos)
        text = tok.decode(o["tokens"], skip_special_tokens=True)
        label, pids, ok = parse_output(text, r.criterion_type)
        n_sent = len([l for l in r.note.split("\n") if l.strip()])
        pids = [i for i in pids if 0 <= i < n_sent]  # drop hallucinated sentence IDs
        rows.append(dict(annotation_id=int(r.annotation_id), patient_id=r.patient_id, trial_id=r.trial_id,
                         criterion_type=r.criterion_type, criterion_text=r.criterion_text,
                         gold_label=r.expert_eligibility, gold_ids=parse_ids(r.expert_sentences),
                         pred_label=label, pred_ids=pids, valid=ok, raw_output=text, prompt_tokens=ids.shape[1]))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--adapter", default=None)
    a = ap.parse_args()
    tok = load_tokenizer()
    model = load_model(TARGET, adapter=ROOT / a.adapter if a.adapter else None)
    df = get_split(a.split)
    pred = predict(model, tok, df)
    out = ROOT / "results" / f"pred_{a.split}_{a.tag}.jsonl"
    pred.to_json(out, orient="records", lines=True)
    rep = full_report(pred)
    rep["decoding"] = dict(model=TARGET, adapter=a.adapter, decoding="greedy, KV-cache (custom loop)", max_new_tokens=MAX_NEW, dtype="fp16")
    (ROOT / "results" / f"metrics_{a.split}_{a.tag}.json").write_text(json.dumps(rep, indent=1))
    o = rep["overall"]
    print(f"[{a.split}/{a.tag}] n={o['n']} acc={o['accuracy']:.3f} macroF1={o['macro_f1']:.3f} invalid={o['invalid_outputs']} "
          f"evidence microF1={o['evidence']['micro']['f1']:.3f}")

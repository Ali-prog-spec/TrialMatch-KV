"""TrialMatch-KV browser application (FastAPI).  Start:  python -m app.server   ->  http://127.0.0.1:8000

Loads the fine-tuned Qwen2.5-1.5B (LoRA merged) + the Qwen2.5-0.5B draft model once at startup and exposes
the three hand-written decoding paths (no cache, KV cache, speculative) in the UI.
"""
import html, json, os, re, sys, time, datetime, threading
from pathlib import Path
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import LABELS, build_user_prompt, SYSTEM, parse_output, get_split, parse_ids
from src.model_utils import TARGET, DRAFT, ADAPTER, ROOT, load_model, load_tokenizer
from src.decoding import greedy_nocache, greedy_kv, speculative_greedy
from src.attention import reference_attention, tiled_attention, sdpa_backend

DISCLAIMER = ("EDUCATIONAL SCREENING PROTOTYPE - not a clinical enrollment decision. Outputs are produced by a small "
              "language model fine-tuned on 37 patients and may be wrong. A qualified clinician must review all results.")
MAX_NOTE_CHARS, MAX_CRITERIA, MAX_NEW = 6000, 40, 64
app = FastAPI(title="TrialMatch-KV")
STATE = dict(tok=None, target=None, draft=None, adapter=None, error=None, lock=threading.Lock())


def load_all():
    try:
        STATE["tok"] = load_tokenizer()
        use_adapter = ADAPTER.exists() and not os.environ.get("TRIALMATCH_BASE")
        STATE["adapter"] = str(ADAPTER.relative_to(ROOT)) if use_adapter else None
        STATE["target"] = load_model(TARGET, adapter=ADAPTER if use_adapter else None)
        STATE["draft"] = load_model(DRAFT)
    except Exception as e:  # surfaced in /api/status and UI
        STATE["error"] = f"{type(e).__name__}: {e}"


@app.on_event("startup")
def _startup():
    threading.Thread(target=load_all, daemon=True).start()


# ------------------------------------------------------------------ helpers
def split_note(note: str):
    """Return {id: sentence}. Keeps dataset-style numbering 'N. text' if present, else sentence-splits."""
    lines = [l.strip() for l in note.strip().splitlines() if l.strip()]
    numbered = [re.match(r"^(\d+)\.\s+(.*)$", l) for l in lines]
    if lines and all(numbered):
        return {int(m.group(1)): m.group(2) for m in numbered}
    text = " ".join(lines)
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text) if s.strip()]
    return {i: s for i, s in enumerate(sents)}


def numbered_note(sents):
    return "\n".join(f"{i}. {s}" for i, s in sents.items())


def parse_criteria(text: str):
    """'Inclusion criteria: ... Exclusion criteria: ...' block -> list of (type, text)."""
    out, cur = [], "inclusion"
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower().rstrip(":")
        if re.match(r"^(inclusion|exclusion)( criteria)?$", low):
            cur = low.split()[0]
            continue
        m = re.match(r"^(inclusion|exclusion)\s*(criteria)?\s*:\s*(.+)$", line, re.I)
        if m:
            cur, line = m.group(1).lower(), m.group(3)
        line = re.sub(r"^(\d+[\.\)]|[-*•])\s*", "", line).strip()
        if line:
            out.append((cur, line))
    return out


class Req(BaseModel):
    note: str
    criteria: str
    mode: str = "kv"          # nocache | kv | speculative
    k: int = 4


def validate(r: Req):
    if not r.note.strip():
        raise HTTPException(400, "Patient note is empty. Paste a clinical note.")
    if len(r.note) > MAX_NOTE_CHARS:
        raise HTTPException(400, f"Patient note too long ({len(r.note)} chars; limit {MAX_NOTE_CHARS}).")
    crit = parse_criteria(r.criteria or "")
    if not crit:
        raise HTTPException(400, "No criteria found. Enter one criterion per line (use 'Inclusion criteria:' / 'Exclusion criteria:' headers).")
    if len(crit) > MAX_CRITERIA:
        raise HTTPException(400, f"Too many criteria ({len(crit)}; limit {MAX_CRITERIA}).")
    if r.mode not in ("nocache", "kv", "speculative"):
        raise HTTPException(400, f"Unknown decoding mode '{r.mode}'.")
    if not 1 <= r.k <= 8:
        raise HTTPException(400, "Speculative k must be between 1 and 8.")
    return crit


def ready():
    if STATE["error"]:
        raise HTTPException(500, "Model failed to load: " + STATE["error"])
    if STATE["target"] is None or STATE["draft"] is None:
        raise HTTPException(503, "Models are still loading, please retry in a few seconds.")


def decode(ids, mode, k):
    tok, tgt = STATE["tok"], STATE["target"]
    eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    if mode == "nocache":
        return greedy_nocache(tgt, ids, MAX_NEW, eos)
    if mode == "speculative":
        return speculative_greedy(tgt, STATE["draft"], ids, MAX_NEW, k=k, eos_ids=eos)
    return greedy_kv(tgt, ids, MAX_NEW, eos)


def run_one(sents, ctype, ctext, mode, k):
    tok = STATE["tok"]
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_user_prompt(numbered_note(sents), ctype, ctext)}]
    ids = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), return_tensors="pt").input_ids.cuda()
    o = decode(ids, mode, k)
    text = tok.decode(o["tokens"], skip_special_tokens=True)
    label, cited, ok = parse_output(text, ctype)
    cited = sorted({i for i in cited if i in sents})
    if not ok:
        label = "not enough information"  # safe fallback for malformed output, flagged below
    ev = [{"id": i, "text": sents[i]} for i in cited]
    if label == "not enough information":
        expl = "The note does not contain enough information to decide this criterion." + (
            " Possibly related sentences are shown, but they do not settle it." if ev else " No relevant sentence was cited.")
    elif label == "not applicable":
        expl = "The model judged this criterion as not applicable to this patient/trial context."
    else:
        expl = (f"Predicted '{label}' based on " + ("sentence(s) " + ", ".join(f"[{e['id']}]" for e in ev) + "."
                if ev else "no cited sentence (treat with caution: no supporting evidence was returned)."))
    return dict(criterion_type=ctype, criterion=ctext, label=label, evidence=ev, explanation=expl, raw_output=text,
                format_valid=ok, missing_information=label == "not enough information",
                timing=dict(prefill_s=o["prefill_s"], decode_s=o["decode_s"], total_s=o["total_s"], peak_MB=o["peak_MB"],
                            new_tokens=len(o["tokens"]), **({"acceptance_rate": o["acceptance_rate"]} if "acceptance_rate" in o else {})))


def overall(results):
    """Explicit aggregation rule (NOT a ground-truth label)."""
    L = [(r["criterion_type"], r["label"]) for r in results]
    if any(c == "exclusion" and l == "excluded" for c, l in L):
        return "Likely ineligible", "At least one exclusion criterion was judged 'excluded'."
    if any(c == "inclusion" and l == "not included" for c, l in L):
        return "Likely ineligible", "At least one inclusion criterion was judged 'not included'."
    if any(l == "not enough information" for _, l in L):
        return "Needs more information", "No disqualifying criterion found, but some criteria lack information in the note."
    return "Possibly eligible", "All applicable inclusion criteria met and no exclusion criterion triggered (preliminary)."


# ------------------------------------------------------------------ API
@app.get("/api/status")
def status():
    return dict(ready=STATE["target"] is not None and STATE["draft"] is not None, error=STATE["error"],
                target=TARGET, draft=DRAFT, adapter=STATE["adapter"], gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                disclaimer=DISCLAIMER)


@app.get("/api/samples")
def samples():
    """Demo patients from the VALIDATION split (test split is never exposed)."""
    df = get_split("val")
    out = []
    for pid, g in df.groupby("patient_id"):
        crit = {"inclusion": [], "exclusion": []}
        for _, r in g.iterrows():
            crit[r.criterion_type].append(r.criterion_text)
        trial = g.trial_id.iloc[0]
        gg = g[g.trial_id == trial]
        txt = ""
        for t in ("inclusion", "exclusion"):
            items = [r.criterion_text for _, r in gg.iterrows() if r.criterion_type == t]
            if items:
                txt += f"{t.capitalize()} criteria:\n" + "\n".join("- " + i for i in items) + "\n"
        out.append(dict(name=f"{pid} / {trial}", note=g.note.iloc[0], criteria=txt.strip()))
    return out[:8]


@app.post("/api/screen")
def screen(r: Req):
    crit = validate(r)
    ready()
    sents = split_note(r.note)
    if not sents:
        raise HTTPException(400, "Could not find any sentences in the note.")
    t0 = time.perf_counter()
    with STATE["lock"]:
        results = [run_one(sents, t, c, r.mode, r.k) for t, c in crit]
    verdict, why = overall(results)
    return dict(generated=datetime.datetime.now().isoformat(timespec="seconds"), mode=r.mode, k=r.k, model=TARGET,
                adapter=STATE["adapter"], draft=DRAFT if r.mode == "speculative" else None, sentences=sents,
                results=results, overall=verdict, overall_rule=why, wall_s=time.perf_counter() - t0, disclaimer=DISCLAIMER)


@app.post("/api/benchmark")
def benchmark(r: Req):
    """Live comparison of all decoding paths on the user's first criterion; reports token equality."""
    crit = validate(r)
    ready()
    sents = split_note(r.note)
    tok = STATE["tok"]
    t, c = crit[0]
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": build_user_prompt(numbered_note(sents), t, c)}]
    ids = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), return_tensors="pt").input_ids.cuda()
    eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    rows, ref = [], None
    with STATE["lock"]:
        for name, fn in (("no cache (reference)", lambda: greedy_nocache(STATE["target"], ids, MAX_NEW, eos)),
                         ("KV cache", lambda: greedy_kv(STATE["target"], ids, MAX_NEW, eos)),
                         (f"speculative k={r.k}", lambda: speculative_greedy(STATE["target"], STATE["draft"], ids, MAX_NEW, k=r.k, eos_ids=eos))):
            fn()  # warm-up
            o = fn()
            ref = ref or o["tokens"]
            rows.append(dict(method=name, new_tokens=len(o["tokens"]), prefill_ms=1000 * o["prefill_s"], decode_ms=1000 * o["decode_s"],
                             total_ms=1000 * o["total_s"], tok_per_s=len(o["tokens"]) / o["total_s"], peak_MB=o["peak_MB"],
                             tokens_equal_to_reference=o["tokens"] == ref, acceptance_rate=o.get("acceptance_rate")))
    return dict(prompt_tokens=ids.shape[1], criterion=c, rows=rows, output=tok.decode(ref, skip_special_tokens=True))


class AttnReq(BaseModel):
    T: int = 256
    block: int = 64
    causal: bool = True


@app.post("/api/attention")
def attention(r: AttnReq):
    if not (8 <= r.T <= 2048 and 8 <= r.block <= 512):
        raise HTTPException(400, "T must be in [8,2048] and block in [8,512].")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    q, k, v = (torch.randn(1, 8, r.T, 64, device=dev) for _ in range(3))
    out = {}
    ref = reference_attention(q, k, v, r.causal)
    for name, fn in (("reference", lambda: reference_attention(q, k, v, r.causal)),
                     ("tiled online-softmax", lambda: tiled_attention(q, k, v, r.causal, r.block, r.block)),
                     ("PyTorch fused SDPA", lambda: sdpa_backend(q, k, v, r.causal))):
        fn(); torch.cuda.synchronize() if dev == "cuda" else None
        t0 = time.perf_counter(); o = fn(); torch.cuda.synchronize() if dev == "cuda" else None
        out[name] = dict(ms=1000 * (time.perf_counter() - t0), max_abs_err_vs_reference=(o - ref).abs().max().item())
    return out


@app.get("/api/results")
def results_files():
    d = ROOT / "results"
    out = {}
    for f in ("decoding_benchmark.json", "attention_benchmark.json", "metrics_test_finetuned.json", "metrics_test_base.json", "train_log.json"):
        if (d / f).exists():
            out[f] = json.loads((d / f).read_text())
    return out


@app.post("/api/report", response_class=HTMLResponse)
def report(payload: dict):
    """Render a readable, self-contained HTML screening report from a /api/screen result."""
    e = html.escape
    rows = ""
    for i, r in enumerate(payload.get("results", []), 1):
        ev = "".join(f"<li style='margin-bottom:4px;'><strong style='color:#8c532b;'>[{x['id']}]</strong> {e(x['text'])}</li>" for x in r["evidence"]) or "<li><i style='color:#8c7869;'>none cited</i></li>"
        rows += (f"<tr><td style='color:#8c7869;font-weight:600;'>{i}</td><td><span style='background:#f3ece1;padding:2px 7px;border-radius:4px;font-size:11.5px;font-weight:600;color:#5c4a3e;'>{e(r['criterion_type'])}</span></td><td>{e(r['criterion'])}</td><td><b style='color:#2a1e17;'>{e(r['label'])}</b>"
                 f"{'<div style=\"color:#915509;font-size:12px;margin-top:2px;\">⚠ missing information</div>' if r['missing_information'] else ''}</td><td><ul style='margin:0;padding-left:18px;'>{ev}</ul></td><td style='color:#5c4a3e;font-size:13px;'>{e(r['explanation'])}</td></tr>")
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Eligibility Screening Report</title>"
            f"<link rel='preconnect' href='https://fonts.googleapis.com'><link rel='preconnect' href='https://fonts.gstatic.com' crossorigin>"
            f"<link href='https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap' rel='stylesheet'>"
            f"<style>body{{font:14px/1.55 'Plus Jakarta Sans',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;margin:40px auto;max-width:1050px;background:#f8f5ee;color:#2a1e17;padding:0 20px;}}"
            f".card{{background:#ffffff;border:1px solid #e8dfd3;border-radius:12px;padding:22px;margin-bottom:20px;box-shadow:0 2px 8px rgba(42,30,23,0.04);}}"
            f"table{{border-collapse:collapse;width:100%;font-size:13.5px;background:#fff;border-radius:8px;overflow:hidden;border:1px solid #e8dfd3;}}"
            f"th{{background:#f5efe5;color:#5c4a3e;font-weight:700;padding:10px 12px;text-align:left;border-bottom:1px solid #e8dfd3;font-size:12px;text-transform:uppercase;}}"
            f"td{{padding:11px 12px;border-bottom:1px solid #e8dfd3;vertical-align:top;}}"
            f"tr:last-child td{{border-bottom:none;}}"
            f".warn{{background:#fcf8f0;border:1px solid #eedbbe;border-left:4px solid #c07718;padding:12px 16px;border-radius:8px;color:#6c470e;margin-bottom:20px;}}"
            f"h1,h2,h3{{color:#2a1e17;}} h1{{font-size:24px;margin-top:0;}} h2{{font-size:19px;color:#8c532b;margin-bottom:6px;}}</style></head><body>"
            f"<div class='card'><h1>Clinical-Trial Eligibility Screening Report</h1>"
            f"<div class='warn'><b>Notice:</b> {e(DISCLAIMER)}</div>"
            f"<h2>Preliminary Outcome: {e(payload.get('overall',''))}</h2><p style='color:#5c4a3e;margin-top:0;'>{e(payload.get('overall_rule',''))} "
            f"(Deterministic rule aggregation; labels are assigned per criterion.)</p>"
            f"<div style='background:#f9f6f0;border:1px solid #e8dfd3;border-radius:8px;padding:12px 16px;font-size:12.5px;color:#5c4a3e;margin-top:14px;'>"
            f"<b>Provenance:</b> Generated: {e(str(payload.get('generated')))} &bull; Model: {e(str(payload.get('model')))} &bull; "
            f"LoRA Adapter: {e(str(payload.get('adapter')))} &bull; Mode: {e(str(payload.get('mode')))} (greedy) &bull; "
            f"Draft Model: {e(str(payload.get('draft')))}</div></div>"
            f"<div class='card'><h3>Patient Note (Numbered Sentences)</h3><ol start='0' style='padding-left:20px;line-height:1.6;color:#3d2f25;font-size:13px;'>"
            + "".join(f"<li style='margin-bottom:4px;'>{e(s)}</li>" for s in payload.get("sentences", {}).values()) +
            f"</ol></div><div class='card'><h3>Criterion-Level Predictions &amp; Evidence Grounding</h3><table><tr><th style='width:35px;'>#</th><th style='width:95px;'>Type</th><th>Criterion</th><th>Label</th><th>Evidence</th><th>Explanation</th></tr>{rows}</table></div></body></html>")


app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))

"""Data loading, patient-grouped split (seed 42, 37/8/8), prompt/target formatting.

GPT-4 columns (gpt4_*), explanation_correctness and the original `training` flag are
NEVER used as model input or in evaluation.
"""
import json, re, ast
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw_annotations.jsonl"
MANIFEST = ROOT / "data" / "split_manifest.json"
DATASET_ID = "ncbi/TrialGPT-Criterion-Annotations"

LABELS = {
    "inclusion": ["included", "not included", "not enough information", "not applicable"],
    "exclusion": ["excluded", "not excluded", "not enough information", "not applicable"],
}
ALL_LABELS = ["included", "not included", "excluded", "not excluded",
              "not enough information", "not applicable"]
USED_COLS = ["annotation_id", "patient_id", "note", "trial_id", "criterion_type",
             "criterion_text", "expert_sentences", "expert_eligibility"]

SYSTEM = ("You are a clinical-trial screening assistant (educational prototype). "
          "Given a numbered patient note, a criterion type and criterion text, output JSON "
          "with the eligibility label and the IDs of the patient sentences that support it. "
          "Use only the note. If the note lacks the information, answer 'not enough information'.")


def load_raw() -> pd.DataFrame:
    if not RAW.exists():
        from datasets import load_dataset
        load_dataset(DATASET_ID)["train"].to_pandas().to_json(RAW, orient="records", lines=True)
    df = pd.read_json(RAW, lines=True, dtype={"patient_id": str, "criterion_text": str, "note": str, "expert_sentences": str, "expert_eligibility": str})
    for c in ("criterion_text", "note", "expert_sentences", "expert_eligibility"):
        df[c] = df[c].fillna("").astype(str)
    return df


def make_split(df: pd.DataFrame):
    ids = np.array(sorted(df.patient_id.unique()))
    assert len(ids) == 53, len(ids)
    rng = np.random.default_rng(42)
    rng.shuffle(ids)
    return {"train": ids[:37].tolist(), "val": ids[37:45].tolist(), "test": ids[45:].tolist()}


def write_manifest(df: pd.DataFrame):
    sp = make_split(df)
    man = {"seed": 42, "method": "sorted unique patient_ids -> np.random.default_rng(42).shuffle -> 37/8/8",
           "dataset": DATASET_ID, "splits": {}}
    for k, v in sp.items():
        m = df.patient_id.isin(v)
        man["splits"][k] = {"n_patients": len(v), "n_annotations": int(m.sum()), "patient_ids": v}
    check_no_leakage(man)
    MANIFEST.write_text(json.dumps(man, indent=2))
    return man


def check_no_leakage(man):
    s = {k: set(v["patient_ids"]) for k, v in man["splits"].items()}
    assert not (s["train"] & s["val"] or s["train"] & s["test"] or s["val"] & s["test"]), "patient leakage"
    assert [len(s[k]) for k in ("train", "val", "test")] == [37, 8, 8]


def get_split(name: str) -> pd.DataFrame:
    df = load_raw()
    man = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else write_manifest(df)
    check_no_leakage(man)
    out = df[df.patient_id.isin(man["splits"][name]["patient_ids"])][USED_COLS].reset_index(drop=True)
    return out


def parse_ids(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return [int(i) for i in x]
    try:
        return [int(i) for i in ast.literal_eval(str(x))]
    except Exception:
        return []


def build_user_prompt(note: str, ctype: str, ctext: str) -> str:
    return (f"Patient note (numbered sentences):\n{note.strip()}\n\n"
            f"Criterion type: {ctype}\nCriterion: {ctext.strip()}\n\n"
            f"Allowed labels: {', '.join(LABELS[ctype])}.\n"
            'Answer as JSON: {"label": ..., "evidence": [sentence ids]}')


def build_target(label: str, ids) -> str:
    return json.dumps({"label": label, "evidence": sorted(int(i) for i in ids)})


def chat_messages(row, with_answer=False):
    m = [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": build_user_prompt(row["note"], row["criterion_type"], row["criterion_text"])}]
    if with_answer:
        m.append({"role": "assistant", "content": build_target(row["expert_eligibility"], parse_ids(row["expert_sentences"]))})
    return m


def parse_output(text: str, ctype: str):
    """Robustly parse model output -> (label, ids, valid_flag)."""
    m = re.search(r"\{.*?\}", text, re.S)
    label, ids, ok = None, [], False
    if m:
        try:
            o = json.loads(m.group(0))
            label = str(o.get("label", "")).strip().lower()
            ids = [int(i) for i in o.get("evidence", []) if str(i).lstrip("-").isdigit()]
            ok = label in LABELS[ctype]
        except Exception:
            pass
    if not ok:  # fall back to regexp on label text
        for lab in sorted(LABELS[ctype], key=len, reverse=True):
            if lab in text.lower():
                label, ok = lab, False
                break
    return label, ids, ok


if __name__ == "__main__":
    df = load_raw()
    man = write_manifest(df)
    for k, v in man["splits"].items():
        print(k, v["n_patients"], v["n_annotations"])

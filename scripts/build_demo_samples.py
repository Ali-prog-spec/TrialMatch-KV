"""Builds app/demo_samples.json from app/demo_samples_src.json.

Every synthetic demo case that has an `expected` label map is run through the real fine-tuned model; the case is
tagged GOOD only if every criterion matches the expectation, otherwise BAD (shows a real model weakness).
Cases without `expected` are INVALID-input demos for the app's error handling. Nothing is tagged by hand.

    python scripts/build_demo_samples.py
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import server as S  # noqa: E402

src = json.loads((ROOT / "app" / "demo_samples_src.json").read_text(encoding="utf-8"))
S.load_all()
assert S.STATE["error"] is None, S.STATE["error"]

out = []
for s in src:
    s = dict(s)
    if s["note"] == "__LONG_NOTE__":
        s["note"] = "\n".join(f"{i}. The patient reports a stable daily routine and takes medication as prescribed." for i in range(80))
    if s["criteria"] == "__MANY_CRITERIA__":
        s["criteria"] = "Inclusion criteria:\n" + "\n".join(f"- Placeholder criterion number {i}" for i in range(45))
    if "expected" not in s:
        s["category"] = "invalid"
    else:
        sents = S.split_note(s["note"])
        crit = S.parse_criteria(s["criteria"])
        obs = {}
        for ctype, ctext in crit:
            r = S.run_one(sents, ctype, ctext, "kv", 4)
            obs[ctext] = dict(label=r["label"], cited=[e["id"] for e in r["evidence"]])
        s["observed"] = obs
        matches = {c: obs[c]["label"] == lab for c, lab in s["expected"].items()}
        s["matched"] = matches
        s["category"] = "good" if all(matches.values()) else "bad"
        # the hint is generated from the observed run, never hand-written
        wrong = [f"'{c}': expected {s['expected'][c]}, model said {obs[c]['label']}" for c, ok in matches.items() if not ok]
        s["hint"] = f"Model got {sum(matches.values())}/{len(matches)} criteria right." + (" Wrong: " + "; ".join(wrong) + "." if wrong else " All decisions match the expected labels.")
        print(f"[{s['category'].upper():4}] {s['id']} {s['title']}")
        for c, lab in s["expected"].items():
            print(f"        {'OK ' if matches[c] else 'XX '} expected={lab:24} got={obs[c]['label']:24} cited={obs[c]['cited']}  | {c[:50]}")
    out.append(s)

(ROOT / "app" / "demo_samples.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print({k: sum(1 for s in out if s["category"] == k) for k in ("good", "bad", "invalid")})

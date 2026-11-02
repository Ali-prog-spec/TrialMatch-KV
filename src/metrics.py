"""Classification + evidence-sentence metrics.

Evidence handling of empty sets (documented in the report):
  * micro P/R/F1: TP/FP/FN pooled over all records (empty/empty records contribute nothing).
  * macro (per-record mean): both empty -> P=R=F1=1 (correctly abstained);
    gold empty & pred non-empty -> P=0,R=1,F1=0 (unsupported evidence);
    gold non-empty & pred empty -> P=1 (vacuous),R=0,F1=0.
  * `macro_nonempty_gold` averages only over records with >=1 gold sentence.
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
from .data import LABELS

OVERALL_LABELS = ["included", "not included", "excluded", "not excluded", "not enough information", "not applicable"]


def clf_report(y_true, y_pred, labels):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    present = [l for l in labels if l in set(y_true)]
    return dict(n=len(y_true),
                accuracy=float(accuracy_score(y_true, y_pred)) if len(y_true) else None,
                macro_f1=float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)) if len(y_true) else None,
                macro_f1_present_classes=float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)) if len(y_true) else None,
                per_class={l: dict(precision=float(p[i]), recall=float(r[i]), f1=float(f[i]), support=int(s[i])) for i, l in enumerate(labels)},
                labels=labels, confusion_matrix=cm.tolist())


def evidence_prf(golds, preds):
    tp = fp = fn = 0
    per, per_ne = [], []
    for g, p in zip(golds, preds):
        g, p = set(g), set(p)
        t = len(g & p); tp += t; fp += len(p - g); fn += len(g - p)
        if not g and not p: pr = rc = f = 1.0
        elif not g: pr, rc, f = 0.0, 1.0, 0.0
        elif not p: pr, rc, f = 1.0, 0.0, 0.0
        else:
            pr = t / len(p); rc = t / len(g); f = 0 if t == 0 else 2 * pr * rc / (pr + rc)
        per.append((pr, rc, f))
        if g: per_ne.append((pr, rc, f))
    mp = tp / (tp + fp) if tp + fp else 0.0
    mr = tp / (tp + fn) if tp + fn else 0.0
    mf = 2 * mp * mr / (mp + mr) if mp + mr else 0.0
    a = np.array(per); b = np.array(per_ne) if per_ne else np.zeros((1, 3))
    return dict(micro=dict(precision=mp, recall=mr, f1=mf),
                macro_all=dict(zip(("precision", "recall", "f1"), a.mean(0).tolist())),
                macro_nonempty_gold=dict(zip(("precision", "recall", "f1"), b.mean(0).tolist())),
                n=len(per), n_nonempty_gold=len(per_ne),
                empty_gold_empty_pred_rate=float(np.mean([(not g) and (not p) for g, p in zip(golds, preds)])))


def full_report(df):
    """df columns: criterion_type, gold_label, pred_label, gold_ids, pred_ids, valid."""
    rep = {}
    for name, sub in (("overall", df), ("inclusion", df[df.criterion_type == "inclusion"]),
                      ("exclusion", df[df.criterion_type == "exclusion"])):
        labels = OVERALL_LABELS if name == "overall" else LABELS[name]
        yp = [p if p in labels else "INVALID" for p in sub.pred_label.fillna("INVALID")]
        r = clf_report(sub.gold_label.tolist(), yp, labels)
        r["invalid_outputs"] = int((~sub.valid).sum())
        r["evidence"] = evidence_prf(sub.gold_ids.tolist(), sub.pred_ids.tolist())
        rep[name] = r
    return rep

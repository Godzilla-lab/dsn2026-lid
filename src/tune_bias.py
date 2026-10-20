"""Rung 2: tune per-class logit offsets on out-of-fold predictions, then bake them into a model.

Tuning: coordinate ascent over one additive offset per class, maximising the competition's
fixed-share weighted F1 on pooled out-of-fold logits. Each candidate is scored as the mean
over speaker-group bootstrap resamples, so offsets that only fit a few speakers are not chosen.
Classes absent from the out-of-fold set (nup: one speaker, never held out) keep offset 0.

    python src/tune_bias.py tune <folds.csv> <oof_A.npz> [<oof_B.npz> ...] --out offsets.json
    python src/tune_bias.py apply <model_dir> offsets.json <out_model_dir>

Applying adds the offsets to classifier.bias in model.safetensors; the model stays a standard
HubertForSequenceClassification and argmax decisions change exactly as tuned.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from score import LABELS, WEIGHTS  # noqa: E402

W = np.array([WEIGHTS[c] for c in LABELS])


def weighted_f1(y, pred, present):
    k = len(LABELS)
    cm = np.bincount(y * k + pred, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
    return float((W[present] * f1[present]).sum() / W[present].sum())


def tune(args):
    folds = pd.read_csv(args.folds).set_index("id")
    ids, logits = [], []
    for f in args.oof:
        z = np.load(f, allow_pickle=True)
        ids.extend(z["ids"].tolist())
        logits.append(z["logits"])
    logits = np.concatenate(logits)
    y = np.array([LABELS.index(l) for l in folds.loc[ids, "label"]])
    groups = folds.loc[ids, "group"].values
    present = np.isin(np.arange(len(LABELS)), np.unique(y))

    rng = np.random.default_rng(0)
    ug = np.unique(groups)
    boots = []
    for _ in range(args.boot):
        pick = rng.choice(ug, len(ug), replace=True)
        boots.append(np.concatenate([np.where(groups == g)[0] for g in pick]))

    def objective(off):
        pred = (logits + off).argmax(1)
        return np.mean([weighted_f1(y[b], pred[b], present) for b in boots])

    off = np.zeros(len(LABELS))
    base = objective(off)
    best = base
    grid = np.linspace(-args.span, args.span, 25)
    for sweep in range(args.sweeps):
        improved = False
        for c in range(len(LABELS)):
            if not present[c]:
                continue
            cand = off.copy()
            scores = []
            for g in grid:
                cand[c] = off[c] + g * (0.5 ** sweep)
                scores.append(objective(cand))
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-5:
                off[c] += grid[j] * (0.5 ** sweep)
                best = scores[j]
                improved = True
        print(f"sweep {sweep}: bootstrap mean F1 {best:.4f}")
        if not improved:
            break
    off -= off[present].mean()  # only differences matter; keep the vector centred
    plain = weighted_f1(y, logits.argmax(1), present)
    tuned = weighted_f1(y, (logits + off).argmax(1), present)
    print(f"pooled OOF weighted F1: {plain:.4f} -> {tuned:.4f}  (bootstrap mean {base:.4f} -> {best:.4f})")
    res = {"offsets": dict(zip(LABELS, off.round(4).tolist())), "oof_f1_before": plain, "oof_f1_after": tuned,
           "n_clips": len(y), "absent_classes": [LABELS[i] for i in np.where(~present)[0]]}
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res["offsets"]))


def apply(args):
    import torch
    from safetensors.torch import load_file, save_file
    src, out = Path(args.model_dir), Path(args.out_dir)
    shutil.copytree(src, out, dirs_exist_ok=True)
    off = json.loads(Path(args.offsets).read_text())["offsets"]
    t = load_file(str(src / "model.safetensors"))
    b = t["classifier.bias"]
    t["classifier.bias"] = (b.float() + torch.tensor([off[c] for c in LABELS])).to(b.dtype)
    save_file(t, str(out / "model.safetensors"), metadata={"format": "pt"})
    print("classifier.bias", b.float().numpy().round(3).tolist(), "->", t["classifier.bias"].float().numpy().round(3).tolist())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tune")
    t.add_argument("folds")
    t.add_argument("oof", nargs="+")
    t.add_argument("--out", required=True)
    t.add_argument("--boot", type=int, default=30)
    t.add_argument("--span", type=float, default=3.0)
    t.add_argument("--sweeps", type=int, default=4)
    a = sub.add_parser("apply")
    a.add_argument("model_dir")
    a.add_argument("offsets")
    a.add_argument("out_dir")
    args = ap.parse_args()
    tune(args) if args.cmd == "tune" else apply(args)


if __name__ == "__main__":
    main()

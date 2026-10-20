"""Local leaderboard: predict a recipe's public-leaderboard score from its CV predictions.

Inputs are the out-of-fold files from a CV run (oof_<fold>_s<step>.npz with clean, noisy and
phone logits for speaker-held-out clips). The predicted score imitates the public test:
- class mix: the clip counts of the public test split, as reported for D1 (2,358 clips)
- conditions: clean / noisy / phone shares fitted to D1's leaderboard accuracies
- efficiency: the official formula with D1's measured memory and latency for HuBERT models
Bootstrap resampling of clips gives a 90% interval. nup has no held-out speaker (one speaker
in all training data), so its F1 cannot be estimated and is reported as missing; its false
positives on other classes are still counted.

    python src/predict_lb.py <folds.csv> oof_A_s4000.npz oof_B_s4000.npz ... [--layers 4] [--offsets f.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from score import LABELS, WEIGHTS, efficiency, final_score  # noqa: E402

TEST_COUNTS = {"yor": 317, "ibo": 217, "hau": 295, "ego": 155, "anw": 123, "efi": 45, "ibb": 23, "idu": 171,
               "igl": 55, "nup": 52, "tiv": 242, "urh": 129, "other": 534}
CONDITIONS = {"clean": 0.47, "noisy": 0.265, "phone": 0.265}  # fitted to D1: 0.641 / 0.210 / 0.436 -> 0.4716 overall
PARAMS = {4: (37.9, 0.076), 6: (51.8, 0.104), 8: (66.0, 0.132), 12: (94.4, 0.189)}  # params M, repo GB (fp16)
MEM_GB, LATENCY_S = 1.72, 0.004  # D1 on the organisers' hardware (HuBERT family)

W = np.array([WEIGHTS[c] for c in LABELS])


def f1_per_class(y, p):
    k = len(LABELS)
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    return np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0), cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folds")
    ap.add_argument("oof", nargs="+")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--offsets")
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--json")
    a = ap.parse_args()

    labels = pd.read_csv(a.folds).set_index("id")["label"]
    ids, L = [], {c: [] for c in CONDITIONS}
    for f in a.oof:
        z = np.load(f, allow_pickle=True)
        ids.extend(z["ids"].tolist())
        for c in CONDITIONS:
            L[c].append(z[c])
    L = {c: np.concatenate(v) for c, v in L.items()}
    y = np.array([LABELS.index(l) for l in labels.loc[ids]])
    off = np.zeros(len(LABELS))
    if a.offsets:
        o = json.loads(Path(a.offsets).read_text())["offsets"]
        off = np.array([o[c] for c in LABELS])
    pred = {c: (L[c] + off).argmax(1) for c in CONDITIONS}

    present = np.array([np.any(y == i) for i in range(len(LABELS))])
    rng = np.random.default_rng(0)
    conds = list(CONDITIONS)
    probs = np.array([CONDITIONS[c] for c in conds])
    by_class = {i: np.where(y == i)[0] for i in range(len(LABELS)) if present[i]}
    wf1s, f1s, accs = [], [], {c: [] for c in conds}
    for _ in range(a.boot):
        yy, pp, cc = [], [], []
        for i, idx in by_class.items():
            n = TEST_COUNTS[LABELS[i]]
            pick = rng.choice(idx, n, replace=True)
            cond = rng.choice(len(conds), n, p=probs)
            yy.append(np.full(n, i))
            cc.append(cond)
            pp.append(np.array([pred[conds[c]][k] for k, c in zip(pick, cond)]))
        yy, pp, cc = np.concatenate(yy), np.concatenate(pp), np.concatenate(cc)
        f1, _ = f1_per_class(yy, pp)
        wf1s.append((W[present] * f1[present]).sum() / W[present].sum())
        f1s.append(f1)
        for j, c in enumerate(conds):
            accs[c].append(float((yy[cc == j] == pp[cc == j]).mean()))
    wf1s, f1s = np.array(wf1s), np.array(f1s)
    params, repo = PARAMS[a.layers]
    eff, parts = efficiency(params, repo, MEM_GB, LATENCY_S)
    finals = np.array([final_score(w, eff) for w in wf1s])
    lo, hi = np.percentile(finals, [5, 95])

    print(f"OOF clips: {len(y)} (classes without held-out speakers: {[LABELS[i] for i in np.where(~present)[0]]})")
    print(f"predicted weighted F1: {wf1s.mean():.3f}  (90%: {np.percentile(wf1s, 5):.3f}-{np.percentile(wf1s, 95):.3f})")
    print(f"efficiency {eff:.3f} -> predicted FINAL: {finals.mean():.4f}  (90%: {lo:.4f}-{hi:.4f})")
    print("accuracy by condition: " + ", ".join(f"{c} {np.mean(v):.3f}" for c, v in accs.items()))
    print("per-class F1: " + ", ".join(f"{c} {'n/a' if not present[i] else f'{f1s[:, i].mean():.2f}'}" for i, c in enumerate(LABELS)))
    if a.json:
        Path(a.json).write_text(json.dumps({
            "pred_weighted_f1": float(wf1s.mean()), "pred_final": float(finals.mean()), "final_90": [float(lo), float(hi)],
            "efficiency": eff, "acc_by_condition": {c: float(np.mean(v)) for c, v in accs.items()},
            "f1": {c: (None if not present[i] else float(f1s[:, i].mean())) for i, c in enumerate(LABELS)}}, indent=1))


if __name__ == "__main__":
    main()

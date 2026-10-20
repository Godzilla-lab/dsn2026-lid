"""Honest check of decision offsets: tune on two CV folds, score on the third.

For each held-out fold, offsets are tuned (predict_lb.tune_offsets) on the other two folds'
out-of-fold logits, then the held-out fold is scored with offsets scaled by each shrink factor.
Offsets for classes listed in --zero are forced to 0 (e.g. nup, whose only held-out speakers are
VoiceAfrica, a different recording domain from the test set).

    python src/crossfit_offsets.py <folds.csv> <oof_dir> <step> [--zero nup] [--va]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from predict_lb import CONDITIONS, TEST_COUNTS, W, f1_per_class, tune_offsets  # noqa: E402
from score import LABELS  # noqa: E402


def load(paths, labels, keep=None):
    ids, L = [], {c: [] for c in CONDITIONS}
    for p in paths:
        z = np.load(p, allow_pickle=True)
        m = np.ones(len(z["ids"]), bool) if keep is None else np.array([labels.loc[i] in keep for i in z["ids"]])
        ids.extend(z["ids"][m].tolist())
        for c in CONDITIONS:
            L[c].append(z[c][m])
    return ids, {c: np.concatenate(v) for c, v in L.items()}


def score(L, y, off, boot=150, seed=0):
    present = np.array([np.any(y == i) for i in range(len(LABELS))])
    rng = np.random.default_rng(seed)
    conds, probs = list(CONDITIONS), np.array(list(CONDITIONS.values()))
    pred = {c: (L[c] + off).argmax(1) for c in conds}
    out = []
    for _ in range(boot):
        yy, pp = [], []
        for i in np.where(present)[0]:
            idx = np.where(y == i)[0]
            n = TEST_COUNTS[LABELS[i]]
            pick, cond = rng.choice(idx, n, replace=True), rng.choice(len(conds), n, p=probs)
            yy.append(np.full(n, i))
            pp.append(np.array([pred[conds[c]][k] for k, c in zip(pick, cond)]))
        f1, _ = f1_per_class(np.concatenate(yy), np.concatenate(pp))
        out.append((W[present] * f1[present]).sum() / W[present].sum())
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folds")
    ap.add_argument("oof_dir")
    ap.add_argument("step")
    ap.add_argument("--zero", default="nup")
    ap.add_argument("--va", action="store_true", help="add held-out VoiceAfrica nup clips to every pool")
    a = ap.parse_args()
    labels = pd.read_csv(a.folds).set_index("id")["label"]
    d, folds = Path(a.oof_dir), ["A", "B", "C"]
    zero = [LABELS.index(c) for c in a.zero.split(",") if c]
    shrinks = [0.0, 0.25, 0.5, 0.75, 1.0]
    res = {s: [] for s in shrinks}
    for f in folds:
        others = [x for x in folds if x != f]
        tr_paths = [d / f"oof_{x}_s{a.step}.npz" for x in others]
        te_paths = [d / f"oof_{f}_s{a.step}.npz"]
        ids_tr, Ltr = load(tr_paths, labels)
        ids_te, Lte = load(te_paths, labels)
        if a.va:
            for ids_, L_, xs in ((ids_tr, Ltr, others), (ids_te, Lte, [f])):
                vi, vL = load([d / f"va_{x}_s{a.step}.npz" for x in xs], labels, keep={"nup"})
                ids_.extend(vi)
                for c in CONDITIONS:
                    L_[c] = np.concatenate([L_[c], vL[c]])
        ytr = np.array([LABELS.index(l) for l in labels.loc[ids_tr]])
        yte = np.array([LABELS.index(l) for l in labels.loc[ids_te]])
        present = np.array([np.any(ytr == i) for i in range(len(LABELS))])
        off = tune_offsets(Ltr, ytr, present, draws=20, sweeps=3)
        off[zero] = 0.0
        for s in shrinks:
            res[s].append(score(Lte, yte, s * off))
        print(f"held-out fold {f}: " + ", ".join(f"x{s}: {res[s][-1]:.4f}" for s in shrinks), flush=True)
    print("MEAN over held-out folds: " + ", ".join(f"x{s}: {np.mean(v):.4f}" for s, v in res.items()))


if __name__ == "__main__":
    main()

"""Local copy of the DSN 2026 LID competition score.

Final score = language-weighted F1 x (0.5 + 0.5 x efficiency), as described on the
competition's Evaluation page. Run `python src/score.py --selftest` to check it against
the organisers' worked example and a published leaderboard row.

Usage:
    python src/score.py preds.csv              # columns: label, pred (ISO codes)
    python src/score.py preds.csv --params-m 38 --repo-gb 0.08 --mem-gb 1.2 --latency-s 0.01
"""
import argparse
import csv
import math
import sys

LABELS = ["yor", "ibo", "hau", "ego", "anw", "efi", "ibb", "idu", "igl", "nup", "tiv", "urh", "other"]
HIGH_RESOURCE = {"yor", "ibo", "hau"}
# 4.3% for high-resource classes and 8.7% for the rest, i.e. exactly 1/23 and 2/23.
WEIGHTS = {c: (1 if c in HIGH_RESOURCE else 2) / 23 for c in LABELS}

PARAM_LIMIT_M = 350.0
REPO_LIMIT_GB = 1.5
MEMORY_LIMIT_GB = 4.0
LATENCY_FLOOR_S = 0.01
LATENCY_LIMIT_S = 0.5
RANKING_F1_FLOOR = 0.60


def per_class_f1(y_true, y_pred):
    f1 = {}
    for c in LABELS:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        if tp + fp + fn == 0:
            f1[c] = None  # class absent from both truth and predictions
        else:
            f1[c] = 2 * tp / (2 * tp + fp + fn)
    return f1


def weighted_f1(y_true, y_pred):
    """Fixed-share F1. Classes absent from the evaluation set (e.g. nup in the official
    validation split) are dropped and the remaining shares renormalised, so local numbers
    stay comparable; the hidden test set contains every class."""
    f1 = per_class_f1(y_true, y_pred)
    present = [c for c in LABELS if f1[c] is not None]
    total = sum(WEIGHTS[c] for c in present)
    return sum(WEIGHTS[c] * f1[c] for c in present) / total, f1


def efficiency(params_m, repo_gb, mem_gb, latency_s):
    clip = lambda x: max(0.0, min(1.0, x))
    size = (clip(1 - params_m / PARAM_LIMIT_M) + clip(1 - repo_gb / REPO_LIMIT_GB)) / 2
    memory = clip(1 - mem_gb / MEMORY_LIMIT_GB)
    t = max(latency_s, LATENCY_FLOOR_S)
    speed = clip(1 - math.log(t / LATENCY_FLOOR_S) / math.log(LATENCY_LIMIT_S / LATENCY_FLOOR_S))
    return (size + memory + speed) / 3, {"size": size, "memory": memory, "speed": speed}


def final_score(wf1, eff):
    return wf1 * (0.5 + 0.5 * eff)


def within_limits(params_m, repo_gb, mem_gb, latency_s):
    return (params_m <= PARAM_LIMIT_M and repo_gb <= REPO_LIMIT_GB
            and mem_gb <= MEMORY_LIMIT_GB and latency_s <= LATENCY_LIMIT_S)


def selftest():
    # Organisers' example: F1 0.90, 100M params, 0.4 GB repo, 1 GB memory, 0.05 s -> about 0.76.
    eff, _ = efficiency(100, 0.4, 1.0, 0.05)
    assert abs(final_score(0.90, eff) - 0.76) < 0.005, final_score(0.90, eff)
    # Leaderboard row (2026-10-20): 20.2M params, 0.04 GB, 1.19 GB, 0.005 s, F1 0.549
    # -> size 0.958, memory 0.702, speed 1.0, efficiency 0.887, final 0.5179.
    eff, parts = efficiency(20.2, 0.04, 1.19, 0.005)
    assert abs(parts["size"] - 0.958) < 0.001 and abs(parts["memory"] - 0.702) < 0.001
    assert abs(eff - 0.887) < 0.001 and abs(final_score(0.549, eff) - 0.5179) < 0.0005
    # Weighted F1 on a perfect prediction is 1.
    assert abs(weighted_f1(LABELS, LABELS)[0] - 1.0) < 1e-12
    print("selftest passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preds", nargs="?", help="CSV with columns label,pred")
    ap.add_argument("--params-m", type=float, help="parameters in millions")
    ap.add_argument("--repo-gb", type=float, help="repository size in GB")
    ap.add_argument("--mem-gb", type=float, help="peak memory in GB (max of RAM and GPU)")
    ap.add_argument("--latency-s", type=float, help="average seconds per clip")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.preds:
        ap.error("give a predictions CSV or --selftest")

    with open(args.preds, newline="") as f:
        rows = list(csv.DictReader(f))
    y_true = [r["label"] for r in rows]
    y_pred = [r["pred"] for r in rows]
    unknown = sorted(set(y_true + y_pred) - set(LABELS))
    if unknown:
        sys.exit(f"unknown labels: {unknown}")

    wf1, f1 = weighted_f1(y_true, y_pred)
    print(f"clips: {len(rows)}")
    print(f"language-weighted F1: {wf1:.4f}" + ("" if wf1 >= RANKING_F1_FLOOR else "  (below the 0.60 ranking floor)"))
    for c in LABELS:
        n = sum(1 for t in y_true if t == c)
        print(f"  {c:<6} F1 {'  n/a' if f1[c] is None else f'{f1[c]:.3f}'}  (n={n}, share {WEIGHTS[c]:.1%})")

    eff_args = (args.params_m, args.repo_gb, args.mem_gb, args.latency_s)
    if all(a is not None for a in eff_args):
        eff, parts = efficiency(*eff_args)
        print(f"efficiency: {eff:.3f}  (size {parts['size']:.3f}, memory {parts['memory']:.3f}, speed {parts['speed']:.3f})")
        print(f"final score: {final_score(wf1, eff):.4f}")
        if not within_limits(*eff_args):
            print("WARNING: breaks a competition limit, so it would not be scored")


if __name__ == "__main__":
    main()

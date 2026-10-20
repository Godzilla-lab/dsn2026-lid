"""Build the 3 speaker-grouped CV folds (A, B, C) over pooled train + validation.

Groups:
- hypa_voices: speaker_id, pooled across labels (bilingual speakers stay in one fold)
- omnilingual_asr: speaker key string, pooled across anw/ego/idu
- afrispeech_ipa: no speaker ids. With --afri-clusters (id,cluster CSV from speaker
  embeddings) each cluster is a group; without it the organisers' split is used as a
  stand-in (validation -> A, train -> B/C by hash), which leaks speakers within
  afrispeech and is only good enough for the rung-0 smoke test.

The scarce-class speakers rotate as in the research report, so every fold holds out one
igl, one ibb, one efi and one hypa-tiv speaker group. nup has a single speaker
(SPK-147), who stays in training for every fold (fold "T").

Usage:
    python src/make_folds.py metadata.parquet out.csv [--afri-clusters clusters.csv]
"""
import argparse
import hashlib
from collections import defaultdict

import pandas as pd

ROTATION = {
    "A": ["SPK-059", "SPK-071", "SPK-011", "SPK-120", "SPK-041"],
    "B": ["SPK-086", "SPK-004", "SPK-040", "SPK-052", "SPK-061"],
    "C": ["SPK-031", "SPK-104", "SPK-140", "SPK-065", "SPK-129", "SPK-096"],
}
ALWAYS_TRAIN = ["SPK-147"]
FOLDS = ["A", "B", "C"]
SILENT_CLIPS = {"69cd0258ddc28532"}


def stable_fold(key, choices):
    h = int(hashlib.md5(key.encode()).hexdigest(), 16)
    return choices[h % len(choices)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metadata")
    ap.add_argument("out")
    ap.add_argument("--afri-clusters")
    args = ap.parse_args()

    m = pd.read_parquet(args.metadata, columns=["id", "split", "label", "source", "speaker_id", "speaker_key",
                                                "duration", "near_dup_of"])
    drop = m["id"].isin(SILENT_CLIPS) | (m["near_dup_of"].fillna("") != "")
    m = m[~drop].copy()

    m["group"] = m["speaker_key"]
    afri = m["source"] == "afrispeech_ipa"
    if args.afri_clusters:
        cl = pd.read_csv(args.afri_clusters, dtype=str).set_index("id")["cluster"]
        m.loc[afri, "group"] = "afrispeech_ipa:cl" + m.loc[afri, "id"].map(cl)
        assert m.loc[afri, "group"].notna().all(), "clusters missing for some afrispeech clips"
    else:
        m.loc[afri, "group"] = "afrispeech_ipa:" + m.loc[afri, "split"] + ":" + m.loc[afri, "id"]

    fold_of = {}
    for f, spks in ROTATION.items():
        for s in spks:
            fold_of[f"hypa_voices:{s}"] = f
    for s in ALWAYS_TRAIN:
        fold_of[f"hypa_voices:{s}"] = "T"

    if not args.afri_clusters:
        for g, split in m.loc[afri, ["group", "split"]].itertuples(index=False):
            fold_of[g] = "A" if split == "validation" else stable_fold(g, ["B", "C"])

    # Greedy stratified assignment of the remaining groups: each group goes to the fold
    # holding the fewest seconds of the group's main label so far.
    seconds = defaultdict(float)  # (label, fold) -> seconds
    for g, f in fold_of.items():
        for lab, dur in m.loc[m["group"] == g, ["label", "duration"]].itertuples(index=False):
            seconds[(lab, f)] += dur
    rest = m[~m["group"].isin(fold_of)]
    sizes = rest.groupby("group")["duration"].sum().sort_values(ascending=False)
    main_label = rest.groupby("group")["label"].agg(lambda s: s.value_counts().index[0])
    for g in sizes.index:
        lab = main_label[g]
        f = min(FOLDS, key=lambda x: (seconds[(lab, x)], x))
        fold_of[g] = f
        for l2, dur in rest.loc[rest["group"] == g, ["label", "duration"]].itertuples(index=False):
            seconds[(l2, f)] += dur

    m["fold"] = m["group"].map(fold_of)
    m[["id", "label", "source", "group", "fold"]].to_csv(args.out, index=False)

    table = m.pivot_table(index="label", columns="fold", values="duration", aggfunc="sum").fillna(0) / 3600
    speakers = m.pivot_table(index="label", columns="fold", values="group", aggfunc="nunique").fillna(0).astype(int)
    print(f"{len(m)} clips ({drop.sum()} dropped)\nhours per fold:\n{table.round(2)}\ngroups per fold:\n{speakers}")


if __name__ == "__main__":
    main()

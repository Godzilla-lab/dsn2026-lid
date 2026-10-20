"""Group the afrispeech clips (no speaker ids) into pseudo-speakers from WavLM x-vectors.

The distance threshold is calibrated on hypa clips, whose speaker ids are known: pick the
smallest threshold at which almost no same-speaker pair ends up split across clusters.
Over-merging is the safe direction here, because a too-coarse group only makes the CV
more conservative, while a split speaker leaks across folds.

Usage:
    python src/cluster_afrispeech.py spk_emb.npy spk_ids.csv metadata.parquet clusters.csv
"""
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

MAX_SPLIT_RATE = 0.03


def cluster(x, threshold):
    if len(x) == 1:
        return np.zeros(1, dtype=int)
    return AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                   distance_threshold=threshold).fit_predict(x)


def split_rate(labels_true, labels_pred):
    """Share of same-speaker clip pairs that land in different clusters."""
    split = total = 0
    for spk in np.unique(labels_true):
        c = labels_pred[labels_true == spk]
        n = len(c)
        if n < 2:
            continue
        _, counts = np.unique(c, return_counts=True)
        same = (counts * (counts - 1)).sum() / 2
        total += n * (n - 1) / 2
        split += n * (n - 1) / 2 - same
    return split / max(total, 1)


def main():
    emb_path, ids_path, meta_path, out_path = sys.argv[1:5]
    emb = np.load(emb_path).astype(np.float32)
    ids = pd.read_csv(ids_path).id.tolist()
    meta = pd.read_parquet(meta_path, columns=["id", "label", "source", "speaker_id"]).set_index("id").loc[ids]

    # Calibrate on hypa, one label at a time, like the afrispeech clustering below.
    hypa = meta.source.values == "hypa_voices"
    print("threshold  split_rate  clusters/speakers")
    chosen = None
    for t in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
        rates, ncl, nspk = [], 0, 0
        for lab in np.unique(meta.label.values[hypa]):
            sel = hypa & (meta.label.values == lab)
            pred = cluster(emb[sel], t)
            rates.append((split_rate(meta.speaker_id.values[sel], pred), sel.sum()))
            ncl += len(np.unique(pred))
            nspk += meta.speaker_id[sel].nunique()
        rate = sum(r * n for r, n in rates) / sum(n for _, n in rates)
        print(f"{t:9.2f}  {rate:10.3f}  {ncl}/{nspk}")
        if chosen is None and rate <= MAX_SPLIT_RATE:
            chosen = t
    chosen = chosen or 0.6
    print(f"chosen threshold {chosen}")

    afri = meta.source.values == "afrispeech_ipa"
    out = []
    for lab in np.unique(meta.label.values[afri]):
        sel = afri & (meta.label.values == lab)
        pred = cluster(emb[sel], chosen)
        sub_ids = np.array(ids)[sel]
        out.append(pd.DataFrame({"id": sub_ids, "cluster": [f"{lab}{p}" for p in pred]}))
        print(f"{lab}: {sel.sum()} clips -> {len(np.unique(pred))} pseudo-speakers")
    pd.concat(out).to_csv(out_path, index=False)


if __name__ == "__main__":
    main()

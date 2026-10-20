"""Do any speakers in a new dataset sound like speakers we already have? (WavLM x-vectors, CPU)

Each speaker gets a centroid from up to --per-speaker clips (first 6 s of each). The threshold is
calibrated on the reference speakers: two disjoint halves of one speaker's clips vs two different
speakers. A new speaker is flagged when its best match beats the 99.5th percentile of the
different-speaker similarities.

    python src/voice_overlap.py --new <dir> --ref <dir> [<dir> ...] --labels ibb,anw,efi --out overlap.csv
"""
import argparse
import glob
import io

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, WavLMForXVector

SR, SECONDS = 16000, 6


def load(dirs, labels, per_speaker, seed=0):
    rows = []
    for d in dirs:
        for shard in sorted(glob.glob(f"{d}/**/*.parquet", recursive=True)):
            t = pq.read_table(shard, columns=["id", "label", "speaker_id", "source"]).to_pandas()
            t = t[t.label.isin(labels)]
            if len(t):
                t["shard"] = shard
                rows.append(t)
    df = pd.concat(rows).sample(frac=1, random_state=seed)
    df = df[df.groupby("speaker_id").cumcount() < per_speaker]
    waves = {}
    for shard, g in df.groupby("shard"):
        t = pq.read_table(shard, columns=["id", "audio"])
        want = set(g.id)
        for i, cid in enumerate(t.column("id").to_pylist()):
            if cid in want:
                waves[cid] = sf.read(io.BytesIO(t.column("audio")[i].as_py()["bytes"]), dtype="float32")[0][: SECONDS * SR]
    df["wave"] = df.id.map(waves)
    return df.reset_index(drop=True)


def embed(df, fe, model, batch=16):
    out = np.zeros((len(df), 512), np.float32)
    order = np.argsort([len(w) for w in df.wave])
    for b in range(0, len(order), batch):
        idx = order[b:b + batch]
        x = fe([df.wave[i] for i in idx], sampling_rate=SR, return_tensors="pt", padding=True, return_attention_mask=True)
        with torch.no_grad():
            out[idx] = torch.nn.functional.normalize(model(**x).embeddings, dim=-1).numpy()
    return out


def centroid(e):
    c = e.mean(0)
    return c / np.linalg.norm(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", nargs="+", required=True)
    ap.add_argument("--ref", nargs="+", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--per-speaker", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    labels = a.labels.split(",")
    torch.set_num_threads(4)
    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval()

    ref = load(a.ref, labels, 2 * a.per_speaker)
    new = load(a.new, labels, a.per_speaker)
    print(f"ref: {ref.speaker_id.nunique()} speakers / {len(ref)} clips; new: {new.speaker_id.nunique()} speakers / {len(new)} clips", flush=True)
    er, en = embed(ref, fe, model), embed(new, fe, model)

    halves, ref_c = [], {}
    for s, g in ref.groupby("speaker_id"):
        idx = g.index.to_numpy()
        ref_c[s] = (centroid(er[idx]), g.label.iloc[0], g.source.iloc[0])
        if len(idx) >= 4:
            h = len(idx) // 2
            halves.append(float(centroid(er[idx[:h]]) @ centroid(er[idx[h:]])))
    names = list(ref_c)
    C = np.stack([ref_c[s][0] for s in names])
    diff = (C @ C.T)[np.triu_indices(len(names), 1)]
    thr = float(np.percentile(diff, 99.5))
    print(f"same speaker (split halves): median {np.median(halves):.3f}, min {np.min(halves):.3f}; "
          f"different speakers: median {np.median(diff):.3f}, 99.5% {thr:.3f}", flush=True)

    res = []
    for s, g in new.groupby("speaker_id"):
        c = centroid(en[g.index.to_numpy()])
        sims = C @ c
        j = int(np.argmax(sims))
        res.append(dict(new_speaker=s, label=g.label.iloc[0], clips=len(g), best_ref=names[j], best_ref_label=ref_c[names[j]][1],
                        best_ref_source=ref_c[names[j]][2], similarity=round(float(sims[j]), 3), flagged=bool(sims[j] > thr)))
    res = pd.DataFrame(res).sort_values("similarity", ascending=False)
    res.to_csv(a.out, index=False)
    print(res.head(15).to_string(index=False))
    print(f"flagged {int(res.flagged.sum())} of {len(res)} new speakers (threshold {thr:.3f})")


if __name__ == "__main__":
    main()

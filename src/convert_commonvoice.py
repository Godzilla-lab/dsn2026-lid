"""Convert one Common Voice locale (Mozilla Data Collective download) to the competition parquet schema.

Uses validated.tsv (plus other.tsv with --with-other), drops anything in invalidated.tsv, caps clips
per speaker so a few heavy contributors don't dominate, and keeps 2-30 s clips. Writes parquet
shards, a folds file (every clip fold X: always train, never evaluated) and a dataset-level manifest
row for the model card.

    python src/convert_commonvoice.py <locale_dir> <label> <out_dir> --release cv-corpus-27.0-2026-09-11
        [--per-speaker 120] [--with-other] [--url https://mozilladatacollective.com/datasets/...]
"""
import argparse
import hashlib
import io
import os
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

LABELS = ["yor", "ibo", "hau", "ego", "anw", "efi", "ibb", "idu", "igl", "nup", "tiv", "urh", "other"]
SR, MAX_SHARD, ROWS_PER_RG = 16000, 450e6, 64
SCHEMA = pa.schema([("id", pa.string()), ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
                    ("label", pa.string()), ("label_id", pa.int64()), ("speaker_id", pa.string()),
                    ("source", pa.string()), ("duration", pa.float64())])


def read_tsv(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=["client_id", "path"])
    return pd.read_csv(path, sep="\t", quoting=3, dtype=str, on_bad_lines="skip")


def decode(path):
    p = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", str(SR),
                        "-f", "s16le", "-"], capture_output=True)
    if p.returncode != 0 or not p.stdout:
        return None
    x = np.frombuffer(p.stdout, np.int16)
    # trim the silent lead-in and tail Common Voice recordings usually have (frames < -50 dBFS)
    f = x[: len(x) // 400 * 400].reshape(-1, 400).astype(np.float32) / 32768
    loud = np.where(10 * np.log10((f ** 2).mean(1) + 1e-10) > -50)[0]
    if len(loud) == 0:
        return None
    a, b = max(0, loud[0] - 4) * 400, min(len(x), (loud[-1] + 5) * 400)
    return x[a:b]


def flac(x):
    buf = io.BytesIO()
    sf.write(buf, x, SR, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("locale_dir")
    ap.add_argument("label", choices=LABELS)
    ap.add_argument("out_dir")
    ap.add_argument("--release", required=True)
    ap.add_argument("--per-speaker", type=int, default=120)
    ap.add_argument("--with-other", action="store_true")
    ap.add_argument("--url", default="https://mozilladatacollective.com")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    src = f"ext_cv_{a.label}"

    rows = read_tsv(os.path.join(a.locale_dir, "validated.tsv"))
    if a.with_other:
        rows = pd.concat([rows, read_tsv(os.path.join(a.locale_dir, "other.tsv"))])
    bad = set(read_tsv(os.path.join(a.locale_dir, "invalidated.tsv"))["path"])
    rows = rows[~rows.path.isin(bad)].drop_duplicates("path")
    print(f"{len(rows)} usable clips from {rows.client_id.nunique()} speakers")
    rng = random.Random(2026)
    pick = []
    for _, g in rows.groupby("client_id"):
        idx = list(g.index) if len(g) <= a.per_speaker else rng.sample(list(g.index), a.per_speaker)
        pick.append(g.loc[idx])
    rows = pd.concat(pick).reset_index(drop=True)

    clips = os.path.join(a.locale_dir, "clips")
    with ThreadPoolExecutor(4) as ex:
        waves = list(ex.map(lambda p: decode(os.path.join(clips, p)), rows.path))
    recs = []
    for r, x in zip(rows.itertuples(), waves):
        if x is None or not 2.0 <= len(x) / SR <= 30.0:
            continue
        recs.append({"id": f"cv_{a.label}_{hashlib.sha1(r.path.encode()).hexdigest()[:16]}",
                     "audio": {"bytes": flac(x), "path": r.path}, "label": a.label, "label_id": LABELS.index(a.label),
                     "speaker_id": "cv_" + hashlib.sha1(r.client_id.encode()).hexdigest()[:12], "source": src,
                     "duration": round(len(x) / SR, 3)})
    rng.shuffle(recs)
    t = pa.Table.from_pylist(recs, schema=SCHEMA)
    per_shard = max(ROWS_PER_RG, int(MAX_SHARD / max(1, t.nbytes / max(1, t.num_rows))) // ROWS_PER_RG * ROWS_PER_RG)
    for k, s in enumerate(range(0, t.num_rows, per_shard)):
        pq.write_table(t.slice(s, per_shard), os.path.join(a.out_dir, f"{src}-{k:05d}.parquet"),
                       row_group_size=ROWS_PER_RG, compression="snappy")

    d = t.select(["id", "label", "speaker_id", "source", "duration"]).to_pandas()
    pd.DataFrame({"id": d.id, "label": d.label, "source": d.source, "group": d.speaker_id, "fold": "X"}).to_csv(
        os.path.join(a.out_dir, f"folds_{src}.csv"), index=False)
    pd.DataFrame([{"dataset_id": f"Common Voice {a.release} ({a.label})", "config_split": "validated" + (" + other" if a.with_other else ""),
                   "revision": a.release.replace("cv-corpus-", ""), "license": "CC0-1.0", "label": a.label, "url": a.url,
                   "clips": len(d), "speakers": d.speaker_id.nunique(), "hours": round(d.duration.sum() / 3600, 2)}]).to_csv(
        os.path.join(a.out_dir, "manifest.csv"), index=False)
    spk = d.groupby("speaker_id").duration.agg(["size", "sum"]).sort_values("sum", ascending=False)
    spk.to_csv(os.path.join(a.out_dir, "speakers.csv"))
    print(f"kept {len(d)} clips, {d.speaker_id.nunique()} speakers, {d.duration.sum() / 3600:.2f} h; "
          f"top speakers (clips, s):\n{spk.head(8).round(0).to_string()}")


if __name__ == "__main__":
    main()

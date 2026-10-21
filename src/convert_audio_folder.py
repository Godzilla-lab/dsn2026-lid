"""Convert any folder of audio files (plus optional metadata table) to the competition parquet schema.

Speaker id comes from, in order: a metadata TSV/CSV column named like speaker/client_id/reader
(matched to the file name), else the first directory level under the root that contains more than one
file, else the whole set is one speaker. Clips longer than --max-s are cut into pieces; shorter than 2 s
are dropped. Silent lead-in and tail are trimmed. Writes parquet shards, folds_<source>.csv (fold X) and a
manifest row.

    python src/convert_audio_folder.py <root> <label> <source> <out_dir> --name "..." --license ... --url ...
        [--per-speaker 120] [--max-s 20]
"""
import argparse
import csv
import glob
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
AUDIO_EXT = (".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".webm", ".aac")
SCHEMA = pa.schema([("id", pa.string()), ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
                    ("label", pa.string()), ("label_id", pa.int64()), ("speaker_id", pa.string()),
                    ("source", pa.string()), ("duration", pa.float64())])


def decode(path):
    p = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", str(SR),
                        "-f", "s16le", "-"], capture_output=True)
    if p.returncode != 0 or len(p.stdout) < 2 * SR:
        return None
    x = np.frombuffer(p.stdout, np.int16)
    f = x[: len(x) // 400 * 400].reshape(-1, 400).astype(np.float32) / 32768
    loud = np.where(10 * np.log10((f ** 2).mean(1) + 1e-10) > -50)[0]
    if len(loud) == 0:
        return None
    return x[max(0, loud[0] - 4) * 400: min(len(x), (loud[-1] + 5) * 400)]


def flac(x):
    buf = io.BytesIO()
    sf.write(buf, x, SR, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def speaker_map(root, files):
    """file path -> speaker id, from a metadata table if one names speakers, else from directories."""
    for t in glob.glob(f"{root}/**/*.tsv", recursive=True) + glob.glob(f"{root}/**/*.csv", recursive=True):
        try:
            df = pd.read_csv(t, sep="\t" if t.endswith(".tsv") else ",", dtype=str, quoting=3, on_bad_lines="skip")
        except Exception:
            continue
        spk_col = next((c for c in df.columns if c.lower() in ("client_id", "speaker_id", "speaker", "reader", "voice", "spk")), None)
        file_col = next((c for c in df.columns if c.lower() in ("path", "file", "filename", "file_name", "audio", "wav", "clip")), None)
        if spk_col and file_col:
            m = {os.path.basename(str(p)): s for p, s in zip(df[file_col], df[spk_col])}
            got = {f: m.get(os.path.basename(f)) for f in files}
            if sum(v is not None for v in got.values()) > len(files) / 2:
                print(f"speakers from {os.path.relpath(t, root)} column '{spk_col}': {len(set(v for v in got.values() if v))} speakers")
                return {f: (v or "unknown") for f, v in got.items()}
    rel = [os.path.relpath(f, root).split(os.sep) for f in files]
    for level in range(max(len(r) for r in rel) - 1):
        keys = {tuple(r[: level + 1]) for r in rel if len(r) > level + 1}
        if len(keys) > 1:
            print(f"speakers from directory level {level}: {len(keys)} groups")
            return {f: "/".join(r[: level + 1]) for f, r in zip(files, rel)}
    print("no speaker information: treating the whole set as one speaker")
    return {f: "all" for f in files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("label", choices=LABELS)
    ap.add_argument("source")
    ap.add_argument("out_dir")
    ap.add_argument("--name", required=True)
    ap.add_argument("--license", required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--per-speaker", type=int, default=120)
    ap.add_argument("--max-s", type=float, default=20.0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    files = sorted(f for f in glob.glob(f"{a.root}/**/*", recursive=True) if f.lower().endswith(AUDIO_EXT))
    print(f"{len(files)} audio files")
    spk = speaker_map(a.root, files)
    rng = random.Random(2026)
    by = {}
    for f in files:
        by.setdefault(spk[f], []).append(f)
    pick = []
    for s, fs in by.items():
        pick += fs if len(fs) <= a.per_speaker else rng.sample(fs, a.per_speaker)
    with ThreadPoolExecutor(4) as ex:
        waves = list(ex.map(decode, pick))
    recs = []
    for f, x in zip(pick, waves):
        if x is None:
            continue
        pieces = [x[i: i + int(a.max_s * SR)] for i in range(0, len(x), int(a.max_s * SR))]
        for j, piece in enumerate(pieces):
            if len(piece) < 2 * SR:
                continue
            h = hashlib.sha1(f"{os.path.relpath(f, a.root)}#{j}".encode()).hexdigest()[:16]
            recs.append({"id": f"{a.source}_{h}", "audio": {"bytes": flac(piece), "path": os.path.basename(f)},
                         "label": a.label, "label_id": LABELS.index(a.label),
                         "speaker_id": f"{a.source}_" + hashlib.sha1(str(spk[f]).encode()).hexdigest()[:10],
                         "source": a.source, "duration": round(len(piece) / SR, 3)})
    rng.shuffle(recs)
    t = pa.Table.from_pylist(recs, schema=SCHEMA)
    per_shard = max(ROWS_PER_RG, int(MAX_SHARD / max(1, t.nbytes / max(1, t.num_rows))) // ROWS_PER_RG * ROWS_PER_RG)
    for k, s in enumerate(range(0, t.num_rows, per_shard)):
        pq.write_table(t.slice(s, per_shard), os.path.join(a.out_dir, f"{a.source}-{k:05d}.parquet"),
                       row_group_size=ROWS_PER_RG, compression="snappy")
    d = t.select(["id", "label", "speaker_id", "source", "duration"]).to_pandas()
    pd.DataFrame({"id": d.id, "label": d.label, "source": d.source, "group": d.speaker_id, "fold": "X"}).to_csv(
        os.path.join(a.out_dir, f"folds_{a.source}.csv"), index=False)
    pd.DataFrame([{"dataset_id": a.name, "config_split": "all", "revision": "mdc", "license": a.license, "label": a.label,
                   "url": a.url, "clips": len(d), "speakers": d.speaker_id.nunique(), "hours": round(d.duration.sum() / 3600, 2)}]).to_csv(
        os.path.join(a.out_dir, "manifest.csv"), index=False)
    print(f"kept {len(d)} clips, {d.speaker_id.nunique()} speakers, {d.duration.sum() / 3600:.2f} h")


if __name__ == "__main__":
    main()

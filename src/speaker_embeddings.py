"""Speaker embeddings (WavLM x-vectors) for afrispeech and hypa clips, on CPU.

The afrispeech embeddings give pseudo-speaker groups for the folds; the hypa ones (known
speaker ids) calibrate the clustering threshold. Uses the first 5 s of each clip.

    python src/speaker_embeddings.py <dir with parquet shards> <out_prefix>
"""
import glob
import io
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, WavLMForXVector

SR, SECONDS, BATCH = 16000, 5, 16
data_dir, out = sys.argv[1], sys.argv[2]
torch.set_num_threads(4)
fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval()

ids, embs, t0 = [], [], time.time()
for shard in sorted(glob.glob(f"{data_dir}/*.parquet")):
    t = pq.read_table(shard, columns=["id", "source", "audio"])
    keep = [i for i, s in enumerate(t.column("source").to_pylist()) if s in ("afrispeech_ipa", "hypa_voices")]
    cids = [t.column("id")[i].as_py() for i in keep]
    waves = [sf.read(io.BytesIO(t.column("audio")[i].as_py()["bytes"]), dtype="float32")[0][: SECONDS * SR] for i in keep]
    del t
    order = np.argsort([len(w) for w in waves])
    for b in range(0, len(order), BATCH):
        idx = order[b:b + BATCH]
        x = fe([waves[i] for i in idx], sampling_rate=SR, return_tensors="pt", padding=True, return_attention_mask=True)
        with torch.no_grad():
            e = model(**x).embeddings
        embs.append(torch.nn.functional.normalize(e, dim=-1).numpy().astype(np.float16))
        ids.extend(cids[i] for i in idx)
    print(f"{shard.split('/')[-1]}: {len(ids)} clips, {time.time() - t0:.0f}s", flush=True)

np.save(f"{out}_emb.npy", np.concatenate(embs))
pd.DataFrame({"id": ids}).to_csv(f"{out}_ids.csv", index=False)
print("done", len(ids))

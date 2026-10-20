"""Speaker embeddings (WavLM x-vectors) on a Kaggle GPU, for the afrispeech pseudo-speaker folds.

Embeds the first 5 s of every afrispeech and hypa clip (hypa has real speaker ids, used to
calibrate the clustering). Outputs /kaggle/working/spk_emb.npy and spk_ids.csv.
"""
import glob
import io
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.6", "soundfile"], check=True)

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, WavLMForXVector

SR, SECONDS, BATCH = 16000, 5, 64
OUT = "/kaggle/working"
t0 = time.time()
shards = sorted(glob.glob("/kaggle/input/**/train-*.parquet", recursive=True) +
                glob.glob("/kaggle/input/**/validation-*.parquet", recursive=True))
os.makedirs("/tmp/shards", exist_ok=True)
with ThreadPoolExecutor(6) as ex:
    local = list(ex.map(lambda p: shutil.copyfile(p, f"/tmp/shards/{os.path.basename(p)}"), shards))


def decode(path):
    t = pq.read_table(path, columns=["id", "source", "audio"])
    ids, waves = [], []
    for cid, src, a in zip(t.column("id").to_pylist(), t.column("source").to_pylist(), t.column("audio").to_pylist()):
        if src in ("afrispeech_ipa", "hypa_voices"):
            ids.append(cid)
            waves.append(sf.read(io.BytesIO(a["bytes"]), dtype="float32")[0][: SECONDS * SR])
    return ids, waves


with Pool(4) as pool:
    parts = pool.map(decode, local)
ids = [i for p in parts for i in p[0]]
waves = [w for p in parts for w in p[1]]
del parts
print(f"decoded {len(ids)} clips in {time.time() - t0:.0f}s", flush=True)

fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to("cuda").half().eval()
order = np.argsort([len(w) for w in waves])
emb = np.zeros((len(ids), 512), dtype=np.float16)
t1 = time.time()
for b in range(0, len(order), BATCH):
    idx = order[b:b + BATCH]
    x = fe([waves[i] for i in idx], sampling_rate=SR, return_tensors="pt", padding=True, return_attention_mask=True)
    with torch.no_grad():
        e = model(input_values=x.input_values.to("cuda").half(), attention_mask=x.attention_mask.to("cuda")).embeddings
    emb[idx] = torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy()
print(f"embedded in {time.time() - t1:.0f}s", flush=True)
np.save(f"{OUT}/spk_emb.npy", emb)
pd.DataFrame({"id": ids}).to_csv(f"{OUT}/spk_ids.csv", index=False)
print("done", flush=True)

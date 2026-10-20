"""Rung 0 smoke test, run as a Kaggle GPU script kernel.

1. Train AfriHuBERT cut to 4 layers on ~200 clips per class (folds B, C, T), evaluate on
   fold A with the competition metric.
2. Save fp16 safetensors + preprocessor, run the organisers' checker.
3. Reload offline the way the organisers do (AutoModelForAudioClassification), check the
   predictions match, and measure latency and peak memory.
4. Extract speaker embeddings (WavLM x-vectors) for every clip, used to build pseudo-speaker
   groups for the afrispeech clips, which have no speaker ids.

Outputs go to /kaggle/working: model/, metrics.json, eval_preds.csv, spk_emb.npy, spk_ids.csv.
"""
import glob
import io
import json
import os
import random
import resource
import subprocess
import sys
import time

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.6", "soundfile"], check=True)

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
import torch

LABELS = ["yor", "ibo", "hau", "ego", "anw", "efi", "ibb", "idu", "igl", "nup", "tiv", "urh", "other"]
L2I = {l: i for i, l in enumerate(LABELS)}
BASE = "ajesujoba/AfriHuBERT"
N_LAYERS = 4
TRAIN_PER_CLASS, EVAL_PER_CLASS = 200, 50
CROP_S, SR = 4.0, 16000
SEED = 0

INPUT = os.path.dirname(glob.glob("/kaggle/input/**/folds_v0.csv", recursive=True)[0])
OUT = "/kaggle/working"
MODEL_DIR = f"{OUT}/model"
sys.path.insert(0, INPUT)
from score import efficiency, final_score, weighted_f1  # noqa: E402

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
dev = "cuda"
metrics = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
print(metrics, flush=True)


def log(**kw):
    metrics.update(kw)
    print(json.dumps(kw), flush=True)
    with open(f"{OUT}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=1)


def load_audio(ids):
    """Decode the FLAC bytes of the given clip ids from the parquet shards."""
    want, got = set(ids), {}
    for shard in sorted(glob.glob(f"{INPUT}/*.parquet")):
        t = pq.read_table(shard, columns=["id", "audio"], filters=[("id", "in", list(want))])
        for cid, a in zip(t.column("id").to_pylist(), t.column("audio").to_pylist()):
            x, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
            assert sr == SR and x.ndim == 1
            got[cid] = x
    return [got[i] for i in ids]


# ---------- 1. data subset ----------
folds = pd.read_csv(f"{INPUT}/folds_v0.csv")
tr_pool, ev_pool = folds[folds.fold.isin(["B", "C", "T"])], folds[folds.fold == "A"]
tr = tr_pool.groupby("label", group_keys=False).apply(lambda d: d.sample(min(len(d), TRAIN_PER_CLASS), random_state=SEED))
ev = ev_pool.groupby("label", group_keys=False).apply(lambda d: d.sample(min(len(d), EVAL_PER_CLASS), random_state=SEED))
t0 = time.time()
tr_x, ev_x = load_audio(tr.id.tolist()), load_audio(ev.id.tolist())
tr_y = torch.tensor([L2I[l] for l in tr.label])
log(n_train=len(tr), n_eval=len(ev), train_per_class=tr.label.value_counts().to_dict(), load_s=round(time.time() - t0, 1))

# ---------- 2. model ----------
from transformers import AutoFeatureExtractor, HubertForSequenceClassification  # noqa: E402

fe = AutoFeatureExtractor.from_pretrained(BASE)
model = HubertForSequenceClassification.from_pretrained(
    BASE, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=L2I,
    num_hidden_layers=N_LAYERS, use_weighted_layer_sum=True).to(dev)
model.freeze_feature_encoder()
head = [p for n, p in model.named_parameters() if n.startswith(("projector", "classifier", "layer_weights"))]
body = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith(("projector", "classifier", "layer_weights"))]
opt = torch.optim.AdamW([{"params": body, "lr": 3e-5}, {"params": head, "lr": 5e-4}], weight_decay=0.01)
scaler = torch.cuda.amp.GradScaler()


def batch_inputs(waves):
    return fe(waves, sampling_rate=SR, return_tensors="pt", padding=True).input_values.to(dev)


def crop_batch(idx):
    # Crop every clip in the batch to the same length, so no padding is needed.
    n = min(int(CROP_S * SR), min(len(tr_x[i]) for i in idx))
    out = []
    for i in idx:
        s = random.randint(0, len(tr_x[i]) - n)
        out.append(tr_x[i][s:s + n])
    return out


EPOCHS, BS = 2, 16
steps = EPOCHS * (len(tr_x) // BS)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[3e-5, 5e-4], total_steps=steps, pct_start=0.1)
model.train()
t0, step = time.time(), 0
for ep in range(EPOCHS):
    order = np.random.permutation(len(tr_x))
    for b in range(len(tr_x) // BS):
        idx = order[b * BS:(b + 1) * BS]
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(batch_inputs(crop_batch(idx)), labels=tr_y[idx].to(dev))
        opt.zero_grad()
        scaler.scale(out.loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        if step % 25 == 0:
            print(f"step {step}/{steps} loss {out.loss.item():.3f}", flush=True)
log(train_s=round(time.time() - t0, 1), steps=steps, final_loss=round(out.loss.item(), 4))


@torch.no_grad()
def predict(m, waves):
    m.eval()
    preds = []
    for w in waves:
        logits = m(batch_inputs([w]).to(next(m.parameters()).dtype)).logits
        preds.append(int(logits.argmax(-1)))
    return preds


ev_pred = predict(model, ev_x)
wf1, f1 = weighted_f1(ev.label.tolist(), [LABELS[p] for p in ev_pred])
log(eval_weighted_f1=round(wf1, 4), eval_f1={k: (None if v is None else round(v, 3)) for k, v in f1.items()})
pd.DataFrame({"id": ev.id, "label": ev.label, "pred": [LABELS[p] for p in ev_pred]}).to_csv(f"{OUT}/eval_preds.csv", index=False)

# ---------- 3. save fp16, check, offline reload ----------
model.half().save_pretrained(MODEL_DIR, safe_serialization=True)
fe.save_pretrained(MODEL_DIR)
chk = subprocess.run([sys.executable, f"{INPUT}/check_model_folder.py", MODEL_DIR], capture_output=True, text=True)
log(model_files=sorted(os.listdir(MODEL_DIR)), checker=chk.stdout.strip().splitlines(), checker_ok=chk.returncode == 0)
np.save(f"{OUT}/ev_lens.npy", np.array([len(x) for x in ev_x]))
np.save(f"{OUT}/ev_pred_inproc.npy", np.array(ev_pred))
del model
torch.cuda.empty_cache()

# The organisers' side, in a fresh process: offline load, default dtype, per-clip inference,
# warm-up then the median of 3 timed passes; peak RAM and GPU memory.
bench = r'''
import json, os, resource, statistics, sys, time, glob, io
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
import numpy as np, pandas as pd, pyarrow.parquet as pq, soundfile as sf, torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
model_dir, inp, out, dtype = sys.argv[1:5]
ids = pd.read_csv(f"{out}/eval_preds.csv").id.tolist()
got = {}
for shard in sorted(glob.glob(f"{inp}/*.parquet")):
    t = pq.read_table(shard, columns=["id", "audio"], filters=[("id", "in", ids)])
    for cid, a in zip(t.column("id").to_pylist(), t.column("audio").to_pylist()):
        got[cid] = sf.read(io.BytesIO(a["bytes"]), dtype="float32")[0]
waves = [got[i] for i in ids]
kw = {"torch_dtype": "auto"} if dtype == "auto" else {}
t0 = time.time()
fe = AutoFeatureExtractor.from_pretrained(model_dir)
m = AutoModelForAudioClassification.from_pretrained(model_dir, **kw).to("cuda").eval()
load_s = time.time() - t0
torch.cuda.reset_peak_memory_stats()
def run():
    preds, t = [], time.time()
    with torch.no_grad():
        for w in waves:
            x = fe(w, sampling_rate=16000, return_tensors="pt").input_values.to("cuda", m.dtype)
            preds.append(int(m(x).logits.argmax(-1)))
    torch.cuda.synchronize()
    return preds, (time.time() - t) / len(waves)
run()  # warm-up
passes = [run() for _ in range(3)]
preds = passes[0][0]
inproc = np.load(f"{out}/ev_pred_inproc.npy").tolist()
res = {"dtype_arg": dtype, "loaded_dtype": str(m.dtype), "load_s": round(load_s, 2),
       "latency_s_median": round(statistics.median(p[1] for p in passes), 5),
       "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
       "ram_peak_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 3),
       "agree_with_inprocess": round(float(np.mean(np.array(preds) == np.array(inproc))), 4),
       "params_m": round(sum(p.numel() for p in m.parameters()) / 1e6, 2)}
print("BENCH " + json.dumps(res))
'''
with open(f"{OUT}/bench.py", "w") as f:
    f.write(bench)
repo_gb = sum(os.path.getsize(f"{MODEL_DIR}/{n}") for n in os.listdir(MODEL_DIR)) / 1e9
for dtype in ["default", "auto"]:
    r = subprocess.run([sys.executable, f"{OUT}/bench.py", MODEL_DIR, INPUT, OUT, dtype], capture_output=True, text=True)
    line = [l for l in r.stdout.splitlines() if l.startswith("BENCH ")]
    if not line:
        log(**{f"bench_{dtype}_error": r.stderr[-2000:]})
        continue
    b = json.loads(line[0][6:])
    eff, parts = efficiency(b["params_m"], repo_gb, max(b["gpu_peak_gb"], b["ram_peak_gb"]), b["latency_s_median"])
    b.update(repo_gb=round(repo_gb, 4), efficiency=round(eff, 3), eff_parts={k: round(v, 3) for k, v in parts.items()},
             est_final_at_eval_f1=round(final_score(wf1, eff), 4))
    log(**{f"bench_{dtype}": b})

# ---------- 4. speaker embeddings for pseudo-speaker groups ----------
try:
    from transformers import AutoFeatureExtractor as AFE, WavLMForXVector
    sfe = AFE.from_pretrained("microsoft/wavlm-base-plus-sv")
    sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(dev).half().eval()
    ids_all, embs = [], []
    t0 = time.time()
    for shard in sorted(glob.glob(f"{INPUT}/*.parquet")):
        t = pq.read_table(shard, columns=["id", "audio"])
        cids, auds = t.column("id").to_pylist(), t.column("audio").to_pylist()
        del t
        for b in range(0, len(cids), 32):
            waves = [sf.read(io.BytesIO(a["bytes"]), dtype="float32")[0][: 8 * SR] for a in auds[b:b + 32]]
            x = sfe(waves, sampling_rate=SR, return_tensors="pt", padding=True, return_attention_mask=True)
            with torch.no_grad():
                e = sv(input_values=x.input_values.to(dev).half(), attention_mask=x.attention_mask.to(dev)).embeddings
            embs.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy())
            ids_all.extend(cids[b:b + 32])
        print(f"embedded {len(ids_all)} clips, {time.time() - t0:.0f}s", flush=True)
    np.save(f"{OUT}/spk_emb.npy", np.concatenate(embs).astype(np.float16))
    pd.DataFrame({"id": ids_all}).to_csv(f"{OUT}/spk_ids.csv", index=False)
    log(spk_emb_clips=len(ids_all), spk_emb_s=round(time.time() - t0, 1))
except Exception as e:  # the smoke test results above must survive a failure here
    log(spk_emb_error=repr(e)[:2000])

log(done=True)

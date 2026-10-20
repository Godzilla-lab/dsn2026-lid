"""Training script for the ladder rungs, run as a Kaggle GPU script kernel.

The run is set by CONFIG below; scripts/launch fills it in per run:
- mode "cv":   train one model per fold in CONFIG["folds"], save out-of-fold logits on
               full clips for the held-out fold (used for CV scores and bias tuning).
- mode "full": train on every clip, save the fp16 model, run the organisers' checker,
               then reload offline and measure latency and memory.

Outputs in /kaggle/working: metrics.json, oof_<fold>.npz (cv), model/ and bench (full).
"""
CONFIG = {"mode": "cv", "folds": ["A", "B", "C"], "folds_file": "folds_v1.csv", "n_layers": 4,
          "steps": 4000, "batch": 32, "crop_min_s": 3.0, "crop_max_s": 6.0, "lr_body": 5e-5, "lr_head": 5e-4,
          "loss": "ce", "tau": 0.0, "group_alpha": 0.5, "aug": False, "seed": 0}

import glob
import io
import json
import os
import random
import subprocess
import sys
import time
from multiprocessing import Pool

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.6", "soundfile"], check=True)

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
import torch

LABELS = ["yor", "ibo", "hau", "ego", "anw", "efi", "ibb", "idu", "igl", "nup", "tiv", "urh", "other"]
L2I = {l: i for i, l in enumerate(LABELS)}
BASE, SR = "ajesujoba/AfriHuBERT", 16000
INPUT = os.path.dirname(glob.glob(f"/kaggle/input/**/{CONFIG['folds_file']}", recursive=True)[0])
OUT = "/kaggle/working"
sys.path.insert(0, INPUT)
from score import efficiency, final_score, weighted_f1  # noqa: E402

dev = "cuda"
metrics = {"config": CONFIG, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}


def log(**kw):
    metrics.update(kw)
    print(json.dumps(kw), flush=True)
    with open(f"{OUT}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=1)


# ---------- audio, decoded once into one int16 array ----------
def decode_shard(path):
    t = pq.read_table(path, columns=["id", "audio"])
    ids, waves = t.column("id").to_pylist(), []
    for a in t.column("audio").to_pylist():
        x, sr = sf.read(io.BytesIO(a["bytes"]), dtype="int16")
        assert sr == SR and x.ndim == 1
        waves.append(x)
    return ids, waves


t0 = time.time()
folds = pd.read_csv(f"{INPUT}/{CONFIG['folds_file']}")
with Pool(4) as pool:
    parts = pool.map(decode_shard, sorted(glob.glob(f"{INPUT}/*.parquet")))
id_to_wave = {i: w for ids, ws in parts for i, w in zip(ids, ws)}
del parts
folds = folds[folds.id.isin(id_to_wave)].reset_index(drop=True)
lens = np.array([len(id_to_wave[i]) for i in folds.id])
offsets = np.concatenate([[0], np.cumsum(lens)])
AUDIO = np.empty(offsets[-1], dtype=np.int16)
for k, i in enumerate(folds.id):
    AUDIO[offsets[k]:offsets[k + 1]] = id_to_wave[i]
del id_to_wave
Y = np.array([L2I[l] for l in folds.label])
log(n_clips=len(folds), audio_hours=round(offsets[-1] / SR / 3600, 2), decode_s=round(time.time() - t0, 1))


def wave(k, start=0, n=None):
    a, b = offsets[k] + start, offsets[k + 1] if n is None else offsets[k] + start + n
    return AUDIO[a:b].astype(np.float32) / 32768.0


# ---------- sampling: class uniform, groups ~ size^alpha, clips uniform within group ----------
def sampling_weights(idx):
    d = folds.iloc[idx]
    w = np.zeros(len(idx))
    for lab, dl in d.groupby("label"):
        gsize = dl.groupby("group").size()
        gw = gsize ** CONFIG["group_alpha"]
        gw = gw / gw.sum()
        w[d.index.get_indexer(dl.index)] = (1 / d.label.nunique()) * gw[dl.group].values / gsize[dl.group].values
    return w / w.sum()


# ---------- augmentation (rung 3) ----------
def augment(x):
    return x  # filled in at rung 3


# ---------- model ----------
from transformers import AutoFeatureExtractor, HubertForSequenceClassification  # noqa: E402

fe = AutoFeatureExtractor.from_pretrained(BASE)


def inputs(waves):
    return fe(waves, sampling_rate=SR, return_tensors="pt", padding=True).input_values.to(dev)


def train(train_idx, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = HubertForSequenceClassification.from_pretrained(
        BASE, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=L2I,
        num_hidden_layers=CONFIG["n_layers"], use_weighted_layer_sum=True).to(dev)
    model.freeze_feature_encoder()
    is_head = lambda n: n.startswith(("projector", "classifier", "layer_weights"))
    params = [{"params": [p for n, p in model.named_parameters() if p.requires_grad and not is_head(n)], "lr": CONFIG["lr_body"]},
              {"params": [p for n, p in model.named_parameters() if is_head(n)], "lr": CONFIG["lr_head"]}]
    opt = torch.optim.AdamW(params, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[CONFIG["lr_body"], CONFIG["lr_head"]],
                                                total_steps=CONFIG["steps"], pct_start=0.08)
    scaler = torch.cuda.amp.GradScaler()
    w = sampling_weights(train_idx)
    draws = np.random.choice(train_idx, size=CONFIG["steps"] * CONFIG["batch"], p=w)
    # Logit adjustment: add tau * log(prior of the sampled batch distribution) during training.
    prior = np.bincount(Y[draws], minlength=len(LABELS)) / len(draws)
    log_prior = torch.tensor(np.log(np.maximum(prior, 1e-8)), dtype=torch.float32, device=dev)
    model.train()
    t0 = time.time()
    for step in range(CONFIG["steps"]):
        b = draws[step * CONFIG["batch"]:(step + 1) * CONFIG["batch"]]
        n = int(min(random.uniform(CONFIG["crop_min_s"], CONFIG["crop_max_s"]) * SR, lens[b].min()))
        batch = [augment(wave(k, random.randint(0, lens[k] - n), n)) if CONFIG["aug"] else wave(k, random.randint(0, lens[k] - n), n) for k in b]
        y = torch.tensor(Y[b], device=dev)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(inputs(batch)).logits.float()
        if CONFIG["loss"] == "logit_adjusted":
            logits = logits + CONFIG["tau"] * log_prior
        loss = torch.nn.functional.cross_entropy(logits, y)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        if (step + 1) % 250 == 0:
            print(f"step {step + 1}/{CONFIG['steps']} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
    return model, round(time.time() - t0, 1)


@torch.no_grad()
def full_clip_logits(model, idx):
    model.eval()
    out = np.zeros((len(idx), len(LABELS)), dtype=np.float32)
    for j, k in enumerate(idx):
        with torch.autocast("cuda", dtype=torch.float16):
            out[j] = model(inputs([wave(k)])).logits.float().cpu().numpy()[0]
    return out


if CONFIG["mode"] == "cv":
    for f in CONFIG["folds"]:
        tr = np.where(folds.fold.values != f)[0]
        ho = np.where(folds.fold.values == f)[0]
        model, train_s = train(tr, CONFIG["seed"])
        lg = full_clip_logits(model, ho)
        np.savez_compressed(f"{OUT}/oof_{f}.npz", ids=folds.id.values[ho], logits=lg)
        wf1, f1 = weighted_f1(folds.label.values[ho].tolist(), [LABELS[i] for i in lg.argmax(1)])
        log(**{f"fold_{f}": {"train_s": train_s, "n_holdout": len(ho), "weighted_f1": round(wf1, 4),
                             "f1": {k: (None if v is None else round(v, 3)) for k, v in f1.items()}}})
        del model
        torch.cuda.empty_cache()
else:
    model, train_s = train(np.arange(len(folds)), CONFIG["seed"])
    log(full_train_s=train_s)
    model_dir = f"{OUT}/model"
    model.half().save_pretrained(model_dir, safe_serialization=True)
    fe.save_pretrained(model_dir)
    chk = subprocess.run([sys.executable, f"{INPUT}/check_model_folder.py", model_dir], capture_output=True, text=True)
    log(model_files=sorted(os.listdir(model_dir)), checker=chk.stdout.strip().splitlines(), checker_ok=chk.returncode == 0)
    del model
    torch.cuda.empty_cache()
    # Organisers' side, in a fresh process: offline load with default dtype, per-clip
    # inference on 300 clips of mixed length, warm-up, median of 3 passes, peak memory.
    rng = np.random.default_rng(0)
    bench_idx = rng.choice(len(folds), 300, replace=False)
    np.save(f"{OUT}/bench_audio.npy", np.concatenate([wave(k) for k in bench_idx]).astype(np.float32))
    np.save(f"{OUT}/bench_lens.npy", lens[bench_idx])
    bench = r'''
import json, os, resource, statistics, sys, time
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
import numpy as np, torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
model_dir, out = sys.argv[1:3]
audio, lens = np.load(f"{out}/bench_audio.npy"), np.load(f"{out}/bench_lens.npy")
offs = np.concatenate([[0], np.cumsum(lens)])
waves = [audio[offs[i]:offs[i + 1]] for i in range(len(lens))]
fe = AutoFeatureExtractor.from_pretrained(model_dir)
m = AutoModelForAudioClassification.from_pretrained(model_dir).to("cuda").eval()
torch.cuda.reset_peak_memory_stats()
def run():
    t = time.time()
    with torch.no_grad():
        for w in waves:
            m(fe(w, sampling_rate=16000, return_tensors="pt").input_values.to("cuda", m.dtype)).logits.argmax(-1).item()
    torch.cuda.synchronize()
    return (time.time() - t) / len(waves)
run()
lat = statistics.median(run() for _ in range(3))
print("BENCH " + json.dumps({"loaded_dtype": str(m.dtype), "latency_s": round(lat, 5),
      "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
      "ram_peak_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 3),
      "params_m": round(sum(p.numel() for p in m.parameters()) / 1e6, 2)}))
'''
    with open(f"{OUT}/bench.py", "w") as f:
        f.write(bench)
    r = subprocess.run([sys.executable, f"{OUT}/bench.py", model_dir, OUT], capture_output=True, text=True)
    line = [l for l in r.stdout.splitlines() if l.startswith("BENCH ")]
    if line:
        b = json.loads(line[0][6:])
        repo_gb = sum(os.path.getsize(f"{model_dir}/{n}") for n in os.listdir(model_dir)) / 1e9
        eff, parts = efficiency(b["params_m"], repo_gb, max(b["gpu_peak_gb"], b["ram_peak_gb"]), b["latency_s"])
        b.update(repo_gb=round(repo_gb, 4), efficiency=round(eff, 3), eff_parts={k: round(v, 3) for k, v in parts.items()})
        log(bench=b)
    else:
        log(bench_error=r.stderr[-2000:])
    for fn in ["bench_audio.npy", "bench_lens.npy"]:
        os.remove(f"{OUT}/{fn}")

log(done=True)

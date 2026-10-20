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
          "loss": "ce", "tau": 0.0, "group_alpha": 0.5, "aug": False, "ext_cap": 0.5, "eval_per_class": 200, "eval_at": [1.0], "seed": 0}

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
FOLDS_PATH = glob.glob(f"/kaggle/input/**/{CONFIG['folds_file']}", recursive=True)[0]
INPUT = os.path.dirname(glob.glob("/kaggle/input/**/score.py", recursive=True)[0])  # main data dataset
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
    """Decode one shard into a single int16 array plus per-clip (id, start, length)."""
    t = pq.read_table(path, columns=["id", "audio"])
    ids, waves = t.column("id").to_pylist(), []
    for a in t.column("audio").to_pylist():
        x, sr = sf.read(io.BytesIO(a["bytes"]), dtype="int16")
        assert sr == SR and x.ndim == 1
        waves.append(x)
    lengths = np.array([len(w) for w in waves])
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    return ids, starts, lengths, np.concatenate(waves)


t0 = time.time()
folds = pd.read_csv(FOLDS_PATH)
# Kaggle's mounted input can be very slow for pyarrow's many small reads, so copy the shards
# to local disk with large sequential reads first.
import shutil  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

LOCAL = "/tmp/shards"
os.makedirs(LOCAL, exist_ok=True)
shard_paths = sorted(glob.glob("/kaggle/input/**/*.parquet", recursive=True))  # competition + external shards
with ThreadPoolExecutor(6) as ex:
    local_paths = list(ex.map(lambda p: shutil.copyfile(p, f"{LOCAL}/{os.path.basename(p)}"), shard_paths))
log(copy_s=round(time.time() - t0, 1), n_shards=len(local_paths))
SHARDS, where = [], {}  # one int16 array per shard; id -> (shard, start, length)
with Pool(4) as pool:
    for ids, starts, lengths, audio in pool.imap(decode_shard, local_paths):
        for i, s, n in zip(ids, starts, lengths):
            where[i] = (len(SHARDS), int(s), int(n))
        SHARDS.append(audio)
shutil.rmtree(LOCAL, ignore_errors=True)  # local copies are no longer needed once decoded
folds = folds[folds.id.isin(where)].reset_index(drop=True)
# Sampling groups: real speakers for hypa/omnilingual; afrispeech clips count individually,
# so sampling is identical whichever pseudo-speaker clustering the folds use.
folds["sgroup"] = np.where(folds.source == "afrispeech_ipa", folds.id, folds.group)
loc = np.array([where[i] for i in folds.id])  # shard, start, length per row of folds
lens = loc[:, 2]
Y = np.array([L2I[l] for l in folds.label])
EVAL_ONLY = folds.fold.values == "V"  # evaluation-only external clips: never used for training
log(n_clips=len(folds), audio_hours=round(lens.sum() / SR / 3600, 2), decode_s=round(time.time() - t0, 1))


def wave(k, start=0, n=None):
    sh, s, length = loc[k]
    n = length - start if n is None else n
    return SHARDS[sh][s + start:s + start + n].astype(np.float32) / 32768.0


# ---------- sampling: class uniform -> source ~ hours^0.5 -> speaker group ~ size^alpha -> clip ----------
# The source level keeps the few hypa-channel clips of the big languages from being drowned
# out by thousands of afrispeech clips (the test set mixes all three sources).
def sampling_weights(idx):
    d = folds.iloc[idx].assign(n=lens[idx])
    w = np.zeros(len(idx))
    pos = {k: j for j, k in enumerate(d.index)}
    for lab, dl in d.groupby("label"):
        src_w = dl.groupby("source").n.sum() ** 0.5
        src_w = src_w / src_w.sum()
        # External sources together get at most ext_cap of a class, so clean studio corpora
        # (e.g. the 2-speaker Efik set) cannot become the class's dominant "sound".
        ext = src_w.index.str.startswith("ext_")
        if ext.any() and (~ext).any() and src_w[ext].sum() > CONFIG["ext_cap"]:
            src_w[ext] *= CONFIG["ext_cap"] / src_w[ext].sum()
            src_w[~ext] *= (1 - CONFIG["ext_cap"]) / src_w[~ext].sum()
        for src, ds in dl.groupby("source"):
            gsize = ds.groupby("sgroup").size()
            gw = gsize ** CONFIG["group_alpha"]
            gw = gw / gw.sum()
            share = (1 / d.label.nunique()) * src_w[src] * gw[ds.sgroup].values / gsize[ds.sgroup].values
            w[[pos[k] for k in ds.index]] = share
    return w / w.sum()


# ---------- augmentation (rung 3): applied to every class alike ----------
# Erases the per-corpus fingerprints (digital silence, noise floor, spectrum) and adds the test
# conditions the training data lacks (phone band and codec, noise, reverb).
from scipy.signal import butter, fftconvolve, resample_poly, sosfilt  # noqa: E402

PHONE_SOS = butter(4, [300, 3400], btype="bandpass", fs=SR, output="sos")
SPEEDS = [(10, 9), (20, 19), (20, 21), (10, 11)]  # resample ratios for speed 0.9, 0.95, 1.05, 1.1


def fit_length(x, n):
    if len(x) >= n:
        s = np.random.randint(0, len(x) - n + 1)
        return x[s:s + n]
    return np.pad(x, (0, n - len(x)), mode="wrap")


def rms(x):
    return np.sqrt(np.mean(x ** 2) + 1e-12)


def colored_noise(n):
    white = np.random.randn(n)
    kind = np.random.randint(3)
    if kind == 0:
        return white
    spec = np.fft.rfft(white)
    f = np.arange(len(spec)) + 1.0
    spec /= f ** (0.5 if kind == 1 else 1.0)  # pink or brown
    return np.fft.irfft(spec, n)


def augment(x, babble=None):
    n = len(x)
    x = x.astype(np.float64)
    if np.random.rand() < 0.5:  # speed perturbation (tempo and pitch together)
        up, down = SPEEDS[np.random.randint(len(SPEEDS))]
        x = fit_length(resample_poly(x, up, down), n)
    if np.random.rand() < 0.5:  # random smooth EQ, +-6 dB over 8 bands
        spec = np.fft.rfft(x)
        curve = np.interp(np.linspace(0, 7, len(spec)), np.arange(8), np.random.uniform(-6, 6, 8))
        x = np.fft.irfft(spec * 10 ** (curve / 20), n)
    if np.random.rand() < 0.3:  # synthetic room reverb
        rt60 = np.random.uniform(0.2, 0.8)
        t = np.arange(int(rt60 * SR)) / SR
        rir = np.random.randn(len(t)) * np.exp(-6.9 * t / rt60)
        rir[0] = 1.0
        x = fftconvolve(x, rir / np.sqrt(np.sum(rir ** 2)))[:n]
    if np.random.rand() < 0.5:  # background noise or babble
        if babble is not None and np.random.rand() < 0.3:
            noise, snr = fit_length(babble.astype(np.float64), n), np.random.uniform(10, 25)
        else:
            noise, snr = colored_noise(n), np.random.uniform(0, 20)
        x = x + noise * rms(x) / rms(noise) / 10 ** (snr / 20)
    if np.random.rand() < 0.25:  # phone channel: 300-3400 Hz, 8 kHz, mu-law 8-bit
        x = sosfilt(PHONE_SOS, x)
        x = resample_poly(resample_poly(x, 1, 2), 2, 1)[:n]
        peak = np.max(np.abs(x)) + 1e-9
        mu = np.sign(x / peak) * np.log1p(255 * np.abs(x / peak)) / np.log(256)
        mu = np.round(mu * 127) / 127
        x = peak * np.sign(mu) * ((256 ** np.abs(mu)) - 1) / 255
    # always: a random noise floor, so exact digital zeros never survive
    x = x + np.random.randn(n) * 10 ** (np.random.uniform(-75, -45) / 20)
    return x.astype(np.float32)


def batch_stream(tasks, pool, window=48):
    """Yield batches in order, preparing at most two windows ahead. An unbounded imap lets the
    workers outrun a slow GPU step and fill RAM with finished batches (D3 was OOM-killed)."""
    if pool is None:
        yield from map(make_batch, tasks)
        return
    pending = pool.map_async(make_batch, tasks[:window])
    for s in range(0, len(tasks), window):
        current = pending.get()
        if s + window < len(tasks):
            pending = pool.map_async(make_batch, tasks[s + window:s + 2 * window])
        yield from current


def make_batch(task):
    """One training batch: (clip indices, start offsets, crop length, babble indices, seed)."""
    ks, starts, n, babble_ks, seed = task
    np.random.seed(seed)
    out = []
    for k, s, bk in zip(ks, starts, babble_ks):
        x = wave(k, s, n)
        out.append(augment(x, wave(bk)) if CONFIG["aug"] else x)
    return out


# ---------- test-like conditions for local evaluation (deterministic per clip) ----------
def degrade(x, cond, seed):
    """'noisy': coloured noise at 0-10 dB SNR, half the time with reverb. 'phone': 300-3400 Hz,
    8 kHz, mu-law 8-bit, low noise floor. Mirrors the leaderboard's clean/noisy/phone split."""
    if cond == "clean":
        return x
    rs = np.random.get_state()
    np.random.seed(seed)
    y = x.astype(np.float64)
    n = len(y)
    if cond == "noisy":
        if np.random.rand() < 0.5:
            rt60 = np.random.uniform(0.3, 0.9)
            t = np.arange(int(rt60 * SR)) / SR
            rir = np.random.randn(len(t)) * np.exp(-6.9 * t / rt60)
            rir[0] = 1.0
            y = fftconvolve(y, rir / np.sqrt(np.sum(rir ** 2)))[:n]
        noise = colored_noise(n)
        y = y + noise * rms(y) / rms(noise) / 10 ** (np.random.uniform(0, 10) / 20)
    else:
        y = sosfilt(PHONE_SOS, y)
        y = resample_poly(resample_poly(y, 1, 2), 2, 1)[:n]
        peak = np.max(np.abs(y)) + 1e-9
        mu = np.round(np.sign(y / peak) * np.log1p(255 * np.abs(y / peak)) / np.log(256) * 127) / 127
        y = peak * np.sign(mu) * ((256 ** np.abs(mu)) - 1) / 255 + np.random.randn(n) * 10 ** (-60 / 20)
    np.random.set_state(rs)
    return y.astype(np.float32)


# ---------- model ----------
from transformers import AutoFeatureExtractor, HubertForSequenceClassification  # noqa: E402

fe = AutoFeatureExtractor.from_pretrained(BASE)


def inputs(waves):
    return fe(waves, sampling_rate=SR, return_tensors="pt", padding=True).input_values.to(dev)


def train(train_idx, seed, on_eval=None):
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
    eval_steps = {max(1, int(round(f * CONFIG["steps"]))) for f in CONFIG["eval_at"]}
    model.train()
    t0 = time.time()
    tasks = []
    for step in range(CONFIG["steps"]):
        b = draws[step * CONFIG["batch"]:(step + 1) * CONFIG["batch"]]
        n = int(min(random.uniform(CONFIG["crop_min_s"], CONFIG["crop_max_s"]) * SR, lens[b].min()))
        starts = [random.randint(0, lens[k] - n) for k in b]
        tasks.append((b, starts, n, np.random.choice(train_idx, len(b)), seed * 100003 + step))
    pool = Pool(3) if CONFIG["aug"] else None
    batches = batch_stream(tasks, pool)
    for step, (task, batch) in enumerate(zip(tasks, batches)):
        b = task[0]
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
        if on_eval and (step + 1) in eval_steps:
            on_eval(model, step + 1)
            model.train()
    if pool:
        pool.close()
    return model, round(time.time() - t0, 1)


@torch.no_grad()
def full_clip_logits(model, idx):
    model.eval()
    out = np.zeros((len(idx), len(LABELS)), dtype=np.float32)
    for j, k in enumerate(idx):
        with torch.autocast("cuda", dtype=torch.float16):
            out[j] = model(inputs([wave(k)])).logits.float().cpu().numpy()[0]
    return out


@torch.no_grad()
def condition_logits(model, idx):
    model.eval()
    out = {c: np.zeros((len(idx), len(LABELS)), dtype=np.float32) for c in ("clean", "noisy", "phone")}
    for j, k in enumerate(idx):
        x = wave(k)
        for ci, c in enumerate(out):
            with torch.autocast("cuda", dtype=torch.float16):
                out[c][j] = model(inputs([degrade(x, c, 7919 * int(k) + ci)])).logits.float().cpu().numpy()[0]
    return out


if CONFIG["mode"] == "cv":
    rng = np.random.default_rng(CONFIG["seed"])
    for f in CONFIG["folds"]:
        tr = np.where((folds.fold.values != f) & ~EVAL_ONLY)[0]
        ho_all = np.where(folds.fold.values == f)[0]
        # evaluation subset: up to eval_per_class held-out clips per class (competition clips only)
        ho = np.concatenate([rng.permutation(ho_all[Y[ho_all] == c])[:CONFIG["eval_per_class"]] for c in range(len(LABELS))]).astype(int)

        va_all = np.where(EVAL_ONLY)[0]
        va = np.concatenate([rng.permutation(va_all[Y[va_all] == c])[:CONFIG["eval_per_class"] // 2] for c in range(len(LABELS))]).astype(int)

        def on_eval(model, step, f=f, ho=ho, va=va):
            t1 = time.time()
            lg = condition_logits(model, ho)
            np.savez_compressed(f"{OUT}/oof_{f}_s{step}.npz", ids=folds.id.values[ho], **lg)
            if len(va):
                lv = condition_logits(model, va)
                np.savez_compressed(f"{OUT}/va_{f}_s{step}.npz", ids=folds.id.values[va], **lv)
            res = {}
            for c, l in lg.items():
                wf1, _ = weighted_f1(folds.label.values[ho].tolist(), [LABELS[i] for i in l.argmax(1)])
                res[c] = {"weighted_f1": round(wf1, 4), "acc": round(float((l.argmax(1) == Y[ho]).mean()), 4)}
            log(**{f"fold_{f}_step_{step}": {**res, "n_eval": len(ho), "eval_s": round(time.time() - t1, 1)}})

        model, train_s = train(tr, CONFIG["seed"], on_eval=on_eval)
        log(**{f"fold_{f}_train_s": train_s})
        del model
        torch.cuda.empty_cache()
else:
    model, train_s = train(np.where(~EVAL_ONLY)[0], CONFIG["seed"])
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
      "gpu_peak_gb": round(torch.cuda.max_memory_reserved() / 1e9, 3),
      "ram_peak_gb": round(int([l for l in open("/proc/self/status") if l.startswith("VmHWM")][0].split()[1]) / 1e6, 3),
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

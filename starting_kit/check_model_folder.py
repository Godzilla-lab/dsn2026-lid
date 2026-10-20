#!/usr/bin/env python
"""
Check a saved model folder against the competition rules before you push it to Hugging Face.

    python check_model_folder.py ./my_model_folder

It looks only at file names, config.json and the weight headers (nothing is loaded), so it is fast.
It cannot measure speed or memory: those are measured on the organisers' hardware.
"""
import json
import struct
import sys
from fnmatch import fnmatch
from pathlib import Path

LABELS = ["yor", "ibo", "hau", "ego", "anw", "efi", "ibb", "idu", "igl", "nup", "tiv", "urh", "other"]
MAX_PARAMS, MAX_BYTES = 350e6, 1.5e9
ALLOWED_DTYPES = {"F32", "F16"}
ALLOWED_FILES = ["config.json", "preprocessor_config.json", "model.safetensors", "model-*.safetensors",
                 "model.safetensors.index.json", "README.md", "LICENSE", "LICENSE.md", "LICENSE.txt", ".gitattributes"]
ALLOWED_TYPES = {"wav2vec2", "wav2vec2-conformer", "wav2vec2-bert", "hubert", "wavlm", "unispeech", "unispeech-sat",
                 "sew", "sew-d", "data2vec-audio", "whisper", "audio-spectrogram-transformer"}


def summary(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    params, dtypes = 0, set()
    for k, v in header.items():
        if k != "__metadata__":
            c = 1
            for d in v["shape"]:
                c *= d
            params += c
            dtypes.add(v["dtype"])
    return params, dtypes


folder = Path(sys.argv[1]) if len(sys.argv) > 1 else sys.exit(__doc__)
problems, files = [], [p for p in folder.rglob("*") if p.is_file() and ".git/" not in p.as_posix() and ".cache" not in p.parts]
names = [p.relative_to(folder).as_posix() for p in files]
for n in names:
    if not any(fnmatch(n, pat) for pat in ALLOWED_FILES):
        problems.append(f"file not allowed: {n}")
for need in ("config.json", "preprocessor_config.json"):
    if need not in names:
        problems.append(f"{need} is missing")
weights = sorted(p for p in files if fnmatch(p.name, "model*.safetensors") and p.suffix == ".safetensors")
if not weights:
    problems.append("no model.safetensors found (save with safe_serialization=True)")
size = sum(p.stat().st_size for p in files)
if size > MAX_BYTES:
    problems.append(f"folder is {size / 1e9:.2f} GB, limit is {MAX_BYTES / 1e9:.1f} GB")
params, dtypes = 0, set()
for w in weights:
    p, d = summary(w)
    params, dtypes = params + p, dtypes | d
if params > MAX_PARAMS:
    problems.append(f"{params / 1e6:.1f}M parameters, limit is {MAX_PARAMS / 1e6:.0f}M")
if dtypes - ALLOWED_DTYPES:
    problems.append(f"weight dtype {sorted(dtypes - ALLOWED_DTYPES)} not allowed; use fp32 or fp16")
if (folder / "config.json").exists():
    cfg = json.loads((folder / "config.json").read_text())
    if "auto_map" in cfg:
        problems.append("config.json has auto_map (custom code is not allowed)")
    if cfg.get("model_type") not in ALLOWED_TYPES:
        problems.append(f"model_type {cfg.get('model_type')!r} is not supported: {sorted(ALLOWED_TYPES)}")
    if set((cfg.get("id2label") or {}).values()) != set(LABELS):
        problems.append(f"id2label must be exactly {LABELS}")

print(f"{params / 1e6:.1f}M parameters, {size / 1e9:.3f} GB, dtypes {sorted(dtypes)}")
if problems:
    print("\nFAIL:")
    print("\n".join(" - " + p for p in problems))
    sys.exit(1)
print("PASS: the folder meets the file, size, parameter and label rules")

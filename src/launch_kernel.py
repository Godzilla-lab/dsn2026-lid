"""Push train_kernel.py to Kaggle as a private GPU script kernel with a given CONFIG.

    KAGGLE_API_TOKEN=... python src/launch_kernel.py <kaggle-user> <slug> '<json overrides>'

The CONFIG line in train_kernel.py is replaced with the defaults updated by the overrides,
so each run's exact settings are stored in its own kernel version.
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

user, slug, overrides = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
src = (Path(__file__).parent / "train_kernel.py").read_text()
m = re.search(r"^CONFIG = (\{.*?\})\n", src, re.S | re.M)
config = eval(m.group(1))  # the literal dict at the top of our own script
config.update(overrides)
src = src[:m.start(1)] + json.dumps(config).replace("false", "False").replace("true", "True") + src[m.end(1):]

with tempfile.TemporaryDirectory() as d:
    (Path(d) / "train_kernel.py").write_text(src)
    (Path(d) / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{slug}", "title": slug, "code_file": "train_kernel.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [f"{user}/dsn2026-lid-data"], "competition_sources": [], "kernel_sources": []}))
    kaggle = str(Path(sys.executable).parent / "kaggle")
    r = subprocess.run([kaggle, "kernels", "push", "-p", d, "--accelerator", "NvidiaTeslaT4"], capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())
    print("config:", json.dumps(config))

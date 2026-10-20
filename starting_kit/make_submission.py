#!/usr/bin/env python
"""Create submission.json and submission.zip for a Hugging Face model repository.

    python make_submission.py username/model-name                       (needs: pip install huggingface_hub)
    python make_submission.py username/model-name --revision <40-char hash>
"""
import argparse
import json
import re
import zipfile

ap = argparse.ArgumentParser()
ap.add_argument("repo", help="Hugging Face repo id, for example username/model-name")
ap.add_argument("--revision", help="40-character commit hash (default: the latest commit of the repo)")
a = ap.parse_args()

if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", a.repo):
    raise SystemExit("repo must look like username/model-name")
revision = a.revision
if not revision:
    from huggingface_hub import HfApi
    revision = HfApi().model_info(a.repo).sha
if not re.fullmatch(r"[0-9a-f]{40}", revision):
    raise SystemExit("revision must be a full 40-character commit hash, not a branch name")

with open("submission.json", "w") as f:
    json.dump({"repo": a.repo, "revision": revision}, f)
with zipfile.ZipFile("submission.zip", "w") as z:
    z.write("submission.json")
print(f"Wrote submission.zip for {a.repo} @ {revision}")

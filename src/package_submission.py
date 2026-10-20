"""Package a trained model folder as a competition submission.

1. Writes the model card (README.md) and LICENSE into the folder.
2. Runs the organisers' checker on the exact folder.
3. Uploads it to a private Hugging Face repo and records the commit hash.
4. Writes submission.zip with the organisers' make_submission.py.

    python src/package_submission.py <model_dir> <hf_repo> <out_dir> <rung> [--ext-manifest manifest.csv]

The repo stays private; make it public only right before uploading the zip to Codabench.
"""
import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CARD = """---
license: cc-by-nc-sa-4.0
library_name: transformers
pipeline_tag: audio-classification
tags: [audio-classification, language-identification, nigerian-languages]
base_model: ajesujoba/AfriHuBERT
---

# Spoken language identification for 12 Nigerian languages (DSN 2026)

Submission of team "godzilla" to the DSN 2026 *Find the Language I Am Speaking* competition.

A standard `HubertForSequenceClassification` model (loads with `AutoModelForAudioClassification`, no custom code).
It predicts one of 13 labels: `yor ibo hau ego anw efi ibb idu igl nup tiv urh other`.

## Approach

{approach}

## Base model

- [ajesujoba/AfriHuBERT](https://huggingface.co/ajesujoba/AfriHuBERT), a continued pretraining of
  [utter-project/mHuBERT-147](https://huggingface.co/utter-project/mHuBERT-147) (CC BY-NC-SA 4.0). This model inherits
  that licence.

## Training data

- Competition training and validation data,
  [Data-Science-Nigeria/dsn-2026-lid-train](https://huggingface.co/datasets/Data-Science-Nigeria/dsn-2026-lid-train),
  which combines afrispeech_ipa (CC-BY-4.0), hypa_voices (Apache-2.0) and omnilingual_asr (CC-BY-4.0); see its
  ATTRIBUTION.md. No other use of those corpora.
{external}

## Usage

```python
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
fe = AutoFeatureExtractor.from_pretrained(REPO)
model = AutoModelForAudioClassification.from_pretrained(REPO)
inputs = fe(waveform_16khz_mono, sampling_rate=16000, return_tensors="pt")
label = model.config.id2label[model(**inputs).logits.argmax(-1).item()]
```
"""

APPROACH = {
    1: "AfriHuBERT truncated to its first 4 transformer layers with a weighted layer sum and mean pooling, fine-tuned on "
       "random 3-6 s crops with speaker- and source-balanced sampling. Inference uses the full clip.",
}


def external_section(manifests):
    if not manifests:
        return "- No external data."
    lines = ["", "External datasets (openly licensed; per-dataset licence shown):", ""]
    seen = set()
    for manifest in manifests:
        rows = list(csv.DictReader(open(manifest)))
        if rows and "dataset_id" in rows[0]:  # one row per dataset / config / label
            for r in rows:
                key = (r["dataset_id"], r["config_split"])
                if key in seen:
                    continue
                seen.add(key)
                lines.append(f"- [{r['dataset_id']}]({r['url']}), {r['config_split']}, revision {r['revision'][:10]}, "
                             f"{r['license']}, used as `{r['label']}`")
        else:  # one row per media file (Wikimedia Commons, Wikitongues, Bloom Library)
            by_source = {}
            for r in rows:
                by_source.setdefault(r["source"], []).append(r)
            names = {"ext_wikimedia": "Wikimedia Commons (Wikimedia Nigeria oral-history project)",
                     "ext_wikitongues": "Wikitongues (via Wikimedia Commons)", "ext_bloom": "Bloom Library talking books"}
            for src, rs in by_source.items():
                lic = sorted({r["licence"] for r in rs})
                lines.append(f"- {names.get(src, src)}: {len(rs)} recordings ({', '.join(lic)}); speech segments only, "
                             f"English and music removed:")
                for r in rs:
                    lines.append(f"  - [{r['title']}]({r['file_page']}), {r['licence']}, by {r['author'] or r['uploader']}, "
                                 f"used as `{r['label']}`")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("hf_repo")
    ap.add_argument("out_dir")
    ap.add_argument("rung", type=int)
    ap.add_argument("--ext-manifest", nargs="*", default=[])
    ap.add_argument("--approach")
    a = ap.parse_args()

    model_dir, out_dir = Path(a.model_dir), Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    card = CARD.format(approach=a.approach or APPROACH[a.rung], external=external_section(a.ext_manifest))
    (model_dir / "README.md").write_text(card.replace("REPO", repr(a.hf_repo)))
    shutil.copy(ROOT / "templates" / "LICENSE_CC_BY_NC_SA_4.0.txt", model_dir / "LICENSE")

    chk = subprocess.run([sys.executable, str(ROOT / "starting_kit" / "check_model_folder.py"), str(model_dir)],
                         capture_output=True, text=True)
    print(chk.stdout.strip())
    if chk.returncode != 0:
        sys.exit("checker failed: not uploading")

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(a.hf_repo, repo_type="model", private=True, exist_ok=True)
    commit = api.upload_folder(repo_id=a.hf_repo, folder_path=str(model_dir), commit_message=f"Rung {a.rung} model")
    sha = commit.oid
    print("uploaded", a.hf_repo, sha)

    subprocess.run([sys.executable, str(ROOT / "starting_kit" / "make_submission.py"), a.hf_repo, "--revision", sha],
                   cwd=out_dir, check=True)
    (out_dir / "INFO.txt").write_text(f"repo: https://huggingface.co/{a.hf_repo}\nrevision: {sha}\nrung: {a.rung}\n")
    print("wrote", out_dir / "submission.zip")


if __name__ == "__main__":
    main()

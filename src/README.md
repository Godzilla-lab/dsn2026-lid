# Training code

Training runs as **Kaggle GPU script kernels** (plain `.py`, not notebooks) on a team member's Kaggle account. This machine only prepares data, launches the runs, and packages the results.

| File | What it does |
|---|---|
| `train_kernel.py` | **The training script.** Decodes all audio once, then trains AfriHuBERT with speaker- and source-balanced sampling, optional augmentation and optional external data. `mode: "full"` trains on everything, saves the fp16 model, runs the organisers' checker, and measures latency and memory. `mode: "cv"` trains per fold and saves out-of-fold logits. Settings live in `CONFIG` at the top. |
| `launch_kernel.py` | Pushes `train_kernel.py` to Kaggle with CONFIG overrides, so each run's exact settings are stored in its kernel version. |
| `package_submission.py` | Writes the model card and licence, runs the checker, uploads to a Hugging Face repo, and writes `submission.zip`. |
| `tune_bias.py` | Rung 2. Tunes per-language decision offsets on out-of-fold logits, then writes them into `classifier.bias`. |
| `score.py` | Exact copy of the competition score (weighted F1 × efficiency). `python src/score.py --selftest` checks it. |
| `make_folds.py` | Builds the speaker-grouped CV folds (`splits/`). |
| `embed_kernel.py`, `speaker_embeddings.py`, `cluster_afrispeech.py` | Speaker embeddings and pseudo-speaker clustering for afrispeech clips. Not used yet; the folds use v0. |
| `rung0_kernel.py` | The original smoke test. Superseded by `train_kernel.py`. |

## Kaggle datasets it expects (private, on the training account)

- `dsn2026-lid-data`: the competition parquet shards, plus `score.py`, `check_model_folder.py` and `folds_v0.csv`
- `dsn2026-lid-meta`: fold files, e.g. `folds_v0_ext.csv`
- `dsn2026-lid-ext`: the external shards built from USE-verdict sources (see the manifest)

## How each submission was made

```bash
# D1 (rung 1): provided data, no augmentation
python src/launch_kernel.py <user> dsn2026-r1-full-b '{"mode": "full", "folds_file": "folds_v0.csv", "datasets": ["dsn2026-lid-data"]}'
# D2 (rung 3): + augmentation, 8000 steps
python src/launch_kernel.py <user> dsn2026-r3-full '{"mode": "full", "folds_file": "folds_v0.csv", "aug": true, "steps": 8000, "datasets": ["dsn2026-lid-data"]}'
# D3 (go-for-#1 stack): + 12 layers + external data
python src/launch_kernel.py <user> dsn2026-r5-full-b '{"mode": "full", "folds_file": "folds_v0_ext.csv", "aug": true, "steps": 8000, "n_layers": 12, "datasets": ["dsn2026-lid-data", "dsn2026-lid-meta", "dsn2026-lid-ext"]}'
```

Run with `KAGGLE_API_TOKEN` set to the training account's token. On Kaggle, only push a new run when nothing else is running: queued runs on this account never start.

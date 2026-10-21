# Why D1 was scored and D2 / D4 were not

## Timeline (UTC)

| When | What |
|---|---|
| Tue 11:46 | datapsalm scored (0.5109) |
| Tue 13:15 | ray_nacido scored (0.5685): the last non-zero score from any other team |
| Tue 13:22 | **D1** (#1005460) runs at **0.004 s per clip** and is scored: 0.3421 (2,358 public clips scored out of the 6,659 run). Its status then stays "Running" with 0.0000 for 1.5 h+ after the results exist |
| Tue 14:23 | **D2** (#1005591): "running inference" for 1,842 s, **0 of 6,659** clips, killed at 1,800 s, `predictions.csv is missing` |
| Tue 14:23 → Wed | At least 10 teams submit; every one shows 0.0. No non-zero score anywhere |
| Wed ~06:00 | **D4**: same failure, 0 of 6,659 clips in 1,800 s |

## It is not our model

- **D2 is D1 with different weight values.** The `config.json` and `preprocessor_config.json` files are byte-identical, and `model.safetensors` is the same size (75,748,084 bytes, fp16, transformers 4.57.6). Weight values cannot make a model hang before its first clip.
- **D4 loads and runs.** An anonymous download of the exact commit loads in 7 s and runs a 30 s clip in 3 s on a plain 4-core CPU.
- **Zero clips in 30 minutes is not "slow".** At D1's measured 0.004 s per clip, 6,659 clips take about 30 s. Zero means inference never produced output, so `predictions.csv` was never written.
- **Every team fails the same way** since the same moment.

## What changed on the organisers' side

1. **Correction: the clip count did not change.** D2's own scoring output says "clips scored: 2358" too. The runner always runs all 6,659 test clips (public + private) and the public leaderboard scores 2,358 of them.
2. **D1's job never closed.** It stayed "Running" after its results were produced, so the worker probably stalled at the end of D1's job and may still hold the GPU.
3. **If the runner fell back to a CPU, every HuBERT-type model would fail exactly like this.** Measured on a 4-core CPU, with clips drawn from the competition's length distribution (mean 11.4 s):

   | Model | Time per clip (4 threads) | 6,659 clips | 2,358 clips |
   |---|---|---|---|
   | D1/D2 (4 layers) | 0.69 s | 77 min | 27 min |
   | D4 (12 layers) | 1.18 s | 131 min | 47 min |

   The limit is 30 min. Most of the CPU time is HuBERT's convolutional front end, which every layer count shares. On the organisers' GPU, D1 took 0.004 s per clip.

## Where inference happens

The Codabench side is a thin client.
- **Programs:** the competition's ingestion program is 2.4 KB and its scoring program 6.2 KB, run in `codalab/codalab-legacy:py39`. They are not available to participants.
- **What they do:** they send the repo and commit to the organisers' own runner server ("Submitting ... to the runner"), poll its status (`queued`, `downloading`, `evaluating`, `done`), then score the predictions it returns.
- **Consequence:** the audio and the GPU are on the organisers' server, so nothing in a submission can change what that server does.

## Probe to tell "hung" from "slow"

`submissions/PROBE/submission.zip` points to `godzilla1215/dsn2026-lid-probe`, a 36k-parameter random-weight model. It runs 6,659 clips in about 3 minutes on one CPU thread.
- **If the probe is scored** (F1 near 0, by design): the runner works but is now too slow for real models, and the organisers need to restore the GPU.
- **If the probe also shows 0 of 6,659:** the runner is hung. Only the organisers can restart it.

## Resolution (Wed 21 Oct, 10:00–11:35 UTC)

The runner recovered around 10:00 UTC: simple0001 (164.5M, 0.025 s/clip), oladahmad (94.6M, 0.014 s/clip), idowuadamo (202M), mustyyyy (20.2M, 0.004 s/clip) and team_lgorithm (94.6M) were all scored. Our probe (36k parameters) was scored at **0.003 s per clip with 1.43 GB peak memory**: a GPU runner with a fixed memory floor. The organisers' "capped on purpose" refers to the 1,800 s limit, not to CPU-only inference. D5 sent for upload.

## What only the organisers can do

Restart or replace the stuck GPU worker, and confirm the evaluation hardware and test size. Then re-run the affected submissions (D2 #1005591, D4).

## What we do

- Don't spend more submissions until another team gets a real score again. Claude is watching the leaderboard.
- Keep the best model (D4, or D5 if it beats D4) ready to upload once the runner works.
- The final phase (Thu 00:00–08:00 UTC) needs a working runner too, so the organisers have to fix it.

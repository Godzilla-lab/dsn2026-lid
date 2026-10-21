# Why D1 was scored and D2 / D4 were not

## Timeline (UTC)

| When | What |
|---|---|
| Tue 11:46 | datapsalm scored (0.5109) |
| Tue 13:15 | ray_nacido scored (0.5685): the last non-zero score from any other team |
| Tue 13:22 | **D1** (#1005460) runs at **0.004 s per clip** and is scored: 0.3421 on **2,358** clips. Its status then stays "Running" with 0.0000 for 1.5 h+ after the results exist |
| Tue 14:23 | **D2** (#1005591): "running inference" for 1,842 s, **0 of 6,659** clips, killed at 1,800 s, `predictions.csv is missing` |
| Tue 14:23 → Wed | At least 10 teams submit; every one shows 0.0. No non-zero score anywhere |
| Wed ~06:00 | **D4**: same failure, 0 of 6,659 clips in 1,800 s |

## It is not our model

- **D2 is D1 with different weight values.** The `config.json` and `preprocessor_config.json` files are byte-identical, and `model.safetensors` is the same size (75,748,084 bytes, fp16, transformers 4.57.6). Weight values cannot make a model hang before its first clip.
- **D4 loads and runs.** An anonymous download of the exact commit loads in 7 s and runs a 30 s clip in 3 s on a plain 4-core CPU.
- **Zero clips in 30 minutes is not "slow".** At D1's measured 0.004 s per clip, 6,659 clips take about 30 s. Zero means inference never produced output, so `predictions.csv` was never written.
- **Every team fails the same way** since the same moment.

## What changed on the organisers' side

1. **The clip count changed** from 2,358 (D1) to 6,659 (every run since). The evaluation setup was changed between D1 and D2. 6,659 is probably public plus private, ahead of the final phase.
2. **D1's job never closed.** It stayed "Running" after its results were produced, so the worker probably stalled at the end of D1's job and may still hold the GPU.
3. **If the runner fell back to a CPU, every HuBERT-type model would fail exactly like this.** Measured on a 4-core CPU, with clips drawn from the competition's length distribution (mean 11.4 s):

   | Model | Time per clip (4 threads) | 6,659 clips | 2,358 clips |
   |---|---|---|---|
   | D1/D2 (4 layers) | 0.69 s | 77 min | 27 min |
   | D4 (12 layers) | 1.18 s | 131 min | 47 min |

   The limit is 30 min. Most of the CPU time is HuBERT's convolutional front end, which every layer count shares. On the organisers' GPU, D1 took 0.004 s per clip.

## What only the organisers can do

Restart or replace the stuck GPU worker, and confirm the evaluation hardware and test size. Then re-run the affected submissions (D2 #1005591, D4).

## What we do

- Don't spend more submissions until another team gets a real score again. Claude is watching the leaderboard.
- Keep the best model (D4, or D5 if it beats D4) ready to upload once the runner works.
- The final phase (Thu 00:00–08:00 UTC) needs a working runner too, so the organisers have to fix it.

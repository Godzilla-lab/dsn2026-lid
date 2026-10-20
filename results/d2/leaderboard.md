# D2: NOT SCORED (platform timeout)

Model: godzilla1215/dsn2026-lid-d2 @ 3e073891. Codabench submission #1005591, uploaded 07:23 server time (14:23 UTC).

Organisers' message:
- limit broken: average time 1800.000 s per clip, limit is 0.5 s
- limit broken: total run time exceeded 1800 s (processed **0 of 6659** clips)
- predictions.csv is missing

## Why this is not our model

- D2's `config.json` and `preprocessor_config.json` are **identical** to D1's. Only the weights differ.
- D1 ran on the organisers' hardware at **0.004 s per clip**.
- D2 loads offline in 0.4 s and predicts correctly here (checked on an anonymous download of the exact commit).
- 0 clips were processed in 1,800 s. The run hung before inference started: the model download or the worker stalled.
- At the same time, D1 (#1005460) stayed "Running" with score 0.0000 for over 1.5 h, after its results were already produced. No team's result reached the leaderboard after about 13:15 UTC.
- Note: D1 was scored on 2,358 clips, while D2's run reports 6,659 clips. The organisers may now run the full test set per submission. At about 0.004 s per clip that is still only about 30 s.

## Action

- Ask the organisers to re-run #1005591 and to check the stuck #1005460.
- Do not re-upload until another team's new result shows the queue is healthy again.
- Keep today's last slot for D3.

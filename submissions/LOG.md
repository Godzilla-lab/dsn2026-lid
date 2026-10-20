# Submission log

One row per Codabench upload. Each submission tests one question (see the training ladder in `docs/plan.md`).

Limits: 3 submissions per day; 20 in the development phase, 5 in the final phase. The development phase ends 2026-10-21 23:59 UTC; the final phase runs 2026-10-22 00:00–08:00 UTC.

| # | Date (UTC) | Ladder rung | Question it tests | HF repo @ commit | Local CV final score | LB final score | LB weighted F1 | Efficiency | Weakest classes | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| D1 | 2026-10-20 | 1 | Calibration: does local CV match the leaderboard? Organisers' latency and memory for a 38M fp16 HuBERT | godzilla1215/dsn2026-lid-d1 @ a9cbed5f0d2f86b8bd090ff2aa3c70657da3e5a6 | n/a (CV not run yet) | **0.3421** | 0.3739 | 0.830 (mem 1.72 GB, 4 ms) | efi 0, ibb 0, igl 0.03, tiv 0.16; noisy acc 0.21 | Idoma is a garbage class: channel shortcut. See results/d1/leaderboard.md |
| D2 | 2026-10-20 | 3 | Does augmentation (noise floor, speed, EQ, reverb, noise/babble, phone channel) fix noisy/phone accuracy and the Idoma dump? | godzilla1215/dsn2026-lid-d2 @ 3e07389119142f68d8ff2c930ab4564be093aee4 | n/a | not scored | | | | Codabench #1005591: platform timeout, 0 of 6659 clips processed; ask organisers to re-run |
| D3 | 2026-10-20 | 3+4+6 | Real check of the 12-layer + augmentation + external data recipe; confirms the organisers' queue works | godzilla1215/dsn2026-lid-d3 @ 41db455c680a879d01d818699f4fdfc65d021210 | fold A held-out F1 ≈0.54 | | | | | sent |

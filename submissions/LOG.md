# Submission log

One row per Codabench upload. Each submission tests one question (see the training ladder in `docs/plan.md`).

Limits: 3 submissions per day; 20 in the development phase, 5 in the final phase. The development phase ends 2026-10-21 23:59 UTC; the final phase runs 2026-10-22 00:00–08:00 UTC.

| # | Date (UTC) | Ladder rung | Question it tests | HF repo @ commit | Local CV final score | LB final score | LB weighted F1 | Efficiency | Weakest classes | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| D1 | 2026-10-20 | 1 | Calibration: does local CV match the leaderboard? Organisers' latency and memory for a 38M fp16 HuBERT | godzilla1215/dsn2026-lid-d1 @ a9cbed5f0d2f86b8bd090ff2aa3c70657da3e5a6 | n/a (CV not run yet) | **0.3421** | 0.3739 | 0.830 (mem 1.72 GB, 4 ms) | efi 0, ibb 0, igl 0.03, tiv 0.16; noisy acc 0.21 | Idoma is a garbage class: channel shortcut. See results/d1/leaderboard.md |
| D2 | 2026-10-20 | 3 | Does augmentation (noise floor, speed, EQ, reverb, noise/babble, phone channel) fix noisy/phone accuracy and the Idoma dump? | godzilla1215/dsn2026-lid-d2 @ 3e07389119142f68d8ff2c930ab4564be093aee4 | n/a | not scored | | | | Codabench #1005591: platform timeout, 0 of 6659 clips processed; ask organisers to re-run |
| D3 | 2026-10-20 | 3+4+6 | Real check of the 12-layer + augmentation + external data recipe; confirms the organisers' queue works | godzilla1215/dsn2026-lid-d3 @ 41db455c680a879d01d818699f4fdfc65d021210 | fold A held-out F1 ≈0.54 | | | | | sent |
| D4 | 2026-10-20 | final | Final recipe: 12 layers + aug + all external data (VoiceAfrica, Wikimedia/Wikitongues/Bloom) + cleaned labels + offsets ×0.75 | godzilla1215/dsn2026-lid-d4 @ a653142c4fbb5a2e7e7ab4302fc7702b1c7c588c | predicted ≈0.62–0.64 | not scored | | | | Wed 21 Oct: platform timeout, 0 of 6659 clips processed (same as D2 and every team since Tue 13:15 UTC). D4 loads anonymously at this commit in 7 s and runs a 30 s clip in 3 s on a 4-core CPU, so the runner hangs before the first clip |
| D5 | 2026-10-21 | final + averaging | Does averaging D4 with a same-data second model (different data order) beat D4? Fold A: yes, 0.5646 vs 0.5465 | godzilla1215/dsn2026-lid-d5 @ e3390666601bfae597bff9a4573a1f3dce46dd72 | fold A 0.5646 (D4-equivalent 0.5465) | | | | | public, sent Wed 11:15 UTC after simple0001's real score showed the runner works again |
| PROBE | 2026-10-21 | diagnostic | Does the runner process clips at all, and how fast is its machine? | godzilla1215/dsn2026-lid-probe @ d9b055dd020e17f4c7254138e2cfb57d9339c529 | n/a (random weights) | 0.0350 | 0.0372 | 0.881 (mem 1.43 GB, 0.003 s) | all | Runner healthy: GPU speed, memory floor ~1.4 GB even for a 36k-parameter model. 11:35 UTC |

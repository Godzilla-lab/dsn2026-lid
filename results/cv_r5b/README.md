# CV of the new recipe (12 layers + augmentation + external data + VoiceAfrica, 7 train / 3 held-out speakers)

Local leaderboard (src/predict_lb.py), public-test class and condition mix. Nupe is scored on held-out VoiceAfrica speakers.

| Checkpoint | Predicted weighted F1 | Predicted final (efficiency 0.791) |
|---|---|---|
| 4,000 steps | 0.654 | 0.585 |
| **8,000 steps** | **0.694** | **0.621** (90%: 0.608–0.636) |
| 8,000 steps + offsets (in-sample, optimistic) | 0.730 | 0.653 |

**Per-class F1 at 8,000 steps:**
- **Strong:** yor 0.93, ibo 0.92, hau 0.92, ego 0.91, urh 0.98, other 0.88.
- **Middling:** anw 0.82, idu 0.86, tiv 0.84.
- **Weak:** **efi 0.17, ibb 0.02, igl 0.19**.
- nup 0.92 is measured on VoiceAfrica speakers. That is likely optimistic for the test's Nupe, which comes from hypa.

**Honest offset check** (`src/crossfit_offsets.py`: tune on 2 folds, score the 3rd, nup offset 0). Mean held-out weighted F1 by offset strength:

| Strength | ×0 | ×0.25 | ×0.5 | ×0.75 | ×1.0 |
|---|---|---|---|---|---|
| Held-out weighted F1 | 0.6956 | 0.7039 | 0.7124 | **0.7171** | 0.7171 |

Chosen: ×0.75, with nup forced to 0 (`offsets_final.json`). That is about +0.02 F1, out of sample.

Calibration reminder: on D1 the local leaderboard was about 0.035 cautious on the final score.

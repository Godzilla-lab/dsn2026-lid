# D1 leaderboard result (public test split, 2,358 clips)

Model: godzilla1215/dsn2026-lid-d1 @ a9cbed5f (rung 1: AfriHuBERT 4 layers, provided data only, no augmentation, 4,000 steps)

| Part | Measured | Score |
|---|---|---|
| Language-weighted F1 | 0.3739 | |
| Parameters | 37.9M | 0.892 |
| Repository | 0.076 GB | 0.949 |
| Peak memory | 1.72 GB | 0.570 |
| Speed | 0.004 s/clip | 1.000 |
| Efficiency | | 0.830 |
| **Final** | | **0.3421** (not ranked: F1 < 0.60) |

| Condition | Accuracy |
|---|---|
| clean | 0.641 |
| noisy | **0.210** |
| phone | 0.436 |

| Class | F1 | P | R | Clips |
|---|---|---|---|---|
| yor | 0.416 | 0.651 | 0.306 | 317 |
| ibo | 0.557 | 0.641 | 0.493 | 217 |
| hau | 0.598 | 0.781 | 0.485 | 295 |
| ego | 0.796 | 0.916 | 0.703 | 155 |
| anw | 0.485 | 0.386 | 0.650 | 123 |
| efi | 0.000 | 0.000 | 0.000 | 45 |
| ibb | 0.000 | 0.000 | 0.000 | 23 |
| idu | 0.402 | **0.254** | 0.965 | 171 |
| igl | 0.032 | 0.125 | 0.018 | 55 |
| nup | 0.280 | 0.273 | 0.288 | 52 |
| tiv | 0.159 | 0.252 | 0.116 | 242 |
| urh | 0.863 | 0.850 | 0.876 | 129 |
| other | 0.497 | 0.520 | 0.476 | 534 |

## Diagnosis

- **Idoma is a garbage class.** The idu column of the confusion matrix collects 82 yor, 40 ibo, 42 hau, 46 ego, 75 tiv and many `other` clips (precision 0.25).
  - Idoma training audio is the quietest in the data (omnilingual, median −42.8 dBFS), so the model learned "quiet or odd channel → Idoma".
  - This is the channel shortcut the audit predicted, and it fires on noisy test audio.
- **Noisy accuracy is 0.21**, against 0.71–0.78 for the leaders. The model has never heard noise.
- **efi and ibb are 0.** efi goes to anw (14), idu (13), ibb (7) and other (8); ibb goes to idu (12) and other (6).
- **igl goes to other (26) and nup (15).**
- **tiv recall is 0.12.** tiv goes to idu (75), other (55) and anw (48).
- **Memory, not size, is the efficiency drag:** 1.72 GB scores 0.57. HuBERT-family models land around 1.7 GB on the organisers' machine (ray_nacido's 94.6M HuBERT: 1.68 GB), against about 1.2 GB for Whisper models.

## Decision

- Run rung 3 (augmentation) before rung 2 (bias tuning). Robustness is the dominant failure, and biases tuned on clean CV logits would not fix it.
- Train for 8,000 steps instead of 4,000.

# Voice-change test (no gain: not used)

Question: the model scores 0.8–1.0 on new VoiceAfrica speakers of igl/nup but only 0.1–0.2 on a new
competition-data speaker of igl/efi/ibb, because it has only 2–5 voices per language to learn from.
Do stronger voice changes during training make it generalise to new speakers?

Change from the D4 recipe: speed/pitch perturbation 0.85–1.15 on 70% of clips (was 0.9–1.1 on 50%),
SpecAugment time masking 0.15 (was 0.05) and feature masking 0.05 (was 0). Same folds (`folds_cv2.csv`), 8,000 steps.

| Run | Predicted weighted F1 | Predicted final | efi | ibb | igl |
|---|---|---|---|---|---|
| D4 recipe (`cv_r5b`) | 0.694 | **0.621** | 0.17 | 0.02 | 0.19 |
| Voice-change test | 0.681 | 0.610 | 0.17 | 0.03 | 0.15 |

Per fold (clean / noisy / phone weighted F1, in-kernel):

| Fold | D4 recipe | Voice-change test |
|---|---|---|
| A | 0.540 / 0.518 / 0.566 | 0.539 / 0.496 / 0.543 |
| B | 0.611 / 0.625 / 0.653 | 0.603 / 0.620 / 0.691 |
| C | 0.644 / 0.559 / 0.584 | 0.596 / 0.566 / 0.595 |

Verdict: no improvement on the weak languages and slightly lower overall. D4's recipe stays. Signal-level
voice changes do not create new speakers; only new real voices (e.g. Common Voice Ibibio) can.

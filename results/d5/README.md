# D5: D4 averaged with a second model trained the same way

**What D5 is:** the weight average of two models, so it is still one standard `HubertForSequenceClassification`, the same size and speed as D4.
- **Model 1:** D4's raw model.
- **Model 2:** the same recipe on the same data (`folds_final`), with the same head initialisation (init seed 0) but a different data order, crops and augmentation (seed 1). Trained in Kaggle run `dsn2026-d5-soup`.

The same per-language offsets as D4 (`results/cv_r5b/offsets_final.json`) are then added to the classifier bias.

The two full models are in the same basin: weight cosine 0.99938, relative distance 0.035.

## Gate: does the average beat a single model? (held-out fold A)

Both single models and their average were trained and scored on fold A, under the same setup as the full models (`folds_cv2`, 8,000 steps). The score is the local leaderboard (public-test class and condition mix, nup from held-out VoiceAfrica speakers).

| Model | Predicted final, no offsets | **Predicted final, with D4's offsets** | Accuracy clean / noisy / phone (with offsets) |
|---|---|---|---|
| seed 0: set up exactly like D4 | 0.5363 | 0.5465 | 0.736 / 0.691 / 0.714 |
| seed 1 | 0.5567 | 0.5640 | 0.756 / 0.674 / 0.727 |
| **average (D5 recipe)** | 0.5563 | **0.5646** | 0.757 / 0.701 / 0.724 |

Fold A scores sit below the 3-fold mean because fold A is the hardest fold.

- With offsets, the average beats both single models: +0.018 over D4's equivalent, and level with the luckier seed.
- It has the best noisy-audio accuracy of the three.
- Expected gain over D4: about +0.01 final. That is the average's lead over the mean single model. The gain is not guaranteed, since only one fold was checked.

## Sanity checks of the packaged D5

- Organisers' checker: PASS (94.6M parameters, 0.189 GB, fp16). The config is byte-identical to D4's.
- **130 competition clips** (10 per class, seen in training): D5 got 128 right and D4 got 127. They agree on 129.

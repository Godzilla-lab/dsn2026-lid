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

## Real result (public leaderboard, Wed 21 Oct 11:17 UTC)

**Final 0.6078, first place** (previous leader 0.5685). Weighted F1 0.6788, efficiency 0.791 (94.6M, 0.189 GB, 1.72 GB, 0.006 s).
Accuracy clean 0.792 / noisy 0.793 / phone 0.736.

| Class | yor | ibo | hau | ego | anw | efi | ibb | idu | igl | nup | tiv | urh | other |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D5 real F1 | 0.75 | 0.92 | 0.92 | 0.96 | 0.97 | 0.28 | 0.06 | 0.83 | 0.37 | 0.73 | 0.53 | 0.99 | 0.80 |
| Leader (ray_nacido) | 0.78 | 0.85 | 0.94 | 0.96 | 0.72 | 0.24 | 0.00 | 0.89 | 0.29 | 0.59 | 0.47 | 1.00 | 0.86 |
| Local 3-fold estimate (D4 recipe) | 0.93 | 0.92 | 0.92 | 0.91 | 0.82 | 0.17 | 0.02 | 0.86 | 0.19 | 0.92 | 0.84 | 0.98 | 0.88 |

Local estimate vs real: the local leaderboard predicted a final of about 0.64 for D4's recipe and D5 scored 0.61, so
it was slightly optimistic this time (it was cautious on D1). The big misses are tiv (0.84 predicted, 0.53 real)
and yor (0.93 vs 0.75): the test's Tiv and Yoruba speakers are harder than our held-out ones. nup (0.92 vs 0.73)
was expected to be optimistic (VoiceAfrica speakers). efi, ibb and igl came out better than predicted.

## D5b: offsets corrected from the confusion matrix (12:15 UTC)

Same weights, `offsets_d5b.json` (yor -2.1→-0.8, efi 3.5→2.6, ibb 3.0→2.2, idu 0.7→0.3, tiv 0.6→1.1, nup 0→0.4).
**Final 0.6222** (D5 0.6078), weighted F1 0.6951, clean / noisy / phone 0.809 / 0.822 / 0.783.

| Class | yor | ibo | hau | ego | anw | efi | ibb | idu | igl | nup | tiv | urh | other |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D5 | 0.75 | 0.92 | 0.92 | 0.96 | 0.97 | 0.28 | 0.06 | 0.83 | 0.37 | 0.73 | 0.53 | 0.99 | 0.80 |
| D5b | 0.80 | 0.92 | 0.93 | 0.96 | 0.97 | 0.34 | 0.08 | 0.85 | 0.35 | 0.71 | 0.61 | 0.99 | 0.82 |
| ray_nacido (11:48) | 0.80 | 0.91 | 0.96 | 0.96 | 0.75 | 0.36 | 0.00 | 0.87 | 0.48 | 0.59 | 0.54 | 0.99 | 0.86 |

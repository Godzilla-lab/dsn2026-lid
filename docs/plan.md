# DSN 2026 "Find the Language I Am Speaking": deeper prompt + plan

## Context

You want the "deeper, not bigger" prompt applied to this Codabench competition: say what a top professional would do that you're not asking for, the rawest material to start from, the specific methods and tools, and what (other than you or me) would prove the result is right. Then rewrite the task as a prompt that asks for exactly that, and run it.

Hard facts pulled from the competition API, the dataset card and the live leaderboard (2026-10-20 ~08:00 UTC):

- **Score** = language-weighted F1 × (0.5 + 0.5 × efficiency). Yoruba, Igbo and Hausa are 4.3% each. The other 9 languages plus `other` are 8.7% each, so **87% of the F1 comes from low-resource classes**.
- **Limits** (breaking any one means the submission is not scored): 350M parameters, 1.5 GB repo, 4 GB peak memory, 0.5 s per clip, 30 min total. The repo must load with `AutoModelForAudioClassification`, contain no `.py`/`.bin`/`.pt` files and no `auto_map`, use fp32/fp16 weights, and have `id2label` set to exactly `yor ibo hau ego anw efi ibb idu igl nup tiv urh other`.
- **Data**: 24,774 train and 4,352 validation clips, 16 kHz, 2–30 s, columns `speaker_id` and `source` (afrispeech_ipa, hypa_voices, omnilingual_asr). Training hours: igl 0.48, nup 0.50 (**0 validation clips**), efi 1.42 (64 validation clips), ibb 1.76. The hidden test set mixes clean, noisy and phone-quality audio.
- **Prohibited**: hypaai/Hypa-Voices, facebook/omnilingual-asr-corpus, AfriSpeech/african-speech-ipa (the test set was drawn from these). Every external dataset must be listed in the model card.
- **Timeline**: the dev phase ends **Oct 21 23:59 UTC** (4:59 PM PDT). The final phase runs **Oct 22 00:00–08:00 UTC** on the private split. Rules allow 3 submissions/day (20 dev, 5 final). That leaves about 6 dev submissions, so each one has to answer a question.
- **Leaderboard now**: the leader (datapsalm) scores 0.5179 final with wF1 0.549, which is below the 0.60 ranking floor, so they are unranked. Their model has 20.2M params, uses 1.19 GB memory and takes 0.005 s/clip. Their F1 is 0.08 on efi, 0.06 on ibb, 0.085 on igl and 0.36 on tiv. The organiser's 316M baseline scores F1 0.01.
- **Efficiency maths** (my formula reproduces the organisers' 0.76 example):

  | Model size | Score multiplier |
  |---|---|
  | ~21M | ≈0.94 |
  | 94M fp16 | ≈0.89 |
  | 66M (truncated) | ≈0.92 |
  | 316M | ≈0.80 |

  A 20M model still needs 1.19 GB, so memory has a fixed ~1 GB floor. Speed is already saturated for small models. **Model size is the only real efficiency lever.** A 95M model must beat a 20M model's F1 by more than 6% relative to be worth it; a 316M model must beat it by more than 18%.

---

## Part 1: Your prompt, applied

### (1) What a top professional would do that you're not asking for

1. **Optimise the actual metric, not accuracy.** The metric is fixed-share F1 times an efficiency multiplier. Efik, Ibibio, Igala and Nupe alone are 35% of the F1, and the current leader scores about 0 on three of them. A pro builds a local copy of the exact score, efficiency included, before training anything.
2. **Distrust the given validation split.** Nupe has none, and Efik and Igala have under 65 clips each. Use speaker-grouped K-fold over train+val so every class, including Nupe, gets a real held-out estimate.
3. **Audit for shortcuts.** Cross-tabulate label × source × speaker. If a language comes from only one corpus, the model may learn the microphone or channel rather than the language. That breaks on the test set's noisy and phone audio, which the leaderboard reports separately (`acc_noisy`, `acc_phone`).
4. **Rebuild the test conditions locally.** Make clean, noisy and phone-codec copies of validation so the local estimate predicts all three leaderboard accuracy columns.
5. **Use the rules creatively, within the rules.**
   - Tune per-class decision offsets on validation and **bake them into `classifier.bias`**. This is still a standard model with no custom code.
   - `use_weighted_layer_sum=True` and `num_hidden_layers=k` are native config flags. They give layer truncation and a better language-ID signal for free.
   - Averaging the weights of several fine-tuned checkpoints still produces one model.
6. **Get an independent teacher.** `facebook/mms-lid-4017` (970M params, CC-BY-NC) is too big to submit, but its label set covers **all 12 targets plus pcm and eng**. Use it to audit training labels, as a zero-shot reference, and as a distillation teacher and filter for external audio.
7. **Treat the submission as an engineering artifact.** Pin the `transformers` version, load the repo with network access off in a clean environment, test fp16 vs fp32 memory, and run the organisers' checker before spending a submission slot.
8. **Spend submissions like experiments**, and pick final-phase entries by grouped cross-validation (CV), not by the public split. The public split is small, and the private split decides.

### (2) The rawest material to start from (instead of the digested version)

- The **parquet audio plus `speaker_id`, `source` and `duration` per clip**, not the README's hours table.
- Actually **listening to 10 clips per class**, especially anw/efi/ibb, `other`, and the shortest and longest clips.
- The **leaderboard's per-class F1 and clean/noisy/phone columns**: the organisers' own measurement of where models fail.
- The **starting kit's `check_model_folder.py`**, which encodes the organisers' real rules. Also the exact scoring formula from the Evaluation page.
- **Pretrained encoders** chosen by language coverage:
  - `utter-project/mHuBERT-147` (94M, CC-BY-NC-SA)
  - `ajesujoba/AfriHuBERT` (95M, continued on 1,226 African languages; tags include ibb, anw, idu, igl, nup, tiv, urh, ego)
  - `openai/whisper-base` encoder (~21M, Apache-2.0)
- For the "edge others won't have": data that is **openly licensed or used with permission**, not scraped.
  - FLEURS (ha/ig/yo, plus many languages for `other`) and other open corpora where licences allow.
  - Teacher-labelled audio.
  - Test-condition augmentation: MUSAN-style noise, room impulses, 8 kHz telephone band and codecs.
  - For efi/ibb/igl/nup, sources such as Global Recordings or Bible audio **only after a licence check and a forum OK**. Scraping radio or YouTube without permission breaks the "ethically sourced / with permission" rule and risks disqualification.

### (3) Specific methods and tools

- **Audit**: DuckDB over `hf://datasets/.../*.parquet` and pandas: label × source × speaker hours, a duration histogram per class, near-duplicate checks.
- **Validation**: `StratifiedGroupKFold` (scikit-learn) by `speaker_id`. Run a local `score.py` that reproduces the weighted F1, efficiency and multiplier. Score each fold on clean, noisy and phone copies.
- **Backbone bake-off** with the same budget for each (4–6 s random crops, 1 short run): whisper-base encoder, mHuBERT-147 and AfriHuBERT, plus layer truncation (8/10/12 layers). Pick by **estimated final score**, not F1.
- **Fine-tuning**:
  - HF `Trainer` on a Kaggle GPU, with a frozen CNN feature encoder (`freeze_feature_encoder`).
  - Class-balanced sampling plus class-weighted CE (cross-entropy loss), and built-in time masking (`mask_time_prob`).
  - Augmentation with `audiomentations`/`torchaudio`: noise, room impulses, telephone band-pass, speed perturbation.
  - Distillation from the MMS-LID-4017 soft labels restricted to the 13 classes, with `other` taken as 1 minus the sum of the 12 target probabilities.
- **Post-hoc**: per-class bias search on out-of-fold predictions, written into `classifier.bias`. Checkpoint weight averaging. fp16 `save_pretrained(safe_serialization=True)`.
- **Efficiency harness**: parameter count, repo size, peak RAM and GPU memory, median latency over 3 passes after warm-up, on 2–30 s clips.
- **Packaging**: model card listing every dataset and licence. Run `check_model_folder.py`, push to the Hub, then run `make_submission.py` with the 40-character commit hash.

### (4) What (other than you or me) shows the result is right

- **The public leaderboard's per-class F1 and clean/noisy/phone columns agree with the local grouped-CV estimate** (within about 0.03). That proves the validation is trustworthy, so the private split should behave the same.
- **The organisers' `check_model_folder.py` passes**, and the repo **loads and runs in a clean, offline environment** pinned to the organisers' Python/transformers versions.
- **Scoring above the current leader (0.5179) and above the 0.60 F1 ranking floor**, with efi, ibb and igl F1 well above 0.1.
- **An independent model (MMS-LID-4017) agrees** with the labels on spot-checked clips. **A native Efik, Ibibio or Annang speaker** checks a sample of confusions.
- **The organisers' forum answers** confirm every external source before it is used.
- Finally, the **private leaderboard** on Oct 22.

---

## Part 2: Your prompt, rewritten

```
Build my submission for DSN 2026 "Find the Language I Am Speaking" (Codabench 18424).
Goal: maximise FINAL score on the PRIVATE test split = language-weighted F1 × (0.5 + 0.5 × efficiency),
not accuracy. Deadline: dev phase ends Oct 21 23:59 UTC; final picks by Oct 22 08:00 UTC.
Depth over breadth: one excellent model, not many mediocre ones.

Start from raw material, not summaries:
- the training parquet audio with speaker_id/source/duration per clip; listen to samples per class
- the Evaluation page formula and limits; the starting kit's check_model_folder.py
- the live leaderboard's per-class F1 and clean/noisy/phone columns

Before training anything:
1. Audit label × source × speaker hours and flag any source/channel shortcuts.
2. Build speaker-grouped K-fold CV over train+val (Nupe has no validation clips) plus
   noisy and phone-codec copies of each fold, and a local score.py that reproduces the exact
   final score, efficiency and multiplier included.
3. Get a zero-shot reference and a label audit from facebook/mms-lid-4017.

Then:
4. Run a fixed-budget bake-off of whisper-base encoder, mHuBERT-147 and AfriHuBERT, with layer truncation;
   choose by estimated final score.
5. Fine-tune the winner: balanced sampling, class-weighted loss, noise/reverb/telephone
   augmentation, optional distillation from MMS-LID. External data only if openly licensed or
   permitted, never from the three prohibited corpora, all listed in the model card.
6. Tune per-class biases on out-of-fold predictions and write them into classifier.bias;
   average checkpoints; export fp16 safetensors; measure params, repo size, peak memory and latency.
7. Pass check_model_folder.py, load the repo with network access off in a pinned clean env, push to HF,
   build submission.zip with the full commit hash.

Treat each leaderboard submission as an experiment that answers one stated question.
Prove it's right with evidence outside us: the leaderboard per-class/condition numbers must
match the local CV within ~0.03, the organisers' checker passes, F1 ≥ 0.60 with efi/ibb/igl
above 0.1, and it beats the current leader (0.5179). Report where it is still weak and why.
```

---

## Part 3: Access check (your question)

| Need | Status |
|---|---|
| Competition pages, rules, live leaderboard | ✅ via your secret link (API) |
| Public HF models (mHuBERT, AfriHuBERT, Whisper, MMS-LID) | ✅ |
| Push model to your HF account (`godzilla1215`) | ✅ token has `repo.write` |
| **Gated training data** | ❌ **403.** Fix: click "Agree" on the dataset page, **and** edit your fine-grained token to allow "read access to contents of all public gated repos you can access" (or give me a new token) |
| Kaggle API (GPU training) | ✅ the API authenticates through the proxy. GPU needs a phone-verified account; we find out on the first push. The Kaggle MCP tools say "Unauthenticated", but the API works without them |
| **Starting kit** (`check_model_folder.py`, `make_submission.py`) | ❌ behind your Codabench login. Download it from Participate → Files and drop it here |
| Codabench upload, forum posts | ❌ your account. I produce `submission.zip` and draft the forum questions; you upload and post |
| This container | 4 CPU, 15 GB RAM, no GPU: enough for the audit, scoring and packaging, not for training |

**On multiple Kaggle accounts:** I won't set those up. Kaggle allows one account per person, and extra accounts to stretch GPU quota risk bans, including on your main account. The legitimate way to get parallel GPUs is to have **teammates** (the Codabench team limit is 3) each run on their **own** Kaggle account. I'd split the backbone bake-off across them and keep the best ("use the winner"). With one account, the plan still fits: the bake-off is short, and only the winner gets a long run.

**On "data others won't get":** yes, that is a real edge, but it has to be legitimate:
- teacher distillation from MMS-LID-4017
- augmentation that matches the test conditions
- open corpora
- permissioned sources for efi/ibb/igl/nup, cleared on the forum first

Undisclosed or scraped audio risks disqualification.

---

## Part 4: Execution plan (after approval, once the token and starting kit are in)

Times are UTC.

1. **Oct 20, ~2 h, this container**
   - pip install duckdb, pandas, scikit-learn, soundfile
   - pull the metadata and audio
   - audit tables plus sample listening notes
   - grouped folds (Nupe included)
   - `score.py` reproducing the metric
   - noisy and phone validation copies
   - draft forum questions about external sources (AfriHuBERT's pretraining data, Global Recordings, FLEURS)
2. **Oct 20, ~4–6 h, Kaggle GPU**
   - MMS-LID-4017 zero-shot on validation (reference plus label audit)
   - bake-off of whisper-base encoder / mHuBERT-147 / AfriHuBERT at truncated depths
   - **Submission 1** of the best quick model, to prove the pipeline end to end and calibrate local CV against the leaderboard
3. **Oct 20 night → Oct 21, Kaggle GPU**
   - full fine-tune of the winner with balanced sampling, augmentation and optional distillation
   - bias tuning, checkpoint averaging, fp16 export
   - **Submissions 2–6** (3/day), each testing one question: augmentation, depth, distillation, bias offsets
4. **Oct 22 00:00–08:00**
   - submit the top 1–3 by grouped CV in the final phase
   - repos frozen at their commit hashes

Each submission goes through the same packaging step: model card with datasets and licences (mHuBERT-147 is NC-SA, so the derived model inherits that licence), checker, offline load test, HF push, `submission.zip`.

### Training ladder: start small, step up gradually (your rule)

Training starts small and moves up **one rung at a time**. Each rung changes one thing, so we know what helped. We only move up when the rung below has passed its gate: the local grouped-CV final score improved, the model is inside every limit, and the packaging checks passed.

**The top rung is only run when you explicitly say so.** I won't jump there on my own, even if time is short; I'll ask first.

| Rung | What changes | Model | Data | Leaderboard? |
|---|---|---|---|---|
| 0. Smoke test | Pipeline only: train, save, checker, offline load, local score | Smallest (AfriHuBERT 4 layers, ~38M) | ~200 clips per class, 1 short run | No |
| 1. Small baseline | Standard recipe + speaker-balanced sampling, 3–6 s crops | AfriHuBERT 4 layers | Provided training data only | Yes: first submission, calibrates local CV against the leaderboard |
| 2. Better loss and decisions | Balanced Softmax / logit-adjusted loss + per-class bias tuning | Same | Same | If CV improves |
| 3. Robustness | Noise, room-echo, phone-codec, speed and formant augmentation | Same | Same | If CV improves |
| 4. More data | Cleared external data (VoiceAfrica igl/nup/urh, Efik set, "other" sets; your voice notes when in) | Same | + external | If CV improves |
| 5. New voices / teacher | kNN-VC shared voice pool on all classes, and/or distillation from MMS-LID | Same | Same | If CV improves |
| 6. Depth | 6 then 8 layers, only if the score maths says it pays | AfriHuBERT 6–8 layers | Same | If CV improves |
| **Top (only on your go)** | Largest allowed setup, longest training, all proven techniques together | Bigger backbone, e.g. AfriHuBERT-large or w2v-BERT, truncated | Everything cleared | Only when you say |

On the leaderboard, this also means our public score climbs in steps rather than showing our strongest model early. The strongest version is kept for the final phase.

## Verification

- `score.py` reproduces the organisers' worked example (0.76) and the leader's numbers from their published sub-scores.
- Every repo passes `check_model_folder.py` and an offline `AutoModelForAudioClassification.from_pretrained(repo, revision=hash)` load in a pinned clean env, staying inside every limit with margin.
- After each submission, compare the leaderboard's per-class F1 and clean/noisy/phone accuracy with the local CV. If they diverge by more than ~0.03, fix the validation before tuning further.

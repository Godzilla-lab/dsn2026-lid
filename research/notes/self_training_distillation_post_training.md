# Semi-supervised, distillation, compression and post-training techniques for a small spoken-LID model (DSN 2026 "Find the Language I Am Speaking", Codabench 18424)

Scope note: this file covers distillation, self-training, continued pretraining, HF-compatible compression, post-hoc calibration/bias tuning, imbalance losses, weight averaging/SAM and model selection. Facts checked in this session are cited. Things I worked out from code or arithmetic are marked **[code-verified]** or **[computed]**. Speculative items are marked **[uncertain]**.

---

## Objective: ranked techniques (expected gain × hours to implement)

### Takeaway
Ranked by expected gain per hour, the best options are:
1. Tune per-class logit offsets for the fixed-share F1 on speaker-grouped out-of-fold (OOF) predictions and fold them into `classifier.bias`. This takes about 1 hour and has no efficiency cost.
2. Replace class-weighted CE with a logit-adjusted or balanced-softmax loss. About 0.5 hours.
3. Distil the 13-way-restricted MMS-LID-4017 teacher into a student of 50M parameters or fewer, built from a truncated multilingual HuBERT/wav2vec2 or the Whisper-base encoder. About 4–8 hours. Its gain depends on how well the teacher scores zero-shot on your validation set, so measure that first.

Weight averaging and fp16 export are cheap extras. SAM, pseudo-labelling, uniform FFN pruning and continued SSL pretraining (TAPT) come after them. Attention-head pruning is not available for wav2vec2/HuBERT in HF.

### Cited Findings
- The teacher, `facebook/mms-lid-4017`, is a standard `Wav2Vec2ForSequenceClassification`: 48 layers, hidden size 1280, FFN 5120, 16 heads, `classifier_proj_size` 1024, 970,077,745 fp32 parameters and 4017 labels. All 14 needed ISO codes are in its `id2label`: yor=35, ibo=1115, hau=21, ego=1265, anw=3342, efi=1336, ibb=2413, idu=1258, igl=1344, nup=1252, tiv=1187, urh=1262, pcm=705, eng=2. **[code-verified, from config.json and the HF API]** — [HF config](https://huggingface.co/facebook/mms-lid-4017/resolve/main/config.json), [HF model page](https://huggingface.co/facebook/mms-lid-4017)
- Post-hoc logit adjustment (predict argmax of f_y(x) − τ·log π_y) lowered balanced error versus ERM (Table 3, ResNet-32/50):

  | Dataset | ERM | Post-hoc LA, τ=1 | Post-hoc LA, tuned τ | LA loss, τ=1 |
  |---|---|---|---|---|
  | CIFAR-10-LT | 27.16 | 22.60 | 19.08 | 22.33 |
  | CIFAR-100-LT | 61.64 | 58.24 | 57.90 | 56.11 |
  | ImageNet-LT | 53.11 | 49.66 | 49.56 | 48.89 |
  | iNaturalist | 38.66 | 33.98 | 33.80 | 33.64 |

  Code: github.com/google-research/google-research/tree/master/logit_adjustment — [Menon et al., ICLR 2021, arXiv:2007.07314](https://arxiv.org/abs/2007.07314)
- Class-balanced re-weighting (CBW), the "class weights" part of the standard recipe, did worse than plain CE on CIFAR-100-LT: 42.3 vs 45.3 top-1 at imbalance 100. Balanced Softmax reached 50.3, and BALMS (Balanced Meta-Softmax) reached 50.8 — [Ren et al., NeurIPS 2020, arXiv:2007.10740](https://ar5iv.labs.arxiv.org/html/2007.10740)
- The efficiency breakpoints in the brief fit a simple parameter model: a 12-layer d=768 encoder is about 94.2M, and a 24-layer d=1024 encoder about 315.2M. These match the "94M" and "316M" examples. **[computed]** (see the compression section)

### Inferences
**Score arithmetic** **[computed from the brief's formula]**
- Each extra 10M parameters exported in fp16 costs about 0.0035 of the multiplier through the size terms alone:
  - The params term loses 10/350 ≈ 0.029.
  - The repo term loses 20 MB/1.5 GB ≈ 0.013.
  - Their mean (≈ 0.021) is divided by 3, then halved.

  Speed and memory add more on top of that.
- Going from the 94M model (×0.89) to the 20M model (×0.94) multiplies the score by about 1.056. A ~20M student therefore beats a 94M model whenever it keeps more than about 94.7% of its F1, i.e. it may lose up to ~5.3% relative (e.g. 0.85 → 0.805).
- Going from 316M (×0.80) to 94M (×0.89) allows an ~11% relative F1 loss before you break even.
- One F1 point on any 8.7%-weight class (efi/ibb/igl/nup/anw/ego/idu/tiv/urh/other) is worth twice a point on yor/ibo/hau. Optimise rare-class F1 first, including its precision: false positives into a rare class lower that class's F1.

**Ranked list** (gain estimates are my inferences, sized from the effect sizes cited in the sections below)

| # | Technique | Hours (Kaggle T4/P100) | Expected gain on final score | Stays standard HF because… |
|---|---|---|---|---|
| 1 | Per-class bias offsets for fixed-share F1 (logit adjustment τ plus coordinate ascent on OOF), added to `classifier.bias` | ~1 | +1 to +4 F1 points on imbalanced, rare-class-heavy metrics; zero efficiency cost | Only the existing bias vector changes |
| 2 | Logit-adjusted CE / Balanced Softmax instead of class-weighted CE | ~0.5 | +0.5 to +3 points (literature: 3–5 points over CE on long-tailed vision; the gap is likely smaller with SSL features) **[uncertain]** | Training-time only |
| 3 | KD from MMS-LID-4017 restricted to 13 classes (store teacher logits offline), soft + hard loss, into a ≤50M student | 4–8 (teacher pass about 0.5–1.5 h on a T4 **[uncertain estimate]**) | Potentially the largest gain, on rare classes with 1–3 training speakers, **if** the teacher is good zero-shot on held-out speakers; also allows a smaller model, which raises the multiplier | Student is a plain `HubertForSequenceClassification` / `Wav2Vec2ForSequenceClassification` / `WhisperForAudioClassification` |
| 4 | Truncated multilingual SSL student (AfriHuBERT / mHuBERT-147 first 6–8 layers, or MMS-300m first 6–8 layers) with `use_weighted_layer_sum=True` | 1–2 per run | Lets a 50–65M model approach the full 94M/316M model; language information peaks in middle layers | `num_hidden_layers=N` is a config field |
| 5 | Weight averaging (EMA, or average of the last-K checkpoints / seeds from one init) | ~0.5 | 0 to +1 point; free at inference | State-dict average, same architecture |
| 6 | fp16 export | ~0.2 | Multiplier +~0.01 (repo term and memory) **[uncertain size of effect]** | `model.half(); save_pretrained(safe_serialization=True)` |
| 7 | SAM (or ImbSAM / high-ρ on tail) | ~0.5 implementation; ~2× training time | 0 to +2 points on minority classes | Training-time only |
| 8 | Pseudo-labelling of extra unlabelled or weakly labelled audio (e.g. extra "other"-class audio such as English/Pidgin), filtered by teacher confidence | 4–10, dominated by data gathering | Uncertain; strong evidence in ASR, little direct evidence for LID | Training-time only |
| 9 | Uniform FFN-width pruning (same `intermediate_size` in every layer, different neurons kept per layer) plus KD recovery | 3–6 | Uncertain; saves up to ~28M parameters on a 12-layer base | `intermediate_size` is one config field **[code-verified]** |
| 10 | TAPT / continued SSL pretraining | Prohibitive at published scale (AfriHuBERT: 100K steps on 4×A100); a short run is possible but unproven | Low within 30 h | Base weights only |
| ✗ | Attention-head pruning | — | Not possible | wav2vec2/HuBERT do not implement `_prune_heads`, and transformers v5 removed head pruning **[code-verified]** |

### Gaps
- I found no published number for MMS-LID-4017 accuracy on Efik, Ibibio, Igala, Nupe, Idoma, Urhobo, Eggon, Anaang or Tiv. Measure the teacher's zero-shot macro-F1 on your speaker-held-out validation set before investing in KD.
- How the speed term behaves on the organiser's hardware, and whether preprocessing (e.g. Whisper log-mel extraction) counts towards "per clip" time, is unknown.

---

## Knowledge distillation for LID / audio classification (MMS-LID teacher → small student; layer-to-layer)

### Takeaway
Logit KD from a large teacher is well established. The published LID-specific KD gains are modest, but they come from settings where teacher and student share the same data (long- vs short-utterance teachers). Your case is different: the teacher has seen far more speakers of the rare languages than your 1–3 training speakers, so KD can carry knowledge your labels do not contain. The main open question is teacher quality on the target domain.

A strong, cheap recipe:
- Cache restricted 13-way teacher logits on the same crops the student sees ("consistent" KD).
- Train with T≈2–4 and α≈0.5–0.9, combining soft KL with a logit-adjusted hard CE.
- Use a 6–8-layer multilingual HuBERT/wav2vec2 or a Whisper-base encoder as the student.

DistilHuBERT-style layer-to-layer distillation is a task-agnostic pretraining method. It is too slow to redo from scratch, but the released 2-layer DistilHuBERT (23.5M) loads directly as a standard HF HuBERT.

### Cited Findings

**LID-specific KD results**
- Interspeech 2025 TF-KD paper (Dey et al.), ECAPA-TDNN on the 10 most common VoxLingua107 languages, 2 s test segments:
  - Conventional KD from a 4 s-trained teacher barely moved EER (9.08 → 9.07) and moved Cavg from 10.06 to 9.47.
  - KD-RL (representation-level KD, Shen et al.) gave Cavg 9.18.
  - Their teacher-free KD (online label smoothing from correctly classified segments of the previous epoch) reached EER 8.94 / Cavg 8.24.
  - Cross-corpus (Vox-10 → Common Voice), Cavg went 20.40 → 16.22.
  - They also report that 36.94% of misclassifications came from out-of-scope segments: non-speech, named entities and filler words.

  — [Dey et al., Interspeech 2025](https://www.isca-archive.org/interspeech_2025/dey25_interspeech.pdf)
- Earlier LID KD work by Shen, Lu, Li and Kawai: interactive teacher-student learning (ICASSP 2019) and KD-based representation learning (IEEE/ACM TASLP 2020), with long-utterance teachers and short-utterance students — cited in [Dey et al., Interspeech 2025](https://www.isca-archive.org/interspeech_2025/dey25_interspeech.pdf)

**Audio-tagging KD recipe (CED)**
- CED stores teacher logits and augmentation parameters on disk, at about 0.3% extra disk space for AudioSet, so the student trains on exactly the augmented view the teacher scored.
- Its students reach mAP on AudioSet-2M of:
  - CED-Tiny, 5.5M: 48.1
  - CED-Mini, 9.6M: 49.0
  - CED-Small, 22M: 49.6
  - CED-Base, 86M: 50.0
- Code: github.com/richermans/ced.
- The CED models themselves are not standard HF wav2vec2/HuBERT classes; use the recipe, not the checkpoints.

— [Dinkel et al., ICASSP 2024, arXiv:2308.11957](https://arxiv.org/pdf/2308.11957); [CED GitHub](https://github.com/richermans/ced)

**General KD**
- "A good teacher is patient and consistent" makes the implicit design choices of KD explicit (consistent teacher/student views, long schedules). It reports an 82.8% top-1 ImageNet ResNet-50 distilled from a large teacher — [Beyer et al., CVPR 2022, arXiv:2106.05237](https://arxiv.org/abs/2106.05237)

**Layer-to-layer distillation**
- DistilHuBERT reduces HuBERT's size by 75% and is 73% faster, while keeping most of its performance on ten SUPERB tasks — [Chang, Yang, Lee, ICASSP 2022, arXiv:2110.01900](https://arxiv.org/abs/2110.01900)
- FitHuBERT (thinner and deeper student, hint-based distillation, time reduction) reaches 23.8% of HuBERT's size and 35.9% of its inference time. Its gains are mostly on content tasks (phoneme recognition, ASR), where layer-dropping students such as DistilHuBERT degrade — [Lee et al., Interspeech 2022, arXiv:2207.00555](https://arxiv.org/pdf/2207.00555)
- The released `ntu-spml/distilhubert` is a standard HF HuBERT: `model_type: hubert`, 2 layers, hidden 768, FFN 3072, 23,491,968 parameters, Apache-2.0. It loads with `AutoModelForAudioClassification` as `HubertForSequenceClassification`. **[code-verified via HF API/config]** — [HF config](https://huggingface.co/ntu-spml/distilhubert)
- DPHuBERT (joint distillation and structured pruning of HuBERT) gives a 23,585,946-parameter model that "outperforms pure distillation methods in most SUPERB tasks". The released models are TorchAudio-style, loaded with the repo's own `wav2vec2_model(**ckpt["config"])`, not HF classes — [Peng et al., Interspeech 2023, arXiv:2305.17651](https://arxiv.org/abs/2305.17651); [DPHuBERT GitHub](https://github.com/pyf98/DPHuBERT)
- CoLLD (contrastive layer-to-layer distillation) compresses multilingual speech encoders and evaluates on ML-SUPERB. I did not retrieve its LID numbers — [arXiv:2309.07707](https://arxiv.org/pdf/2309.07707)

**Whisper-encoder students** **[code-verified]**
- In transformers 5.20.2, `WhisperForAudioClassification`:
  - is in the `AutoModelForAudioClassification` mapping;
  - supports `use_weighted_layer_sum`;
  - requires mel inputs of exactly `max_source_positions × conv strides` frames (3000 = 30 s), and raises an error otherwise;
  - mean-pools over all 1500 encoder frames with no attention mask, so a 2 s clip is pooled mostly over padding.

  — [transformers source, modeling_whisper.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py)
- Encoder configs:
  - `openai/whisper-base`: encoder_layers=6, d_model=512, FFN 2048.
  - `openai/whisper-tiny`: encoder_layers=4, d_model=384, FFN 1536.

  — [whisper-base config](https://huggingface.co/openai/whisper-base), [whisper-tiny config](https://huggingface.co/openai/whisper-tiny)

### Inferences

**Restricting the teacher to 13 classes** **[code-verified indices; recipe is my design]**
```python
TGT = ['yor','ibo','hau','ego','anw','efi','ibb','idu','igl','nup','tiv','urh']
idx = torch.tensor([teacher.config.label2id[c] for c in TGT])
other = torch.ones(4017, dtype=torch.bool); other[idx] = False
z = teacher(x).logits.float()                      # fp16 forward on a T4 is fine
z13 = torch.cat([z[:, idx], torch.logsumexp(z[:, other], -1, keepdim=True)], -1)
```
- Softmax over `z13` gives p(class | class ∈ the 13 classes). The `logsumexp` for "other" adds up all 4005 non-target languages, including eng and pcm.
- If the competition's "other" clips are mostly English/Pidgin, compare this against `logsumexp` over {eng, pcm} only. Pick whichever gives the better teacher F1 on validation.

**Fix the teacher's bias before distilling**
- Apply the same per-class offset tuning (next sections) to `z13` on your labelled training/validation data. MMS-LID's priors differ from the competition's.
- Optionally build an ensemble teacher: a weighted logit average of restricted MMS-LID and your best fine-tuned 300M model(s), with weights tuned on speaker-grouped OOF.

**Caching teacher logits** **[uncertain throughput]**
- Run the 1B teacher once over fixed windows of every training clip (e.g. 4–8 s windows with a 2–4 s hop, plus the whole clip) and store `z13`.
- The student samples only those windows, so its crops are exactly the ones the teacher scored (the "consistent" KD of Beyer and CED).
- Waveform augmentations such as speed or noise break that consistency. Either accept some mismatch, or score a few fixed augmentation seeds per window and store their indices.
- Rough cost: about 1B parameters × 50 frames/s ≈ 100 GFLOP per audio-second. 95 h of audio is about 3.4e5 s, so the pass is plausibly 0.5–1.5 h on a T4 in fp16.

**Student loss**
```
L = α·T²·KL( softmax(z_t/T) || softmax(z_s/T) ) + (1−α)·CE( z_s + τ·log π , y )
```
- The second term is the logit-adjusted CE (see the losses section).
- Start with T=2, α=0.7, then sweep T ∈ {1, 2, 4} and α ∈ {0.5, 0.9} on OOF.
- Fall back to α≈0.3 if the teacher's zero-shot F1 on a class is below the student's.

**Student choice** (all `AutoModelForAudioClassification`; parameter counts from my formula)

| Student | Params | Notes |
|---|---|---|
| AfriHuBERT / mHuBERT-147, first 6 layers | ≈ 51.6M | Multilingual, African-adapted |
| Same, first 4 layers | ≈ 37.5M | |
| DistilHuBERT | 23.5M | English-only pretraining **[uncertain LID quality for Nigerian languages]** |
| Whisper-base encoder | ≈ 20M | Multilingual, LID-trained during Whisper pretraining |
| Whisper-tiny encoder | ≈ 8M | |

- The Whisper students need a fix for padding dilution. One option is to always crop or tile training and test audio to fill 30 s. Another is to set `max_source_positions=750` (15 s), slice the `embed_positions` weights and set the feature extractor's `chunk_length=15`. **[uncertain]** That second option depends on the grader using the saved `preprocessor_config.json`, so treat it as risky.
- A Whisper student could start from Whisper-small fine-tuned weights, but Whisper-small's ~88M encoder **[uncertain count]** gives up most of the size advantage.

**Layer-to-layer KD (DistilHuBERT style) for a task-specific student**
- While fine-tuning a truncated student, add an L1 + cosine loss between the student's last hidden state and a frozen teacher layer. The teacher here is a fine-tuned 12-layer AfriHuBERT/mHuBERT, and its layer ≈8 is the natural target given that language information peaks mid-network.
- The prediction heads are discarded at export, so the saved model stays standard.
- I expect this to be a second-order gain over logit KD within a 1-day budget.

**Ensemble → single-model distillation**
- An ensemble cannot be submitted, but it can be distilled.
- Train K fold models, average their logits as the teacher (plus MMS-LID if it helps), and distil into the final small model trained on all data.

### Gaps
- I found no paper that distils MMS-LID (any size) into a small LID student, and no wav2vec2/HuBERT → small-student LID KD result with numbers.
- I did not retrieve ML-SUPERB LID scores for DistilHuBERT/FitHuBERT/CoLLD students.
- I did not verify whether Whisper's training languages include Igbo or the minority Nigerian languages.
- The MMS paper's description of its LID training data (reportedly largely religious-domain recordings) was not re-checked this session — [MMS, arXiv:2305.13516](https://arxiv.org/abs/2305.13516). If the teacher has a domain bias, the competition data may differ in domain.

---

## Self-training / noisy student / pseudo-labelling with confidence filtering

### Takeaway
The evidence is strong for ASR and indirect for LID: I found no paper applying noisy-student training to spoken LID. Within a day, use pseudo-labelling only where unlabelled in-domain audio is cheap. The most promising case is extra "other"-class audio (English, Pidgin, other languages) filtered by the MMS teacher. Otherwise, treat the MMS-LID teacher's soft labels on the existing training crops as the "pseudo-labels" (that is just KD).

Labelling test clips (transductive use) is only valid if the rules allow it. That needs checking.

### Cited Findings
- Improved Noisy Student Training for ASR uses (adaptive) SpecAugment, plus filtering, balancing and augmentation of the generated data between iterations. Results (LibriSpeech clean/other WER):

  | Setting | Prior SOTA | NST |
  |---|---|---|
  | 100h labelled + 860h unlabelled | 4.74 / 12.20 | 4.2 / 8.6 |
  | 960h + LibriLight 60k | 1.9 / 4.1 | 1.7 / 3.4 |

  — [Park et al., Interspeech 2020, arXiv:2005.09629](https://arxiv.org/abs/2005.09629)
- Self-training and wav2vec 2.0 pretraining are complementary. With 10 minutes of labelled data the combination reaches 3.0 / 5.2 WER; with all labelled LibriSpeech, 1.5 / 3.1 — [Xu et al., arXiv:2010.11430](https://arxiv.org/abs/2010.11430)
- Continued pretraining (CoPT) of XLSR-53 on unlabelled in-language audio gives WER "equal to or slightly better than" pseudo-label semi-supervised training (SST), at lower compute (no pseudo-labelling). Pseudo-labels from the CoPT model give further gains — [DeHaven & Billa, arXiv:2207.00659](https://arxiv.org/abs/2207.00659)
- In federated audio self-training, a cosine-scheduled dynamic confidence threshold is used to discard low-confidence pseudo-labels — [arXiv:2107.06877](https://arxiv.org/pdf/2107.06877)
- In semi-supervised speaker recognition (SpeakerMatch), fixed-threshold selection is reported to separate reliable from unreliable pseudo-labels poorly, so the authors model the confidence distribution instead — [SpeakerMatch, Signal Processing 2025](https://www.sciencedirect.com/science/article/abs/pii/S0165168425003779)
- A teacher-student, few-shot prototype-alignment method (ProtoAlign) targets cross-domain spoken language recognition with minimal target supervision. It is the closest LID-specific semi-supervised work found — [ResearchGate](https://www.researchgate.net/publication/399712276_Bridging_Domain_Gaps_with_ProtoAlign_Teacher-Student_Few-Shot_Prototype_Alignment_for_Cross-Domain_Spoken_Language_Recognition)

### Inferences

**Recipe within the budget**
1. Gather unlabelled or weakly labelled audio. Easiest: English/Pidgin/other-language clips for "other" from permissively licensed corpora; any Nigerian-language radio or speech you are allowed to use.
2. Score it with the restricted MMS-LID teacher, ensembled with your fine-tuned model.
3. Keep clips with ensemble max-probability ≥ 0.9 **and** agreement between teacher and student.
4. Cap pseudo-labelled data per class at about 1–2× that class's real data, to avoid swamping the rare classes (the NST "balancing").
5. Train the student on real + pseudo data with soft targets and strong augmentation (speed, noise, SpecAugment-style masking); this is the "noisy student".

- One iteration is realistic in the time available.
- For classes with 1–3 speakers, any genuinely new speaker in correctly pseudo-labelled audio is worth more than more hours from the same speakers. This is a speaker-diversity argument, not a published result.

### Gaps
- No LID-specific noisy-student or pseudo-labelling effect sizes found.
- Unknown whether the competition rules allow external data or pseudo-labelling of test audio. Check the Codabench rules page.

---

## Task-adaptive continued SSL pretraining (TAPT) before fine-tuning

### Takeaway
Continued pretraining helps African-language LID: AfriHuBERT gains +3.6 F1 over mHuBERT-147. But it took 100K steps on 4×A100 over more than 10K hours of audio, which is out of reach in a day on a T4. Use the released AfriHuBERT checkpoint (a standard 95M HF HuBERT) as the TAPT shortcut instead of running TAPT yourself. A short wav2vec2-contrastive TAPT on the 95 h is possible in HF but unproven at that scale, so rank it last.

### Cited Findings
- AfriHuBERT (v2, Interspeech 2025):
  - Continued pretraining of mHuBERT-147 (95M) on more than 10K hours covering 1,226 African languages plus Arabic, English, French and Portuguese.
  - Multilingual adaptive fine-tuning with one iteration and temperature sampling α=0.8.
  - 100K steps, 4×A100-40GB, learning rate 5e-5, 32K warm-up steps.
  - New k-means targets from layer 9 of mHuBERT-147, built with up to 1 h of speech per language.
  - FLEURS SLID F1: 88.0 → 91.6 (25-language average) and 85.8 → 90.0 (21 African languages). For comparison, MMS scores 86.3 and XEUS 96.2.
  - Release: HF `ajesujoba/AfriHuBERT`, GitHub `nii-yamagishilab/AfriHuBERT`.

  — [AfriHuBERT, arXiv:2409.20201v2](https://arxiv.org/html/2409.20201v2); [Interspeech 2025](https://www.isca-archive.org/interspeech_2025/alabi25_interspeech.html)
- The AfriHuBERT v1 preprint reported 39 languages, 6,500+ hours, and +4% F1 for LID — [arXiv:2409.20201v1](https://arxiv.org/html/2409.20201v1)
- Checkpoint configs **[code-verified via HF config]**:
  - `ajesujoba/AfriHuBERT`: `model_type: hubert`, 12 layers, hidden 768, FFN 3072, group-norm feature extractor; licence field "cc".
  - `utter-project/mHuBERT-147`: same shape, 94,371,712 parameters, licence cc-by-nc-sa-4.0.
  - `facebook/mms-300m`: `Wav2Vec2ForPreTraining`, 24 layers, hidden 1024, stable layer norm, cc-by-nc-4.0.

  — [AfriHuBERT config](https://huggingface.co/ajesujoba/AfriHuBERT), [mHuBERT-147 config](https://huggingface.co/utter-project/mHuBERT-147), [mms-300m config](https://huggingface.co/facebook/mms-300m)
- Continued pretraining matches or slightly beats pseudo-label SST for low-resource ASR, at lower compute — [DeHaven & Billa, arXiv:2207.00659](https://arxiv.org/abs/2207.00659)
- Continued pretraining for classroom-domain ASR — [CPT-Boosted wav2vec2, arXiv:2409.14494](https://arxiv.org/pdf/2409.14494); [arXiv:2405.13018](https://arxiv.org/pdf/2405.13018) (I did not extract numbers)
- In transformers 5.20.2, `Wav2Vec2ForPreTraining` implements the contrastive and diversity losses, but HuBERT has no pretraining class (only `HubertModel`, `HubertForCTC`, `HubertForSequenceClassification`). HuBERT TAPT would need fairseq or a custom masked-prediction loss with k-means targets. **[code-verified]** — [transformers wav2vec2 source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/modeling_wav2vec2.py), [hubert source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/hubert/modeling_hubert.py)

### Inferences
- **Fastest TAPT-like edge:** pick AfriHuBERT over mHuBERT-147 or English HuBERT as the initialisation for the student and the fold teachers. It is architecture-identical, so it costs nothing to try. Check the licence ("cc" on the hub, which is ambiguous) against the competition rules.
- **If you still try a short TAPT on MMS-300m:** use `Wav2Vec2ForPreTraining` with HF's `run_wav2vec2_pretraining_no_trainer.py`-style masking and negative sampling on the 95 h (plus test audio if allowed), with LR about 1e-5–5e-5 and a few thousand updates. Then truncate and fine-tune.
- Expect a T4 to do only around 5–10K small-batch updates in about 8 h **[uncertain]**, roughly 5–10% of AfriHuBERT's step count with much smaller batches. Any gain is speculative.

### Gaps
- No published TAPT result at the hours-on-one-GPU scale for LID.
- I did not confirm whether AfriHuBERT's pretraining data includes Efik, Ibibio, Tiv, Nupe, Igala, Idoma, Urhobo, Eggon or Anaang; the paper's text does not list them.

---

## Compression that keeps the HF architecture (layer truncation, head/FFN pruning, fp16, classifier_proj_size)

### Takeaway
In HF, the only compression options that keep a standard architecture are:
- **Layer truncation**, via `num_hidden_layers`. Middle layers carry most language information, so keep the bottom 50–70% of layers.
- **Uniform reduction** of `intermediate_size` / `hidden_size`, with the same width in every layer.
- **fp16 export.**
- **Choosing a narrow architecture** (DistilHuBERT, Whisper-tiny/base encoders).

Head pruning cannot be done for wav2vec2/HuBERT in HF, and non-uniform per-layer sizes such as DPHuBERT's cannot be expressed. `classifier_proj_size` has a negligible effect on size.

Best accuracy per parameter is most likely a 4–8-layer multilingual base (37–66M) or a distilled 20–24M student, with fp16.

### Cited Findings
**What the HF source shows** **[code-verified]**
- transformers 4.46.3:
  - `PreTrainedModel.prune_heads` stores heads in `config.pruned_heads` and calls `self.base_model._prune_heads`.
  - Neither `Wav2Vec2Model` nor `HubertModel` (nor WavLM/Whisper) defines `_prune_heads`, so the call fails.
  - `init_weights()` re-applies `config.pruned_heads` at load time, so a config carrying pruned heads would also fail.
- transformers 5.20.2 (latest on PyPI, 2026-10-20) has no `prune_heads` or `pruned_heads` anywhere in `modeling_utils.py` or `configuration_utils.py`.
- Every wav2vec2 encoder layer is `Wav2Vec2EncoderLayer(config)`, with one global `intermediate_size` and `num_attention_heads`, so per-layer sizes cannot be expressed.
- `Wav2Vec2ForSequenceClassification`:
  - is `projector = Linear(hidden, classifier_proj_size)` followed by `classifier = Linear(classifier_proj_size, num_labels)`;
  - mean-pools with the feature-level attention mask;
  - with `use_weighted_layer_sum`, learns softmax weights over `num_hidden_layers + 1` hidden states.

— [transformers modeling_utils (v4.46)](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/modeling_utils.py); [modeling_wav2vec2.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/modeling_wav2vec2.py); [PyPI transformers](https://pypi.org/project/transformers/)

**Which layers carry language identity**
- Analysis of MMS layers: "the middle layers capture language-related information, and the high layers encode content-related information", and content information "gradually decreases in the final layers" — [SSHR, ICME 2024, arXiv:2309.16937](https://arxiv.org/abs/2309.16937)
- A search summary attributed peak language information to MMS layers 8–12 on ML-SUPERB. I could not confirm the exact layer numbers from the primary text. **[uncertain]** — [SSHR](https://arxiv.org/abs/2309.16937)
- Pasad et al. find an acoustic → linguistic progression across wav2vec 2.0 layers, with the final layers of pretrained (not fine-tuned) wav2vec2 reverting towards input-like features — [arXiv:2107.04734](https://arxiv.org/pdf/2107.04734)
- A wav2vec 2.0 probe of Mandarin/English suprasegmentals found the best probe performance in the middle third of the network — [Interspeech 2024](https://www.isca-archive.org/interspeech_2024/delafuente24_interspeech.pdf)
- Distillation at about 25% of HuBERT's size keeps most SUPERB performance (DistilHuBERT, above). DPHuBERT's non-uniform pruning beats pure distillation but outputs TorchAudio models — [arXiv:2110.01900](https://arxiv.org/abs/2110.01900), [DPHuBERT GitHub](https://github.com/pyf98/DPHuBERT)

**Parameter counts by depth** **[computed]** (CNN about 4.2M + positional conv + layers + head)

| Encoder | Layers | Params |
|---|---|---|
| Base, d=768, FFN 3072 | 2 | 23.3M |
| | 3 | 30.4M |
| | 4 | 37.5M |
| | 6 | 51.6M |
| | 8 | 65.8M |
| | 12 | 94.2M |
| Large, d=1024, FFN 4096 | 4 | 63.2M |
| | 6 | 88.4M |
| | 8 | 113.6M |
| | 12 | 164.0M |
| | 24 | 315.2M |
| Base, 12 layers, FFN 2048 | 12 | 75.3M |
| Base, 12 layers, FFN 1536 | 12 | 65.8M |

The head (`projector` 768×256 plus classifier) is about 0.2M.

### Inferences
**Truncation recipe** (standard HF):
```python
m = AutoModelForAudioClassification.from_pretrained(
    "ajesujoba/AfriHuBERT", num_hidden_layers=6, use_weighted_layer_sum=True,
    num_labels=13, label2id=..., id2label=..., classifier_proj_size=256)
```
- The unused layer weights are dropped with a warning, and the new config is saved with `save_pretrained`.
- For stable-layer-norm models (MMS/XLS-R), the encoder's final LayerNorm, which was trained on layer-24 outputs, is now applied to layer-N outputs. Fine-tuning absorbs this, but expect the first epoch to look worse.
- A 6-layer MMS-300m (88M) is about the same size as a full 12-layer base, so for small budgets prefer truncated base-sized multilingual models (AfriHuBERT / mHuBERT-147).

**Pick depth by a probe, not a guess** (~1 h):
1. Fine-tune or linear-probe the full model with `use_weighted_layer_sum=True`.
2. Read the learned `layer_weights` and per-layer linear-probe F1 on OOF.
3. Truncate just above the last layer with substantial weight.

**Uniform FFN pruning (HF-compatible)**
- In each layer, rank FFN neurons by an importance score (e.g. |W_out| × mean |activation|, or a Taylor/gradient score) and keep the top-k. The same k is used in every layer, but different indices per layer.
- Slice `intermediate_dense` (rows) and `output_dense` (columns), set `config.intermediate_size = k`, then recover with KD from the unpruned model.
- Going from FFN 3072 to 1536 saves about 28M parameters on 12 layers.
- **[uncertain]** I found no paper quantifying this exact uniform-width variant for LID. Truncation is simpler and probably gives a similar trade-off.

**Head count and hidden size**
- `head_dim = hidden/num_heads`, and the Q/K/V projections are hidden×hidden. Reducing heads without shrinking `hidden_size` everywhere is therefore impossible.
- Shrinking `hidden_size` means a new narrow student (distillation from scratch). That is too slow except via existing narrow checkpoints such as Whisper-tiny/base.

**fp16 export**
- Halves the repo size (the 1.5 GB term) and weight memory. It leaves the parameter count unchanged.
- Validate that argmax agreement with fp32 on OOF is about 100%. Wav2vec2-large-style models occasionally overflow in fp16; check for NaNs.

**`classifier_proj_size`**: dropping 256 → 64 saves only about 0.15M parameters, so it does not matter for the score.

### Gaps
- No direct LID-vs-depth curve for HuBERT-base or AfriHuBERT. The SSHR layer numbers are unverified.
- No numbers on how the grader's speed metric scales with depth (the CNN front end is a fixed cost for every wav2vec2/HuBERT student).

---

## Post-training for macro / fixed-share F1 (bias/threshold tuning, temperature, weight averaging, SAM)

### Takeaway
- **Do:** post-hoc logit adjustment, plus per-class offsets tuned directly for the fixed-share F1, folded into `classifier.bias`. This is cheap and theory-backed, and it removes 3–8 points of balanced error in the vision long-tail literature.
- **Temperature scaling** does not change an argmax and is useless for the submission itself. It only matters for KD and for ensembling teachers.
- **EMA / checkpoint averaging:** cheap and harmless, gains usually small.
- **SAM:** helps minority classes (+6.2% minority accuracy reported) at 2× training cost.

### Cited Findings
- Post-hoc logit adjustment predicts argmax_y f_y(x) − τ·log π_y, where π is the empirical training prior.
  - τ=1 corresponds to the Bayes-optimal rule for balanced error under a calibrated model.
  - Tuning τ gave CIFAR-10-LT balanced error 19.08 vs 27.16 for ERM, and ImageNet-LT 49.56 vs 53.11.
  - Post-hoc *weight normalisation* (Kang et al.'s τ-norm) is multiplicative and "not consistent for minimising the balanced error, unlike logit adjustment".

  — [Menon et al., arXiv:2007.07314](https://arxiv.org/abs/2007.07314)
- Decoupled classifier re-training (cRT / LWS) beat end-to-end CE on CIFAR-100-LT at imbalance 100 (50.0 / 50.5 vs 45.3) and ImageNet-LT (41.8 / 41.4) — [Ren et al., arXiv:2007.10740](https://ar5iv.labs.arxiv.org/html/2007.10740)
- SAM escapes saddle points on minority classes, giving +6.2% minority-class accuracy over Vector-Scaling loss and +4% average across imbalanced datasets. Code: github.com/val-iisc/Saddle-LongTail — [Rangwani et al., NeurIPS 2022, arXiv:2212.13827](https://arxiv.org/abs/2212.13827)
- Model soups: averaging the weights of models fine-tuned with different hyperparameters improves on the best single model in the sweep at no inference cost. ViT-G reached 90.94% ImageNet top-1. Code: github.com/mlfoundations/model-soups — [Wortsman et al., ICML 2022, arXiv:2203.05482](https://arxiv.org/abs/2203.05482)
- Weight averaging in ASR fine-tuning mitigates catastrophic forgetting, which the authors attribute to wider optima — [arXiv:2210.15282](https://arxiv.org/pdf/2210.15282)
- Caution: a 2026 study of 8 fine-tuning variants on 9 wav2vec 2.0 / HuBERT / WavLM checkpoints across 3 SUPERB classification tasks found many reported gains to be seed- and instance-dependent "elicitation match" rather than a higher performance ceiling. Its title and authors were not retrieved — [arXiv:2607.13864](https://arxiv.org/abs/2607.13864)

### Inferences
**Fixed-share-F1 bias search** (exact, standard HF):
1. Pool speaker-grouped OOF logits Z (N×13) and labels.
2. Initialise b = −τ·log π, with τ swept over {0, 0.25, …, 1.5}.
3. Run coordinate ascent: for each class c, try b_c + δ for δ in a grid such as linspace(−3, 3, 61) and keep the value that maximises Σ_c w_c·F1_c(argmax(Z + b)), with w = 0.087 for the 10 rare classes and 0.043 for yor/ibo/hau. Run 3–5 passes.
4. Regularise: average b over bootstrap resamples of *speakers*, or shrink towards the LA solution, to avoid overfitting a 1–3-speaker class.
5. Fold in with `model.classifier.bias.data += torch.tensor(b)` and save.

- The logits are `classifier(projector(pool))`, so the added bias is exactly the post-hoc offset. Nothing non-standard is introduced.
- Re-do the search after any weight averaging or fp16 cast.

**Temperature**
- A scalar T>0 rescales all logits equally and never changes the argmax; it is irrelevant for the leaderboard.
- Tune T only for KD soft targets, or for calibrating fold or teacher models before averaging their probabilities.

**EMA / SWA / soups**
- Keep an EMA of the weights (decay about 0.999 per step) during fine-tuning and/or average the last 3–5 epoch checkpoints, then evaluate on OOF.
- Averaging separately trained fold models is riskier: their data differ, though they share the pretrained init **[uncertain]**. Use greedy soup (add a checkpoint only if OOF/holdout improves).
- Wav2vec2/HuBERT use LayerNorm or per-sample GroupNorm, so no BatchNorm recalibration is needed after averaging (it would be for SWA on BN networks).

**SAM**
- Use ρ about 0.05; optionally use a larger ρ on tail-class samples (ImbSAM-like).
- It doubles the step cost, so use it only for the final small-student run if time allows.

### Gaps
- No speech-LID paper quantifies post-hoc bias tuning, soups or SAM. The effect sizes come from vision long-tail benchmarks.
- I did not obtain the title or authors of the 2026 caution paper.

---

## Directly optimising fixed-share F1: cost-sensitive / F1-surrogate losses, CB loss, LDAM-DRW, balanced softmax

### Takeaway
With very small classes (0.5 h, 1–3 speakers), logit-adjusted CE (τ=1), which is equivalent to Balanced Softmax, is the best-supported drop-in. On CIFAR-100-LT it beat class-balanced weighting by about 8 points (50.3 vs 42.3), and beat CE by 5. LDAM(-DRW) and the CB loss give smaller or inconsistent gains. Re-weighting by inverse frequency, the competitors' "class weights", can underperform plain CE.

Combine LA-CE training with post-hoc bias tuning for the fixed-share F1 weights; this is a better fit than differentiable F1 surrogates.

### Cited Findings
- Logit-adjusted loss: CE(f(x) + τ·log π, y), applied in training and removed at inference. At τ=1 it achieved the lowest balanced error on CIFAR-100-LT (56.11 vs ERM 61.64), ImageNet-LT (48.89 vs 53.11) and iNaturalist (33.64 vs 38.66), beating LDAM ("Adaptive": 60.40, 52.15, 35.42) — [Menon et al., arXiv:2007.07314](https://arxiv.org/abs/2007.07314)
- Balanced Softmax: p_j = n_j·e^{η_j} / Σ_i n_i·e^{η_i}, i.e. softmax over η + log n.

  | Method | CIFAR-10-LT (200 / 100 / 10) | CIFAR-100-LT (200 / 100 / 10) |
  |---|---|---|
  | CE | 71.2 / 77.4 / 90.0 | 41.0 / 45.3 / 61.9 |
  | Class-balanced weighting (CBW) | 72.5 / 78.6 / 90.1 | 36.7 / 42.3 / 61.4 |
  | CB loss (Cui) | 72.6 / 78.2 / 89.9 | 39.9 / 44.6 / 59.8 |
  | LDAM | 73.6 / 78.9 / 90.3 | 41.3 / 46.1 / 62.1 |
  | Focal | 71.8 / 77.1 / 90.3 | 40.2 / 43.8 / 60.0 |
  | Balanced Softmax (ablation) | 79.0 / 83.1 / 90.9 | 45.9 / 50.3 / 63.1 |

  - LDAM-DRW's originally reported numbers were 77.03 / 88.16 (CIFAR-10-LT, imbalance 100 / 10) and 42.04 / 58.71 (CIFAR-100-LT); these come from a different baseline and are not directly comparable.
  - Code: github.com/jiawei-ren/BalancedMetaSoftmax.

  — [Ren et al., NeurIPS 2020, arXiv:2007.10740](https://ar5iv.labs.arxiv.org/html/2007.10740)
- Class-balanced loss re-weights by the inverse "effective number" (1−β^n)/(1−β). Code: github.com/richardaecn/class-balanced-loss — [Cui et al., CVPR 2019, arXiv:1901.05555](https://arxiv.org/abs/1901.05555)
- LDAM: label-distribution-aware margins with deferred re-weighting (DRW). Compared above — [Cao et al., NeurIPS 2019, arXiv:1906.07413](https://arxiv.org/abs/1906.07413) (numbers via the two papers above)

### Inferences
**Loss recipe**
```
loss = CE(logits + τ·log π, y)   # τ = 1
```
- Combine with balanced or square-root-balanced *sampling* only if the LA loss alone under-recalls the rare classes. LA + balanced sampling double-corrects; if you combine them, set τ≈0.
- Optionally add label smoothing about 0.05.

**Why the priors matter here**
- With a fixed-share F1, test class frequencies do not enter the weights. The training prior π (efi/igl/nup at 0.5–1.4 h versus tens of hours for majority classes) is the bias to remove.
- The remaining offset that is optimal for F1 (as opposed to balanced error) is found by the post-hoc search above.

**Measure speakers, not hours**
- Count imbalance by speaker as well as by hours. With 1–3 speakers per class, speaker-level generalisation, not class frequency, may dominate the error.
- Strong speaker and channel augmentation (speed/pitch perturbation, RIR/noise, codec) and KD from a teacher that has heard other speakers probably matter more than the choice of loss **[uncertain; not quantified in the sources found]**.

**F1 surrogates**
- Soft-F1 / Lovász losses are noisy with small batches containing few rare-class samples.
- Use them only as a short fine-tune on top of LA-CE, if at all **[no source found for speech]**.

### Gaps
- No speech/LID benchmark comparing LA vs CB vs LDAM-DRW vs Balanced Softmax found. All effect sizes are from long-tailed vision.

---

## Model selection: grouped cross-validation vs single split, and choosing final submissions between public and private leaderboards

### Takeaway
With 1–3 speakers in several classes, a single random split is close to meaningless for those classes. Either their validation shares speakers with training (leakage), or validation holds one speaker, giving a single-speaker F1 estimate.

Use speaker-grouped CV (`StratifiedGroupKFold` by speaker, or leave-one-speaker-out for the tiny classes). Pool OOF predictions for bias tuning and model choice. Then retrain on all data with the CV-chosen settings.

Kaggle evidence shows little *adaptive* overfitting to public leaderboards on average. The bigger risk here is that a small, speaker-sparse test set is high-variance, so trust pooled grouped-CV over small public-LB differences.

### Cited Findings
- Across 120 Kaggle competitions with public and private holdouts, Roelofs et al. found "little evidence of substantial overfitting" from holdout reuse, which they read as robustness of the holdout method across domains, loss functions and model classes — [Roelofs et al., NeurIPS 2019](https://papers.nips.cc/paper/9117-a-meta-analysis-of-overfitting-in-machine-learning)
- Seed- and instance-dependence of fine-tuning gains for wav2vec2/HuBERT/WavLM classification argues for multi-seed or multi-fold evaluation before believing small improvements — [arXiv:2607.13864](https://arxiv.org/abs/2607.13864)

### Inferences
**Protocol**
1. Assign folds by speaker ID, or by recording/session if speaker IDs are missing (cluster with an off-the-shelf speaker-embedding model).
2. Use 5 folds. For classes with 1–3 speakers, rotate so that every speaker is held out once.
3. Report the fixed-share F1 on pooled OOF, plus the per-class F1 spread across folds.
4. Choose the epoch count, τ, KD settings and the bias vector b on pooled OOF.
5. Train the final student on 100% of the data and apply b (tuned on OOF of identically configured models).

**Public-LB use**
- Treat the public LB as a sanity check: label mapping, preprocessing, efficiency measurement.
- Do not chase differences under about 1 F1 point. A class with one test speaker can swing that class's F1 by tens of points, and the whole score by several.

**Final pick** (if two final submissions are allowed)
- (a) the best pooled-OOF fixed-share-F1 × estimated multiplier;
- (b) a hedge: a more conservative or larger model, or the same model with LA τ=1 only, without the fully tuned b.

If only the last submission counts, submit (a).

**Score-aware selection**: rank candidates by OOF-F1 × the measured multiplier (measure memory, latency and repo size locally), not by F1 alone. A 23M student that loses less than about 5% relative F1 beats a 94M model.

### Gaps
- No LID-specific study of single-split vs grouped-CV variance at these data sizes found.
- The number of final submissions allowed, and the public/private split sizes on Codabench 18424, were not checked (out of this file's scope).

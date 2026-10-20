# Training techniques and HF no-custom-code deployment gotchas for small spoken LID on closely related low-resource Nigerian languages (DSN 2026, Codabench 18424)

Conventions used below:
- **[LT]** = a local test I ran on 2026-10-20 in a scratch venv with **transformers 4.57.6 + torch 2.8.0+cpu** (Python 3.13, 4 vCPU sandbox, random-init models). Scripts: `/tmp/claude-0/-home-user/a55f94d8-bdde-575d-8d9b-69b0f597d9e3/scratchpad/tests/hf_checks.py` and `.../tests/timing.py` (scratchpad is session-local and may not persist). Absolute timings are sandbox-specific; use only the ratios.
- Source-code citations point at the exact git tag I read (`v4.57.6` = the last transformers release that installs on Python 3.9; `v5.20.2` = latest on 2026-10-20).
- "(canonical; not re-fetched)" = a well-known paper whose core claim I cite from memory of its abstract. I did not re-open it this session.
- Metric arithmetic: the 9 low-resource classes × 8.7% = 78.3%, and yor/ibo/hau × 4.3% = 12.9%. That leaves about 8.8% for "other" (my inference; not confirmed by organisers). **Each of igl/nup/efi/ibb/anw/… is worth 2× yor/ibo/hau.**

---

## Q0. Ranked technique shortlist (expected impact vs. implementation time) for this exact setting

### Takeaway
With about 30 hours left, the best returns come from (1) fine-tuning a multilingual SSL encoder (MMS/XLS-R family, truncated for efficiency) with a stock mean-pooling head; (2) class-balanced/temperature sampling plus post-hoc per-class bias tuning folded into `classifier.bias`; (3) noise/RIR/telephone augmentation and random 3–6 s crops with full-length test pooling. KD from MMS-LID, weight averaging and the Whisper short-context trick are second-tier.

### Cited Findings
- In NIST LRE22 (African languages, 8 kHz telephone/broadcast), the ABC team's **fine-tuned XLS-R-1B** systems reached Cprimary of about **0.095–0.114** (dev/eval min/act). Their best fixed-condition ResNet/ECAPA systems got **0.23–0.46**, and ECAPA-TDNN scored 0.342–0.458. That is a large gap in favour of fine-tuned multilingual SSL — [Matějka et al., Interspeech 2023 (ABC LRE22)](https://www.isca-archive.org/interspeech_2023/matejka23_interspeech.pdf)
- The ML-SUPERB 2.0 (Interspeech 2025) top-scoring LID combined a frozen W2V-BERT 2.0 (SeamlessM4T-v2) encoder, learned per-dimension layer weights, 4-head multi-resolution attention with mean+std pooling, MUSAN noise and OpenSLR-28 reverb (each p=0.5), and an LDA/logistic-regression backend. It scored **86.8% LID** on eval vs. baselines at 53.8%/74.6%, and 56.6% dialect-LID vs. 18.8%/0.0% — [TalTech system paper, arXiv 2506.01458](https://arxiv.org/html/2506.01458v1)
- The second ML-SUPERB 2.0 system (CMU) used data augmentation for few-shot languages plus an auxiliary LID-CTC loss. It reported a **14% relative LID-accuracy gain** over baseline (abstract only; no absolute numbers on the page) — [Wang et al., Interspeech 2025](https://www.isca-archive.org/interspeech_2025/wang25z_interspeech.html)
- MMS-LID recipe: a linear classifier on MMS-1B with **all parameters fine-tuned**. It used Adam (β=0.9/0.98) and a tri-stage LR (10% warm-up, 40% hold, then linear decay), a grid over LR {1e-5, 3e-5, 3e-6, 5e-6, 7e-6}, 10K–50K updates and batches of 1.5/3/6 minutes of audio. Language-balanced temperature sampling used β ∈ {0, 0.3, 0.5, 0.7, 1} — [Pratap et al. 2023, MMS §6.1](https://arxiv.org/abs/2305.13516)
- Post-hoc logit adjustment (τ=1) reduced balanced error vs. ERM from **27.16→22.60** (CIFAR-10-LT), **53.11→49.66** (ImageNet-LT) and **38.66→33.98** (iNaturalist). With tuned τ, CIFAR-10-LT went to 19.08. The logit-adjusted loss gave 22.33 / 48.89 / 33.64 — [Menon et al. 2021, Table 3](https://arxiv.org/abs/2007.07314)
- In a Zindi African audio-classification win (Swahili keywords), a single wav2vec-pretrained encoder with a classifier head beat mel-spectrogram approaches. The winner noted ensembling might have helped further — [Zindi winners blog](https://zindi.africa/blog/meet-the-winners-of-the-swahili-audio-classification-hackathon)
- I found **no public DSN 2026 / Codabench 18424 winner write-up**. No search returned one.

### Inferences
Ranked list. "Impact" means the expected effect on fixed-share F1 × efficiency in this task; "Time" means implementation plus one training run.

| # | Technique | Impact | Time | Why / evidence |
|---|---|---|---|---|
| 1 | Full fine-tune of a multilingual SSL encoder (e.g. `facebook/mms-300m`, XLS-R-300m), **truncated to ~8–12 layers** via `num_hidden_layers`, stock `*ForSequenceClassification` head, `use_weighted_layer_sum: true` | Very high | 2–4 h GPU | LRE22 SSL ≫ ECAPA; lower layers carry LID info (Q2); truncation is the main efficiency lever (Q7) |
| 2 | **Post-hoc per-class bias tuning** on a speaker-disjoint val set, maximising the *fixed-share* F1 directly; fold into `classifier.bias` | High (esp. rare/confusable classes) | <1 h | Menon et al. gains; zero inference cost; no custom code |
| 3 | Class-balanced or temperature sampling (β≈0.3–0.5) **plus** logit-adjusted CE (τ≈1), not both + post-hoc LA | High for igl/nup/efi/ibb | <1 h | MMS used β-sampling; LA loss is Fisher-consistent for balanced error (Q3) |
| 4 | Augmentation: MUSAN noise (SNR 0–15 dB), RIRs, 8 kHz band-limit/μ-law/GSM-style codec on ~20–30% of samples, speed 0.9/1.0/1.1, SpecAugment via `mask_time_prob` | Medium–high on noisy/phone test | 1–2 h | TalTech/SpeechBrain recipes; MMS BABEL telephone drop (Q5) |
| 5 | Random 3–6 s crops for training; **full-clip** inference (no crop) | Medium | trivial | SpeechBrain 3 s crops; LRE22 accuracy rises with duration up to 15–20 s (Q8) |
| 6 | Explicit "other" class with diverse English/Pidgin/other-language data (outlier exposure) + tuned "other" bias | Medium (~8.8% weight + FP impact on all classes) | 1–2 h data | Q4 |
| 7 | Weight averaging of last K checkpoints / soups of 2–3 seeds | Low–medium, free at inference | <30 min | Q7 |
| 8 | KD from `facebook/mms-lid-1024/2048/4017` (covers all 12 codes + eng + pcm) restricted to 12+other | Uncertain (teacher trained mostly on New Testament readings) | 3–6 h (teacher inference over 95 h) | Q6 — validate teacher on val first |
| 9 | Whisper-tiny/base encoder with shortened context (`max_source_positions`, `chunk_length`) | Uncertain; small params but fixed-length cost | 2–3 h | Q11 |
| 10 | Hierarchical/pairwise classifiers, attentive stats pooling | Not deployable as-is (no custom code; Auto head = mean pooling) | — | Q2/Q12 |

### Gaps
- No public leaderboard or write-up for this specific challenge was found, so the ranking is my synthesis, not challenge-verified.
- The efficiency multiplier's exact formula (params? file size? latency? CPU or GPU?) is unknown. Items 1 and 9 depend on it.

---

## Q1. Which recipes win recent LID challenges/benchmarks with SSL encoders?

### Takeaway
Winning or SOTA LID systems from 2023–2025 all start from a large multilingual SSL or speech-foundation encoder (XLS-R-1B, MMS-1B, W2V-BERT 2.0). They add noise/reverb augmentation, language-balanced sampling, multi-layer features and long-context or windowed pooling. Most residual errors are between closely related languages.

### Cited Findings
- **ML-SUPERB 2.0 challenge** (Interspeech 2025): 154 languages, >200 varieties, 56 dialects/accents in dev. 5 submissions came from 3 teams. TalTech had the top overall score: LID 86.8, dialect-LID 56.6 (eval). Embedding-only LID was 85.3 dev / 80.5 dev-dialect; interpolating with an MMS-zeroshot generative LID raised this to 89.9 / 84.9 — [TalTech, arXiv 2506.01458](https://arxiv.org/html/2506.01458v1)
- TalTech LID details: the head was trained on VoxLingua107 (two 512-d ReLU layers, softmax CE), with a final LDA(100)+logistic-regression backend trained on up to 1 h/language — [arXiv 2506.01458](https://arxiv.org/html/2506.01458v1)
- TalTech error analysis: "most errors occur between related languages" (Tamil→Telugu, German→Luxembourgish, Urdu→Hindi) — [arXiv 2506.01458](https://arxiv.org/html/2506.01458v1)
- **NIST LRE22**: 14 African target languages, CTS and broadcast narrowband, all **8 kHz a-law**. 21 organisations / 16 teams / 65 submissions. Xhosa and Zulu were hardest. Confusability was highest among **Zulu, Xhosa, Ndebele, Tsonga, Venda** (closely related Bantu languages) and between South African and Indian-accented South African English — [Lee et al., The 2022 NIST LRE, arXiv 2302.14624](https://arxiv.org/abs/2302.14624)
- ABC LRE22: fine-tuned XLS-R-1B (HF `facebook/wav2vec2-xls-r-1b`) with a TDNN head, softmax loss, Adam, **LR 1e-6 or 3e-6**. 8 kHz input was **upsampled to 16 kHz** for XLS-R, while all other systems ran at 8 kHz. Inference used a **20 s sliding window shifted by 2 s, averaging embeddings** — [ABC, Interspeech 2023](https://www.isca-archive.org/interspeech_2023/matejka23_interspeech.pdf)
- MMS-LID scales to 4,017 languages with little degradation: FLEURS accuracy is 97.5 (126 langs) and 97.2 (4,017). Out of domain on **BABEL conversational telephone (23 African/Asian languages)**, accuracy is only **80.5–84.1%** — [MMS paper, Table 7](https://arxiv.org/abs/2305.13516)
- Self-supervised adaptive pre-training (continued SSL on target-domain audio) of XLSR-128 gave "gains up to 40.1%" for under-represented FLEURS languages (metric unspecified in the abstract) — [Shaik et al., arXiv 2312.07338](https://arxiv.org/abs/2312.07338)
- SpeechBrain's VoxLingua107 ECAPA recipe: **3 s training crops**, 40 epochs, LR 1e-3→1e-4 linear, batch 128. Augmenter applies 1–3 of {reverb, noise SNR 0–15 dB, speed perturb} with `concat_original: True` — [SpeechBrain train_ecapa.yaml](https://github.com/speechbrain/speechbrain/blob/develop/recipes/VoxLingua107/lang_id/hparams/train_ecapa.yaml)

### Inferences
- The fine-tuned-SSL vs. ECAPA gap in LRE22 (about 2–3× lower cost) was on African languages with telephone audio. That is the closest public analogue to this test set, so starting from MMS/XLS-R weights is the single most important choice.
- TalTech's attention/stats pooling and LDA backend are **not reproducible** in a stock `AutoModelForAudioClassification` (see Q12). The deployable subset of their recipe is: multi-layer features (`use_weighted_layer_sum`), MUSAN/RIR augmentation, and balanced data.

### Gaps
- I found no AfriSpeech/"Naija" LID challenge with published winning recipes. Nigerian-speech SSL benchmarks found are ASR-focused ([engrXiv 6650](https://engrxiv.org/preprint/view/6650)).
- I couldn't open the full CMU ML-SUPERB 2.0 paper, so its augmentation types and LID-CTC weight are unknown.

---

## Q2. Closely related languages (Annang/Ibibio/Efik): longer context, ASP vs. mean pooling, intermediate layers, hierarchical classifiers

### Takeaway
The strongest evidence supports (a) longer context and full-length test pooling, and (b) multi-layer or lower-layer features. Attentive statistics pooling helps in the literature but is **not available** in AutoModelForAudioClassification heads. I found no evidence for hierarchical or pairwise classifiers.

### Cited Findings
- LRE22: "when speech duration increased, system performance significantly increased up to a certain duration (between 15 s and 20 s)", then plateaued — [NIST LRE22, arXiv 2302.14624](https://arxiv.org/abs/2302.14624)
- ABC scored whole utterances with 20 s windows (2 s shift) and averaged them — [ABC LRE22](https://www.isca-archive.org/interspeech_2023/matejka23_interspeech.pdf)
- "Pre-trained speech models optimally encode language discriminatory information in **lower layers**". Those embeddings were robust to unseen languages and acoustic environments — [Bartley et al., arXiv 2211.05103](https://arxiv.org/abs/2211.05103)
- TalTech combined all 24 W2V-BERT 2.0 layers with learned per-dimension weights and pooled with multi-head attention + mean/std (8192-d) — [arXiv 2506.01458](https://arxiv.org/html/2506.01458v1)
- On a frozen BEST-RQ encoder, swapping the attention type before mean+std pooling changed average macro-F1 from **63.27 (self-attention) to 74.28 (Performer r=128)**. Pooling design matters a lot for macro-F1 (VoxLingua F1 53.5→77.9) — [arXiv 2502.05841](https://arxiv.org/html/2502.05841v1)
- HF stock heads: `Wav2Vec2/Hubert/WavLM/Wav2Vec2Bert ForSequenceClassification` do **projector → (masked) mean pooling → linear classifier** only. Optional `use_weighted_layer_sum` takes a softmax-weighted sum over all hidden states — [modeling_wav2vec2.py v4.57.6 L1997–2014](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/wav2vec2/modeling_wav2vec2.py)
- Heads with TDNN + statistics pooling exist (`Wav2Vec2ForXVector`, `WavLMForXVector`, `Wav2Vec2BertForXVector`…). They are registered under `MODEL_FOR_AUDIO_XVECTOR_MAPPING`, **not** `MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING`, so `AutoModelForAudioClassification` will not load them — [modeling_auto.py v4.57.6](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/auto/modeling_auto.py)

### Inferences
- Do not crop at test time. Feed the full clip, which the stock pipeline already does (Q14). Train with random crops (Q8), but consider a final short fine-tune epoch on longer crops (8–15 s) so the mean pool sees test-like lengths.
- To "simulate" a hierarchical classifier without custom code, use confusion-aware **bias tuning** on {anw, ibb, efi}: coordinate-ascent on their three biases against the fixed-share F1. Also oversample hard pairs (e.g. weight anw/ibb/efi more in the sampler).
- `use_weighted_layer_sum: true` is the only deployable multi-layer mechanism. It adds `num_hidden_layers+1` scalars (see the Q7 truncation gotcha).

### Gaps
- I found no paper directly comparing hierarchical/pairwise vs. flat classifiers for closely related LID with SSL encoders.
- No layer-wise LID curve for MMS-300m/XLS-R-300m was found, so the "optimal truncation depth" must be measured on val. Candidates are 6/8/12 of 24 layers.

---

## Q3. Extreme imbalance under fixed-share macro F1: sampling vs. weighted CE vs. focal vs. logit adjustment vs. post-hoc bias tuning

### Takeaway
Use balanced or temperature sampling *or* logit-adjusted CE during training. Then tune per-class additive biases on a speaker-disjoint val set against the exact fixed-share F1, and write them into `classifier.bias`. Post-hoc logit adjustment is theoretically consistent for balanced error, empirically beats weight normalisation and reweighting, and costs nothing at inference.

### Cited Findings
- Post-hoc rule: predict `argmax_y f_y(x) − τ·log π_y`, where π = training class priors. The logit-adjusted loss is softmax-CE on `f_y(x) + τ·log π_y`, predicting with plain argmax. **Do not combine** the LA loss with post-hoc adjustment ("expected to be harmful") — [Menon et al. 2021, §4.1, §5.1](https://arxiv.org/abs/2007.07314)
- Table 3 balanced error: ERM 27.16/61.64/53.11/38.66 (CIFAR-10-LT/CIFAR-100-LT/ImageNet-LT/iNat). Post-hoc LA τ=1 gave 22.60/58.24/49.66/33.98; τ=τ* gave 19.08/57.90/49.56/33.80; LA loss gave 22.33/56.11/48.89/33.64. Weight-norm (Kang) τ=1 gave 24.02/58.89/52.00/48.05, and "Equalised" 26.02/57.26/54.02/38.37 — [Menon et al. 2021](https://arxiv.org/abs/2007.07314)
- Menon et al. show post-hoc **weight normalisation is not Fisher-consistent** for balanced error; additive logit offsets are — [Menon et al. 2021 §4.2](https://arxiv.org/abs/2007.07314)
- MMS-LID handled language imbalance with two-stage temperature sampling (βL across corpora, βD across languages, β ∈ {0…1}) — [MMS §6.1](https://arxiv.org/abs/2305.13516)
- Decoupling (Kang et al. 2020): instance-balanced sampling is best for representation learning. Re-balancing only the classifier (cRT / τ-norm) recovers tail accuracy — [arXiv 1910.09217](https://arxiv.org/abs/1910.09217) (canonical; not re-fetched)
- Focal loss down-weights easy examples ([arXiv 1708.02002](https://arxiv.org/abs/1708.02002)), and class-balanced loss reweights by "effective number" ([arXiv 1901.05555](https://arxiv.org/abs/1901.05555)) (both canonical; not re-fetched). In Menon's Table 3, LA matched or beat the margin/reweighting alternatives it compared (LDAM "Adaptive", "Equalised").
- In HF heads the final layer is `self.classifier = nn.Linear(classifier_proj_size, num_labels)` applied to the pooled vector. Logits = `W·h + b`, so any per-class additive offset δ is exactly `classifier.bias += δ` — [modeling_wav2vec2.py v4.57.6 L1931, L2014](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/wav2vec2/modeling_wav2vec2.py)

### Inferences
- Concrete recipe:
  1. Train with sampler weights ∝ n_c^(β−1), β≈0.3–0.5, **or** plain sampling + LA loss τ=1.
  2. Score a speaker-disjoint held-out split.
  3. Initialise δ = 0 (or −log π if trained plainly).
  4. Run coordinate ascent over δ_c ∈ [−3, 3] in steps of 0.05, maximising Σ_c w_c·F1_c with the official shares.
  5. `model.classifier.bias.data += torch.tensor(δ)`.
  6. `save_pretrained`.
  ```python
  with torch.no_grad(): model.classifier.bias += torch.tensor(delta, dtype=model.classifier.bias.dtype)
  model.save_pretrained(out_dir, safe_serialization=True)
  ```
- **nup has zero validation clips**, so its δ cannot be tuned on val. Either carve a speaker-disjoint slice of nup train (if multiple speakers exist) or fix δ_nup = −τ·log π_nup and tune the rest. Over-fitted biases on tiny val sets (igl, nup) are the main risk. Regularise by tuning on pooled cross-validation folds, or cap |δ|.
- Because fixed-share F1 weights each rare class at 8.7%, the optimal biases will deliberately trade some yor/ibo/hau recall (4.3% each) for rare-class recall.

### Gaps
- I found no speech-LID paper that directly compares focal vs. LA vs. balanced sampling under macro-F1. The evidence above is from image long-tail benchmarks.

---

## Q4. Open-set "other" rejection with a closed-set classifier (must be baked into weights)

### Takeaway
Train an explicit "other" class on diverse real outlier speech (English, Pidgin, many other languages) — this is outlier exposure. Then tune the "other" bias like any other class. A pure max-logit threshold *can* be baked into a linear head: give the "other" row zero weights and bias T. Energy/max-softmax thresholds cannot be expressed exactly.

### Cited Findings
- Threshold-based open-set LID: no single threshold fits all. The EER point was at softmax 0.65, while total accuracy peaked at 0.81, and "total accuracy numbers are highly reflective of the proportion of in-set and out-of-set samples" — [CU MultiLang, arXiv 2308.14951](https://arxiv.org/abs/2308.14951)
- An earlier open-set LID paper compared threshold rejection vs. treating rejection as verification. I could not extract its results — [arXiv 2205.10397](https://arxiv.org/abs/2205.10397)
- Outlier Exposure: training against an auxiliary outlier dataset improves OOD detection on unseen outliers — [Hendrycks et al. 2019, arXiv 1812.04606](https://arxiv.org/abs/1812.04606) (canonical; not re-fetched). Energy score = logsumexp(logits) — [Liu et al. 2020, arXiv 2010.03759](https://arxiv.org/abs/2010.03759) (canonical). MaxLogit as an OOD score — [Hendrycks et al., arXiv 1911.11132](https://arxiv.org/abs/1911.11132) (canonical).
- `facebook/mms-lid-256` and larger include `pcm` (Nigerian Pidgin) and `eng` labels, so MMS-LID can help pseudo-label or curate "other" data — [mms-lid-256 config](https://huggingface.co/facebook/mms-lid-256/blob/main/config.json) (checked id2label)

### Inferences
- Baked max-logit rejection (derivation, not tested on real data): set `W[other] = 0` and `b[other] = T`. Then argmax = "other" iff max_k f_k(x) < T, which is MaxLogit OOD detection with zero custom code. A hybrid (learned "other" row + tuned b_other) is likely better, because test "other" includes English/Pidgin, which are *in-distribution* if trained. Tune b_other jointly with the Q3 coordinate ascent.
- "Other" errors also create false positives in the 8.7%-weight classes. Make the "other" training pool acoustically diverse, including phone/noisy audio, so channel cues don't leak into the label.
- Energy (logsumexp) or max-softmax thresholds are not linear in logits, so they can't be folded exactly into a single linear layer.

### Gaps
- I found no direct LID comparison of explicit-other-class vs. threshold rejection on the same data.

---

## Q5. Noise/telephone robustness: MUSAN, RIR, 8 kHz band-limit, codecs, speed perturbation, SpecAugment — effect sizes

### Takeaway
MUSAN+RIR augmentation is standard in every strong LID recipe found. Telephone/out-of-domain conditions cost MMS-LID about 13–17 accuracy points. Codec augmentation gains are real but inconsistent in the speaker-verification literature, and I found no LID-specific effect size. SpecAugment is cheap through config flags but adds little on top of noise+reverb.

### Cited Findings
- TalTech (ML-SUPERB 2.0 winner): reverb (OpenSLR 28 RIRs) and MUSAN noise, each with p=0.5 — [arXiv 2506.01458](https://arxiv.org/html/2506.01458v1)
- SpeechBrain VoxLingua107: noise SNR 0–15 dB, reverb, speed perturbation; 1–3 augmentations per sample, original also kept — [train_ecapa.yaml](https://github.com/speechbrain/speechbrain/blob/develop/recipes/VoxLingua107/lang_id/hparams/train_ecapa.yaml)
- MMS-LID accuracy: FLEURS 97.2–97.5% vs. **BABEL telephone 80.5–84.1%** (out-of-domain) — [MMS Table 7](https://arxiv.org/abs/2305.13516)
- LRE22: CTS (telephone) was harder than broadcast narrowband — [NIST LRE22](https://arxiv.org/abs/2302.14624). ABC trained fixed systems on 8 kHz with MUSAN+RIR and fed XLS-R 16 kHz-upsampled audio — [ABC LRE22](https://www.isca-archive.org/interspeech_2023/matejka23_interspeech.pdf)
- Phone-based LID degrades as codec bit rate drops (GSM 12.2k → G.729 → G.723.1) — [Quatieri et al., Eurospeech 1999](https://www.isca-archive.org/eurospeech_1999/quatieri99_eurospeech.html)
- In speaker verification, codec augmentation + adaptation improved SRE16-tgl EER 13.09%→12.12% but **hurt other test sets**. Codec effects were inconsistent — [arXiv 2104.01433](https://arxiv.org/abs/2104.01433)
- On VoxCeleb, adding SpecAugment on top of noise+reverb gave **no further benefit**, while pitch+speed perturbation improved Vox1-O EER from 0.947 to 0.861 — [VoxSRC/CNSRC 2022 lessons, arXiv 2211.00815](https://arxiv.org/abs/2211.00815)
- For HuBERT/wav2vec2 downstream tasks, SpecAugment slightly improved clean performance. Noise/speed-perturbed training improved robustness on augmented test sets — [S3PRL augmentation comparison, arXiv 2303.00510](https://arxiv.org/abs/2303.00510)
- The ML-SUPERB 2.0 CMU system attributes a 14% relative LID-accuracy gain partly to augmentation (combined with LID-CTC) — [wang25z](https://www.isca-archive.org/interspeech_2025/wang25z_interspeech.html)
- HF SpecAugment knobs (applied only in `model.train()` to CNN latent features): `apply_spec_augment`, `mask_time_prob`, `mask_time_length`, `mask_feature_prob`, `mask_feature_length`. `facebook/mms-300m` ships with `mask_time_prob: 0.075`, `layerdrop: 0.1`, `feat_extract_norm: "layer"` — [mms-300m config.json](https://huggingface.co/facebook/mms-300m/blob/main/config.json); [configuration_wav2vec2.py v4.57.6](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/wav2vec2/configuration_wav2vec2.py)

### Inferences
- Practical on-the-fly chain (numpy/torchaudio in the training loop; nothing ships in the model):
  - p=0.5 MUSAN noise/music/babble at SNR U(0,15) dB.
  - p=0.3 RIR.
  - p=0.25 telephone channel: resample 16k→8k→16k (band-limit to about 300–3400 Hz), optional μ-law 8-bit quantisation or ffmpeg GSM/AMR/Opus round-trip at 6–12 kbps.
  - p=0.3 speed ∈ {0.9, 1.1}.
  - Keep `mask_time_prob` ≈0.05–0.075.
- Pre-compute codec-degraded copies offline with ffmpeg, which is faster than per-step codec calls. Given SV results, codecs should be a *minority* of samples.

### Gaps
- I found no LID-specific ablation quantifying codec/8 kHz augmentation gains. The effect sizes above are from SV/ASR/LRE context.

---

## Q6. Knowledge distillation from a large LID teacher (MMS-LID) into a small student

### Takeaway
MMS-LID-1024/2048/4017 cover **all 12 target codes plus eng and pcm**, so a label-subset KD target is constructible. However, I found no LID-specific KD evidence, and the teacher's training data for these languages is mostly New Testament read speech. Measure teacher accuracy on your val set before investing.

### Cited Findings
- Coverage check of id2label in each HF config, for yor ibo hau ego anw efi ibb idu igl nup tiv urh (+eng, pcm):
  - `mms-lid-126`: only yor/ibo/hau/eng.
  - `mms-lid-256`: adds ibb, tiv, pcm.
  - `mms-lid-512`: all except **ego**.
  - `mms-lid-1024/2048/4017`: **all 14 present**.
  - All are wav2vec2 with **48 layers, hidden 1280** (~1B), `classifier_proj_size 1024`, `feat_extract_norm "layer"`, `use_weighted_layer_sum false`.
  - Preprocessor: `do_normalize true`, `return_attention_mask true`, 16 kHz.
  - Sources: [mms-lid-1024 config](https://huggingface.co/facebook/mms-lid-1024/blob/main/config.json), [mms-lid-512 config](https://huggingface.co/facebook/mms-lid-512/blob/main/config.json), [mms-lid-126 preprocessor](https://huggingface.co/facebook/mms-lid-126/blob/main/preprocessor_config.json)
- MMS-lab data = recordings of people reading the **New Testament**. For 1K/2K/4K-language LID models, per-language data was cut to 15/7/3 minutes during development — [MMS §3, §6.1](https://arxiv.org/abs/2305.13516)
- KD (Hinton et al.): match temperature-softened teacher distributions, loss scaled by T² — [arXiv 1503.02531](https://arxiv.org/abs/1503.02531) (canonical)
- Wav2Small (emotion, not LID): teacher soft labels produced better tiny students than human labels — [arXiv 2408.13920](https://arxiv.org/html/2408.13920v3). DistillW2V2 (ASR): 8× faster, 12× smaller student via hidden-layer MSE + output loss — [arXiv 2303.09278](https://arxiv.org/abs/2303.09278)
- DistilHuBERT: a 2-layer HuBERT student, about 75% smaller and 73% faster with most SUPERB performance retained — [arXiv 2110.01900](https://arxiv.org/abs/2110.01900) (canonical). It loads as `model_type: "hubert"`, so it is AutoModelForAudioClassification-compatible (inference).

### Inferences
- Recipe if the teacher is good on val:
  - Teacher logits z (4017-d) → keep the 12 target indices; "other" = logsumexp(z over all remaining indices, incl. eng/pcm).
  - Soft target p_T = softmax(z_sub / T), T≈2.
  - Student loss = 0.5·CE(y) + 0.5·T²·KL(p_T ‖ softmax(s/T)).
  - Pre-compute teacher logits once on full clips (1B params: roughly 1–3 GPU-hours for 95 h audio; my estimate).
- Use the teacher mainly for **rare and confusable classes** (igl, nup, anw/ibb/efi) and to pseudo-label external unlabeled audio, if challenge rules allow it.

### Gaps
- MMS-LID accuracy on these specific Nigerian languages is not reported in sources I found.
- No LID KD papers were found.

---

## Q7. Layer truncation, use_weighted_layer_sum, checkpoint averaging/soups, fp16 export

### Takeaway
Truncate with `num_hidden_layers=N` when loading the *pre-trained SSL* checkpoint, which loads cleanly and drops the unused layers. On CPU, encoder cost is roughly linear in layers plus a fixed CNN front-end cost. **Gotcha:** re-truncating an already fine-tuned model that has `use_weighted_layer_sum=true` fails with a `layer_weights` size mismatch. fp16 export halves file size, and under transformers ≤4.57.x default loading it is upcast to fp32.

### Cited Findings
- [LT] `from_pretrained(dir, num_hidden_layers=2)` on a 4-layer checkpoint with `use_weighted_layer_sum=true` **raised** `RuntimeError: size mismatch for layer_weights: ... [5] ... [3]`. With `ignore_mismatched_sizes=True` it loaded: 2 layers, `layer_weights` shape (3,), re-initialised. The log reported "weights … not used" for dropped layers — [LT hf_checks.py §4](file:///tmp/claude-0/-home-user/a55f94d8-bdde-575d-8d9b-69b0f597d9e3/scratchpad/tests/hf_checks.py)
- `layer_weights = nn.Parameter(torch.ones(num_layers)/num_layers)`, where `num_layers = num_hidden_layers + 1`. Forward forces `output_hidden_states=True` and uses a softmax-weighted sum — [modeling_wav2vec2.py v4.57.6 L1928–2003](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/wav2vec2/modeling_wav2vec2.py)
- [LT] CPU latency for a 5 s clip, wav2vec2-base dims (768-d), random init, fp32, 4 threads: **12 layers 3.05 s / 94.6M params; 6 layers 1.62 s / 52.0M; 4 layers 1.29 s / 37.9M**. That implies about 0.9 s of fixed CNN front-end and ~0.18 s per transformer layer in this sandbox — [LT timing.py](file:///tmp/claude-0/-home-user/a55f94d8-bdde-575d-8d9b-69b0f597d9e3/scratchpad/tests/timing.py)
- [LT] Same model in **fp16 on CPU: 66.5 s vs 3.08 s fp32 (~20× slower)** — [LT timing.py](file:///tmp/claude-0/-home-user/a55f94d8-bdde-575d-8d9b-69b0f597d9e3/scratchpad/tests/timing.py)
- [LT] fp16-saved safetensors loaded with default `from_pretrained` (4.57.6) → **float32** params. Loaded with `torch_dtype="auto"` → float16. A fp16 model fed fp32 input raised `RuntimeError: Input type (torch.FloatTensor) and weight type (torch.HalfTensor) should be the same`. fp16-vs-fp32 logits max diff on the tiny model was 9.7e-6 — [LT hf_checks.py §1–2](file:///tmp/claude-0/-home-user/a55f94d8-bdde-575d-8d9b-69b0f597d9e3/scratchpad/tests/hf_checks.py)
- Model soups: averaging weights of models fine-tuned from the same initialisation improves accuracy and robustness at zero extra inference cost — [Wortsman et al. 2022, arXiv 2203.05482](https://arxiv.org/abs/2203.05482) (canonical; not re-fetched)
- Lower layers carry LID information — [arXiv 2211.05103](https://arxiv.org/abs/2211.05103)

### Inferences
- Set `num_hidden_layers` (and `use_weighted_layer_sum: true`) **before** fine-tuning, loading from the SSL base checkpoint: no `layer_weights` exist there, so there is no mismatch. If you truncate after fine-tuning, pass `ignore_mismatched_sizes=True` and re-train briefly, or slice `layer_weights[:N+1]` manually before saving.
- The CNN front-end is a fixed cost, so very aggressive truncation (<4 layers) yields diminishing latency gains. Smaller-dim encoders (base 768-d vs. 1024-d MMS-300m) matter as much as depth.
- Checkpoint averaging: average the state_dicts of the last 3–5 checkpoints of one run (same trajectory), then re-run bias tuning on the averaged model.
- fp16 export: safe under default loading in ≤4.57.x (it is upcast to fp32 in RAM, so no speed or memory benefit at inference) and halves the download/file size. It is risky only if the organiser passes `torch_dtype="auto"/float16` **on CPU** (≈20× slower) or runs a hand-written loop with fp32 inputs (crash). The pipeline itself casts inputs to the model dtype (Q14). Ship fp32 unless file size is scored.

### Gaps
- No published layer-truncation curve for LID on MMS-300m/XLS-R was found.
- I did not test fp16 on GPU here.

---

## Q8. Random-crop training length vs. full clips; test-time full-length pooling

### Takeaway
Train on random 3–6 s crops (cheap, augments, balances long/short clips) and evaluate on full clips. LRE evidence shows accuracy keeps improving with speech duration up to about 15–20 s.

### Cited Findings
- SpeechBrain VoxLingua107 recipe: `sentence_len: 3` seconds — [train_ecapa.yaml](https://github.com/speechbrain/speechbrain/blob/develop/recipes/VoxLingua107/lang_id/hparams/train_ecapa.yaml)
- LRE22: performance improves significantly with SAD duration up to 15–20 s, then plateaus. Test segments ranged 3–35 s — [NIST LRE22](https://arxiv.org/abs/2302.14624)
- ABC scored with 20 s windows (2 s shift) and averaged embeddings over the whole utterance — [ABC LRE22](https://www.isca-archive.org/interspeech_2023/matejka23_interspeech.pdf)
- HF audio-classification pipeline passes the **whole clip** to the feature extractor, with no chunking or truncation for Wav2Vec2-family models — [pipelines/audio_classification.py v4.57.6](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/pipelines/audio_classification.py)

### Inferences
- Mean pooling over a long clip ≈ averaging over windows, so full-length inference with the stock head approximates ABC's window averaging for free. Attention cost is O(T²), but this is negligible at ≤30 s for small models.
- If test clips are short (<3 s), include 1.5–3 s crops in training as well. Mixed crop lengths U(2, 8) s are a safe default.

### Gaps
- Test-clip duration distribution unknown.

---

## Q9. Which transformers/torch versions support Python 3.9? Which audio-classification model types exist there? Which config keys may break across versions?

### Takeaway
Python 3.9 caps you at **transformers 4.57.6** (released 2026-01-16) and **torch 2.8.0** (2025-08-06). Dependency ceilings are tokenizers 0.22.2, safetensors 0.7.0 and huggingface-hub <1.0. Train and export with transformers 4.57.x, and test-load in a Python 3.9 venv before submitting.

### Cited Findings
- PyPI `requires_python` (queried 2026-10-20):
  - **transformers**: `>=3.9.0` from 4.47.0 (2024-12-05) through **4.57.6 (2026-01-16)**; `>=3.10.0` from **5.0.0 (2026-01-26)**. Latest is 5.20.2 (2026-10-19) — [PyPI transformers JSON](https://pypi.org/pypi/transformers/json)
  - **torch**: `>=3.9.0` from 2.6.0 through **2.8.0 (2025-08-06)**, which has cp39 wheels; `>=3.10` from **2.9.0 (2025-10-15)**, with no cp39 wheels. Latest is 2.14.1 — [PyPI torch JSON](https://pypi.org/pypi/torch/json)
  - **safetensors**: last allowing 3.9 is 0.7.0 (2025-11-19); 0.8.0 needs ≥3.10 — [PyPI safetensors JSON](https://pypi.org/pypi/safetensors/json). **tokenizers**: last for 3.9 is 0.22.2 (2026-01-05) — [PyPI tokenizers JSON](https://pypi.org/pypi/tokenizers/json). **huggingface-hub**: last allowing 3.9 is 1.8.0 — [PyPI huggingface-hub JSON](https://pypi.org/pypi/huggingface-hub/json)
  - transformers 4.57.6 pins `huggingface-hub>=0.34.0,<1.0`, `tokenizers>=0.22.0,<=0.23.0`, `safetensors>=0.4.3` (from the wheel METADATA, [LT install](https://pypi.org/project/transformers/4.57.6/))
- The `codalab/codalab-legacy:py39` image is described as a basic image "including numpy, sklearn, etc. — python3.9", last pushed about 4 years ago, so transformers/torch are presumably installed by the organisers — [Docker Hub codalab-legacy](https://hub.docker.com/r/codalab/codalab-legacy); [codalab-dockers](https://github-redirect.dependabot.com/codalab/codalab-dockers)
- `MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES` in **both** 4.57.6 and 5.20.2: audio-spectrogram-transformer (AST), data2vec-audio, hubert, sew, sew-d, unispeech, unispeech-sat, wav2vec2, **wav2vec2-bert (Wav2Vec2BertForSequenceClassification)**, wav2vec2-conformer, wavlm, **whisper (WhisperForAudioClassification)** — [modeling_auto.py v4.57.6](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/auto/modeling_auto.py); [v5.20.2](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/auto/modeling_auto.py)
- [LT] 4.57.6 `save_pretrained` writes `"dtype": "float16"` (not `torch_dtype`) plus `"transformers_version": "4.57.6"`. 4.57.6 still accepts legacy `torch_dtype` via a back-compat pop — [LT]; [configuration_utils.py v4.57.6 L264–268](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/configuration_utils.py)
- v5 configs are `@strict` dataclasses (`huggingface_hub.dataclasses.strict`). The model-specific field names of `Wav2Vec2Config` are identical between 4.57.6 and 5.20.2 (diffed) — [configuration_wav2vec2.py v5.20.2](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/wav2vec2/configuration_wav2vec2.py)
- 4.57.6 raises `OSError` if safetensors metadata exists but `format` ∉ {pt, tf, flax, mlx}. Missing metadata is treated as pt. Both 4.57.6 and 5.20.2 `save_pretrained` write `{"format": "pt"}` — [modeling_utils.py v4.57.6 L484–491](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/modeling_utils.py)
- 4.57.6 renames weight-norm keys both ways (`weight_g` ↔ `parametrizations.weight.original0`), so the wav2vec2 `pos_conv_embed` weights load across torch versions — [modeling_utils.py v4.57.6 L5182–5188](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/modeling_utils.py)
- [LT] After loading, `config.id2label` keys are `int` (JSON stores strings) and the label for id 12 was preserved — [LT hf_checks.py §1](file:///tmp/claude-0/-home-user/a55f94d8-bdde-575d-8d9b-69b0f597d9e3/scratchpad/tests/hf_checks.py)

### Inferences
- Safest path: train and `save_pretrained` with **transformers==4.57.6, torch==2.8.0** (works on Python 3.9–3.13). Then run `AutoFeatureExtractor.from_pretrained` + `AutoModelForAudioClassification.from_pretrained` in a `python:3.9` container with unpinned `pip install transformers torch` to mimic the organisers.
- Avoid shipping files produced only by v5 (`@strict` configs, new keys). If you trained in v5, reload and re-save under 4.57.6.
- Keep `config.json` minimal and standard: `model_type`, `architectures`, `id2label`/`label2id` with exactly the 13 codes, and no `auto_map` (that implies custom code). Use `model.safetensors` only — no `pytorch_model.bin` or `training_args.bin` in the repo.
- If the organisers pin an *old* transformers, newer model types such as wav2vec2-bert might be missing. wav2vec2/hubert/wavlm are the lowest-risk choices.

### Gaps
- I couldn't inspect the actual organiser image or ingestion script (GitHub repo access was not enabled for codalab-dockers in this session).
- Which transformers version first added Wav2Vec2BertForSequenceClassification / WhisperForAudioClassification was not verified.

---

## Q10. (merged into Q9) Config-key breakage across save/load versions
See Q9.

### Takeaway
Model-specific keys for wav2vec2 match between 4.57.6 and 5.20.2. The main drift is `torch_dtype`→`dtype` and v5's strict config validation. Re-save with 4.57.6.

### Cited Findings
- See Q9 citations ([configuration_utils.py v4.57.6](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/configuration_utils.py), [configuration_utils.py v5.20.2](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/configuration_utils.py)).

### Inferences
- Older transformers (pre-`dtype` rename) would ignore a `dtype` key and could only matter for `torch_dtype="auto"` loading. This is harmless for the default fp32 load.

### Gaps
- I didn't test loading a v5-saved checkpoint in 4.57.6.

---

## Q11. WhisperForAudioClassification: 3000-frame requirement, longer/shorter clips, cost

### Takeaway
Yes: the stock encoder **requires exactly `max_source_positions×2` mel frames (3000 = 30 s)**. Shorter audio is zero-padded to 30 s, longer audio is **truncated to 30 s**, and mean pooling has **no mask**, so short clips are dominated by padding frames. A no-custom-code workaround exists: set `max_source_positions` and `chunk_length` smaller and slice the positional embeddings. I verified it loads and runs in 4.57.6.

### Cited Findings
- `WhisperEncoder.forward` raises `ValueError` if `input_features.shape[-1] != max_source_positions * conv1.stride * conv2.stride` (=3000). The check is present in both 4.57.6 and 5.20.2 — [modeling_whisper.py v4.57.6 L668–671](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/whisper/modeling_whisper.py); [v5.20.2 L609–612](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/whisper/modeling_whisper.py)
- `WhisperForAudioClassification`: projector → `hidden_states.mean(dim=1)` over all 1500 frames (**no attention mask**); supports `use_weighted_layer_sum` — [modeling_whisper.py v4.57.6 L1506–1602](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/whisper/modeling_whisper.py)
- `WhisperFeatureExtractor.__call__` defaults: `padding="max_length"`, `truncation=True`, `max_length=n_samples` (`chunk_length*sampling_rate`), `return_attention_mask=False`. `__init__` recomputes `n_samples`/`nb_max_frames` from `chunk_length` *after* setting JSON kwargs, so **`chunk_length` is the controlling key** — [feature_extraction_whisper.py v4.57.6](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/whisper/feature_extraction_whisper.py)
- [LT] Default FE: 3 s → (1, 80, 3000); 45 s → (1, 80, 3000), i.e. truncated. A 1000-frame input into a 1500-position encoder raised `ValueError: Whisper expects the mel input features to be of length 3000, but found 1000` — [LT §5](file:///tmp/claude-0/-home-user/a55f94d8-bdde-575d-8d9b-69b0f597d9e3/scratchpad/tests/hf_checks.py)
- [LT] **Shortened context works with stock classes**:
  - Config `max_source_positions: 500`, with `encoder.embed_positions.weight[:500]` sliced from the original.
  - Preprocessor `"chunk_length": 10`, which saved `n_samples 160000, nb_max_frames 1000`.
  - Reloaded via AutoFeatureExtractor/AutoModelForAudioClassification, a 4 s clip → features (1, 80, 1000) → logits (1, 13) — [LT §5](file:///tmp/claude-0/-home-user/a55f94d8-bdde-575d-8d9b-69b0f597d9e3/scratchpad/tests/hf_checks.py)
- vLLM's modified Whisper encoder also handles <30 s inputs by slicing positional embeddings (the same mechanism, but with custom code) — [vLLM Ultravox docs](https://docs.vllm.ai/en/v0.11.0/api/vllm/model_executor/models/ultravox.html)
- [LT] Encoder+head params: whisper-tiny dims (384-d, 4 layers) **8.3M**; whisper-base dims (512-d, 6 layers) **20.7M**. CPU time: tiny 1.22 s (3000 frames) vs 0.99 s (1000 frames); base 9.76 s vs 5.36 s (noisy sandbox timings) — [LT timing.py](file:///tmp/claude-0/-home-user/a55f94d8-bdde-575d-8d9b-69b0f597d9e3/scratchpad/tests/timing.py)

### Inferences
- Stock Whisper classification pays 30 s of compute for every clip, whatever its length. Its unmasked mean pool also dilutes short-clip evidence with silence frames, though fine-tuning learns around this partly. Long clips lose everything after 30 s.
- If you use Whisper, fine-tune with the shortened context (e.g. 10–15 s) so the training and inference length match. The pretrained positions beyond the slice are simply unused.
- Whisper-tiny is attractive if the efficiency score is parameter-based (8M). If it is latency-based, compare against a truncated wav2vec2 at the actual test-clip lengths.

### Gaps
- I found no published accuracy comparison of Whisper-encoder vs. MMS/XLS-R for LID on these languages.
- The accuracy effect of truncating positional context was not tested on real data.

---

## Q12. Wav2Vec2/HuBERT/WavLM ForSequenceClassification pooling, attention masks, feat_extract_norm, preprocessor_config fields

### Takeaway
Pooling is the mean over frames, masked only if `attention_mask` is passed. For `feat_extract_norm: "layer"` models (MMS, XLS-R, large-lv60, MMS-LID), set `"return_attention_mask": true`, which makes batched inference exactly match per-clip inference. For `"group"` models (wav2vec2-base, HuBERT-base, DistilHuBERT), padding changes outputs even with a mask, so per-clip inference is exact and batching is approximate.

### Cited Findings
- Pooling code (identical in Wav2Vec2/Hubert/WavLM/Wav2Vec2Bert): `if attention_mask is None: pooled = hidden_states.mean(dim=1)`; otherwise padded frames are zeroed and `pooled = sum / n_valid`. The frame mask comes from `_get_feature_vector_attention_mask` (CNN output lengths) — [modeling_wav2vec2.py v4.57.6 L1096–1114, L2006–2012](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/wav2vec2/modeling_wav2vec2.py); [modeling_hubert.py](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/hubert/modeling_hubert.py); [modeling_wavlm.py](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/wavlm/modeling_wavlm.py); [modeling_wav2vec2_bert.py](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/wav2vec2_bert/modeling_wav2vec2_bert.py)
- HF doc tip: models with `feat_extract_norm == "group"` (e.g. wav2vec2-base) were **not** trained with `attention_mask`, so pad with 0 and pass no mask. Models with `"layer"` (e.g. wav2vec2-large-lv60) **should** get `attention_mask` for batched inference. FE defaults: `return_attention_mask=False`, `do_normalize=True`, `padding_value=0.0` — [feature_extraction_wav2vec2.py v4.57.6 L40–78](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/wav2vec2/feature_extraction_wav2vec2.py)
- `do_normalize` = per-utterance zero-mean/unit-variance. It is computed over the unpadded length **only if** an attention mask is available when padding. Otherwise it runs over the full (possibly padded) vector. If `sampling_rate` is passed and ≠ FE rate it raises `ValueError`; if omitted it only warns — [feature_extraction_wav2vec2.py v4.57.6](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/wav2vec2/feature_extraction_wav2vec2.py)
- [LT] Max |logit diff| for a 3 s clip scored alone vs. batched with a 9 s clip (padding=True):
  - **layer-norm model: 3.7e-9 with mask, 0.023 without mask**
  - **group-norm model: 0.0070 with mask, 0.018 without**
  - [LT §3](file:///tmp/claude-0/-home-user/a55f94d8-bdde-575d-8d9b-69b0f597d9e3/scratchpad/tests/hf_checks.py)
- MMS-LID preprocessor ships `do_normalize: true, return_attention_mask: true, sampling_rate: 16000, feature_extractor_type: Wav2Vec2FeatureExtractor` — [mms-lid-126 preprocessor_config.json](https://huggingface.co/facebook/mms-lid-126/blob/main/preprocessor_config.json)
- Pipeline batching collate pads `input_values` with the FE `padding_value` and `attention_mask` with 0 — [pipelines/base.py v4.57.6 L147–204](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/pipelines/base.py)

### Inferences
- Preprocessor_config.json for an MMS-300m/XLS-R-based submission: `{"feature_extractor_type": "Wav2Vec2FeatureExtractor", "feature_size": 1, "sampling_rate": 16000, "padding_value": 0.0, "padding_side": "right", "do_normalize": true, "return_attention_mask": true}`. Copy `do_normalize` from the *base* model's own preprocessor_config, because training/inference mismatch here is a silent accuracy killer.
- With group-norm bases, train with batch padding too, or bucket by length, so the model has seen padded inputs.
- Unknown ingestion (per-clip vs. batched): a layer-norm base + `return_attention_mask: true` is invariant to both.

### Gaps
- I did not verify each candidate base's `feat_extract_norm` (e.g. WavLM-base-plus, HuBERT-large). Check each `config.json`.

---

## Q13. fp16 safetensors loaded with from_pretrained without torch_dtype: resulting dtype, memory, CPU vs. GPU

### Takeaway
In transformers ≤4.57.x (all that Python 3.9 can run), the default load is **fp32** regardless of stored dtype: the file is half-size, but RAM and speed are fp32. In v5 the default is `dtype="auto"`, which loads fp16. On CPU, fp16 was about 20× slower in my test.

### Cited Findings
- 4.57.6 docstring: if `dtype` is not specified, "the model will get loaded in `torch.float` (fp32)"; `"auto"` uses config `dtype`/`torch_dtype` or the first floating weight's dtype — [modeling_utils.py v4.57.6 L4508–4521](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/modeling_utils.py)
- 5.20.2 docstring: `dtype (str or torch.dtype, optional, defaults to "auto")` — [modeling_utils.py v5.20.2 L3953](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/modeling_utils.py)
- [LT] 4.57.6 default load of fp16 safetensors → float32; `torch_dtype="auto"` → float16 — [LT §1](file:///tmp/claude-0/-home-user/a55f94d8-bdde-575d-8d9b-69b0f597d9e3/scratchpad/tests/hf_checks.py)
- [LT] CPU fp16 forward of a wav2vec2-base-size model (5 s): 66.5 s vs. 3.08 s fp32. A fp16 model with fp32 input raised a dtype-mismatch RuntimeError — [LT timing.py](file:///tmp/claude-0/-home-user/a55f94d8-bdde-575d-8d9b-69b0f597d9e3/scratchpad/tests/timing.py)
- The pipeline casts processed inputs to `self.dtype` when set — [audio_classification.py v4.57.6 preprocess](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/pipelines/audio_classification.py)

### Inferences
- fp16 storage costs nothing for accuracy (fp16 rounding of weights is tiny for inference) and helps only if download/file size is scored. If the organisers might pass `torch_dtype="auto"` or `float16` on a **CPU** box, fp16 storage could make you about 20× slower in the efficiency multiplier. Ship **fp32** unless file size is explicitly part of efficiency. On GPU, fp16 would be fast either way.
- Peak RAM when loading fp16 into fp32 is about the fp32 model size.

### Gaps
- Whether the organisers run on CPU or GPU is unknown. CPU fp16 slowness may vary with CPU ISA (AVX512-FP16/AMX); I tested only this sandbox CPU.

---

## Q14. HF pipeline("audio-classification"): long audio, sampling rate, top_k

### Takeaway
The pipeline does **no chunking**: the whole clip goes to the feature extractor, and Whisper's FE truncates at 30 s. A raw numpy array is **assumed to already be at the FE sampling rate**, with no check. A dict with a different `sampling_rate` needs **torchaudio**. File paths and bytes need the **ffmpeg** binary. In 4.57.6, calling without `top_k` returned all 13 labels sorted by softmax score.

### Cited Findings
- Preprocess paths:
  - `str` → open file (or HTTP GET) → bytes.
  - bytes → `ffmpeg_read(bytes, fe.sampling_rate)`, which resamples via ffmpeg.
  - dict `{"raw"|"array", "sampling_rate"}` → `torchaudio.functional.resample` if the rate ≠ FE rate, else `ImportError`.
  - A numpy array must be 1-D (mono). It is passed to the FE with `sampling_rate=fe.sampling_rate`, so a wrong rate is not detected.
  - Then `feature_extractor(inputs, sampling_rate=..., return_tensors="pt")` runs with **default** FE args.
  - [audio_classification.py v4.57.6 L169–242](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/pipelines/audio_classification.py)
- Postprocess: `function_to_apply` default `"softmax"` (alternatives `"sigmoid"`/`"none"`), then `probs.topk(top_k)`; `top_k=None` → `num_labels`. `__init__` sets `top_k=5` if not provided — [audio_classification.py v4.57.6 L96–103, L147–166, L248–257](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/pipelines/audio_classification.py)
- [LT] In 4.57.6, `clf(np_array)` with no `top_k` returned **13** results. A dict input at 8 kHz raised `ImportError: torchaudio is required to resample audio samples` (torchaudio not installed). A file path worked when the ffmpeg binary was present — [LT §6](file:///tmp/claude-0/-home-user/a55f94d8-bdde-575d-8d9b-69b0f597d9e3/scratchpad/tests/hf_checks.py)
- v5.20.2 preprocess is the same except `requests`→`httpx` for URLs — [audio_classification.py v5.20.2](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/pipelines/audio_classification.py)

### Inferences
- Results are sorted by score, so `[0]["label"]` is the argmax. Bias edits in `classifier.bias` change both argmax and scores consistently.
- Because sampling-rate handling depends on how the organisers decode audio (librosa/soundfile → numpy at 16 kHz is most likely), keep `sampling_rate: 16000` in preprocessor_config. Make training robust to 8 kHz-origin audio upsampled to 16 kHz (Q5).
- Long test clips (>30 s) are processed whole by wav2vec2-family models; for Whisper they are truncated.

### Gaps
- I don't know whether the organisers' image has ffmpeg or torchaudio, or whether they call the pipeline or the model directly.
- The `top_k` default behaviour may differ in other transformers versions. I tested only 4.57.6.

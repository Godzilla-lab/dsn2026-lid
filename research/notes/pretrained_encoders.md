# Pretrained speech encoders and distillation teachers for DSN 2026 "Find the Language I Am Speaking" (12 Nigerian languages + other)

Research date: 2026-10-20. Most "facts" below come from live Hugging Face API, config.json or safetensors-header reads made today, or from the primary papers. Parameter counts marked "(computed)" come from summing tensor shapes in each repo's `model.safetensors` header. Truncated-model sizes are my arithmetic from those per-layer counts. Uncertain items are marked **[uncertain]**.

## Q1. Which compact pretrained encoders are the best starting point? (comparison table)

### Takeaway
Of the AutoModelForAudioClassification-compatible encoders, only the **AfriHuBERT family** has pretraining data that lists nearly all 12 targets: 11 of 12 in the AfriHuBERT-n language metadata, all except Efik. It also has the best published African-LID evidence among small models: FLEURS African SLID F1 of 91.6 (-n) and 93.2 (-s), against 88.0 for mHuBERT-147. Its licences are messy, though: -n is "cc" and derived from NC-SA, -o is NC-ND, and -large/-xlarge are labelled Apache-2.0. **Whisper-base's encoder** (about 20 M parameters) is what the current competitors use. It is the efficiency-optimal choice, but Whisper's pretraining saw only Yoruba and Hausa among the targets. A HuBERT-base model truncated to 4–6 layers (about 38–52 M) sits between the two and is the natural compromise to test. **Do not use omniASR-W2V encoders**: they were trained on the prohibited `facebook/omnilingual-asr-corpus`.

### Cited Findings

**HF classes available without custom code.** The `AutoModelForAudioClassification` mapping includes hubert→HubertForSequenceClassification, wav2vec2→Wav2Vec2ForSequenceClassification, **wav2vec2-bert→Wav2Vec2BertForSequenceClassification**, wav2vec2-conformer, wavlm→WavLMForSequenceClassification, **whisper→WhisperForAudioClassification**, data2vec-audio, sew, sew-d, unispeech and unispeech-sat — [transformers modeling_auto.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/auto/modeling_auto.py). E-Branchformer (XEUS) is not in the mapping.

**Main comparison table.** Params are encoder-only as used by the classification head.

| HF id | model_type → AutoModelForAudioClassification class | Params (full encoder) | Licence (HF card) | Pretraining data | Target-language coverage (of yor, ibo, hau, ego, anw, efi, ibb, idu, igl, nup, tiv, urh) | Overlap with prohibited corpora | Published LID evidence | Truncation notes |
|---|---|---|---|---|---|---|---|---|
| [utter-project/mHuBERT-147](https://huggingface.co/utter-project/mHuBERT-147) | hubert → HubertForSequenceClassification ([config](https://huggingface.co/utter-project/mHuBERT-147/blob/main/config.json): 12 layers, 768-d, group norm) | 94.37 M (computed; safetensors present) | CC-BY-NC-SA-4.0 | 90,430 h, 147 languages; MLS 50.7k h, VoxPopuli 15.5k, Common Voice v11 14.9k, VoxLingua107 5.8k, BibleTTS 358 h, etc. ([paper](https://arxiv.org/abs/2406.06371)) | Language tags include **yo, ha, ig**. No Lower Cross languages (no efi, ibb or anw) and no other targets ([card](https://huggingface.co/utter-project/mHuBERT-147)) | None listed. CV v11 is in its pretraining and Hypa-Voices includes "Common Voice-derived subsets", so there is a small theoretical overlap **[uncertain]** | ML-SUPERB LID accuracy 85.3 (10 min) / 91.0 (1 h), against MMS-1B 84.8/86.1, XLS-R-300M 66.9/87.9 and MMS-300M 62.3/84.3 ([paper Table 3](https://arxiv.org/html/2406.06371)). FLEURS-African SLID F1 88.0 (25 langs) / 85.8 (21 African) ([AfriHuBERT paper](https://arxiv.org/html/2409.20201)) | Per transformer layer about 7.09 M. Fixed part (CNN 4.2 M + pos-conv 4.72 M + proj 0.39 M) about 9.3 M (computed) |
| [ajesujoba/AfriHuBERT](https://huggingface.co/ajesujoba/AfriHuBERT) (= AfriHuBERT-**n**) | hubert → HubertForSequenceClassification (12 layers, 768-d; `pytorch_model.bin` only, so re-save as safetensors) | about 94.4 M (same architecture as mHuBERT-147) | Card says only "cc". It is continued pretraining of CC-BY-NC-SA mHuBERT-147, so treat it as **NC-SA** | "over 10,000 hours" in 1,230 languages (1,226 African), 11 sources: BibleTTS 357.6 h (6 langs), CSRC 0.1 h, Jesus Dramas 99.6 h (88 langs), Kallaama 124.9 h, MCV 1,606 h (4 langs), **MMS ulab v2 2,835 h (1,230 langs)**, **NaijaVoices 1,873.9 h (3 langs)**, NCHLT 1,889 h, Nicolingua 142 h, VoxLingua107 886 h (13 langs), Zambezi Voice 176 h ([card + sources table](https://huggingface.co/ajesujoba/AfriHuBERT/blob/main/images/afrihuberts_sources.png)) | Card language metadata (1,228 codes) includes **yor, ibo, hau, ego, anw, ibb, idu, igl, nup, tiv, urh**. **efi is not listed** (checked from the card's YAML). So 2 of the 3 Lower Cross languages are covered (ibb, anw) | No prohibited corpus listed. Created 2024-09, before the omnilingual corpus (Nov 2025) existed ([HF API](https://huggingface.co/api/models/ajesujoba/AfriHuBERT)) | FLEURS-African SLID F1 91.6 / 90.0, i.e. +3.6 over mHuBERT-147 ([paper](https://arxiv.org/html/2409.20201)). No layer-wise analysis in the paper | Same as mHuBERT-147. k-means targets were taken from layer 9 of mHuBERT-147 ([paper](https://arxiv.org/html/2409.20201)) |
| [ajesujoba/AfriHuBERTs](https://huggingface.co/ajesujoba/AfriHuBERTs) (= AfriHuBERT-**s**, trained from scratch, one iteration) | hubert (12 layers, 768-d) | about 94.4 M | "cc" (unspecified) | Same 11 sources, about 10k h | Same 1,230-language claim | None listed | **Best small model on FLEURS-African SLID: F1 93.2 / 92.0** ([paper](https://arxiv.org/html/2409.20201)) | Same as above |
| [ajesujoba/AfriHuBERTo](https://huggingface.co/ajesujoba/AfriHuBERTo) | hubert (12 layers) | about 94.4 M | **CC-BY-NC-ND-4.0**: derivatives cannot be shared, so it is **unusable for a public fine-tuned repo** | Same | Same | None listed | F1 90.3 / 88.9 | — |
| [ajesujoba/AfriHuBERT-large](https://huggingface.co/ajesujoba/AfriHuBERT-large) (new, 2026-06-23) | hubert, 24 layers, 1024-d, from `facebook/hubert-large-ll60k` | about 316 M (300M+ per card) | **Apache-2.0** on card. Note that some of its training sources are NC-SA or NC-ND, so the label may not match the data licences **[uncertain]** | Same 11 sources, more than 10k h | 1,230 languages | None listed | No published LID numbers found (it cites an unpublished 2026 Saarland PhD thesis) | Wav2vec2-large-style layer about 12.6 M. Keeping 6–8 layers gives about 89–114 M |
| [ajesujoba/AfriHuBERT-xlarge](https://huggingface.co/ajesujoba/AfriHuBERT-xlarge) | hubert, 48 layers, 1280-d | about 1 B | Apache-2.0 on card | Same | Same | None listed | None found | Teacher only (exceeds 350 M) |
| [Orange/SSA-HuBERT-base-60k](https://huggingface.co/Orange/SSA-HuBERT-base-60k) (also -V2, [-large-60k](https://huggingface.co/Orange/SSA-HuBERT-large-60k) 315 M, -XL 962 M) | hubert (12 layers) | 94.37 M (safetensors) | CC-BY-NC-4.0 | About 60k h of studio and street-interview speech, 21 Sub-Saharan languages | Only **hau** among the targets | None listed | FLEURS-African SLID F1 89.6 / 88.0 ([AfriHuBERT paper](https://arxiv.org/html/2409.20201)) | Same as HuBERT-base |
| [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) / [-base](https://huggingface.co/openai/whisper-base) / [-small](https://huggingface.co/openai/whisper-small) | whisper → WhisperForAudioClassification (encoder only; 4 / 6 / 12 layers; 384 / 512 / 768-d) | Encoder: **8.2 M / 20.6 M / 88.2 M** (computed; full seq2seq 37.8 / 72.6 / 241.7 M) | Apache-2.0 | 680k h weakly supervised, 99 languages ([paper](https://arxiv.org/abs/2212.04356)) | Language tokens include **yo, ha**. **ig is not one of them.** None of the minority targets ([generation_config](https://huggingface.co/openai/whisper-base/blob/main/generation_config.json)) | None | Zero-shot Whisper LID on FLEURS 64.5% (102 langs) and 80.3% on the 82 languages it covers ([Whisper paper Table 5, §3.6](https://arxiv.org/abs/2212.04356)) | Per-layer: tiny 1.77 M, base 3.15 M, small 7.09 M (computed). The HF encoder demands mel length = `max_source_positions`×2 (default 3000 frames = 30 s) ([modeling_whisper.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py)) |
| [facebook/mms-300m](https://huggingface.co/facebook/mms-300m) | wav2vec2 → Wav2Vec2ForSequenceClassification (24 layers, 1024-d, stable layer norm) | about 315 M (MMS paper: 317 M) | CC-BY-NC-4.0 | About 491k h, 1,406 languages: MMS-lab-U (55k h, 1,362 langs, New Testament readings), FLEURS, VoxPopuli-400k, MLS, CV, VoxLingua107, BABEL ([MMS paper](https://arxiv.org/abs/2305.13516)) | Not listed per language in what I read. Its sibling MMS-LID-1024 covers all 12 targets using the same MMS-lab-U/unlab sources, so the pretraining very likely includes them **[inference]** | None. Predates all three prohibited corpora | ML-SUPERB LID 62.3 (10 min) / 84.3 (1 h). FLEURS-African SLID F1 86.3 / 85.6 ("MMS") | Per layer 12.6 M. Fixed part (CNN + pos-conv + proj) about 13.1 M |
| [facebook/wav2vec2-xls-r-300m](https://huggingface.co/facebook/wav2vec2-xls-r-300m) | wav2vec2 (24 layers, 1024-d) | about 315 M | Apache-2.0 | 436k h, 128 languages: VoxPopuli, MLS, CommonVoice, BABEL, VoxLingua107 ([card](https://huggingface.co/facebook/wav2vec2-xls-r-300m)) | yo and ha via VoxLingua107. Igbo possibly via BABEL **[uncertain]** | None | ML-SUPERB LID 66.9 / 87.9. FLEURS-African SLID F1 80.3 / 78.2 (the weakest in the AfriHuBERT table). VoxLingua107 LID 94.3% ([MMS paper Table 7](https://arxiv.org/abs/2305.13516)) | Tjandra et al.: keeping 8 of 24 blocks loses no accuracy (300 M → 100 M) (see Q2) |
| [microsoft/wavlm-base-plus](https://huggingface.co/microsoft/wavlm-base-plus) | wavlm → WavLMForSequenceClassification (12 layers, 768-d) | about 94.7 M | **Not stated on the HF card** (the upstream unilm repo is MIT) **[uncertain]** | 94k h of **English only**: Libri-Light 60k, GigaSpeech 10k, VoxPopuli 24k ([card](https://huggingface.co/microsoft/wavlm-base-plus)) | None | None | No African LID evidence found | Monolingual, so a poor prior for this task |
| [facebook/w2v-bert-2.0](https://huggingface.co/facebook/w2v-bert-2.0) | **wav2vec2-bert → Wav2Vec2BertForSequenceClassification** (24 Conformer layers, 1024-d) | **580.49 M** (computed). **24.18 M per layer**; the fixed part is only 0.16 M | MIT | 4.5M h unlabeled, "more than 143 languages" ([card](https://huggingface.co/facebook/w2v-bert-2.0)) | Per-language list not checked **[gap]** | None listed | FLEURS-African SLID F1 92.7 / 91.3 (second best overall) ([AfriHuBERT paper](https://arxiv.org/html/2409.20201)) | Too big as a whole. Truncated to k layers it is about 24.2·k M: k=4 about 97 M, k=6 about 145 M, k=8 about 194 M, k=12 about 290 M (computed). Loadable with `num_hidden_layers=k` |
| [espnet/xeus](https://huggingface.co/espnet/xeus) | **No HF transformers class** (E-Branchformer, ESPnet fork only) | 577 M | CC-BY-NC-SA-4.0 | More than 1M h, 4,000+ languages. Tags include espnet/mms_ulab_v2, YODAS, FLEURS, VoxPopuli, MLS, GigaSpeech ([card](https://huggingface.co/espnet/xeus)) | Likely broad (4,000 langs) **[uncertain]** | None listed | **Best on FLEURS-African SLID: F1 96.2 / 95.5** ([AfriHuBERT paper](https://arxiv.org/html/2409.20201)) | Cannot be submitted, so teacher only |
| [ntu-spml/distilhubert](https://huggingface.co/ntu-spml/distilhubert) | hubert (2 transformer layers, 768-d) | 23.49 M (safetensors) | Apache-2.0 | LibriSpeech, English | None | None | None for African LID | Shows that a 2-layer HuBERT is about 23.5 M; the same size as AfriHuBERT truncated to k=2 |
| [ylacombe/omniASR_W2V_300M_SSL](https://huggingface.co/ylacombe/omniASR_W2V_300M_SSL), [aadel4/omniASR-W2V-300M](https://huggingface.co/aadel4/omniASR-W2V-300M), [gclose19/omnilingual-asr-ssl-hf](https://huggingface.co/gclose19/omnilingual-asr-ssl-hf) | wav2vec2 (24 layers, 1024-d; HF ports of Meta Omnilingual fairseq2 checkpoints) | 317.4 M / 315.4 M | Apache-2.0 | Omnilingual ASR pretraining; the HF card tags dataset **facebook/omnilingual-asr-corpus** | Very broad (1,600+ languages) | **PROHIBITED OVERLAP.** The omnilingual corpus is listed as training data, and its public configs include **ego_Latn, anw_Latn, idu_Latn** ([corpus card](https://huggingface.co/datasets/facebook/omnilingual-asr-corpus)) | — | **Do not use** |

**Approximate truncated sizes (computed from the per-layer counts above).**
- **HuBERT-base family** (mHuBERT-147, AfriHuBERT-n/s, SSA-HuBERT-base), about 9.3 M + 7.09 M·k: k=2 about 23.5 M, k=3 about 30.6 M, k=4 about 37.7 M, k=6 about 51.8 M, k=8 about 66.0 M, k=12 about 94.4 M. The classification head adds about 0.2 M.
- **wav2vec2/HuBERT-large family** (XLS-R-300M, MMS-300M, AfriHuBERT-large), about 13.1 M + 12.6 M·k: k=4 about 63.5 M, k=6 about 88.7 M, k=8 about 114 M.
- **Whisper encoders**: tiny about 8.2 M, base about 20.6 M, small about 88.2 M.

**Existing fine-tuned African or DSN LID models on HF** (from today's [HF API](https://huggingface.co/api/models?search=dsn2026) search):
- **Competitor models.** `datapsalm/dsn2026-lid-w01/w02/w03/f02a/f02b` were created 2026-10-19/20, which matches the leader's profile.
  - Each is **WhisperForAudioClassification with Whisper-base's 6-layer, 512-d encoder**, **20,213,268 params in fp16**, 13 labels and `use_weighted_layer_sum=True`.
  - They set **`max_source_positions=500`** with a WhisperFeatureExtractor of **`chunk_length=10`, `n_samples=160000`, `nb_max_frames=1000`**. That means 10-s input windows instead of 30 s, done purely through config with no custom code ([config](https://huggingface.co/datapsalm/dsn2026-lid-w03/blob/main/config.json), [preprocessor](https://huggingface.co/datapsalm/dsn2026-lid-w03/blob/main/preprocessor_config.json)).
- `EmmanuelHardey/afrihubert` (created 2026-10-20): HubertForSequenceClassification, 12 layers, 94,571,917 params in fp32, with exactly the 13 competition labels ([HF](https://huggingface.co/EmmanuelHardey/afrihubert)).
- `Ololade/dsn-lid-baseline`: Wav2Vec2ForSequenceClassification, 24 layers, 315.7 M in fp16, 13 labels ([HF](https://huggingface.co/Ololade/dsn-lid-baseline)).
- `drrobot9/wav2vec2-nigerian-lid-v4`: XLS-R-300M-style 24-layer model, labels hausa/igbo/yoruba/english, Apache-2.0. It claims 0.9989 eval accuracy on undocumented data ([HF](https://huggingface.co/drrobot9/wav2vec2-nigerian-lid-v4)).
- `olaolugbenle/african-lid-v2`: XLS-R-300M fine-tuned on Google WaxalNLP for ful/lin/sna only ([HF](https://huggingface.co/olaolugbenle/african-lid-v2)).
- `UBC-NLP/afrolid_1.5` is a **text** LID model, not speech ([HF search](https://huggingface.co/api/models?search=afrolid)).
- No public speech LID model was found that covers the minority targets (efi, ibb, anw, igl, nup, tiv, urh, ego, idu) apart from Meta's MMS-LID.

### Inferences
- **Trade-offs in efficiency multipliers.** These use linear interpolation of the user-supplied multipliers (21 M ≈ 0.94, 94 M ≈ 0.89, 316 M ≈ 0.80), about −0.0007 per extra million params.
  - A 4-layer HuBERT (about 38 M) gets roughly 0.93 and needs only about 1% relative F1 over Whisper-base to break even.
  - A 6-layer HuBERT (about 52 M) gets about 0.92 and needs about 2%.
  - A full 94 M model needs more than 5.6%.
  - A 316 M model needs more than 17.5% (0.94/0.80).
  - With the leader at F1 ≈ 0.08 on efi, ibb and igl, an encoder that has actually heard Lower Cross and Igala speech could plausibly gain far more than these margins. AfriHuBERT-n lists ibb, anw and igl; mHuBERT-147 and Whisper list none of them.
- **Recommended shortlist.** In order:
  1. AfriHuBERT-n or -s truncated to 4–8 layers. Note that -s and -n differ by licence clarity, not quality.
  2. Whisper-base encoder, the 20 M baseline, possibly with a 10-s window as the competitor does.
  3. AfriHuBERT-large truncated to 4–6 layers (64–89 M), if its Apache-2.0 label is acceptable to the organisers.
  4. w2v-BERT 2.0 truncated to 2–4 layers (48–97 M). It has a strong African SLID score at full depth, but there is no evidence for such a shallow truncation.
- AfriHuBERT and mHuBERT-147 repos ship `pytorch_model.bin` (AfriHuBERT) or a safetensors conversion (mHuBERT). Either way, `save_pretrained()` after fine-tuning writes safetensors by default, so the no-.bin rule is not a blocker.
- Whisper's fixed-length input means latency is set by the window, not the clip. A 10-s window (`max_source_positions=500`) cuts encoder cost about 3×, but it truncates 10–30 s clips to their first 10 s. HuBERT/wav2vec2 models process variable-length audio, and their cost scales with clip length.

### Gaps
- No per-language LID results were found for AfriHuBERT on efi, ibb, anw, igl, nup, tiv, urh, ego or idu. Its SLID benchmark is FLEURS, which contains none of them.
- The per-language pretraining hours for the minority targets inside MMS ulab v2 (in AfriHuBERT) and inside MMS-300M were not obtained.
- I did not check the w2v-BERT 2.0 per-language pretraining list (Seamless paper) for the 12 targets.
- Whether XLS-R's BABEL subset includes Igbo was not verified.
- WavLM-base-plus licence: none on the HF card.
- Actual CPU/GPU latency per 30-s clip on the scorer's hardware is unknown, and no benchmark was found.
- No 2025–2026 SSL encoders from Masakhane, Lelapa or Intron were found on HF in these searches. The only new African SSL releases found were AfriHuBERT-large/-xlarge/-s (2026-06) and SSA-HuBERT V2/large/XL (2025-02).

## Q2. Which layers carry the most language-ID information in HuBERT/wav2vec2-family models, and does truncation keep LID accuracy?

### Takeaway
Two direct studies show that truncation to about 1/3–1/2 of depth keeps (or even improves) LID accuracy, and that top layers are not needed. They disagree on whether the very lowest layers are good: XLSR shows layers 8+ of 24 far better than 2–6, while an NVIDIA Conformer shows the lower third to half best. For a 12-layer HuBERT-base, keeping about 4–8 layers is supported by analogy, but no study tested HuBERT-base or mHuBERT/AfriHuBERT specifically.

### Cited Findings
- **Tjandra et al., ICASSP 2022** (Meta; XLSR-53-style 300 M wav2vec 2.0, 24 blocks, 26-language in-house LID):
  - Feeding the output of each block to the LID classifier showed "the middle and upper parts of the pre-trained model (from 8th to 24th layer) perform significantly better than the lower part (from 2nd to 6th layer)".
  - "We could prune up to 2/3 of the context encoder blocks… while maintaining good accuracy". Keeping 8 blocks cuts parameters from 300 M to 100 M.
  - With 10 min of labelled data per language, XLSR reaches more than 89.2%.
  - On pooling, mean+max (93.2%) and mean (described as "works very well") beat [CLS] (91.7%) and self-attention (92.2%).
  - Source: [arXiv 2107.04082](https://arxiv.org/abs/2107.04082).
- **Bartley et al., "Accidental Learners"** (NVIDIA; 18-layer multilingual Conformer pretrained on VoxPopuli, evaluated on VoxLingua107):
  - Fine-tuned sub-models of 7 and 10 layers outperformed the others.
  - Benchmark eval error was 7 layers 7.27%, 8 layers 6.90%, **9 layers 5.41%**, 10 layers 6.46%, and **all 18 layers 6.71%**. XLS-R (300 M) scored 5.7%.
  - With frozen encoders, sub-models up to the lower third to bottom half were best. Middle and upper layers helped more for **languages unseen in pretraining**.
  - Source: [arXiv 2211.05103](https://arxiv.org/abs/2211.05103).
- **Neural factor analysis paper**: language and speaker discriminability rise from layer 1 to layer 6, then deeper layers perform worse; supervised LID was also best at layer 6. This comes from the search-result summary only; I did not read the model details in full **[uncertain]** — [arXiv 2305.08099](https://arxiv.org/abs/2305.08099).
- The MMS LID models fine-tune the full 48-layer MMS-1B with a linear classifier on top (no truncation) ([MMS paper §6.1](https://arxiv.org/abs/2305.13516)).
- AfriHuBERT built its iteration-4 targets from **layer 9** of mHuBERT-147 and reports no layer analysis. mHuBERT-147 used layers 6 and 9 from earlier iterations for clustering ([AfriHuBERT](https://arxiv.org/html/2409.20201), [mHuBERT-147](https://arxiv.org/html/2406.06371)).
- HF `WhisperForAudioClassification` supports `use_weighted_layer_sum` (a learnable softmax over layer outputs) together with a `projector` of size `classifier_proj_size` ([modeling_whisper.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py)). The competitor Whisper models enable it ([config](https://huggingface.co/datapsalm/dsn2026-lid-w03/blob/main/config.json)).

### Inferences
- **Practical recipe.** Load the encoder with `config.num_hidden_layers = k` (HuBERT, wav2vec2, w2v-BERT) or `encoder_layers = k` (Whisper). The checkpoint loader then simply drops the unused upper layers, so no custom code is needed.
- **Choice of k.** Sweep k ∈ {3, 4, 6, 8} for HuBERT-base derivatives with `use_weighted_layer_sum=True` and mean pooling. Tjandra's "8 of 24 blocks" corresponds to about 4 of 12 for base models. Bartley's "9 of 18" corresponds to about 6 of 12.
- **Teacher layers.** Since mid layers carry LID information, distillation can also match a teacher's intermediate-layer features (DistilHuBERT-style) rather than only logits.

### Gaps
- No published layer-wise LID probe was found for mHuBERT-147, AfriHuBERT, WavLM or w2v-BERT 2.0.
- No study was found of truncation below 1/3 depth (e.g. 2 of 12 layers) for LID.

## Q3. Whisper: which targets are in its training set, and how well do Whisper encoders transfer to LID on unseen African languages?

### Takeaway
Of the 12 targets, Whisper's 99-language vocabulary has only **Yoruba (yo) and Hausa (ha)**. Igbo is absent, and so are all 9 minority languages. The training-data figure lists Yoruba (432 h) and Hausa (8 h) only in the X→en translation portion. Zero-shot Whisper LID is weak, and comparative work found XLS-R adapts better than Whisper to unseen Atlantic-Congo languages. Whisper-base's appeal is therefore efficiency (about 20 M params), not language prior.

### Cited Findings
- **Language coverage**: Whisper's `lang_to_id` has 99 languages and includes `<|yo|>`, `<|ha|>`, `<|sw|>`, `<|ln|>`, `<|sn|>`, `<|so|>`, `<|am|>` and `<|af|>`. It **does not include `<|ig|>`** ([whisper-base generation_config.json](https://huggingface.co/openai/whisper-base/blob/main/generation_config.json)).
- **Training data**: in Whisper's Figure 11 (training dataset statistics), Yoruba (432 h) and Hausa (8 h) appear in the translation column (125,739 h total), alongside Swahili 282, Shona 279 and Lingala 20. They do not appear in the multilingual ASR column (117,113 h), where Swahili has 5.4 h and Afrikaans 4.1 h. This is my reading of the extracted figure text **[medium confidence]** ([Whisper paper](https://arxiv.org/abs/2212.04356)).
- **Zero-shot LID**: Whisper's FLEURS LID accuracy is 64.5%. Whisper has "no training data for 20 of the 102 languages in Fleurs, upper-bounding accuracy at 80.4%", and it reaches 80.3% on the 82 overlapping languages. That is 13.6% below supervised SOTA ([Whisper paper §3.6, Table 5](https://arxiv.org/abs/2212.04356)). MMS lists Whisper at 64.5% against MMS-LID at 97%+ on FLEURS ([MMS Table 7](https://arxiv.org/abs/2305.13516)).
- **Transfer to unseen languages** (Interspeech 2023, Rouditchenko et al.): fine-tuning on 13 FLEURS languages unseen by both models, **XLS-R beat Whisper-Medium on Atlantic-Congo, Afro-Asiatic and Nilo-Saharan languages**. Performance tracked pretraining exposure to the language family. This is from the search-result summary; the paper evaluates ASR, not LID **[uncertain whether LID was tested]** — [arXiv 2305.12606](https://arxiv.org/abs/2305.12606).
- **African ASR benchmark** (2025; 13 African languages, 1–400 h): MMS and w2v-BERT were more data-efficient in very low-resource regimes, XLS-R scaled better with more data, and Whisper had advantages in mid-resource conditions. This is ASR, not LID, and comes from the search summary — [arXiv 2512.10968](https://arxiv.org/abs/2512.10968).
- **HF input handling**: the Whisper encoder raises an error unless the mel input has exactly `max_source_positions × conv strides` frames (3000 by default) ([modeling_whisper.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py)). Competitors set `max_source_positions=500` with a 10-s feature extractor ([datapsalm config](https://huggingface.co/datapsalm/dsn2026-lid-w03/blob/main/config.json)).

### Inferences
- The leader's F1 ≈ 0.08 on efi, ibb and igl fits Whisper having no exposure to Lower Cross or Igala phonotactics. An Africa-pretrained encoder such as AfriHuBERT (which lists ibb, anw, igl, nup, tiv, urh, ego and idu) is the most direct fix. Distillation from MMS-LID-1024+, which has explicit efi/ibb/igl classes, is another.
- Whisper-tiny's encoder (about 8.2 M) is an even cheaper option, but it has no evidence for this task.

### Gaps
- No paper was found that fine-tunes Whisper encoders for LID on unseen African languages and compares them with SSL encoders. The evidence above is ASR-based or zero-shot.

## Q4. Which large models are the best teachers for distillation or pseudo-labels, and what is their label coverage?

### Takeaway
**facebook/mms-lid-1024, -2048 and -4017 are the only off-the-shelf speech LID models with explicit classes for all 12 targets**. mms-lid-512 covers all except Eggon. They are about 965 M Wav2Vec2ForSequenceClassification models under CC-BY-NC-4.0, with about 97% FLEURS accuracy but only about 80–82% on out-of-domain African/Asian telephone speech. Their training is dominated by single-speaker Bible-reading data, so check them on the DSN training set before trusting pseudo-labels. SpeechBrain's VoxLingua107 ECAPA and Whisper only cover yo and ha, which makes them useful for "other" detection rather than for the minority classes. Fine-tuning a large encoder (XEUS, w2v-BERT 2.0 full, AfriHuBERT-xlarge or MMS-1B) on the competition data is a stronger teacher option.

### Cited Findings
- **MMS-LID label coverage** (checked against each `config.json` `id2label` today):

| Teacher | # labels | yor | ibo | hau | ego | anw | efi | ibb | idu | igl | nup | tiv | urh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [mms-lid-126](https://huggingface.co/facebook/mms-lid-126) | 126 | ✓ | ✓ | ✓ | – | – | – | – | – | – | – | – | – |
| [mms-lid-256](https://huggingface.co/facebook/mms-lid-256) | 256 | ✓ | ✓ | ✓ | – | – | – | ✓ | – | – | – | ✓ | – |
| [mms-lid-512](https://huggingface.co/facebook/mms-lid-512) | 512 | ✓ | ✓ | ✓ | **–** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| [mms-lid-1024](https://huggingface.co/facebook/mms-lid-1024) | 1024 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| [mms-lid-2048](https://huggingface.co/facebook/mms-lid-2048) | 2048 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| [mms-lid-4017](https://huggingface.co/facebook/mms-lid-4017) | 4017 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

- **MMS-LID architecture**: Wav2Vec2ForSequenceClassification with 48 layers, 1280-d. Safetensors totals are 966.1 M (126) to 970.1 M (4017), CC-BY-NC-4.0 ([HF API](https://huggingface.co/api/models/facebook/mms-lid-1024)).
- **MMS-LID test accuracy** ([MMS paper Table 7](https://arxiv.org/abs/2305.13516)):

| Model | FLEURS (102 langs) | VoxLingua107 (33) | BABEL (23 African+Asian, out-of-domain) | VoxPopuli (25) |
|---|---|---|---|---|
| 126 | 97.5 | 93.9 | 84.1 | 87.3 |
| 256 | 97.2 | 93.4 | 80.1 | 87.6 |
| 512 | 96.8 | 92.9 | 81.6 | 85.6 |
| 1024 | 97.0 | 92.8 | 80.5 | 86.2 |
| 2048 | 97.3 | 92.8 | 81.5 | 86.6 |
| 4017 | 97.2 | 93.9 | 80.5 | 87.1 |

- **Prior work in the same MMS table**: Whisper 64.5 (FLEURS, 82 langs); SpeechBrain 93.3 (VoxLingua107); XLS-R 94.3 (VoxLingua107); AmberNet 95.3 (VoxLingua107) ([MMS paper](https://arxiv.org/abs/2305.13516)).
- **MMS-LID training data**: the models are trained by fine-tuning MMS-1B on MMS-lab-U + MMS-unlab + FLEURS + VoxLingua107. MMS-lab is New Testament readings that are "almost always single speaker" in a narrow domain. MMS-unlab has 7.7k h in 3,809 languages ([MMS paper §3, §6](https://arxiv.org/abs/2305.13516)).
- No per-language accuracy was reported for any of the 9 minority targets.
- **SpeechBrain [lang-id-voxlingua107-ecapa](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa)**: ECAPA-TDNN covering 107 languages, including **Hausa and Yoruba but not Igbo** ([label_encoder.txt](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa/blob/main/label_encoder.txt)). It reports a 6.7% error rate on the VoxLingua107 dev set, and the card warns that "accuracy on smaller languages is quite limited". Apache-2.0, SpeechBrain library (not HF transformers).
- **Whisper LID**: covers yo and ha only, with 64.5% zero-shot FLEURS accuracy (see Q3).
- **XEUS** (577 M, ESPnet) has the best FLEURS-African SLID after fine-tuning (F1 96.2) ([AfriHuBERT paper](https://arxiv.org/html/2409.20201)). **w2v-BERT 2.0** (580 M) scores 92.7. Both require fine-tuning on labelled data to act as a teacher.
- **AfriHuBERT-xlarge** (about 1 B, Apache-2.0 label, 2026-06) is a further large Africa-centric candidate teacher. No LID results found ([HF](https://huggingface.co/ajesujoba/AfriHuBERT-xlarge)).

### Inferences
- **Best teacher strategy**:
  1. Score the DSN train/dev clips with mms-lid-1024 or -4017, restricting the softmax to the 12 target codes plus "other" (pooling the remaining labels' mass) to measure zero-shot per-class accuracy.
  2. If it is good on efi, ibb and igl, use its restricted logits as soft targets or pseudo-labels for unlabelled Nigerian audio.
  3. Separately, fine-tune a large encoder (MMS-1B/mms-lid-4017 itself, w2v-BERT 2.0 or AfriHuBERT-large) on DSN data as an in-domain teacher, then distil into the truncated student.
- Efik, Ibibio and Annang are closely related Lower Cross languages. Expect MMS-LID confusions among them, so check them first.
- MMS-LID licence (CC-BY-NC) only matters if its weights end up in the submission. Using it for pseudo-labels does not redistribute it, though the competition rules on teacher models are unknown.

### Gaps
- Zero-shot accuracy of MMS-LID on the minority targets (or on DSN data) is not published.
- No reported accuracy was found for any teacher specifically on efi, ibb, anw, igl, nup, tiv, urh, ego or idu.
- Whether Meta released an omnilingual LID model was not checked; it would be prohibited-data-trained anyway.

## Q5. Licences and overlap with the prohibited corpora

### Takeaway
Permissive (submission-safe) backbones are Whisper (Apache-2.0), XLS-R-300M (Apache-2.0), w2v-BERT 2.0 (MIT), DistilHuBERT (Apache-2.0) and, by label, AfriHuBERT-large/-xlarge (Apache-2.0). mHuBERT-147, XEUS (NC-SA), MMS (NC) and SSA-HuBERT (NC) would force a non-commercial (and for NC-SA, share-alike) licence on the public submission. AfriHuBERTo (NC-ND) cannot be redistributed as a derivative at all. Only the omniASR encoders have documented overlap with a prohibited corpus.

### Cited Findings
- **Licence labels on HF cards** (all from today's HF API):
  - CC-BY-NC-SA-4.0: [mHuBERT-147](https://huggingface.co/utter-project/mHuBERT-147), [XEUS](https://huggingface.co/espnet/xeus).
  - "cc", unspecified: [AfriHuBERT (-n)](https://huggingface.co/ajesujoba/AfriHuBERT), [AfriHuBERTs](https://huggingface.co/ajesujoba/AfriHuBERTs).
  - **CC-BY-NC-ND-4.0**: [AfriHuBERTo](https://huggingface.co/ajesujoba/AfriHuBERTo).
  - **Apache-2.0**: [AfriHuBERT-large](https://huggingface.co/ajesujoba/AfriHuBERT-large), [AfriHuBERT-xlarge](https://huggingface.co/ajesujoba/AfriHuBERT-xlarge), Whisper tiny/base/small, [XLS-R-300M](https://huggingface.co/facebook/wav2vec2-xls-r-300m), [DistilHuBERT](https://huggingface.co/ntu-spml/distilhubert), [ylacombe omniASR port](https://huggingface.co/ylacombe/omniASR_W2V_300M_SSL).
  - CC-BY-NC-4.0: [MMS-300M](https://huggingface.co/facebook/mms-300m), MMS-1B, all MMS-LID models, [SSA-HuBERT](https://huggingface.co/Orange/SSA-HuBERT-base-60k).
  - MIT: [w2v-BERT 2.0](https://huggingface.co/facebook/w2v-bert-2.0).
  - Not stated: [WavLM-base-plus](https://huggingface.co/microsoft/wavlm-base-plus).
- **AfriHuBERT training-data licences**: BibleTTS CC BY-SA 4.0, CSRC CC-BY, Jesus Dramas CC BY-NC-SA 4.0, Kallaama CC BY-SA 4.0, MCV CC-0, MMS ulab v2 CC BY-NC-SA 4.0, **NaijaVoices CC BY-NC-SA 4.0**, NCHLT CC BY 3.0, Nicolingua CC BY-SA 4.0, VoxLingua107 CC BY 4.0, Zambezi Voice **CC BY-NC-ND 4.0** ([AfriHuBERT sources table](https://huggingface.co/ajesujoba/AfriHuBERT/blob/main/images/afrihuberts_sources.png)). The -large and -xlarge cards show the same sources table yet carry Apache-2.0.
- **Prohibited corpus 1, facebook/omnilingual-asr-corpus**: CC-BY-4.0, 348 under-served languages. Its public configs include **ego_Latn, anw_Latn, idu_Latn**; none for yor, ibo, hau, efi, ibb, igl, nup, tiv or urh (checked from the card YAML) ([corpus card](https://huggingface.co/datasets/facebook/omnilingual-asr-corpus)). omniASR SSL ports list it as training data ([ylacombe card](https://huggingface.co/ylacombe/omniASR_W2V_300M_SSL)).
- **Prohibited corpus 2, hypaai/Hypa-Voices**: Apache-2.0. The full pool is about 8,442 h over 22 source languages (public subset 8,800 examples).
  - Language tags include yo, ha, ig, ibb, tiv, efi, anw, igl, nup, ego, urh and "idc". `idu` itself is not tagged.
  - Its sources include "AfroVoices recordings… and supplementary open speech material (including Mozilla Common Voice-derived subsets for some languages)" ([dataset card](https://huggingface.co/datasets/hypaai/Hypa-Voices)).
- **Prohibited corpus 3, AfriSpeech/african-speech-ipa**: CC-BY-4.0. It is AfriSpeech `african-speech-public_v1` audio with IPA added, covering 141 languages ([card](https://huggingface.co/datasets/AfriSpeech/african-speech-ipa)). The underlying recording sources were not documented in what I read.
- None of the candidate backbone cards (mHuBERT-147, the AfriHuBERT family, SSA-HuBERT, Whisper, MMS, XLS-R, WavLM, w2v-BERT 2.0, XEUS) lists Hypa-Voices, the omnilingual corpus or AfriSpeech as pretraining data (cards cited above).

### Inferences
- **Safest high-prior choice**: AfriHuBERT-n/s, if NC-SA output is acceptable, or AfriHuBERT-large truncated, if the organisers accept its Apache-2.0 label. On data overlap, neither lists a prohibited source, and AfriHuBERT-n predates the omnilingual corpus.
- **Residual overlap risk**: AfriHuBERT, mHuBERT-147, XLS-R and MMS all pretrain on Common Voice, and Hypa-Voices includes CV-derived subsets. Identical CV clips could therefore appear in both. This is SSL pretraining only (no labels), so it is probably a low risk, but it is worth a line in the write-up **[uncertain]**.
- **Unverified risk**: if AfriSpeech public_v1 was built from Bible or Jesus-film recordings, then MMS (MMS-lab), AfriHuBERT (MMS ulab v2, Jesus Dramas) and XEUS (mms_ulab_v2) could share source audio with the test set **[uncertain]**.
- The submission repo must be public, so an NC-SA base forces the fine-tuned model to be published as CC-BY-NC-SA-4.0. Whether the competition forbids NC licences is not known.

### Gaps
- Competition rules on backbone licences (NC/NC-SA allowed?) and on teacher-model use were not found in my sources.
- The provenance of AfriSpeech public_v1 audio is unknown, so overlap with Bible-based pretraining corpora can be neither confirmed nor ruled out.
- I could not verify how AfriHuBERT-large/-xlarge reconcile an Apache-2.0 label with NC-SA/NC-ND training data.

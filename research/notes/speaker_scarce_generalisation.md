# Speaker-scarce generalisation for DSN 2026 Nigerian spoken LID (speaker augmentation, synthetic speech, speaker/channel-invariant training)

Scope note: the setting is a ≤350M-param HF AutoModelForAudioClassification (e.g. Wav2Vec2ForSequenceClassification) and a speaker-disjoint test set. The five 1–3-speaker classes (nup 1 spk, igl 2, ibb 2, tiv 2, efi 3) together carry about 43.5% of the fixed-share F1 (5 × 8.7%). The deadline is 2026-10-21 23:59 UTC. All HF/GitHub checks below were run on 2026-10-20 from this session.

## Objective: ranked techniques (expected gain for this setting × hours to implement)

### Takeaway
The best-supported edge is manufacturing speaker diversity that is independent of the class label. Do it with (1) kNN-VC conversion to a shared pool of target voices applied to every class, plus (2) cheap NANSY-style formant/F0/EQ "information perturbation" and channel/codec augmentation, also on every class. The direct evidence: a 315M MMS-300m Wav2Vec2ForSequenceClassification dialect-ID model gained +34% relative cross-domain accuracy from kNN-VC, while class-specific target voices collapsed it to chance. Two more methods add value in hours: distilling from facebook/mms-lid-1024/2048/4017, which already cover all 12 target languages plus pcm and eng but are about 967M params, so usable only as a teacher; and a cosine/AAM or LDA-refit head that folds into the standard classifier. TTS is essentially a dead end for the scarce classes. Speaker-adversarial GRL is low-confidence and medium effort.

### Cited Findings
Ranked list. "Gain" is my expected-gain judgement for this competition, "h" is my estimate of hours to implement, and the evidence is in the cited items in each section below.

1. **kNN-VC speaker augmentation to a class-independent target pool, applied to all classes.** Gain: high. 3–5 h, most of it GPU conversion time.
   - Evidence: in Arabic dialect ID with MMS-300m, kNN-VC with 4 target voices took cross-domain accuracy from 60.22% to 80.73% and in-domain from 75.94% to 85.32%. "Biased" (dialect-specific) target speakers collapsed to 24–27%, near chance — [Abdullah et al., arXiv 2505.24713](https://arxiv.org/html/2505.24713v1).
   - The released model is a 315.7M-param Wav2Vec2ForSequenceClassification fine-tuned from facebook/mms-300m (CC-BY-4.0), the same form factor as this competition — [HF API: badrex/mms-300m-arabic-dialect-identifier](https://huggingface.co/api/models/badrex/mms-300m-arabic-dialect-identifier).
   - Code: [bshall/knn-vc](https://github.com/bshall/knn-vc) (MIT, `torch.hub.load('bshall/knn-vc','knn_vc', prematched=True)`).
2. **On-the-fly "information perturbation" plus speed perturbation on all classes.** Formant scaling, F0 scaling and random parametric EQ, with speed 0.9–1.1. Gain: medium-high. 1–2 h.
   - Evidence: speaker-disjoint wav2vec2 accent ID used exactly this perturbation to break speaker reliance — [Yang et al., arXiv 2306.06524](https://arxiv.org/html/2306.06524v1).
   - Pitch shift alone gave +10.6% relative cross-domain in dialect ID — [arXiv 2505.24713](https://arxiv.org/html/2505.24713v1).
   - Augmentation including speed perturbation cut LID Cavg from 0.166 to 0.140 — [Snyder et al., Odyssey 2018](https://danielpovey.com/files/2018_odyssey_xvector_lid.pdf).
3. **Channel augmentation for phone and noisy test audio, on all classes.** GSM/AMR or other codecs, 8 kHz band-limiting, RIR, MUSAN noise, random EQ; optionally Freq-MixStyle-like statistic mixing. Gain: medium-high for the phone/noisy part of the test set. 1–2 h.
   - Evidence: LID systems simulate GSM-AMR plus MUSAN plus RIR — [Snyder 2018](https://danielpovey.com/files/2018_odyssey_xvector_lid.pdf). The ML-SUPERB 2.0 winner used noise and reverb at p=0.5 — [TalTech, arXiv 2506.01458](https://arxiv.org/html/2506.01458v1).
   - Freq-MixStyle is the standard device-generalisation trick in DCASE — [CP-JKU DCASE2022](https://dcase.community/documents/challenge2022/technical_reports/DCASE2022_Schmid_77_t1.pdf).
4. **Knowledge distillation from facebook/mms-lid-1024/2048/4017** into the ≤350M student. Map every non-target label to "other". Gain: potentially high for the scarce classes, because the teacher has heard other speakers of efi/ibb/igl/nup/tiv. Unverified, so check zero-shot accuracy on dev first. 3–5 h.
   - The coverage check is my own (config.json id2label, see Section 3).
5. **Cosine/AAM-softmax training, or an LDA/Gaussian-backend refit, folded into the HF `classifier` Linear.** Gain: medium. 1–2 h.
   - Evidence: the embedding plus Gaussian backend beat direct softmax classification out-of-domain, VAST Cavg 0.149 vs 0.256 — [Snyder 2018](https://danielpovey.com/files/2018_odyssey_xvector_lid.pdf).
   - The ML-SUPERB 2.0 winner used LDA plus logistic regression on embeddings — [arXiv 2506.01458](https://arxiv.org/html/2506.01458v1).
6. **Speaker-balanced sampling, short random crops and leave-one-speaker-out validation.** Gain: medium (it keeps the 4.27 h of tiv from being memorised as 2 voices, and model selection becomes honest). Under 1 h. This is my own inference; see Section 5.
7. **Latent mixup / Map-Mix.** Gain: low-medium. 1–2 h. Evidence: +2% weighted F1 over random mixup with XLSR LID — [Map-Mix, arXiv 2302.08229](https://arxiv.org/abs/2302.08229); code at [skit-ai/map-mix](https://github.com/skit-ai/map-mix).
8. **Gradient-reversal adversary against VC-target-speaker, gender or source labels**, with the head dropped at export. Gain: low-medium and uncertain. 2–3 h. Evidence is mixed — [Meng et al., arXiv 1804.00732](https://arxiv.org/abs/1804.00732); [Adi et al., arXiv 1812.03483](https://arxiv.org/pdf/1812.03483).
9. **Stronger SpecAugment** (HF built-in `mask_time_prob`, `mask_feature_prob`). Gain: low-medium; the evidence conflicts. Minutes to change config values. Sources: [HF Wav2Vec2Config](https://huggingface.co/docs/transformers/model_doc/wav2vec2); [Baas & Kamper, arXiv 2111.02674](https://arxiv.org/abs/2111.02674v1); [arXiv 2505.24713](https://arxiv.org/html/2505.24713v1).
10. **TTS synthetic speech.** Gain: about zero for the scarce classes, because no checkpoints exist (Section 2). The only multilingual zero-shot TTS covering anw/ego/ibb/idu (OmniVoice) was trained on the banned Omnilingual ASR corpus, so do not use it.

### Inferences
- Methods 1–3 are complementary and all live purely in the training data, so they carry no export risk.
- Methods 4–5 live in the weights and export cleanly into a standard ForSequenceClassification.
- Method 8 needs a wrapper during training and a head stripped at save time.
- If only about 6 working hours are available, do 2 + 3 first (about 2 h), then 1 (about 3–4 h including conversion), then 5. Do 4 if a GPU is free to run the 1B teacher.
- The standard recipe that most competitors will follow (class weights plus noise) does nothing to break the "class = 1–2 voices" shortcut. Speaker-identity randomisation that is uniform across classes is the main differentiator.

### Gaps
- No paper evaluates kNN-VC or information perturbation for LID (as opposed to dialect/accent ID) with 1–3 speakers per class. The effect size for this setting is an extrapolation.
- kNN-VC conversion throughput on the available GPU was not measured. Benchmark it on 1 minute of audio before committing.

## 1. Speaker augmentation to manufacture new "speakers" (VTLP, pitch/formant, speed, voice conversion)

### Takeaway
There is strong evidence that VC-based speaker diversification improves speaker- and domain-independent classification in dialect ID (+34% relative cross-domain accuracy) and in very-low-resource ASR (up to 7% absolute WER). This holds only when target voices are not correlated with the label: dialect-specific targets drove accuracy to chance. Apply VC, and every other speaker transform, to all classes with the same target pool and the same probability. Keep the original audio. Prefer 4 or more target voices.

### Cited Findings
**Voice conversion for dialect ID** (Abdullah et al., Interspeech 2025, [arXiv 2505.24713](https://arxiv.org/html/2505.24713v1))
- Setup: MMS-300m fine-tuned for 5-way dialect ID on MGB-3 ADI-5 (about 53.6 h), tested on a new out-of-domain set (MADIS-5: radio, TEDx, TV drama, theatre).
- VC system: kNN-VC. Target voices were native-language LibriVox speakers with about 1 minute of reference audio each. Each training sample was converted once and combined with the natural speech.
- Cross-domain average accuracy:

  | Condition | Accuracy | Relative change |
  |---|---|---|
  | Baseline | 60.22% | |
  | SpecAugment | 66.29% | |
  | Pitch shift | 66.62% | |
  | RIR | 62.69% | |
  | Additive noise | 66.76% | |
  | All four of those augmentations (5×N data) | 70.75% | |
  | VC, 1 voice | 72.72% | |
  | VC, 2 voices | 76.65% | |
  | VC, 4 voices | 80.73% | +34.07% |

- In-domain accuracy: 75.94% → 85.32%.
- Shortcut test: "unbiased" VC (12 target speakers uniformly distributed across classes, no natural speech) reached 83.38% in-domain and 76.61% cross-domain. "Biased" VC (target speakers specific to each dialect) collapsed to 27.33% and 24.32%, near the 20% chance level.
- The tables are slightly inconsistent: the baseline is 60.22% vs 59.60% in different tables.
- Released model: badrex/mms-300m-arabic-dialect-identifier is a Wav2Vec2ForSequenceClassification with 315,702,405 params, base_model facebook/mms-300m, licence CC-BY-4.0 — [HF API](https://huggingface.co/api/models/badrex/mms-300m-arabic-dialect-identifier).

**Retrieval-based VC for German dialect classification** — [Fischbach et al., Interspeech 2025, arXiv 2507.03641](https://arxiv.org/abs/2507.03641v1)
- Converting all audio to one uniform target speaker improved low-resource German dialect classification. Combining it with frequency masking and segment removal gave further gains.
- The abstract gives no numbers.

**Voice conversion for very-low-resource ASR** (Baas & Kamper, Interspeech 2022, [arXiv 2111.02674](https://arxiv.org/abs/2111.02674v1); [ISCA](https://www.isca-archive.org/interspeech_2022/baas22_interspeech.html))
- Method: an English-trained VC used to augment unseen low-resource languages (Afrikaans, Setswana, isiXhosa, Sepedi). XLSR-53 was fine-tuned on 10 minutes of data per language, and the splits had no speaker overlap.
- Adding 100% extra VC data improved all four languages, by 7% absolute WER for isiXhosa.
- 500% VC data, and VC chained with SpecAugment, made results worse than no augmentation. SpecAugment alone also hurt in that very-low-resource setting.
- On English (10 min), VC gave 47.7 → 43.8 WER and VC+SpecAugment at 500% gave 42.4. At 1 h and 10 h of real data, extra VC data hurt (30.4 → 32.7/34.4 at 1 h).
- The authors attribute this to VC's intelligibility cost once natural speaker variety is already sufficient.

**kNN-VC** (Baas, van Niekerk & Kamper, Interspeech 2023, [arXiv 2305.18975](https://arxiv.org/abs/2305.18975); [repo](https://github.com/bshall/knn-vc))
- Mechanism: WavLM-Large layer-6 features of the source are replaced by the mean of the k nearest reference-speaker frames, then vocoded with HiFi-GAN.
- Only torch, torchaudio and numpy are needed, and it loads through torch.hub.
- The README recommends clean reference audio. Quality improves with more reference audio, with diminishing returns beyond about 5 minutes.
- LibriSpeech dev-clean: WER 6.29%, EER 35.73% (prematched vocoder).
- Licence: MIT (LICENSE file, raw.githubusercontent.com check).

**Other VC licences** (from each repo's LICENSE file on raw.githubusercontent.com)
- [Seed-VC](https://github.com/Plachtaa/seed-vc): GPL-3.0.
- [OpenVoice](https://github.com/myshell-ai/OpenVoice): MIT (2024 MyShell.ai).
- [FreeVC](https://github.com/OlaWod/FreeVC): MIT.
- [RVC WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI): MIT.

**Formant/F0/EQ perturbation for speaker-disjoint accent ID** — [Yang et al., arXiv 2306.06524](https://arxiv.org/html/2306.06524v1)
- Finding: a wav2vec2 accent identifier "strongly relies on speaker identity", which gives high training accuracy but poor generalisation to unseen speakers. The test split is speaker-disjoint: 28 speakers, one held out per accent.
- Intervention, applied on the fly with probability 0.75:
  - scale all formants by β1;
  - scale F0 by β2;
  - apply a random frequency-shaping equaliser;
  - β1 and β2 are drawn from U(1, 1.4) and inverted with probability 0.5.
- No before/after ablation numbers are reported.

**The same idea in content-representation work**
- "Information perturbation" (formant, pitch, frequency response) is the speaker-removal mechanism of NANSY — [Choi et al., NeurIPS 2021](https://papers.neurips.cc/paper_files/paper/2021/file/87682805257e619d49b8e0dfdc14affa-Paper.pdf).
- It is also used in ContentVec's speaker-disentangled HuBERT — [Qian et al., ICML 2022](https://proceedings.mlr.press/v162/qian22b.html).

**Speed perturbation**
- Factors 0.9/1.0/1.1 gave an average 4.3% relative WER gain over 4 LVCSR tasks — [Ko et al., Interspeech 2015](https://www.isca-archive.org/interspeech_2015/ko15_interspeech.html).
- Implemented as torchaudio.transforms.SpeedPerturbation — [torchaudio docs](https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.SpeedPerturbation.html).
- In x-vector LID, a 6-fold augmentation (speed 0.9/1.1, MUSAN music/noise, reverb) lowered overall Cavg from 0.166 to 0.140; "removing augmentation significantly degrades performance" — [Snyder et al., Odyssey 2018](https://danielpovey.com/files/2018_odyssey_xvector_lid.pdf).

**Augmentation for low-resource SSL pre-training**
- Combined noise and pitch augmentation outperformed adding accented or other-language speech — [arXiv 2309.12763](https://arxiv.org/abs/2309.12763). The abstract gives no numbers.

**Zindi Wolof ASR write-up**
- Speed reduction plus reverb (p=0.5) took WER from about 12% to 9.9%. Noise, gain and pitch shift did not help in that setting — [Zindi discussion](https://zindi.africa/competitions/ai4d-baamtu-datamation-automatic-speech-recognition-in-wolof/discussions/5763) (via search snippet; not opened in full).

### Inferences
**Recipe for kNN-VC augmentation (about 3–5 h)**
- Target pool: 8–20 target voices that are not tied to any class.
  - Candidate sources: ethically licensed multi-speaker corpora such as LibriSpeech/LibriVox (public domain or CC BY), VCTK (CC BY) or Common Voice (CC0).
  - Plus training-set speakers from the many-speaker classes, used across all classes.
  - Give each target voice 1–5 minutes of reference audio.
- Mix genders: nup's single speaker presumably has one gender, and a gender shortcut is likely.
- Convert every training clip of the scarce classes 4–8 times (each time to a different random target). Convert the other classes 1–2 times with the same pool, so that "converted-sounding" audio is not class-correlated.
- Keep all originals, and keep the converted:real ratio the same across classes if possible. If budget forces asymmetry, at least ensure every class contains VC audio.
- Abdullah's "biased" result shows the catastrophic failure mode is target voices correlated with class, not VC per se.
- Chain on-the-fly channel augmentation after VC, so vocoder artifacts are further masked.
- Sanity check: train a quick probe to predict "is VC" from the model's embeddings. If it also predicts class above chance on VC audio only, there is leakage.

**kNN-VC caveat for LID (uncertain)**
- kNN-VC outputs are averages of target-speaker WavLM frames, so phones absent from the reference audio's language (for example labial-velars /kp/, /gb/, or tone contours) may be approximated by the nearest frames. Language cues could be partially overwritten.
- Mitigations:
  - use ≥3–5 min of phonetically varied reference audio;
  - prefer references drawn from several languages, or from Nigerian-language speech of the large classes, used uniformly across all classes;
  - keep the originals;
  - optionally compare with OpenVoice or FreeVC, which synthesise from source content features rather than retrieving target frames.
- The Abdullah gains used same-language (Arabic) target speakers, so the cross-language case is untested.

**Recipe for information perturbation (about 1–2 h)**
- Apply on the fly to every class with p≈0.5–0.75.
- Formant shift via praat-parselmouth "Change gender" with a formant ratio drawn from 1/1.3–1.3 (my implementation suggestion, not from a cited source).
- F0 median scaling ×U(0.7, 1.4), which preserves relative tone contours, important for tonal yor/ibo/tiv/nup/igl/idu.
- A random 3–7-band parametric EQ (audiomentations has SevenBandParametricEQ, BandPassFilter, PitchShift, TimeStretch, ApplyImpulseResponse and Mp3Compression — names from my knowledge, verify in the installed version).
- Speed perturbation 0.9–1.1.
- Pitch shift alone is weaker than VC: +10.6% vs +34% relative in ADI.

**VTLP**
- Formant scaling plus speed perturbation covers what VTLP does. There is no reason to implement spectrogram-domain VTLP separately for a raw-waveform wav2vec2 model.

### Gaps
- No verified effect-size numbers for VTLP (Jaitly & Hinton 2013); the original paper was not retrieved.
- No published study of VC augmentation for LID (as opposed to dialect ID) with 1–3 speakers per class, and none on tonal or Niger-Congo languages.
- Seed-VC, OpenVoice and RVC have no LID/ASR augmentation studies that I found. Their cross-lingual content preservation for Nigerian tone languages is untested.
- The Fischbach RVC effect sizes are not available from the abstract.

## 2. Synthetic speech via TTS for the low-resource classes

### Takeaway
MMS-TTS has no checkpoint for any of the 5 speaker-scarce classes, nor for idu, anw, ego, urh or even ibo. Only yor and hau (plus pcm and eng) exist, all CC-BY-NC-4.0. No public TTS exists on HF for efi, ibb, igl, nup, tiv, urh, anw or ego.

The one multilingual zero-shot TTS that covers anw/ego/ibb/idu (k2-fsa/OmniVoice) lists the banned Meta Omnilingual ASR corpus as training data. Using it risks disqualification. TTS is therefore not a viable lever for the scarce classes, and for yor/hau/ibo it is unnecessary because those classes already have many speakers.

### Cited Findings
**MMS-TTS availability**
- HF API check of https://huggingface.co/api/models/facebook/mms-tts-<code> on 2026-10-20. "Exists" returned model JSON. "Absent" returned HTTP 401 `{"error":"Invalid username or password."}`, which is HF's response for a repo that does not exist or is not public.

  | Code | HF transformers checkpoint | Licence | Original fairseq tarball |
  |---|---|---|---|
  | yor | exists — [facebook/mms-tts-yor](https://huggingface.co/api/models/facebook/mms-tts-yor), VitsModel, lastModified 2023-09-01 | cc-by-nc-4.0 | 200 |
  | hau | exists — [facebook/mms-tts-hau](https://huggingface.co/api/models/facebook/mms-tts-hau), VitsModel | cc-by-nc-4.0 | 200 |
  | pcm ("other" class) | exists — [facebook/mms-tts-pcm](https://huggingface.co/api/models/facebook/mms-tts-pcm) | cc-by-nc-4.0 | 200 |
  | eng ("other" class) | exists — [facebook/mms-tts-eng](https://huggingface.co/api/models/facebook/mms-tts-eng) | cc-by-nc-4.0 | not checked |
  | efi, ibb, igl, nup, tiv, urh, idu, anw, ego, ibo | absent | n/a | 403 |

- Original fairseq tarballs were checked at https://dl.fbaipublicfiles.com/mms/tts/<code>.tar.gz.
- The [MMS TTS language list](https://dl.fbaipublicfiles.com/mms/tts/all-tts-languages.html) contains "yor Yoruba", "hau Hausa" and "pcm Pidgin, Nigerian", and no entry for Igbo or the other 9 target codes.
- MMS-TTS is single-speaker per language (VITS) — [MMS paper arXiv 2305.13516](https://arxiv.org/abs/2305.13516). This is from my knowledge of the model; the per-language card was not re-read.

**Community TTS on HF**
- Search of HF models with pipeline_tag=text-to-speech, by language name and ISO code, on 2026-10-20:
  - Igbo: [multilingual-tts/VITS-OpenBible-Igbo](https://huggingface.co/multilingual-tts/VITS-OpenBible-Igbo) (CC-BY-SA-4.0), F5-TTS-OpenBible-Igbo (CC-BY-SA-4.0), samuelolubukun/xtts-v2-igbo-openbible ("other"), Shinzmann/soro-tts-ibo (CC-BY-NC-4.0), rnjema-unima/mms-tts-ibo-baseline (CC-BY-NC-4.0).
  - Yoruba and Hausa: many, e.g. multilingual-tts/VITS-OpenBible-Yoruba and -Hausa (CC-BY-SA-4.0), CLEAR-Global/TWB-Voice-Hausa-TTS-1.0 (CC-BY-NC-4.0), naijaml/f5-tts-yoruba (CC-BY-NC-4.0).
  - Idoma: only "mrheartng/idoma-mms-tts-eng" (0 downloads, unverified).
  - No results for efi/Efik, ibb/Ibibio, tiv, nupe, igala, urhobo, annang or eggon.

**OmniVoice** ([arXiv 2604.00688](https://arxiv.org/html/2604.00688v1); HF [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice), about 1.8M downloads, base Qwen/Qwen3-0.6B)
- Claims 600+ languages and was trained on 581k h aggregated from about 50 open datasets.
- Appendix A explicitly includes the "Meta Omnilingual ASR Corpus", alongside Emilia, YODAS, Common Voice, FLEURS and others.
- Appendix D training hours: Yoruba 15.66, Igbo 13.69, Hausa 17.75, Idoma 11.16, Annang 9.65, Eggon 9.95, Ibibio 7.38.
- The HF language tags include ibb, idu, anw, ego, yo, ig, ha, pcm and en, but not efi, tiv, nup, igl or urh (checked via HF API).
- No licence field in the HF card metadata.

**Literature on TTS augmentation**
- Speaker diversity is what makes synthetic speech useful. A zero-shot voice-cloning TTS with many voices was "much more effective" than a conventional TTS with few voices, few-voice synthetic data saturates quickly, and high-variability synthetic data kept helping even after training data grew by 30% — [Czyżnikiewicz et al., FedCSIS 2024, arXiv 2406.07090](https://arxiv.org/abs/2406.07090).
- Other findings from the search snippet only (paper not opened):
  - Hao et al. (LREC): TTS helps most with under about 1 h of real speech or over about 10 h of TTS data, at a sensible synthetic:real ratio;
  - Gronings: up to 25.5% relative WER reduction.
- I found no paper on TTS-augmented spoken LID; this was searched and nothing direct turned up.

**Bible-domain shortcut risk**
- Single-speaker Bible training data (CMU Wilderness: mostly one, typically male, speaker per language, often dramatised with music) generalised poorly to other speakers and domains.
- In the SIGTYP 2021 shared task, the baseline got 0.96 F1 in-domain but 13.7% accuracy on multi-speaker Common Voice/OpenSLR test data. The best team reached 53.1%, mainly by adding multi-speaker validation samples to training — [SIGTYP 2021, arXiv 2106.03895](https://arxiv.org/html/2106.03895v1).

### Inferences
- Do not use OmniVoice, or any model trained on facebook/omnilingual-asr-corpus, to synthesise anw/ego/ibb/idu. Even though OmniVoice itself is not on the banned list, its outputs for these languages largely encode the banned corpus (about 7–11 h each), and organisers could reasonably treat that as indirect use.
- Possibly the banned corpus overlaps the competition's own data sources (speculative).
- Training your own TTS on the 1-speaker nup data just reproduces the same voice, so it is pointless without VC on top. VC on real recordings (Section 1) gives the same speaker diversity with natural prosody and no text requirement.
- MMS-TTS for yor/hau/pcm/eng would add a single synthetic voice to classes that already have many speakers. The risk that "TTS timbre = yor/hau" becomes a shortcut outweighs any benefit. If used at all, also pass the TTS output through VC, and synthesise for "other" (eng/pcm) as well.
- Text sources, only if TTS were pursued: Bible translations (eBible.org) exist for most of these languages, but they are religious-register text, a distinct domain. Wikipedia exists for yo/ig/ha. NaijaVoices supplies Igbo/Hausa/Yoruba text and audio. Not verified per language.
- CC-BY-NC licences (MMS-TTS, MMS-LID, NaijaVoices) may or may not be acceptable under the competition rules; check and disclose.

### Gaps
- Did not verify eBible.org or Wikipedia coverage per language (efi, ibb, igl, nup, tiv, urh, idu, anw, ego).
- No direct evidence on TTS augmentation for LID, positive or negative.
- The OmniVoice licence was not confirmed: blog claims Apache-2.0, unverified.

## 3. Speaker- and channel-invariant training (GRL/DANN, speaker normalisation, instance-norm/MixStyle)

### Takeaway
Speaker-adversarial GRL has only modest, mixed evidence: about 5% relative WER in ASR, and "no benefit for large models" in another study. I found no direct LID-with-SSL result. The LID evidence for adversarial or unsupervised domain adaptation (9–77% relative accuracy) needs target-domain audio.

A GRL head can be dropped at export trivially. The bigger risk is that with 1 speaker in nup, a speaker adversary is confounded with the language label. If used, adversarially remove VC-target-speaker identity (class-independent by construction), gender or recording-source labels instead.

Channel invariance is better bought with augmentation (codecs, EQ, RIR, noise) and parameter-free statistic mixing (Freq-MixStyle style), which needs no export changes. A strong, cheap, in-the-weights speaker-diversity source is distillation from facebook/mms-lid-* (all 12 languages covered).

### Cited Findings
**Gradient-reversal and domain adaptation evidence**
- Speaker-invariant training via a GRL speaker adversary: 4.99% relative WER reduction on CHiME-3 vs a speaker-independent baseline — [Meng et al., ICASSP 2018, arXiv 1804.00732](https://arxiv.org/abs/1804.00732).
- Adversarial vs multi-task speaker learning push features in opposite directions. Deep models trained on large data already learn speaker-invariant representations without an auxiliary loss, so the adversary's benefit is small — [Adi et al., ICASSP 2019, arXiv 1812.03483](https://arxiv.org/pdf/1812.03483).
- Unsupervised domain adaptation for 6-language Slavic LID (read speech → radio broadcast) gave 9%–77% relative accuracy improvements, depending on source-domain diversity — [Abdullah et al., Interspeech 2020, arXiv 2008.00545](https://arxiv.org/abs/2008.00545).
- Channel-robust LID without domain labels: an MMD regulariser on RATS reached supervised-level performance on target domains without labelled target data — [Duroselle et al., Odyssey 2020](https://superlectures.com/odyssey2020/unsupervised-regularization-of-the-embedding-extractor-for-robust-language-identification).
- A within-sample similarity loss as an alternative to adversarial multi-task learning when channel labels are unavailable — [Muralikrishna et al., ICASSP 2021](https://rc.signalprocessingsociety.org/conferences/icassp-2021/spsicassp21vid1659) (search snippet only).
- Cross-corpus Indian LID: domain-invariant learning suited cross-corpora generalisation better than domain-aware learning — Dey, Sahidullah & Saha 2023, Computer Speech & Language (search snippet only; [summary search result](https://researcher.manipal.edu/en/publications/spoken-language-identification-in-unseen-channel-conditions-using/)).

**Speaker-controlled LID: TidyLang 2026 (Odyssey 2026)**
- Challenge design: speakers each speak 2–10 languages (Tidy-X from Common Voice), so speaker cues cannot help.
- Winner: 96.8% macro accuracy, using an acoustic plus LLM-over-transcripts ensemble, which is not usable here — [Olev & Alumäe](https://www.isca-archive.org/odyssey_2026/olev26_odyssey.html).
- Another team used margin-based losses on ECAPA to reduce speaker interference: 85.95% macro accuracy — [Fang et al., arXiv 2605.01905](https://arxiv.org/abs/2605.01905).
- A third used a speaker-aware MoE on fine-tuned wav2vec2 — [Penagarikano et al.](https://www.isca-archive.org/odyssey_2026/penagarikano26_odyssey.html).
- No ablation numbers for speaker-removal tricks were obtained.

**Freq-MixStyle**
- Normalises per-frequency-band statistics and re-mixes them with another sample's (Beta(α) weights, probability p).
- Used by CP-JKU's rank-1 DCASE 2022 low-complexity ASC system — [DCASE2022 Schmid](https://dcase.community/documents/challenge2022/technical_reports/DCASE2022_Schmid_77_t1.pdf); [workshop paper](https://dcase.community/documents/workshop2022/proceedings/DCASE2022Workshop_Schmid_27.pdf).
- Adopted as the device-mismatch remedy in the DCASE 2024/2025 baselines — [arXiv 2505.01747](https://arxiv.org/pdf/2505.01747).
- Complementary to device-impulse-response augmentation — [JKU EUSIPCO 2023](https://research.jku.at/en/publications/device-robust-acoustic-scene-classification-via-impulse-response-/).

**Where speaker information sits in SSL layers**
- In wav2vec2, phone information peaks in late-middle layers — [arXiv 2406.09200](https://arxiv.org/pdf/2406.09200).
- Larger SSL models recover speaker information in deep layers — [arXiv 2501.05310](https://arxiv.org/pdf/2501.05310).
- Phone, tone and speaker subspaces are largely orthogonal — [arXiv 2506.10855](https://arxiv.org/abs/2506.10855v1).

**HF head structure (exportability)**
- `Wav2Vec2ForSequenceClassification` = optional softmax-weighted layer sum (`use_weighted_layer_sum`) → `projector` Linear(hidden→`classifier_proj_size`=256) → masked mean-pool → `classifier` Linear(256→num_labels). Read from source lines 1656–1733 — [modeling_wav2vec2.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/modeling_wav2vec2.py); config defaults in the [HF docs](https://huggingface.co/docs/transformers/model_doc/wav2vec2).

**MMS-LID as a teacher (own check)**
- Read config.json id2label from HF on 2026-10-20:

  | Checkpoint | Coverage of the 12 targets + pcm + eng |
  |---|---|
  | [facebook/mms-lid-1024](https://huggingface.co/facebook/mms-lid-1024), [-2048](https://huggingface.co/facebook/mms-lid-2048), [-4017](https://huggingface.co/facebook/mms-lid-4017) | all of efi, ibb, igl, nup, tiv, urh, idu, anw, ego, yor, ibo, hau, pcm, eng |
  | mms-lid-512 | all except ego |
  | mms-lid-256 | ibb, tiv, yor, ibo, hau, pcm, eng only |
  | mms-lid-126 | yor, ibo, hau, eng only |

- mms-lid-1024 is a Wav2Vec2ForSequenceClassification with 967,009,920 params; mms-lid-4017 has 970,077,745. Both are cc-by-nc-4.0 (HF API). They are too big to submit (cap 350M) but usable as teachers.

### Inferences
**Recipe for GRL (2–3 h; optional, after Sections 1–2)**
- Wrap the HF model and run the forward pass with `output_hidden_states`.
- Take the mean-pooled `projector` output (256-d) and feed it through GRL(λ) → 2-layer MLP → adversary logits.
- Adversary targets, best first:
  - (a) the VC target-speaker ID of converted clips (uniform across classes, so removing it cannot remove language);
  - (b) gender;
  - (c) recording source/session or channel type, if metadata exists;
  - (d) original speaker ID only within multi-speaker classes. Mask the loss for nup, because nup's speaker ID equals its class.
- Ramp λ with the DANN schedule λ_p = 2/(1+e^(−10p)) − 1, capped at about 0.1–0.3. That schedule is from Ganin & Lempitsky ([arXiv 1409.7495](https://arxiv.org/abs/1409.7495)), cited from my knowledge, not re-read.
- At save time, call `model.save_pretrained()` on the inner ForSequenceClassification only. The adversary weights then never enter the checkpoint, and AutoModelForAudioClassification loads it without custom code.

**Freq-MixStyle analogue for wav2vec2 (inference, untested for SSL LID)**
- Apply MixStyle to the CNN feature-encoder output (512 channels × frames): per-channel mean/std over time, mixed between batch items with Beta(0.1–0.3), p≈0.5, training only.
- It is parameter-free, so export is unaffected. A forward hook on `model.wav2vec2.feature_extractor` that is active only in training mode needs no architecture change.

**Per-utterance normalisation**
- Per-utterance normalisation (instance-norm) of the input waveform is already standard in the Wav2Vec2FeatureExtractor (`do_normalize`).
- Per-utterance mean subtraction of hidden features cannot be expressed in the standard HF head, apart from the existing layer norms. Only use tricks that live in the training data or the weights.

**Recipe for MMS-LID distillation (3–5 h)**
1. Run mms-lid-1024 (or -4017) once over all training audio (originals and VC/perturbed copies).
2. Collapse its posteriors to 13 classes: the 12 targets, with all other labels summed into "other".
3. Check its zero-shot macro-F1 on a speaker-disjoint dev split. Expect weakness on noisy/phone audio and among anw/ibb/efi, since MMS-lab is largely Bible read speech (uncertain).
4. If it beats the student on the scarce classes, train the student with CE(hard labels) + α·T²·KL(teacher_T ∥ student_T), with α≈0.5 and T≈2.
5. Disclose this as use of a pretrained external model. It is not trained on the banned corpora; MMS (2023) predates the Omnilingual corpus (2025).

**LID head on a weighted layer sum**
- Setting `use_weighted_layer_sum=True` in config is a standard HF option. It lets the head emphasise middle layers (phonetic/language) over late layers that may carry speaker information. Cheap to try, but unverified for this task.

### Gaps
- No paper found applying GRL speaker-adversarial training to SSL-based LID with an effect size.
- No TidyLang ablations isolating speaker-removal techniques.
- Did not retrieve MMS-LID's accuracy for these specific Nigerian languages.
- Freq-MixStyle's effect on raw-waveform SSL models for speech LID is untested in the literature I found.

## 4. Metric learning / prototypical approaches before a standard classifier head

### Takeaway
Margin or cosine softmax (AM/AAM) is the norm in LID embedding systems, and it is claimed to reduce speaker interference. Embedding plus generative-backend classifiers beat direct softmax classification out-of-domain: Cavg 0.149 vs 0.256 on mismatched VAST data.

Both can be folded exactly into the HF `classifier` Linear layer. Use L2-normalised class weights with zero bias, or refit the layer in closed form as LDA, a shared-covariance Gaussian classifier whose pooled within-class covariance borrows strength from the many-speaker classes. Supervised-contrastive pre-fine-tuning for speech LID has no direct evidence that I found.

### Cited Findings
- **Snyder et al. 2018**, x-vector LID, direct softmax vs embeddings + Gaussian classifier:
  - Direct softmax: Cavg 0.155 (MLS14) and 0.256 (VAST).
  - Embeddings + Gaussian classifier: 0.130 and 0.149.
  - The out-of-domain gap is large — [Odyssey 2018](https://danielpovey.com/files/2018_odyssey_xvector_lid.pdf).
- **ABC LRE22** (African languages; [Matějka et al., Interspeech 2023](https://www.isca-archive.org/interspeech_2023/matejka23_interspeech.pdf)):
  - Embedding extractors trained with AM/AAM-softmax, mostly with margin 0 and scale 30–32, i.e. normalised-cosine softmax, then a GLC/PLDA backend.
  - XLS-R-1B fine-tuning gave >30% relative gain over ResNet embeddings.
  - Fusion gave 15–20% relative.
- **ML-SUPERB 2.0 winner** (TalTech): frozen W2V-BERT 2.0 → pooling → 512-d embedding → length-norm → LDA(100) → logistic regression on ≤1 h per language — [arXiv 2506.01458](https://arxiv.org/html/2506.01458v1).
- **TidyLang 2026** (speaker-controlled LID):
  - Margin losses on ECAPA-TDNN "help reduce the interference caused by speaker variation".
  - 85.95% macro accuracy, +45.7% over the baseline; no CE-only ablation retrieved.
  - [Fang et al., arXiv 2605.01905](https://arxiv.org/abs/2605.01905); code [PunkMale/TidyLang2026](https://github.com/PunkMale/TidyLang2026) ("will be released").
- **Triplet Entropy Loss** (CE + triplet) improved LID generalisation to new speakers and domains for short utterances (NCHLT → Lwazi, South African languages), "though not significantly" — [arXiv 2012.03775](https://arxiv.org/abs/2012.03775).
- **ConLID** is supervised contrastive learning for low-resource *text* LID, not speech — [arXiv 2506.15304](https://arxiv.org/pdf/2506.15304).
- **HF already ships an `AMSoftmaxLoss`** (scale 30, margin 0.4) in modeling_wav2vec2.py, used by Wav2Vec2ForXVector — [source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/modeling_wav2vec2.py).

### Inferences
**Recipe A: AM-softmax head (1 h)**
- Train with logits = s·(cos θ − m·onehot), with s≈30 and m≈0.1–0.2 (small, since there are few classes and "other" is heterogeneous).
- Compute it from L2-normalised pooled 256-d features and L2-normalised `classifier.weight` (bias frozen at 0).
- At export, write the normalised weight into `classifier.weight` with bias=0.
- HF computes W·x with an un-normalised x, but argmax(W·x) = argmax(cos θ), since ‖x‖ is a positive per-utterance constant. Decisions are identical; only softmax temperature differs.
- If calibrated probabilities matter for thresholding, multiply the weights by a scalar instead (still argmax-equivalent).

**Recipe B: LDA/GLC refit (1 h)**
- After fine-tuning, extract pooled 256-d features for all training clips, including augmented ones and VC copies of the scarce classes.
- Estimate class means μ_k and one pooled within-class covariance Σ.
- Set `classifier.weight[k] = Σ⁻¹μ_k` and `bias[k] = −½ μ_kᵀΣ⁻¹μ_k + log π_k`, where π_k is a uniform or F1-tuned prior.
- This is exactly linear, so it fits the standard head. The class-bias term is also the natural place to bake in per-class threshold/bias tuning for the fixed-share F1.
- Compare against the CE head on a speaker-disjoint dev split.

**Supervised contrastive pre-fine-tuning (lower priority, 2–3 h)**
- A SupCon stage with positives defined as the same language but a different speaker or a VC-converted version is theoretically aligned with the goal.
- No speech-LID evidence was found, so rank it below A and B under deadline pressure.

### Gaps
- No direct SSL-LID ablation comparing CE vs AAM vs SupCon on speaker-disjoint low-resource data was retrieved.
- The ConLID speech analogue is not found.

## 5. Mixup/CutMix, SpecAugment strength, random-crop length

### Takeaway
Mixup has modest evidence for SSL-based LID: Map-Mix with XLSR gave +2% weighted F1 over random latent mixup on a low-resource LRE17 subset. SpecAugment evidence conflicts: it helped dialect-ID cross-domain by +10% relative but hurt very-low-resource ASR fine-tuning.

Keep the HF built-in masking moderate, and spend effort on speaker and channel augmentation instead. Short random crops, around 2–6 s against test clips of 2–30 s, multiply the effective samples per speaker and match the short-test regime. This is standard in LID (x-vector training on 2–4 s chunks), though not ablated here.

### Cited Findings
- **Map-Mix** (datamaps-guided latent mixup on fine-tuned XLSR for LID): +2% weighted F1 over the random-mixup baseline, better calibration, aimed at low-resource dialect classification on an LRE 2017 subset — [Rajaa et al., ICASSP 2023, arXiv 2302.08229](https://arxiv.org/abs/2302.08229); [repo skit-ai/map-mix](https://github.com/skit-ai/map-mix).
- **MixRep** (hidden-representation mixup) for low-resource ASR — [Xie & Hansen, Interspeech 2023, arXiv 2310.18450](https://ar5iv.labs.arxiv.org/html/2310.18450).
- **SpecAugment results conflict**:
  - +10.08% relative cross-domain in Arabic dialect ID with MMS-300m — [arXiv 2505.24713](https://arxiv.org/html/2505.24713v1).
  - Worse than no augmentation in 10-min ASR fine-tuning of XLSR-53, and harmful when chained with VC — [Baas & Kamper, arXiv 2111.02674](https://arxiv.org/abs/2111.02674v1).
- **HF built-in SpecAugment** on feature-encoder outputs: `apply_spec_augment=True`, `mask_time_prob=0.05`, `mask_time_length=10`, `mask_feature_prob=0.0`, `mask_feature_length=10`, `layerdrop=0.1` by default — [HF Wav2Vec2Config](https://huggingface.co/docs/transformers/model_doc/wav2vec2).
- **Crop lengths**: x-vector LID training used 2–4 s chunks. NIST LRE test durations cluster around 3, 10 and 30 s — [Snyder 2018](https://danielpovey.com/files/2018_odyssey_xvector_lid.pdf).
- **Spectral masking** improved LID generalisation to new speakers and domains, "though not significantly" — [arXiv 2012.03775](https://arxiv.org/abs/2012.03775).

### Inferences
**Recipe**
- Random-crop 3 s (p=0.4), 6 s (p=0.4) or a full clip up to 15 s (p=0.2) per sample.
- For long recordings, sample many crops per epoch, but sample speakers, not seconds. For example, draw class → speaker → recording → crop, so tiv's 4.27 h from 2 speakers does not dominate.
- Evaluate on full clips; HF masked mean-pooling handles variable length.
- Set `mask_time_prob` to 0.05–0.1 and `mask_feature_prob` to 0.0–0.05. Keep `layerdrop` 0.05–0.1.

**Mixup choices**
- Prefer within-class, cross-speaker mixing for the scarce classes. One way: concatenate two crops from different speakers or VC voices of the same class into one 4–8 s training example. This breaks any "single consistent voice" cue without label ambiguity (my inference; no direct evidence).
- Cross-class waveform mixup or CutMix produces multi-language audio that is unlike the test data. If used, apply it in latent space (MixRep/Map-Mix style) at low α (0.2–0.4) and low probability, ideally excluding the closely related anw/ibb/efi triplet, to avoid blurring a decision boundary that is already hard.

**Validation**
- With 2–3 speakers, use leave-one-speaker-out folds for igl/ibb/tiv/efi.
- For nup (1 speaker), no speaker-disjoint validation is possible. Use held-out VC'd/perturbed and phone-simulated versions of held-out nup recordings as a weak proxy, and avoid early-stopping on nup accuracy.

### Gaps
- No direct evidence on CutMix for speech LID.
- No ablation on crop length for SSL LID fine-tuning found.
- Map-Mix's absolute numbers and the no-mixup baseline were not available from the abstract.

## 6. Lessons from 2021–2026 low-resource African and speaker-scarce speech challenges

### Takeaway
Across SIGTYP 2021, NIST LRE22, ML-SUPERB 2.0 (2025), TidyLang 2026 and Zindi, the repeatable score movers were:
1. getting some multi-speaker or in-domain data into training/backends (SIGTYP: 13.7% → 53.1%; LRE22: 5× worse without target-domain dev data);
2. a strong multilingual SSL backbone (LRE22: XLS-R >30% relative over ResNet);
3. noise/reverb/codec/speed augmentation;
4. embedding backends and fusion (15–20% relative).

Closely related languages (Xhosa/Zulu in LRE22) were consistently the hardest, which is a direct analogue of Annang/Ibibio/Efik.

### Cited Findings
- **SIGTYP 2021 robust LID** (16 languages; training on mostly single-speaker CMU Wilderness Bible speech; test on speaker-disjoint Common Voice/OpenSLR) — [arXiv 2106.03895](https://arxiv.org/html/2106.03895v1):
  - The baseline CNN got 0.96 F1 in-domain but 13.7% test accuracy.
  - Best team (Lipsia, ResNet-50, adding 60% of the multi-speaker validation set to training): 53.1% accuracy, 0.508 macro-F1.
  - NTR (QuartzNet with noise/time-shift/SpecAugment-style perturbation, no added multi-speaker data): 6.3%.
- **NIST LRE 2022** (14 target languages emphasising African ones; CTS + broadcast narrowband) — [NIST overview arXiv 2302.14624](https://arxiv.org/pdf/2302.14624); [Lee et al. Interspeech 2023](https://www.isca-archive.org/interspeech_2023/lee23c_interspeech.pdf):
  - 21 organisations in 16 teams submitted 65 systems.
  - Oromo and Tigrinya were easiest; Xhosa and Zulu (closely related) were hardest.
- **ABC team LRE22** — [Matějka et al.](https://www.isca-archive.org/interspeech_2023/matejka23_interspeech.pdf):
  - XLS-R-1B gave >30% relative gain over ResNet embeddings.
  - A backend that never saw target-domain data was 5× worse, so even a small target-domain dev set is essential.
  - Training the backend on dev plus an augmented copy of dev helped.
  - Fusion gave 15–20% relative.
- **IITKGP LRE22** (no pretrained models; augmentation + classifier fusion): EER 11.43% and cost 0.41 on dev — [arXiv 2501.08616](https://arxiv.org/html/2501.08616v1).
- **ML-SUPERB 2.0 (Interspeech 2025)** — [TalTech arXiv 2506.01458](https://arxiv.org/html/2506.01458v1); [overview arXiv 2505.24200](https://arxiv.org/html/2505.24200v2):
  - Winner TalTech: frozen SeamlessM4T-v2 W2V-BERT 2.0 encoder plus a VoxLingua107-trained pooling head with noise and reverb (p=0.5 each). LDA + LR backend on ≤1 h per language. 86.8% LID accuracy on the blind set.
  - Second place (CMU): few-shot data augmentation plus LID-aware CTC, 14% relative LID accuracy gain over baseline.
- **TidyLang 2026**: speaker-controlled LID (speakers using 2–10 languages) to stop speaker shortcuts. Winner 96.8% macro accuracy — [Olev & Alumäe, Odyssey 2026](https://www.isca-archive.org/odyssey_2026/olev26_odyssey.html); [flyer](https://odyssey2026.inesc-id.pt/wp-content/uploads/2026/02/Odyssey26_TidyLang_flyer.pdf).
- **Zindi**:
  - Wolof ASR: speed + reverb helped (WER about 12% → 9.9%); noise, gain and pitch shift did not. Another top solution found random validation splits misleading because of transcript overlap — [Zindi discussion 5763](https://zindi.africa/competitions/ai4d-baamtu-datamation-automatic-speech-recognition-in-wolof/discussions/5763); [6290](https://zindi.africa/competitions/ai4d-baamtu-datamation-automatic-speech-recognition-in-wolof/discussions/6290) (search snippets).
  - Swahili ASR winners used simple ensembles and noise robustness — [Zindi blog](https://zindi.africa/blog/winning-solution-to-the-swahili-asr-challenge).
- **NaijaVoices** (Interspeech 2025): Igbo/Hausa/Yoruba, about 1,838 h, 5,455 speakers. ASR fine-tuning WER improvements averaged 75.86% (Whisper), 52.06% (MMS) and 42.33% (XLSR). No LID results — [arXiv 2505.20564](https://arxiv.org/abs/2505.20564v1).
  - HF dataset [naijavoices/naijavoices-dataset](https://huggingface.co/datasets/naijavoices/naijavoices-dataset) is CC-BY-NC-SA-4.0 and gated (HF API).

### Inferences
- The SIGTYP result is the closest analogue to nup/igl: models trained on 1–2 voices per language can score in-domain F1 above 0.95 and still collapse on new speakers. Expect validation on same-speaker held-out clips to be badly over-optimistic.
- The main lever in SIGTYP was more real speakers. Here, the legal substitutes are VC-manufactured speakers and teacher knowledge (MMS-LID).
- NaijaVoices' thousands of yor/ibo/hau speakers are a candidate source of ethically sourced, disclosed target voices for VC. It requires accepting a gated CC-BY-NC-SA licence; check whether the competition permits NC data.
- The organisers' ban on facebook/omnilingual-asr-corpus, hypaai/Hypa-Voices and AfriSpeech/african-speech-ipa suggests these corpora contain the target languages, or overlap the test speakers (speculative). Derived models (OmniVoice) inherit that risk.
- For anw/ibb/efi, the LRE22 Xhosa/Zulu experience suggests confusions will dominate errors among the three. Speaker augmentation should be applied within the triplet with the same target-voice pool, so voice cannot separate them. Per-class bias tuning in the classifier bias (Section 4) is the cheapest fix for systematic over-prediction of the better-resourced member, ibb (1.76 h).

### Gaps
- No published post-mortem of a Nigerian-language LID challenge (Zindi or otherwise) was found.
- I found no DSN 2026 / Codabench 18424 forum material; not searched in depth and it may require login.
- AfriSpeech-related Interspeech special-session LID lessons were not found; AfriSpeech work is ASR-focused.

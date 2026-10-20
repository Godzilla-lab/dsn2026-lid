# DSN 2026 LID: training data audit

Dataset: `Data-Science-Nigeria/dsn-2026-lid-train` @ `87cf6d4b1c7599f97f44b6e6aade86438369a2b1` (audited 2026-10-20).
I downloaded all 11 shards and decoded every one of the 29,126 clips: 24,774 train and 4,352 validation, 92.52 h in total (78.55 h train, 13.97 h validation). The numbers below cover the whole population. A line says "sample" when it does not.
Per-clip table (metadata, FLAC header, acoustic features, hashes, near-duplicate neighbour): `/tmp/claude-0/-home-user/a55f94d8-bdde-575d-8d9b-69b0f597d9e3/scratchpad/audit/metadata.parquet`.
Scripts: `/tmp/claude-0/-home-user/a55f94d8-bdde-575d-8d9b-69b0f597d9e3/scratchpad/audit/scripts/`.

## TL;DR

* **Source and label are tightly coupled.**
  * urh, efi, ibb, igl and nup each come from a single source.
  * omnilingual supplies only ego, anw and idu; afrispeech supplies only yor, ibo, hau, tiv, urh and other.
  * Train contains no hypa-channel Yoruba; the only hypa yor clips are 90 in validation.
  * Scalar channel statistics alone (no content) predict the source of a validation clip with 99.5% accuracy and the label at fixed-share F1 0.34. Duration plus source reaches 0.17, against chance of about 0.08.
* **The afrispeech fingerprints:** 24% of samples are exact zeros, the noise floor is about −92 dBFS and trailing silence is long. The exception is tiv·afrispeech, with only 2% exact zeros and a noise floor of −73 dBFS. Inside omnilingual, loudness separates the languages: idu has a median of −42.8 dBFS against −26.6 for anw.
* **62.5% of clips have no speaker id** (all of afrispeech, 53 h). yor, urh and most of ibo, hau and tiv therefore have no speaker information. An MFCC-statistics test finds **no sign** of speaker leakage across the organiser's afrispeech split, but it cannot prove the split is disjoint.
* **The speaker ids reuse across labels in two different ways:**
  * The omnilingual ids `spk01`–`spk15` collide across anw, ego and idu. They belong to different people, which the acoustics confirm.
  * 42 hypa speakers record both a target language and `other`, which is probably English. These are genuinely the same voice.
* **No 8 kHz or phone-band audio** is in train: 0.7% of clips are narrowband-like. **No clip sits at the 30 s cap**; the maximum is 29.99 s. There are no exact duplicates, but there are **36 near-exact duplicate pairs inside hypa**, all within one speaker and split. One clip is silent: `69cd0258ddc28532`, at −83 dBFS.
* **The small classes rest on 1–3 real speakers each:** nup has 1, igl 3 and ibb 3. efi has 6 ids, but one of them holds 72% of its hours. A speaker-disjoint CV estimate for nup is impossible, and every fold holds out at most one igl, ibb or efi speaker.

---

## 1. Label × source: hours (clips)

**Train**

| label | afrispeech_ipa h (clips) | hypa_voices h (clips) | omnilingual_asr h (clips) | total h (clips) | sources |
|---|---|---|---|---|---|
| yor | 9.05 (3160) | - | - | 9.05 (3160) | 1 |
| ibo | 7.17 (2316) | 0.79 (276) | - | 7.96 (2592) | 2 |
| hau | 7.43 (2469) | 0.88 (286) | - | 8.31 (2755) | 2 |
| ego | - | 0.57 (191) | 6.75 (1440) | 7.32 (1631) | 2 |
| anw | - | 1.19 (446) | 6.93 (1466) | 8.12 (1912) | 2 |
| efi | - | 1.42 (483) | - | 1.42 (483) | 1 |
| ibb | - | 1.76 (607) | - | 1.76 (607) | 1 |
| idu | - | 0.96 (320) | 6.20 (1329) | 7.16 (1649) | 2 |
| igl | - | 0.48 (170) | - | 0.48 (170) | 1 |
| nup | - | 0.50 (176) | - | 0.50 (176) | 1 |
| tiv | 3.60 (1220) | 0.67 (254) | - | 4.27 (1474) | 2 |
| urh | 9.22 (3122) | - | - | 9.22 (3122) | 1 |
| other | 9.83 (3530) | 3.15 (1513) | - | 12.98 (5043) | 2 |
| **total** | 46.29 (15817) | 12.37 (4722) | 19.88 (4235) | 78.55 (24774) | |

**Validation**

| label | afrispeech_ipa h (clips) | hypa_voices h (clips) | omnilingual_asr h (clips) | total h (clips) | sources |
|---|---|---|---|---|---|
| yor | 0.70 (240) | 0.18 (90) | - | 0.88 (330) | 2 |
| ibo | 2.08 (704) | - | - | 2.08 (704) | 1 |
| hau | 1.42 (495) | - | - | 1.42 (495) | 1 |
| ego | - | - | 1.05 (222) | 1.05 (222) | 1 |
| anw | - | - | 1.83 (387) | 1.83 (387) | 1 |
| efi | - | 0.15 (64) | - | 0.15 (64) | 1 |
| ibb | - | 0.28 (107) | - | 0.28 (107) | 1 |
| idu | - | - | 2.12 (450) | 2.12 (450) | 1 |
| igl | - | 0.12 (49) | - | 0.12 (49) | 1 |
| nup | - | - | - | 0.00 (0) | 0 |
| tiv | 0.10 (35) | 0.52 (178) | - | 0.63 (213) | 2 |
| urh | 1.57 (550) | - | - | 1.57 (550) | 1 |
| other | 1.07 (359) | 0.78 (422) | - | 1.85 (781) | 2 |
| **total** | 6.94 (2383) | 2.04 (910) | 4.99 (1059) | 13.97 (4352) | |

### Source shortcut risk

* **Single-source classes.** In train, yor and urh come only from afrispeech_ipa, and efi, ibb, igl and nup come only from hypa_voices. Pooling train and validation changes this little:
  * urh is 100% afrispeech.
  * efi, ibb, igl and nup are 100% hypa.
  * yor has just 90 hypa clips (0.18 h, 11 speakers), and every one of them is in validation. So the train split contains no hypa-channel Yoruba.
* **Each source covers only part of the label set:**
  * omnilingual_asr supplies only ego, anw and idu. It supplies no `other` and none of the other 9 classes.
  * afrispeech_ipa supplies only yor, ibo, hau, tiv, urh and other.
  * hypa_voices supplies every class except urh.
  * As a result, a model that recognises the omnilingual channel can narrow the answer to 3 classes, and one that recognises the afrispeech channel can narrow it to 6.
* **Train and validation have different source mixes:**
  * ibo and hau validation clips are 100% afrispeech, although train contains hypa ibo (0.79 h) and hypa hau (0.88 h).
  * ego, anw and idu validation clips are 100% omnilingual, although train contains 0.57, 1.19 and 0.96 h of hypa.
  * The organiser validation set therefore never tests hypa-channel Igbo, Hausa, Eggon, Annang or Idoma.
* **Where `other` comes from:**
  * Train: 9.83 h afrispeech (3,530 clips, no speaker ids) and 3.15 h hypa (1,513 clips, 51 speaker ids).
  * Validation: 1.07 h afrispeech and 0.78 h hypa (31 speakers).
  * No `other` comes from omnilingual.
* **The ids and paths reveal nothing about language:**
  * `id` is a 16-hex hash with uniform character frequencies.
  * `audio.path` is always `<id>.flac`.
  * Rows are sorted by (label, source, speaker_id, id).
* **Whisper-base language-ID probe** (sample: 40 + 40 clips of 4 s or longer; weak evidence):
  * hypa `other`: Whisper's top-1 guess is English for 26/40 clips, and P(en) is at least 0.5 for 16/40. These clips are mostly English, and since they come from the same speakers, it is probably Nigerian-accented English.
  * afrispeech `other`: top-1 is English for 2/40 and P(en) is at least 0.5 for 0/40. The top-1 guesses are spread over yo 13, la 7, es 4, he 3, …, which is Whisper's usual fallback for Nigerian languages it has no label for. These clips are mostly non-English African-language speech.
  * Caveat: Whisper also says "en" for some genuine target-language hypa clips (hau 4/6 top-1, nup 3/6, yor 2/6). The probe therefore cannot rule out English or code-switched content inside hypa target-language labels; see §7.

## 2. Speaker structure

| label | train ids | val ids | pooled ids | % clips no id (train / val) | top-1 / top-3 share of label hours, train (all clips) | top-1 / top-3 share of known-speaker hours, train | top-1 / top-3 share of known-speaker hours, val |
|---|---|---|---|---|---|---|---|
| yor | 0 | 11 (11 hypa) | 11 | 100.0 / 72.7 | - / - | - / - | 24.4 / 50.8 |
| ibo | 7 (7 hypa) | 0 | 7 | 89.4 / 100.0 | 5.3 / 9.4 | 52.8 / 94.2 | - / - |
| hau | 3 (3 hypa) | 0 | 3 | 89.6 / 100.0 | 5.2 / 10.5 | 48.9 / 100.0 | - / - |
| ego | 10 (2 hypa + 8 omni) | 3 (3 omni) | 13 | 0.0 / 0.0 | 14.8 / 41.1 | 14.8 / 41.1 | 58.9 / 100.0 |
| anw | 20 (4 hypa + 16 omni) | 3 (3 omni) | 23 | 0.0 / 0.0 | 8.8 / 24.7 | 8.8 / 24.7 | 41.9 / 100.0 |
| efi | 3 (3 hypa) | 3 (3 hypa) | 6 | 0.0 / 0.0 | 79.4 / 100.0 | 79.4 / 100.0 | 98.1 / 100.0 |
| ibb | 2 (2 hypa) | 1 (1 hypa) | 3 | 0.0 / 0.0 | 51.2 / 100.0 | 51.2 / 100.0 | 100.0 / 100.0 |
| idu | 11 (3 hypa + 8 omni) | 3 (3 omni) | 14 | 0.0 / 0.0 | 13.8 / 40.7 | 13.8 / 40.7 | 48.1 / 100.0 |
| igl | 2 (2 hypa) | 1 (1 hypa) | 3 | 0.0 / 0.0 | 54.7 / 100.0 | 54.7 / 100.0 | 100.0 / 100.0 |
| nup | 1 (1 hypa) | - (no val) | 1 | 0.0 / - | 100.0 / 100.0 | 100.0 / 100.0 | - |
| tiv | 2 (2 hypa) | 2 (2 hypa) | 4 | 82.8 / 16.4 | 11.6 / 15.6 | 74.5 / 100.0 | 92.5 / 100.0 |
| urh | 0 | 0 | 0 | 100.0 / 100.0 | - / - | - / - | - / - |
| other | 51 (51 hypa) | 31 (31 hypa) | 82 | 70.0 / 46.0 | 2.7 / 6.9 | 11.1 / 28.5 | 15.1 / 38.0 |

* **No train/validation speaker leakage among known ids.** 68 ids appear in train and 38 in validation, with 0 shared. This holds for raw id strings, for (label, id) and for (source, label, id).
* **afrispeech_ipa has no speaker ids at all.** All 18,200 of its clips (62.5% of clips, 53.2 h, 57.5% of hours) have an empty `speaker_id`. Missing ids are all empty strings, never null. The card says speakers were kept within one split "where speaker information exists". For yor, ibo, hau and urh, and for 74% of tiv, the organiser split is therefore **not guaranteed speaker-disjoint**; see §5 for the acoustic leakage test. One hypa clip has `SPK-UNK` (other, train, 24.7 s).
* **Speaker id reuse across labels: 53 of 106 ids appear under more than one label.** There are two different causes:
  * **omnilingual_asr: these are id collisions.** The 11 ids spk01–03, 05, 07, 08, 10 and 12–15 each appear under all three of anw, ego and idu:
    * ego and idu have exactly the same 11 ids, and anw has those 11 plus spk16, 17 and 19–24.
    * The numbering has gaps at spk04, 06, 09, 11 and 18, which are probably test speakers.
    * The ids are numbered separately for each language, so they label different people, not trilingual speakers. An acoustic check confirms they are different people. Comparing the MFCC-statistic centroid of each (label, id) group, the same id string in two languages scores a cosine of 0.311 and two different ids score 0.317, which is indistinguishable. Two halves of the same (label, id) group score 0.996.
    * The organiser split kept each id string in one split across languages: spk07, spk14 and spk15 are validation in all three. Grouping by the raw string is conservative and harmless.
  * **hypa_voices: these are real bilingual speakers.**
    * 42 hypa ids record one target language and also `other`. For example, SPK-147 has 176 nup clips and 46 `other` clips, and SPK-011 has 363 efi and 68 `other`.
    * 40 ids appear only in `other`. 5 ids appear only in a target language and have tiny volumes (SPK-023 ibo, SPK-044 and SPK-144 yor, SPK-052 and SPK-129 efi).
    * No hypa id appears under two target languages.
    * 911 of the 1,935 hypa `other` clips (1.28 of 3.93 h) come from target-language speakers.
    * An acoustic check confirms they are the same voice. A speaker's target-language centroid and `other` centroid score a cosine of 0.922, against 0.410 for different speakers, and 93% of same-id pairs lie above the 90th percentile of different-id pairs.
    * This is good for forcing the model to learn language rather than voice. It also means a fold must keep each person's target-language and `other` clips together.

## 3. Duration

### By label (train + validation, seconds)

| label | clips | min | median | p90 | max | % ≥ 29.9 s | % ≥ 25 s | % < 3 s |
|---|---|---|---|---|---|---|---|---|
| yor | 3490 | 2.30 | 8.27 | 21.02 | 29.98 | 0.14 | 5.1 | 0.2 |
| ibo | 3296 | 2.05 | 8.62 | 22.44 | 29.96 | 0.24 | 6.7 | 0.3 |
| hau | 3250 | 2.13 | 9.37 | 21.09 | 29.99 | 0.06 | 5.5 | 0.3 |
| ego | 1853 | 2.22 | 15.51 | 25.29 | 29.99 | 0.22 | 10.4 | 0.2 |
| anw | 2299 | 2.05 | 15.34 | 24.31 | 29.99 | 0.35 | 8.5 | 1.0 |
| efi | 547 | 2.05 | 7.08 | 21.16 | 25.94 | 0.00 | 1.6 | 5.5 |
| ibb | 714 | 2.05 | 6.36 | 21.39 | 26.88 | 0.00 | 2.2 | 6.2 |
| idu | 2099 | 2.13 | 15.36 | 24.24 | 29.99 | 0.14 | 8.9 | 0.6 |
| igl | 219 | 2.05 | 6.32 | 21.33 | 26.20 | 0.00 | 2.3 | 8.2 |
| nup | 176 | 2.05 | 6.87 | 21.59 | 27.39 | 0.00 | 5.7 | 4.5 |
| tiv | 1687 | 2.13 | 8.96 | 20.43 | 29.87 | 0.00 | 4.6 | 2.0 |
| urh | 3672 | 3.00 | 8.15 | 22.43 | 29.99 | 0.16 | 6.7 | 0.0 |
| other | 5824 | 2.05 | 7.00 | 19.54 | 29.92 | 0.02 | 4.2 | 3.0 |

### By source

| source | clips | min | median | p90 | max | % ≥ 29.9 s | % ≥ 25 s | % < 3 s |
|---|---|---|---|---|---|---|---|---|
| afrispeech_ipa | 18200 | 3.00 | 8.53 | 21.71 | 29.99 | 0.12 | 6.1 | 0.0 |
| hypa_voices | 5632 | 2.05 | 6.14 | 20.48 | 27.39 | 0.00 | 1.7 | 6.6 |
| omnilingual_asr | 5294 | 3.64 | 15.59 | 25.21 | 29.99 | 0.28 | 10.3 | 0.0 |

**By label × source (median / p90 seconds):** anw-hypa 6.5/21.2; anw-omnilingual 15.6/25.0; efi-hypa 7.1/21.2; ego-hypa 7.0/23.6; ego-omnilingual 15.6/25.9; hau-afrispeech 9.4/21.1; hau-hypa 7.9/21.1; ibb-hypa 6.4/21.4; ibo-afrispeech 8.6/23.0; ibo-hypa 7.9/19.8; idu-hypa 7.1/23.1; idu-omnilingual 15.5/24.8; igl-hypa 6.3/21.3; nup-hypa 6.9/21.6; other-afrispeech 8.0/21.1; other-hypa 5.4/17.2; tiv-afrispeech 9.3/20.0; tiv-hypa 6.6/21.1; urh-afrispeech 8.1/22.4; yor-afrispeech 8.3/21.3; yor-hypa 5.9/14.1

* **No clip is exactly at the 30 s cap.** The maximum is 29.993 s, 37 clips (0.13%) are 29.9 s or longer, and 1,760 (6.0%) are 25 s or longer. Each source has its own minimum: afrispeech 3.000 s, hypa 2.048 s, omnilingual 3.637 s (9.66 s for ego and 13.01 s for idu).
* **hypa never reaches 30 s.** Its maximum is 27.39 s, so no hypa recording was cut.
* **The omnilingual durations are bimodal.** 4,253 of 5,294 clips (80.3%) fall in 13–17 s. The rest form a thin tail: 64 clips in 17–18 s, 120 in 18–20 s, 302 in 20–25 s and 545 in 25–30 s. This matches recordings of about 26–34 s that were cut in two at a quiet point, so most omnilingual clips probably have a **sibling segment**. Both halves carry the same speaker_id, so speaker grouping already covers this.
* **Sample-count fingerprints identify the source:**
  * Exactly one third of hypa clips (1,879/5,632) are multiples of 4,096 samples. This fits originals made of fixed-size blocks that were resampled by 2:3 or 3:1, and it holds at 0.30–0.35 for every label, so it says nothing about the label.
  * 100% of afrispeech clips are multiples of 16 samples, against 2.4% of omnilingual clips.
  * A model can only use these fingerprints when it sees the exact clip length, which random cropping removes.
* **Duration alone is a weak shortcut. Duration together with source is a moderate one.** Classifiers trained on train and scored on validation (12 classes, since nup has no validation clips; chance macro-F1 is about 0.083):

| features | model | val macro-F1 | val fixed-share F1 | val acc | train 5-fold (not grouped) macro-F1 |
|---|---|---|---|---|---|
| duration | decision tree (depth 6, balanced) | 0.089 | 0.084 | 0.114 | 0.102 |
| duration | logistic regression (balanced) | 0.088 | 0.082 | 0.152 | 0.085 |
| duration | gradient-boosted trees (balanced) | 0.129 | 0.128 | 0.146 | 0.120 |
| source | logistic regression (balanced) | 0.078 | 0.089 | 0.202 | 0.113 |
| duration + source | decision tree (depth 6, balanced) | 0.169 | 0.160 | 0.194 | 0.172 |
| duration + source | logistic regression (balanced) | 0.121 | 0.115 | 0.162 | 0.140 |
| **duration + source** | **gradient-boosted trees (balanced)** | **0.176** | **0.173** | 0.195 | 0.175 |

Per-class validation F1 for the best shortcut (gradient-boosted trees, duration + source): yor .15, ibo .19, hau .27, ego .29, anw .28, efi .08, ibb .16, idu .34, igl .05, tiv .03, urh .27, other .01. Most of the lift comes from source restricting the possible labels; long omnilingual clips point to ego, anw or idu. The test set comes from the same corpora, so this shortcut would partly "work" on test, but it caps out at about 0.17 and is fragile under phone or noisy audio.

## 4. Channel and acoustic shortcut check

**Format check.** All 29,126 clips decode as **16 kHz, mono, 16-bit FLAC (PCM_16)**. The STREAMINFO sample count equals the decoded frame count and `duration × 16000` for every clip. No other sample rate or channel layout occurs.

**Method.** I decoded every clip. The features are:
* RMS in dBFS;
* leading and trailing silence, measured with 25 ms frames and counting a frame as silent when it is more than 35 dB below the clip's loudest frame;
* the per-frame spectral rolloff at 95% and 99%, taking the median over speech frames;
* the energy share above 4 kHz and above 7 kHz in the long-term spectrum;
* the bandwidth edge, i.e. the highest frequency within 50 dB of the 0.3–3 kHz level;
* the noise floor, from the 10th-percentile frame;
* the fraction of samples that are exactly zero;
* the DC offset.

A 15-clip-per-cell stratified sample (315 clips over 21 label × source cells, seed 0) gives the same picture; for example, the sample median RMS of idu·omnilingual is −47.1 dBFS against −42.8 in the full population. The tables below use the full population.

| source | clips | RMS dBFS | % RMS < −40 | lead sil s | trail sil s | rolloff95 Hz | rolloff99 Hz | energy >4 kHz dB | energy >7 kHz dB | bw edge Hz | noise floor dBFS | % exact-zero samples | % ending in ≥2 s digital zero | abs DC ×1e-4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| afrispeech_ipa | 18200 | -20.1 | 0.0 | 0.48 | 0.86 | 719 | 1938 | -20.6 | -29.4 | 8000 | -92.5 | 23.7 | 2.4 | 0.2 |
| hypa_voices | 5632 | -24.0 | 1.6 | 0.34 | 0.24 | 625 | 1375 | -22.8 | -38.5 | 7844 | -69.4 | 3.4 | 0.0 | 1.1 |
| omnilingual_asr | 5294 | -32.5 | 27.0 | 0.13 | 0.36 | 594 | 1250 | -25.8 | -38.1 | 8000 | -73.7 | 1.8 | 0.0 | 0.0 |

**By label × source (all clips, medians):**

| label · source | clips | RMS dBFS | % RMS < −40 | lead sil s | trail sil s | rolloff95 Hz | rolloff99 Hz | energy >4 kHz dB | energy >7 kHz dB | bw edge Hz | noise floor dBFS | % exact-zero samples | % ending in ≥2 s digital zero | abs DC ×1e-4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| anw · hypa | 446 | -30.2 | 9.0 | 0.28 | 0.18 | 531 | 1094 | -25.7 | -41.9 | 7844 | -70.6 | 2.7 | 0.0 | 0.8 |
| anw · omnilingual | 1853 | -26.6 | 0.2 | 0.19 | 0.56 | 734 | 1844 | -24.0 | -36.3 | 8000 | -72.8 | 1.5 | 0.0 | 0.1 |
| efi · hypa | 547 | -23.5 | 0.2 | 0.40 | 0.29 | 594 | 1219 | -24.9 | -35.2 | 7875 | -74.4 | 7.7 | 0.0 | 1.0 |
| ego · hypa | 191 | -18.7 | 0.0 | 0.33 | 0.30 | 1031 | 1812 | -23.8 | -37.5 | 7844 | -71.6 | 3.2 | 0.0 | 0.9 |
| ego · omnilingual | 1662 | -32.7 | 16.7 | 0.11 | 0.33 | 594 | 969 | -31.2 | -42.5 | 8000 | -73.0 | 1.7 | 0.0 | 0.0 |
| hau · afrispeech | 2964 | -20.0 | 0.0 | 0.40 | 0.89 | 688 | 2156 | -17.7 | -25.4 | 8000 | -93.2 | 23.2 | 1.3 | 0.1 |
| hau · hypa | 286 | -25.9 | 0.7 | 0.41 | 0.23 | 625 | 1406 | -16.6 | -29.4 | 7906 | -69.8 | 3.0 | 0.0 | 1.1 |
| ibb · hypa | 714 | -31.0 | 2.5 | 0.46 | 0.22 | 594 | 1469 | -25.4 | -49.8 | 7750 | -73.6 | 3.1 | 0.0 | 0.6 |
| ibo · afrispeech | 3020 | -20.5 | 0.0 | 0.51 | 0.91 | 688 | 1828 | -21.4 | -29.0 | 8000 | -88.0 | 25.2 | 1.9 | 0.3 |
| ibo · hypa | 276 | -25.7 | 4.0 | 0.24 | 0.25 | 578 | 1156 | -24.2 | -36.6 | 7875 | -65.3 | 2.8 | 0.0 | 1.3 |
| idu · hypa | 320 | -21.3 | 0.0 | 0.39 | 0.23 | 594 | 1219 | -23.0 | -40.3 | 7844 | -66.5 | 2.0 | 0.0 | 1.6 |
| idu · omnilingual | 1779 | -42.8 | 64.6 | 0.10 | 0.21 | 562 | 1156 | -23.1 | -27.2 | 8000 | -75.2 | 2.3 | 0.0 | 0.0 |
| igl · hypa | 219 | -25.1 | 0.0 | 0.22 | 0.17 | 562 | 1188 | -23.6 | -37.7 | 7844 | -69.5 | 8.0 | 0.0 | 1.4 |
| nup · hypa | 176 | -18.8 | 0.0 | 0.40 | 0.23 | 500 | 1055 | -20.9 | -34.1 | 7875 | -67.3 | 1.6 | 0.0 | 1.3 |
| other · afrispeech | 3889 | -19.8 | 0.0 | 0.51 | 0.65 | 531 | 1406 | -21.6 | -29.8 | 8000 | -93.2 | 25.3 | 2.5 | 0.2 |
| other · hypa | 1935 | -23.3 | 0.9 | 0.31 | 0.24 | 656 | 1500 | -20.9 | -35.4 | 7875 | -68.6 | 3.8 | 0.0 | 1.1 |
| tiv · afrispeech | 1255 | -22.2 | 0.0 | 0.49 | 1.33 | 656 | 1562 | -26.1 | -33.1 | 8000 | -72.9 | 2.0 | 4.5 | 0.2 |
| tiv · hypa | 432 | -21.7 | 0.2 | 0.28 | 0.20 | 625 | 1375 | -23.0 | -42.2 | 7844 | -63.8 | 2.8 | 0.0 | 2.4 |
| urh · afrispeech | 3672 | -19.5 | 0.0 | 0.33 | 0.82 | 1156 | 2375 | -19.8 | -32.8 | 8000 | -95.9 | 22.3 | 1.6 | 0.1 |
| yor · afrispeech | 3400 | -20.2 | 0.0 | 0.48 | 1.05 | 688 | 1969 | -21.0 | -28.4 | 8000 | -93.6 | 25.2 | 3.7 | 0.1 |
| yor · hypa | 90 | -21.3 | 0.0 | 0.26 | 0.21 | 562 | 1328 | -26.0 | -40.4 | 7844 | -61.5 | 2.4 | 0.0 | 1.8 |

What the numbers show:
* **No 8 kHz (telephone-band) source.** The median bandwidth edge is 7.8–8.0 kHz in every cell. Only 205 clips (0.7%) have less than −40 dB of their energy above 4 kHz. These cluster in ego·omnilingual (63), other·afrispeech (45), efi·hypa (32) and other·hypa (19). 42 afrispeech clips have a bandwidth edge below 4.5 kHz. **The "phone-quality" audio in the test set therefore has almost no counterpart in train**, and has to come from augmentation.
* **afrispeech has a strong digital fingerprint.**
  * A median of 23.7% of its samples are exactly zero, against 3.4% in hypa and 1.8% in omnilingual. The audio looks noise-gated or zero-padded, giving a noise floor of about −92 dBFS against −69 to −74 elsewhere.
  * Trailing silence is long: a median of 0.86 s, and 36.6% of clips have more than 1 s, against 0.5% in hypa and 10.5% in omnilingual.
  * 437 afrispeech clips end with at least 2 s of pure digital zero, and 25 start with it.
  * **tiv·afrispeech is the exception inside afrispeech.** Only 2.0% of its samples are exact zeros and its noise floor is −72.9 dBFS. A model can therefore tell afrispeech-Tiv from afrispeech-Yoruba, Igbo, Hausa, Urhobo or other by the silence pattern alone.
* **Loudness separates the omnilingual languages.**
  * The median RMS is −42.8 dBFS for idu·omnilingual (64.6% of clips below −40), −32.7 for ego·omnilingual (16.7% below −40) and −26.6 for anw·omnilingual (0.2% below −40).
  * Within omnilingual, then, gain alone separates idu, ego and anw.
* **Spectral tilt differs by language within afrispeech.**
  * urh·afrispeech is much brighter than the other afrispeech languages: a rolloff95 of 1,156 Hz against 531–688 Hz, and a rolloff99 of 2,375 Hz against 1,406–2,156 Hz.
  * hau·afrispeech carries more energy above 4 kHz (−17.7 dB) than tiv·afrispeech (−26.1 dB).
  * These look like differences in recording session or device for each language.
* **DC offset is negligible everywhere.** No clip has an absolute DC offset above 0.01, and 265 clips (afrispeech 128, hypa 137) are above 0.001. There is no clipping: no clip has more than 0.1% of samples at full scale, and 96 clips peak at or above 0.999 (afrispeech 70, omnilingual 26).
* **Channel statistics alone are a strong shortcut.** I trained gradient-boosted trees on train and scored them on validation, using 15 scalar channel statistics: no MFCCs and no content.
  * The statistics predict **source with 99.5% accuracy**, or 99.7% with duration added.
  * They predict **label at macro-F1 0.344 and fixed-share F1 0.340**, or 0.356 and 0.353 with duration. That is twice the duration+source baseline and about 4× chance.
  * Per-class F1: anw .87, hau .61, urh .58, other .55, idu .52, ego .30, yor .25, ibo .25, ibb .14, tiv .04, efi .02, igl .00.
  * The big classes from afrispeech and omnilingual are where channel differences give the label away. The hypa small classes are not separable by channel, which is good, but they get no free help either.

## 5. Duplicates and parent recordings

* **No exact duplicates.** Neither the FLAC file bytes (MD5) nor the decoded PCM (the STREAMINFO MD5) collide among the 29,126 clips.
* **Hashes of the first and last 2 s.** These were computed on the whole population, not only the sample.
  * The only collision is the hash of 2 s of pure digital zero. 25 afrispeech clips start with it (other 23, urh 2), and 437 afrispeech clips end with it (yor 126, other 99, ibo 58, tiv 57, urh 57, hau 40).
  * After quantising to 8 bits, 61 clips share an effectively silent first 2 s.
  * So these collisions are padding, not repeated content.
* **There are 36 near-exact duplicate pairs, all in hypa_voices.**
  * I found them two ways: a log-mel fingerprint of the first 2 s with a cosine of at least 0.99 followed by a waveform cross-correlation above 0.9, and a zero-lag waveform correlation above 0.99 between clips with the same length and speaker. The union is 36 pairs covering 72 clips.
  * The two clips in a pair have the same length (in 97% of pairs) and waveform correlation of about 1.0. Their samples differ by up to about 4,000 LSB, which points to re-encoding or a gain or dither change, and is why the exact-hash de-duplication missed them.
  * By label: other 13, ibb 7, anw 4, efi 4, tiv 3, idu 2, ego 1, ibo 1, igl 1. SPK-104 (ibb) alone has 5.
  * All 36 pairs share speaker, label and split, so they cause **no cross-split leakage**. They do slightly inflate the small classes; dropping one clip of each pair removes about 0.09 h.
  * The second clip of each pair is flagged in `metadata.parquet` as `near_dup_of`.
  * Afrispeech fingerprint pairs with a cosine of at least 0.99 (29 pairs) are not duplicates: the median waveform correlation is 0.22 and none exceeds 0.9. They are pairs of quiet, padded openings.
* **The ids and row order carry no parent-recording information.** The id is a uniform 16-hex hash, and rows are sorted by (label, source, speaker_id, id). Nothing ties sibling segments together except speaker_id and duration:
  * **omnilingual.** 80% of clips are 13–17 s, which points to roughly 30–34 s recordings cut in half (see §3). Siblings share speaker_id, so grouping by speaker id covers grouping by parent.
  * **hypa.** No clip is longer than 27.4 s, so hypa recordings were never cut. The 36 near-duplicate pairs are the only within-speaker coupling.
  * **afrispeech.** The duration distribution is smooth, and 6.1% of clips are at least 25 s, so some clips come from cut recordings. With no speaker or parent id, the parent links cannot be recovered from the metadata.
* **Is afrispeech's train/validation split speaker-leaky?** I used clip-level MFCC mean and standard deviation, z-scored, as a crude voice-plus-session embedding. For each class I compared two medians:
  * how similar each **train** clip's nearest other-train clip is (train→train), and
  * how similar each **validation** clip's nearest train clip is (val→train).
  
  A split that holds out speakers should show a clear gap, because a train clip's nearest neighbour is usually the same speaker; this embedding finds a same-speaker nearest neighbour 92–100% of the time for target-language classes in hypa and omnilingual, and 78% of the time for hypa `other`.

| source | class | train→train median | val→train median | gap |
|---|---|---|---|---|
| hypa (speaker-disjoint, reference) | efi / ibb / igl / tiv / other | .907 / .901 / .891 / .890 / .863 | .657 / .800 / .594 / .802 / .824 | .25 / .10 / .30 / .09 / .04 |
| omnilingual (speaker-disjoint, reference) | anw / ego / idu | .929 / .942 / .971 | .734 / .834 / .765 | .20 / .11 / .21 |
| **afrispeech (no ids)** | hau / ibo / other / tiv / urh / yor | .866 / .887 / .878 / .861 / .881 / .863 | .795 / .663 / .800 / .640 / .753 / .729 | **.07 / .22 / .08 / .22 / .13 / .13** |

The afrispeech gaps fall in the same range as the known speaker-disjoint splits. **There is no sign that the same speakers or sessions sit on both sides of the afrispeech train/validation split.** The split was probably made on a speaker or session key that was not published. The gaps for hau (.07) and other (.08) are the smallest and deserve a second check with a real speaker-verification embedding. The test cannot prove the split is disjoint.

## 6. Grouped-CV design for the low-speaker classes

**Group key.** Pool train and validation, then assign each clip a `group` as follows:
* hypa: `speaker_id`, pooled across labels, so that a person's target-language clips and `other` clips always land in the same fold. This gives 87 groups.
* omnilingual: the raw `spkNN` string, pooled across anw, ego and idu. This gives 19 groups. It matches the organiser split, and it also keeps sibling half-segments together.
* afrispeech: there is no id, so build **pseudo-speaker groups**. Cluster a speaker embedding (for example ECAPA or WavLM-SV computed on the first 6 s) with a tight agglomerative threshold, then run StratifiedGroupKFold over those clusters. If this is not done, at least treat the organiser's afrispeech validation clips as a block that is never mixed back into training folds.

**Use 3 folds with forced speaker rotation for the small classes, and keep nup in every training set.** With K=5, two folds would contain no igl or ibb speaker at all. With K=3, each fold holds out exactly one speaker of each small class. Each listed speaker's `other` clips go to the same fold.

| class (speakers pooled) | fold A held out | fold B held out | fold C held out | independent speakers held out per fold | speakers left to train on |
|---|---|---|---|---|---|
| igl (3) | SPK-059: 49 clips, 0.12 h | SPK-086: 73 clips, 0.22 h | SPK-031: 97 clips, 0.27 h | **1** | 2 |
| ibb (3) | SPK-071: 107 clips, 0.29 h | SPK-004: 329 clips, 0.90 h | SPK-104: 278 clips, 0.86 h | **1** | 2 (fold B keeps only 1.15 h) |
| efi (6; 3 with real volume) | SPK-011: 363 clips, 1.13 h | SPK-040 + SPK-052: 83 clips, 0.20 h | SPK-140 + SPK-065 + SPK-129: 101 clips, 0.24 h | **1 real** (+ tiny ones) | 3–5 (fold A keeps only 0.44 h of efi) |
| tiv hypa (4) | SPK-120 + SPK-041: 89 clips, 0.21 h | SPK-061: 184 clips, 0.50 h | SPK-096: 159 clips, 0.49 h | 1–2 hypa speakers, plus about a third of the 1,255 afrispeech tiv clips via pseudo-groups | 2–3 hypa speakers + afrispeech |
| nup (1) | not held out | not held out | not held out | **0** | 1 (SPK-147, all 176 nup clips + 46 `other` clips, in every training set) |
| hau hypa (3), ibo hypa (7), yor hypa (11), ego/anw/idu omni (19 strings), other-only hypa (40) | StratifiedGroupKFold(3) over the remaining groups | | | | |

Rules for reading the CV:
* Compute per-class F1 and the fixed-share F1 on the **pooled out-of-fold predictions**, never as a mean of per-fold F1. For igl and ibb, each fold's number is a single-speaker estimate. The pooled number reflects only 3 speakers (igl, ibb), 4 hypa speakers plus afrispeech (tiv) or 3–4 real speakers (efi). Also report recall for each held-out speaker, because one bad speaker can swing that class's F1, and with it 8.7% of the score.
* **nup cannot be validated speaker-disjointly with this data.** Report two separate numbers and keep both out of the CV mean:
  * an optimistic, speaker-seen nup recall from a 5-fold utterance split inside SPK-147;
  * nup precision, i.e. the false-positive rate, which *is* measured fairly on every held-out speaker of every other class.
  * Only external Nupe speakers can give a true estimate.
* The organiser validation set on its own holds out 1 igl speaker, 1 ibb speaker, 3 efi speakers (98% of efi validation hours is SPK-140), 2 tiv-hypa speakers and 0 nup speakers. It is too thin to choose between models on the 8.7% classes.

## 7. Surprises and label-noise hints

* **Two clips are empty or near-empty.** `69cd0258ddc28532` (ibo, hypa, SPK-007, train, 4.5 s) has an RMS of −82.9 dBFS and a peak of 0.0007, so it is effectively silent; drop it. `70f78361a8c74cb9` (idu, omnilingual, spk02, train, 13.7 s) has an RMS of −60.2 dBFS. In total, 323 clips are below −50 dBFS: idu·omnilingual 254, ego·omnilingual 59, ibo·hypa 6, anw·hypa 2, efi·hypa 1 and tiv·hypa 1. Apart from the first clip, they are quiet but not empty.
* **Label-noise hints in hypa.**
  * Each of the 42 bilingual hypa speakers has `other` clips that Whisper mostly scores as English.
  * Whisper-base also puts English top-1, at low confidence (P = 0.2–0.7), on 4/6 hau·hypa, 3/6 nup·hypa and 2/6 yor·hypa sample clips. It puts English top-1 on 0/36 afrispeech target-language clips.
  * So either hypa prompts include English or code-switched material, or Whisper is fooled by the hypa channel. Whisper-base cannot tell which.
  * **Action:** listen to about 10 clips for each hypa (speaker, label), or run a stronger language-ID model such as MMS-LID, starting with SPK-147 (nup) and SPK-132 and SPK-024 (hau). A mislabelled 20% inside nup would directly cap that 8.7% share.
* **Gaps in the id numbering point to the test speakers.**
  * omnilingual skips spk04, 06, 09, 11 and 18.
  * hypa uses 86 of SPK-001…150; 64 numbers are absent.
  * These are probably the speakers held out for test, which fits a test set that is "speaker-disjoint, same corpora".
* **The validation set does not represent the hypa channel for most big languages.** ibo, hau, ego, anw and idu validation clips contain no hypa audio. **efi validation is essentially one speaker**: SPK-140 holds 98.1% of efi validation hours (60 of 64 clips).
* **The dataset card is inaccurate in places.**
  * "Exact duplicates were removed" is true for byte-exact copies only; 36 re-encoded copies remain.
  * "2 to 30 s": the per-source minimums are 2.048, 3.000 and 3.637 s, and no clip reaches 30 s.
  * The card does not mention that afrispeech has no speaker ids.
* **Odd sample counts.** Exactly one third of hypa clips are multiples of 4,096 samples, and every afrispeech clip is a multiple of 16 samples. These are harmless resampling artefacts, but they are perfect source tags if a model ever sees the exact clip length.
* **Missing ids.** One hypa clip has `SPK-UNK` (`f2429de9a83b6966`, other, train). Empty speaker ids are stored as `''`, never as null.

## Decision implications

### Validation design
1. **Pool train and validation into a 3-fold grouped CV.** Groups are hypa speaker ids (pooled across labels), omnilingual id strings (pooled across anw, ego and idu), and afrispeech pseudo-speaker clusters from a speaker-verification embedding. Force the small-class speaker rotation from §6, and keep SPK-147 (nup) in every training set.
2. **Score on pooled out-of-fold predictions** with the fixed-share F1 (8.7% / 4.3%). Alongside it, report:
   * recall for each held-out speaker in efi, ibb, igl and tiv;
   * nup precision, plus a separately labelled speaker-seen nup recall.
3. **Add a per-source breakdown, and above all an F1 on the hypa clips only.** hypa is the one source that contains every class except urh. Its F1 is the most shortcut-free number available, because inside hypa the channel says almost nothing about the label (channel-only F1 for efi .02 and igl .00).
4. **Use the organiser validation set only as a secondary sanity check.** It holds 1 speaker each for igl and ibb, effectively 1 for efi, 0 for nup, and no hypa audio for ibo, hau, ego, anw or idu.

### Sampling
1. **Drop the bad clips:** one clip from each of the 36 near-duplicate pairs, plus the silent clip `69cd0258ddc28532`.
2. **Sample speakers, not clips.** Use a sampler weighted per (label, source, speaker), with a per-speaker cap, so that no single speaker dominates a class. SPK-011 holds 72% of efi hours; SPK-004 and SPK-104 together hold 87% of ibb. Use loss or class weights that follow the metric shares.
3. **Boost the hypa cells of the big languages:** ibo·hypa 0.79 h, hau·hypa 0.88 h, ego·hypa 0.57 h, anw·hypa 1.19 h and idu·hypa 0.96 h. They are the only examples of those languages in the hypa channel, so the model can learn them in hypa conditions rather than only in afrispeech or omnilingual conditions.
4. **Move the 90 validation yor·hypa clips (11 speakers, 0.18 h) into training** in the final fit. Otherwise yor is never seen through the hypa channel.

### Augmentation (ordered by the shortcut each removes)
1. **Remove the silence and digital-zero cue.**
   * Trim leading and trailing silence with an energy-based voice-activity detector, or crop on speech frames.
   * Add a random noise floor of −75 to −45 dBFS, or noise at 10–30 dB SNR, to every clip, so that exact-zero runs and the −92 dBFS afrispeech floor disappear.
2. **Remove the loudness cue.** Normalise loudness at both train and inference, and randomise gain by ±15 dB. This kills the idu, ego and anw loudness separation inside omnilingual.
3. **Use random crops of 2–8 s** in training, with multi-crop averaging at test time. This removes the duration and sample-count cues, which alone gave a validation macro-F1 of 0.13–0.18.
4. **Simulate the channel:**
   * telephone band (300–3,400 Hz, or resample to 8 kHz and back), plus codecs such as AMR-NB, GSM, Opus at 6–12 kbps and MP3 at 32 kbps, because train has essentially no phone-band audio;
   * random EQ and spectral tilt, against the urh brightness and the hau high-frequency difference;
   * room impulse responses and babble or MUSAN noise at 0–20 dB SNR.
5. **Add speaker diversity for the 1–3-speaker classes:** speed perturbation of 0.9–1.1, pitch and formant shifts, and possibly voice conversion. This is the only lever on speaker generalisation short of external data.

### External data (respect `PROHIBITED_SOURCES.txt`: do not use Hypa-Voices, the omnilingual-asr-corpus or AfriSpeech-IPA directly)
| priority | class | why | current train-able data |
|---|---|---|---|
| 1 | **nup** | 8.7% share, 1 speaker, 0 validation clips, no fair CV estimate | 0.50 h, 1 speaker |
| 2 | **igl** | 8.7% share, 3 speakers, second-smallest hours (0.60 h) | 0.60 h, 3 speakers |
| 3 | **ibb** | 8.7% share, 3 speakers, 1 hypa channel | 2.05 h, 3 speakers |
| 4 | **efi** | 8.7% share, 1 dominant speaker (72% of hours) | 1.57 h, 6 ids (3 with real volume) |
| 5 | **tiv** | 8.7% share, 4 known hypa speakers; the afrispeech part has its own channel signature | 4.89 h (3.70 h of it afrispeech, without ids) |
| 6 | **urh** | 8.7% share, afrispeech only: 1 channel, 0 speaker ids. It needs other channels more than more hours | 10.79 h |
| 7 | **other** | 8.7% share, but no omnilingual-channel or phone-channel non-target speech. Add English, Pidgin and other Nigerian-language speech from new channels | 14.83 h |
| low | yor / ibo / hau | 4.3% share each, about 9–10 h each, many afrispeech speakers. For yor, a non-afrispeech channel is what is missing | |

Candidate sources to vet for licence and language coverage: Bible-reading corpora (for example CMU Wilderness or Open.Bible-derived sets, which often cover Nupe, Igala, Ibibio, Efik, Tiv and Urhobo) and FLEURS or Common Voice for yor, ibo and hau and for English or Pidgin "other". Check each language's coverage before relying on it. Bible audio is read speech in a single studio channel, so it should be paired with the channel augmentation above.

# External data for DSN 2026 LID: (A) Yoruba/Igbo/Hausa, (B) the open-set "other" class, (C) noise/RIR/telephone augmentation

Scope: metadata only. Facts were checked on 2026-10-20 against the HF Hub API and dataset cards, OpenSLR, the Mozilla Data Collective (MDC) public catalogue API, and project pages. No audio was downloaded; the only files fetched were FLEURS TSV metadata and READMEs. Hours marked "≈" are estimates (method stated). "Overlap risk" means risk of overlap with the three prohibited corpora: hypaai/Hypa-Voices, facebook/omnilingual-asr-corpus and AfriSpeech/african-speech-ipa.

---

## Q0 (cross-cutting): Which organisations and datasets count as "prohibited by association"?

### Takeaway
Treat everything from the HF orgs **hypaai**, **AfriSpeech** and (to be safe) **intronhealth**, plus re-uploads derived from them, as banned. AfriSpeech/african-speech-ipa is built directly on AfriSpeech/african-speech-public_v1, which is a 141-language corpus covering Edo, Esan, Isoko, Ijaw, Itsekiri, Tiv, Urhobo, Hausa, Igbo, Yoruba and others. Bible-audio corpora carry some *unverified* overlap risk because that corpus's source is not stated.

### Cited Findings
- african-speech-ipa is "[AfriSpeech](https://huggingface.co/datasets/AfriSpeech/african-speech-public_v1) audio paired with IPA phoneme transcriptions, for 141 languages", and its splits "are AfriSpeech's own" — [AfriSpeech/african-speech-ipa card](https://huggingface.co/datasets/AfriSpeech/african-speech-ipa)
- african-speech-public_v1 Nigerian and nearby subsets include Edo (bin, 26.0 h), Esan (ish, 24.2 h), Gokana (7.98 h), Hausa (28.4 h), Igbo (38.1 h), Ijaw (ijc, 0.88 h), Isoko (26.8 h), Itsekiri (6.78 h), Khana (3.62 h), Tiv (4.03 h), Urhobo (25.7 h), Yoruba (14.2 h), Pidgin West Africa (wes, CM, 21.1 h), Fon (33.2 h), Gun (40.0 h), Ewe (29.4 h), Ga (35.1 h), Twi (50.3 h), Kabiye, Dagaare, Fante and others. The card does not say where the audio came from. — [AfriSpeech/african-speech-public_v1](https://huggingface.co/datasets/AfriSpeech/african-speech-public_v1)
- The AfriSpeech HF org also hosts: open-bible-speech-african, youversion-african-speech, african-speech-public_v1, afrispeeches(+aligned, v2), africa-female-speech(+v2), multivoice-synthetic-speech, grn-african-speech, afri2universal, afri-speech-pause-aligned and more — [HF API author=AfriSpeech](https://huggingface.co/api/datasets?author=AfriSpeech)
- AfriSpeech/open-bible-speech-african is an unchanged re-host of the African subset of davidguzmanr/open-bible-resources (Open.Bible / Faith Comes By Hearing): 19 languages, about 1,741 h, including Hausa 99.7 h, Igbo 94.2 h and Yoruba 89.9 h, CC BY-SA 4.0 — [card](https://huggingface.co/datasets/AfriSpeech/open-bible-speech-african)
- AfriSpeech/youversion-african-speech has Bible-app audio, including Hausa (3.13 h), Igbo (4.18 h), Fon, Fulani and Burkina Faso Fulfulde — [card](https://huggingface.co/datasets/AfriSpeech/youversion-african-speech)
- The hypaai org hosts Hypa_Fleurs, Hypa_Fleurs_v2, Hypa-AfriSpeech, Hypa-Speech-10k, Hypa-LibreSpeech, Hypa-Voices-snac, Hypa-Voices-ASR/TTS/Translation-Benchmark and others — [HF API author=hypaai](https://huggingface.co/api/datasets?author=hypaai)
- Hypa_Fleurs is FLEURS English text translated into ig/yo/ha, with "Audio recordings by AfroVoices experts" (en, ig, yo, ha) — [hypaai/Hypa_Fleurs](https://huggingface.co/datasets/hypaai/Hypa_Fleurs)
- The full Hypa-Voices pool is about 8,442 h across 22 source languages: en, fr, es, pt, ar, sw, pcm, yo, ha, ig, ibb, tiv, efi, anw, idc, igl, nup, ego, fon, urh, tw, ibr. The card glosses the non-major ones as Ibibio, Tiv, Efik, Annang, Idoma, Igala, Nupe, Eggon, Fongbe, Urhobo, Twi and Ebira. Records come in "transcribe" or "translate" mode with anonymised speakers — [hypaai/Hypa-Voices](https://huggingface.co/datasets/hypaai/Hypa-Voices)
- The intronhealth org (creators of AfriSpeech-200) hosts afrispeech-200 (gated manual, CC BY-NC-SA 4.0, has a loading script `afrispeech-200.py`), afrispeech-dialog, afrivox(-v2, -transcribe, -translate, _in_the_wild), afrispeech-parliament, afrispeech-countries, med-convo-nig and others — [HF API author=intronhealth](https://huggingface.co/api/datasets?author=intronhealth); [afrispeech-200 API](https://huggingface.co/api/datasets/intronhealth/afrispeech-200)
- Third-party re-uploads of AfriSpeech data on HF include Professor/afrispeech-200-16khz, Johniblazee/naija-speech-afrispeech-ng (CC BY-NC-SA), okezieowen/afrispeech, okezieowen/afrispeech_yoruba / _tiv / _idoma, and hypaai/Hypa-AfriSpeech — [HF search "afrispeech"](https://huggingface.co/api/datasets?search=afrispeech&sort=downloads&direction=-1)
- The omnilingual-asr-corpus is CC BY 4.0, covers hundreds of languages (including anw) and is ungated — [HF API](https://huggingface.co/api/datasets/facebook/omnilingual-asr-corpus). A third-party dataset, "Chiz/omniASR-igbo-blindspots" (CC BY 4.0), is named after omniASR — [HF search "igbo"](https://huggingface.co/api/datasets?search=igbo&sort=downloads&direction=-1)

### Inferences
- **Hard-ban list (same org or derived):** all `hypaai/*`, all `AfriSpeech/*` (including open-bible-speech-african and youversion-african-speech even though their content comes from elsewhere), all `intronhealth/*` (the AfriSpeech brand; banning is the conservative call), every AfriSpeech-200 re-upload, and anything derived from omnilingual (e.g. Chiz/omniASR-igbo-blindspots, judging by its name).
- **Bible-audio caution:** african-speech-public_v1 does not state its source. Its language list (Gokana, Khana, Isoko, Esan, Kabiye...) and clip counts look like Bible-recording-derived corpora, but this is unverified. If it is Bible-derived, davidguzmanr/open-bible-resources and BibleTTS (Biblica/Open.Bible recordings) could contain the *same recordings* as some test clips. Rate the risk **medium (unverified)**. If Bible audio is used at all, use it only sparingly for the target classes, and note this in the model card.
- **Name-similarity flag:** Hypa_Fleurs credits "AfroVoices experts". The gated HF org **Africanvoice** (WaZoBiaSpeech) has a similar name but no stated link. Verify provenance before using it, or treat it as medium risk.
- **Strong clue about the test "other" class:** a Hypa-Voices-derived test set probably labels en, fr, es, pt, ar, sw, pcm, fon, tw and Ebira (ibr) as "other". An african-speech-ipa-derived one adds many West African languages (Edoid Edo/Esan/Isoko, Ijaw, Itsekiri, Ga, Ewe, Fon, Gun, Kabiye, Cameroonian Pidgin...).

### Gaps
- Could not confirm where AfriSpeech/african-speech-public_v1 audio came from: the card has no provenance text. It is also unknown whether the "AfriSpeech" HF org is legally the same entity as Intron Health.
- Could not confirm who is behind the Africanvoice HF org: the card is gated.

---

## Q1 (A): Best open Yoruba / Igbo / Hausa speech corpora

### Takeaway
Ungated sources that are fast to get today: **FLEURS** (10–14 h train per language, speaker-disjoint splits, CC BY 4.0), **Common Voice 27.0 via MDC** (CC0, crowd-sourced, many speakers, mixed consumer mics), **VoxLingua107** (yo 94 h, ha 106 h of YouTube/broadcast-like audio, CC BY 4.0; no Igbo) and **NaijaS2ST** (yo/ig/ha/pcm/Nigerian English, CC BY 4.0, very large). NaijaVoices (1,800 h, 5,000+ speakers) is the best for diversity, but it is CC BY-NC-SA, needs registration, and comes as a 500 GB+ (or 84 GB compressed) download. Phone or noisy channel material for the three languages is scarce: VoxLingua107, Common Voice and the Gamayun/TWB Hausa sets (mobile phones) are the best proxies.

### Cited Findings

**Table A — Yoruba / Igbo / Hausa**

| Dataset | Langs | Hours | Licence | Access (gated?) | Speech type / channel | Obtainable today? | Overlap risk | Source |
|---|---|---|---|---|---|---|---|---|
| google/fleurs (`ha_ng`, `ig_ng`, `yo_ng`) | ha, ig, yo (+99 others) | ha 13.6/1.5/3.3 h (train/dev/test); ig 13.7/1.9/4.8 h; yo 10.0/1.7/3.8 h (≈, summed from num_samples in the TSVs at 16 kHz) | CC BY 4.0 | HF, ungated; repo now has `data/` + `parquet-data/` and no loading script | Read FLoRes sentences; train speakers differ from dev/test speakers | Yes (small per-language download) | Low. Hypa_Fleurs reuses FLEURS *English text* with new recordings, so audio overlap is unlikely | [card](https://huggingface.co/datasets/google/fleurs), [API](https://huggingface.co/api/datasets/google/fleurs), [ha_ng TSVs](https://huggingface.co/datasets/google/fleurs/tree/main/data/ha_ng) |
| Common Voice Scripted Speech 27.0 (MDC) | ha, ig, yo (also pcm, ibb, tw, dag) | ha 273 MB, ig 399 MB, yo 175 MB MP3 (≈12–18 h each by size). CV 23.0 Hausa was 10,288 clips ≈13 h | CC0-1.0, plus MDC terms: no speaker re-identification, no re-hosting | MDC account + API key + accept terms in the web UI; released 2026-09-17 | Read prompts; crowd-sourced; browser/phone mics; many speakers | Yes (minutes once the account exists) | Low | [ha 27.0](https://mozilladatacollective.com/datasets/cmu5u0ood006tnq07tmgdpgwi), [ig 27.0](https://mozilladatacollective.com/datasets/cmu5v190x009pnq07l18h4qir), [yo 27.0](https://mozilladatacollective.com/datasets/cmu5q07ui011gmi07vflkzmej), [ha 23.0 hours](https://datacollective.mozillafoundation.org/datasets/cmflnuzw5y1pg5pgsavb9382w), [ha 24.0 terms](https://datacollective.mozillafoundation.org/datasets/cmj8u3p6w00alnxxbby9yqbms) |
| VoxLingua107 | yo, ha (no ig); 107 langs | yo 94 h, ha 106 h, en 49 h | CC BY 4.0 (copyright stays with video owners) | Ungated: per-language zips (yo.zip 9.8 GB, ha.zip 10.7 GB, en.zip 4.8 GB, HTTP 200 checked) or HF `TalTechNLP/voxlingua107_wds` (WebDataset shards of 500 clips per language) | YouTube, auto-labelled; varied mics, broadcast/vlog, background noise; about 2% wrong-language or non-speech | Yes (grab a few shards) | Low | [HF card](https://huggingface.co/datasets/TalTechNLP/voxlingua107_wds), [zips](https://cs.taltech.ee/staff/tanel.alumae/data/voxlingua107/) |
| McGill-NLP/NaijaS2ST | yo, ig, ha, pcm, en (Northern-, Southern- and British-Nigerian accents) | 65,980 train + 7,500 dev utterances; ≈78 GB of 48 kHz WAV (per-language hours not given) | CC BY 4.0 | HF, ungated | Read speech (S2ST benchmark); has `snr_db` and `silence_ratio` fields | Yes, but partial download advised | Low (Masakhane/McGill; IWSLT 2026) | [card](https://huggingface.co/datasets/McGill-NLP/NaijaS2ST) |
| naijavoices/naijavoices-dataset | yo, ig, ha | 1,800 h, 5,000+ speakers (≈600 h per language) | CC BY-NC-SA 4.0 + NaijaVoices T&Cs | HF gated "auto", but you must first register a membership tier at naijavoices.com; 500 GB+ (84 GB compressed copy) | Community-recorded read speech (per paper) | Probably (auto-gate), but the size is a problem | Low | [HF API/card](https://huggingface.co/datasets/naijavoices/naijavoices-dataset), [paper](https://arxiv.org/abs/2505.20564v1), [Interspeech 2025](https://www.isca-archive.org/interspeech_2025/emezue25_interspeech.html), [survey](https://arxiv.org/pdf/2505.18436) |
| Africanvoice/{yoruba,igbo,hausa}_african_voice(s) "WaZoBiaSpeech" | yo, ig, ha | "500+ hour" each (v. 30 Nov 2025) | CC BY 4.0 (card metadata) | HF gated "auto" | Not visible (card gated) | Probably | **Medium (unverified)**: provenance unknown; name resembles Hypa's "AfroVoices" | [igbo](https://huggingface.co/datasets/Africanvoice/igbo_african_voice), [org API](https://huggingface.co/api/datasets?author=Africanvoice) |
| BibleTTS (OpenSLR 129) | ha, yo (+ ewe, lin, Akuapem and Asante Twi) | hausa.tgz 21 GB, yoruba.tgz 6.3 GB | CC BY-SA 4.0 | OpenSLR, ungated; Yoruba mirror on HF: `Emulated-Inc/yoruba-speech-slr86-bibletts` (7,594 verses, 1.3 GB, 16 kHz FLAC) | Studio-quality, single-speaker Bible reading (Biblica) | Yes | **Medium (unverified)**: Bible audio (see Q0) | [OpenSLR 129](https://www.openslr.org/129/), [HF mirror](https://huggingface.co/datasets/Emulated-Inc/yoruba-speech-slr86-bibletts) |
| OpenSLR 86 (Google crowdsourced Yoruba) | yo | 462 MB female + 445 MB male zips; HF subset has 2,804 utts from 30 speakers | CC BY-SA 4.0 | OpenSLR, ungated | Read speech, high quality | Yes | Low | [OpenSLR 86](https://www.openslr.org/86/), [HF mirror](https://huggingface.co/datasets/Emulated-Inc/yoruba-speech-slr86-bibletts) |
| Lagos-NWU Yoruba Speech Corpus | yo | 165 min, 4,316 utts, 33 speakers (16F/17M) | CC BY 2.5 ZA (click-through) | SADiLaR repository | Read, phonetically balanced, 16 kHz, recorded in Lagos | Yes (small) | Low | [SADiLaR](https://repo.sadilar.org/handle/20.500.12185/431) |
| IroyinSpeech (validated part) | yo | 5 h | CC BY 4.0 | HF `Tundragoon/IroyinSpeech` (third-party upload; original on Kaggle) | Read news-domain text | Yes | Low | [HF](https://huggingface.co/datasets/Tundragoon/IroyinSpeech) |
| google/WaxalNLP `hau_tts`, `ibo_tts`, `yor_tts` | ha, ig, yo | Parquet sizes: hau 0.76 GB, ibo 1.18 GB, yor 2.59 GB (whole Waxal TTS is >180 h across 17 languages) | CC BY-SA 4.0 (Media Trust) | HF, ungated | "High-quality, single-speaker" read scripts (studio) | Yes | Low | [card](https://huggingface.co/datasets/google/WaxalNLP), [tree](https://huggingface.co/api/datasets/google/WaxalNLP/tree/main/data/TTS) |
| Data-Science-Nigeria/voice-of-care-health-dataset | ha | 52 h, 30,000 clips, 50 speakers (25F/25M), 4 dialects (Kano, Katsina, Sokoto, Zaria) | CC BY 4.0 | HF, ungated | Health-domain read utterances | Yes | Low (from the organiser's org, but not a prohibited source) | [card](https://huggingface.co/datasets/Data-Science-Nigeria/voice-of-care-health-dataset) |
| CLEAR-Global Gamayun Hausa mini-kit | ha | 4,986 clips (≈6.7 h) | CC BY-NC 4.0 | HF ungated; also on MDC | Read; recorded "on mobile phones or desktop computers" by paid voice talent | Yes | Low | [HF](https://huggingface.co/datasets/CLEAR-Global/Gamayun-hausa-voice-minikit), [MDC](https://mozilladatacollective.com/datasets/cmv5enx99004707ntkne42vaq) |
| TWB Voice 1.0 – Hausa | ha | 58 h (12.8 GB) | CC BY-NC 4.0 | MDC | Read speech | Yes | Low | [MDC](https://mozilladatacollective.com/datasets/cmp74k47s02u2mp07zr9zwhpt) |
| IADH Hausa/Igbo/Yoruba-TTS-Dataset | ha (5 h 25 m), ig, yo | 290 / 181 / 335 MB | NOODL-1.0 | MDC | TTS (single or few speakers) | Yes | Low | [ha](https://mozilladatacollective.com/datasets/cmnopto3q00t0mf07v2dtc0ej), [ig](https://mozilladatacollective.com/datasets/cmp0f5u2n02a8mp0771ryoly8), [yo](https://mozilladatacollective.com/datasets/cmo1nlaah0071mk077mw0qhpv) |
| benjaminogbonna/nigerian_common_voice_dataset | en, ha, ig, yo | 10K–100K rows | Labelled Apache-2.0, but it looks like a Common Voice re-upload | HF ungated | Read (CV) | Yes | Low; **licence/ethics concern** (MDC forbids re-hosting) | [API](https://huggingface.co/api/datasets/benjaminogbonna/nigerian_common_voice_dataset) |

- In the fixie-ai Common Voice 17 mirror, the configs include `ha` and `en` but no `ig` or `yo` — [card](https://huggingface.co/datasets/fixie-ai/common_voice_17_0)
- Excluded (banned family): AfriSpeech open-bible-speech-african (ha 99.7 h / ig 94.2 h / yo 89.9 h), youversion (ha 3.13 h, ig 4.18 h), AfriSpeech-200 and its re-uploads, and Hypa_Fleurs (ig/yo/ha/en) — see Q0 sources.

### Inferences
- **Minimal high-value set (A):** (1) FLEURS ha_ng/ig_ng/yo_ng, all splits (≈50 h total). (2) CV 27.0 ha/ig/yo from MDC (many speakers and consumer microphones; about 1 GB in total). (3) VoxLingua107 yo and ha, a subset of shards (e.g. 10–20 h each) for real-world YouTube channel variety. (4) A slice of NaijaS2ST (a few shards per language) for more Nigerian speakers, especially Igbo, since VoxLingua107 has none. Add NaijaVoices only if the 84 GB copy can be pulled in a few hours and the NC-SA licence is acceptable for the competition. Skip single-speaker TTS sets (Waxal TTS, IADH, BibleTTS) or down-weight them: they add little speaker diversity and BibleTTS carries Bible-overlap risk.
- Even though yor/ibo/hau are only 4.3% each of macro-F1, the "other" class is graded on not swallowing them. Phone-channel augmentation of the clean read data (FLEURS/CV) matters more than adding more clean hours.
- Using HF mirrors of Common Voice (fixie-ai, fsicoli, benjaminogbonna) conflicts with MDC's "no re-hosting" terms. To keep the model-card provenance clean, download from MDC directly.

### Gaps
- Exact hours and speaker counts for CV 27.0 ha/ig/yo: the MDC public API returns size but not hours; the figures above are size-based estimates.
- Per-language hours for NaijaS2ST, and the recording channel and speaker counts for WaZoBiaSpeech (gated).
- Whether NaijaVoices' "auto" gate approves instantly after naijavoices.com registration (not tested).

---

## Q2 (B): Nigerian English, Nigerian Pidgin and broad "other" coverage

### Takeaway
Pidgin is now easy to get: **CV 27.0 pcm** (CC0), **asr-nigerian-pidgin/nigerian-pidgin-1.0** (CC BY 4.0, recorded on an Android app), **NaijaS2ST pcm** and **Waxal pcm_tts**. For Nigerian-accented English, use **NaijaS2ST's English** (Northern, Southern and British-Nigerian accents, CC BY 4.0). AfriSpeech-200 and afrispeech-dialog are the classic sources but belong to the banned family. For broad "other" coverage, FLEURS gives en/fr/es/pt/ar/sw (the exact non-Nigerian languages in Hypa-Voices) plus Fula and Wolof, and WaxalNLP gives many-speaker natural speech in Akan/Twi, Ewe, Dagbani, Dagaare, Ikposo and Fula. Close Nigerian relatives (Edoid, Ebira, Itsekiri, Ijaw) are essentially only in banned AfriSpeech data, which is a real coverage gap.

### Cited Findings

**Table B — "other" class candidates**

| Dataset | Langs | Hours | Licence | Access | Speech type / channel | Today? | Overlap risk | Source |
|---|---|---|---|---|---|---|---|---|
| CV Scripted 27.0 – Nigerian Pidgin | pcm | 297 MB MP3 (≈13 h, size estimate) | CC0 + MDC terms | MDC (API key) | Read, crowd-sourced, mixed mics | Yes | Low | [MDC](https://mozilladatacollective.com/datasets/cmu5ufsxn007snq07xuxba0uc) |
| asr-nigerian-pidgin/nigerian-pidgin-1.0 | pcm | 4,277 utterances (≈976 MB, 16 kHz); 10 speakers (5M/5F, aged 20–28) | CC BY 4.0 | HF, ungated | Read; recorded with the LIG-Aikuma Android app "in quiet settings"; some door/table noises | Yes | Low | [card](https://huggingface.co/datasets/asr-nigerian-pidgin/nigerian-pidgin-1.0) |
| McGill-NLP/NaijaS2ST (pcm + en) | pcm; en with accent codes EN (Northern Nigerian), EY (Southern Nigerian), EB (British Nigerian) | Part of 73,480 utterances, 48 kHz | CC BY 4.0 | HF, ungated | Read | Yes (partial) | Low | [card](https://huggingface.co/datasets/McGill-NLP/NaijaS2ST) |
| google/WaxalNLP `pcm_tts` | pcm | 0.77 GB parquet | CC BY-SA 4.0 | HF, ungated | Single-speaker studio TTS | Yes | Low | [card](https://huggingface.co/datasets/google/WaxalNLP) |
| Africanvoice/naija_african_voices (WaZoBiaSpeech) | pcm | "1,000+ hour" | CC BY 4.0 | HF gated auto | Unknown | Probably | Medium (provenance unknown) | [API](https://huggingface.co/api/datasets/Africanvoice/naija_african_voices) |
| IADH Naija-TTS-Dataset | pcm | 341 MB | NOODL-1.0 | MDC | TTS | Yes | Low | [MDC](https://mozilladatacollective.com/datasets/cmnykldcz010knu0737o8bgh9) |
| benjaminogbonna/nigerian_accented_english_dataset | en (Nigerian accents) | ≈8 h | Labelled Apache-2.0; gating prompt quotes Common Voice terms, so likely CV-derived | HF ungated | Read | Yes | Low; licence provenance doubtful | [card](https://huggingface.co/datasets/benjaminogbonna/nigerian_accented_english_dataset) |
| intronhealth/afrispeech-dialog | Accented English (Hausa, Isoko, Idoma, Urhobo, Ijaw, Yoruba, Igbo, Igala, Ebira and other L1s) | 6 h, 50 conversations | CC BY-NC-SA 4.0 | HF ungated | Spontaneous dialogue, "collected remotely across various acoustic environments" | Yes | **High (banned family)** | [card](https://huggingface.co/datasets/intronhealth/afrispeech-dialog) |
| google/fleurs (non-target languages) | 102 langs incl. en_us 7.3 h train, sw_ke 12.1 h, ff_sn (Fula) 13.9 h, wo_sn 7.3 h, ln_cd 18.2 h, plus fr, es, pt, ar, am, so, xh, zu, lg, luo... | ≈10–18 h train per language (computed from TSVs) | CC BY 4.0 | HF, ungated | Read | Yes | Low | [card](https://huggingface.co/datasets/google/fleurs), [data tree](https://huggingface.co/api/datasets/google/fleurs/tree/main/data) |
| VoxLingua107 (non-targets) | 107 langs incl. en 49 h, fr 67 h, pt 64 h, ar 59 h, sw 64 h, ln 90 h, am 81 h, so 103 h, sn 30 h, mg 109 h | 6,628 h total | CC BY 4.0 | Ungated zips or HF WebDataset | YouTube; real-world noise and channel variety | Yes | Low | [card](https://huggingface.co/datasets/TalTechNLP/voxlingua107_wds) |
| google/WaxalNLP ASR | Akan (aka), Ewe, Dagbani, Dagaare, Ikposo, Fula (ful) + East African languages | ≈1,250 h transcribed across 19 languages; per-language parquet is 29 GB (ful) to 62 GB (ewe), including unlabeled audio | CC BY 4.0 (U. Ghana langs) / CC BY-SA 4.0 (Digital Umuganda) | HF, ungated; shards of about 0.2–0.5 GB, so a few shards are enough | "Transcribed natural speech from a wide variety of voices"; speaker-disjoint v2 splits | Yes (partial shards) | Low | [card](https://huggingface.co/datasets/google/WaxalNLP), [tree](https://huggingface.co/api/datasets/google/WaxalNLP/tree/main/data/ASR) |
| google/WaxalNLP TTS (non-targets) | twi, ewe, ful, fat, bau (Baoulé), + bam and wol folders present but commented out of configs | >180 h total TTS | CC BY / CC BY-SA | HF | Single-speaker studio | Yes | Low | [card](https://huggingface.co/datasets/google/WaxalNLP) |
| CV Scripted 27.0 – Twi / Dagbani / Swahili; CV Spontaneous 5.0 – Swahili / French | tw (26 MB), dag (553 MB), sw (22.4 GB), fr spontaneous (51 MB) | — | CC0 | MDC | Read (scripted) or spontaneous answers | Yes | Low | [tw](https://mozilladatacollective.com/datasets/cmu5oplv600yemi07eiv63hz9), [dag](https://mozilladatacollective.com/datasets/cmu5vkji5009lo107rm1763p9), [sw](https://mozilladatacollective.com/datasets/cmu5sgles002zo1071nphgtby), [fr spont.](https://mozilladatacollective.com/datasets/cmu5z135s00ginq072mliz1as) |
| Central Kanuri Speech Dataset | knc | 12.04 h approved, 3,576 recordings (2,134 read + 1,442 free-form), 24 contributors | CC BY-NC 4.0 | HF `ciatech-africa/central-kanuri-speech-dataset` (ungated) and MDC | Read + spontaneous (TWB Voice platform) | Yes | Low | [HF](https://huggingface.co/datasets/ciatech-africa/central-kanuri-speech-dataset), [MDC](https://mozilladatacollective.com/datasets/cmuo1rfsj01runn073id66vg7) |
| TWB Voice 1.0 – Kanuri | kau | 52 h (11.8 GB) | CC BY-NC 4.0 | MDC | Read | Yes | Low | [MDC](https://mozilladatacollective.com/datasets/cmp77rkiv02wao007n5ut2oxr) |
| NaijaVoices Tyap (Atyap) | kcg | 264 MB | CC BY-NC-SA 4.0 | MDC (NaijaVoices / Lanfrica Labs) | Community-recorded | Yes | Low | [MDC](https://mozilladatacollective.com/datasets/cmiodqzui00wwnx07j549ek4c) |
| Wolof sets | wo | various | e.g. mansourr/wolof-asr-commonvoice CC BY 4.0; galsenai/wolof_tts CC BY 4.0; soynade-research/Wolof-ASR-Data CC BY-SA 4.0 | HF | Mixed | Yes | Low | [HF search "wolof"](https://huggingface.co/api/datasets?search=wolof&sort=downloads&direction=-1) |
| Fula sets | ff | badrex/fulani-speech, Pullo-Africa-Protagonist/FulaData (no licence tag) | unknown | HF | unknown | Yes | Low | [HF search "fula"](https://huggingface.co/api/datasets?search=fula&sort=downloads&direction=-1) |
| Ewe / Twi Bible & navigation corpora | ee, tw | e.g. ghanaopenai/* (CC BY-NC 4.0), abiyo27/BibleTTS_Ewe-Bible (CC BY-SA 4.0) | — | HF | Bible/TTS or read prompts | Yes | Medium for Bible-derived (see Q0) | [HF search "ewe"](https://huggingface.co/api/datasets?search=ewe&sort=downloads&direction=-1) |

- HF searches for "ijaw", "urhobo", "isoko" and "edo" returned no audio datasets; Tiv and Idoma hits were okezieowen/afrispeech_* re-uploads (banned family) — [HF search](https://huggingface.co/api/datasets?search=ijaw)
- The MDC catalogue has no entries for the locales ff/ful, bin, gaa, fon, wo/wol, mos, kri, ak/aka or ee/ewe. kr/knc/kau, tw, dag, ln, sw and fr do have entries — [MDC API, e.g. locale=dag](https://mozilladatacollective.com/api/datasets?locale=dag)
- Fon: GbeBenin/FFSTC-2 (Fon speech translation, no licence tag) is on HF — [HF search "gbe"](https://huggingface.co/api/datasets?search=gbe)
- Hypa-Voices' non-Nigerian / "other-like" languages are en, fr, es, pt, ar, sw, pcm, fon, tw and Ebira — [Hypa-Voices card](https://huggingface.co/datasets/hypaai/Hypa-Voices)

### Inferences
- **The test "other" set very likely includes English, French, Spanish, Portuguese, Arabic, Swahili, Pidgin, Fon, Twi and Ebira** (if derived from Hypa-Voices). If derived from african-speech-ipa it adds Edoid languages, Ijaw, Itsekiri, Ga, Ewe, Gun, Kabiye, Cameroonian Pidgin and others. If derived from omnilingual, it could be almost any language. FLEURS covers en, fr, es, pt, ar and sw cheaply, with speaker-disjoint splits.
- **Yes, "other" should include acoustically close languages ("hard negatives")** so the model does not map any tonal Niger-Congo speech to the nearest target. Linguistic relationships:
  - Target urh (Urhobo) is Edoid, close to Edo, Esan and Isoko.
  - Targets ibb/efi/anw (Ibibio, Efik, Annang) are Lower Cross.
  - Target igl (Igala) is Yoruboid, close to Yoruba and Itsekiri.
  - Target nup (Nupe) is Nupoid, close to Ebira and Gbagyi.
  - Target idu (Idoma) is Idomoid.
  - Target ego (Eggon) is Plateau; Tyap (kcg) is a nearby Plateau language available on MDC.

  Of these hard negatives only Tyap, Kanuri, Fula, Hausa-adjacent Chadic (none found), Ewe, Twi, Dagbani and Fon-ish data are obtainable outside the banned family. Edoid, Ebira, Itsekiri and Ijaw are not. Mitigation: use whatever close-but-allowed languages exist (Tyap, Dagbani, Ikposo, Ewe, Akan, Fula, Kanuri, Wolof) and rely on open-set thresholding or calibration for the rest. This is an inference, not tested.
- **Pidgin and Nigerian English are probably the highest-value "other" sources** because they share Nigerian speakers, accents and code-switching with the targets. Without them the model may tag Pidgin as yor/ibo/hau, or the reverse.
- **BBC Pidgin radio:** no open licence was found during this session. BBC content is copyrighted, and permission cannot be obtained within the deadline, so exclude it (inference; not separately sourced).
- **NaijaSenti** is a tweet sentiment corpus; no speech component turned up in HF searches for "naija" (inference from search results).
- **Minimal high-value set (B):** (1) CV 27.0 pcm + nigerian-pidgin-1.0 (Android-recorded) + a NaijaS2ST pcm slice. (2) NaijaS2ST Nigerian English (EN/EY/EB) plus FLEURS en_us and a few VoxLingua107 en shards. (3) FLEURS fr/es/pt/ar/sw/ff_sn/wo_sn/ln_cd/am/so and a handful more Sub-Saharan languages (1–2 h each is enough for a "reject" class). (4) Waxal ASR shards for aka, ewe, dag, dga, kpo and ful (natural, many-speaker West African speech). (5) The Central Kanuri HF set (12 h) and MDC Tyap. (6) VoxLingua107 shards from a few dozen random languages for broad open-set coverage with real-world channels.

### Gaps
- No allowed corpus found for Edo, Esan, Isoko, Urhobo-adjacent Edoid languages, Ijaw, Itsekiri, Ebira, Ga or Krio outside the banned AfriSpeech family (searches of HF and MDC came up empty).
- English Common Voice: did not verify how many clips carry a Nigerian / West-African accent tag, or the size of the MDC English download.
- BBC Pidgin licensing terms were not checked directly.

### Side findings (outside B but relevant to the parent task: target languages)
- CV Scripted 27.0 – **Ibibio (ibb)**, 380 MB, CC0 — [MDC](https://mozilladatacollective.com/datasets/cmu5uwjhw008znq07h4vu9kxp)
- IADH **Efik**-TTS (312 MB) and **Tiv**-TTS (327 MB), NOODL-1.0 — [Efik](https://mozilladatacollective.com/datasets/cmp1ma8ty007eo607klisho1a), [Tiv](https://mozilladatacollective.com/datasets/cmo4nmfam00nxny07rssox2tj)
- NaijaVoices **Nupe** speech (1.57 GB, CC BY-NC-SA 4.0) — [MDC](https://mozilladatacollective.com/datasets/cmihkoth10246md07dtojxehg)
- HF: offiongbassey/efik_speech_dataset (CC BY 4.0); UmarBaba1/nupe_speech_corpus (CC BY-NC 4.0); mrheartng/idoma-tts-* (gated) — [HF search "efik"](https://huggingface.co/api/datasets?search=efik), [HF search "nupe"](https://huggingface.co/api/datasets?search=nupe), [HF search "idoma"](https://huggingface.co/api/datasets?search=idoma)

---

## Q3 (C): Noise, RIR and telephone-channel augmentation resources and tools

### Takeaway
The fastest licence-clean stack is **OpenSLR SLR28 RIRS_NOISES** (1.3 GB, Apache 2.0: real and simulated RIRs plus noises) and **MUSAN** (CC BY 4.0, 11 GB; or a noise-only HF mirror). Simulate phone channels with ffmpeg (GSM, G.711 µ/A-law, G.722/G.726, Opus, Speex, Codec2), pedalboard's `GSMFullRateCompressor`, or lhotse `Narrowband`/`Compress`. torchaudio's `apply_codec`/`AudioEffector` are gone as of 2.9. WHAM! and ESC-50 are non-commercial. DNS is huge (≈550 GB archived), so use an HF mirror or skip it.

### Cited Findings

**Table C1 — noise and RIR corpora**

| Resource | Content | Size | Licence | Access | Today? | Source |
|---|---|---|---|---|---|---|
| MUSAN (SLR17) | Music, speech and noise recordings | musan.tar.gz 11 GB | CC BY 4.0 | OpenSLR; HF mirrors `FluidInference/musan`, `bilguun/musan-noise` (CC BY 4.0, noise-only) | Yes | [OpenSLR 17](https://www.openslr.org/17/), [HF musan-noise](https://huggingface.co/datasets/bilguun/musan-noise), [HF FluidInference/musan](https://huggingface.co/datasets/FluidInference/musan) |
| SLR26 simulated RIRs | Simulated room IRs, various room configs | sim_rir_8k.zip 128 MB; sim_rir_16k.zip 178 MB | Apache 2.0 | OpenSLR | Yes (seconds to minutes) | [OpenSLR 26](https://www.openslr.org/26/) |
| SLR28 RIRS_NOISES | Simulated + real RIRs, isotropic and point-source noises, 16 kHz / 16-bit | rirs_noises.zip 1.3 GB | Apache 2.0 | OpenSLR | Yes | [OpenSLR 28](https://www.openslr.org/28/) |
| DNS Challenge (ICASSP 2023 / DNS5) | Clean speech, noise_fullband (58 GB), impulse_responses (5.9 GB) | ≈550 GB archived, ≈1 TB unpacked | Repo content CC BY 4.0, code MIT. Data keeps its source licences: AudioSet CC BY 4.0, Freesound CC0-only, DEMAND CC BY-SA 3.0, RIRs SLR26/28 Apache 2.0 | GitHub download scripts (dry-run by default); HF mirror `humanify/DNS-Noise` (CC BY 4.0) | Only partially / via mirror | [GitHub README](https://github.com/microsoft/DNS-Challenge), [HF DNS-Noise](https://huggingface.co/datasets/humanify/DNS-Noise) |
| WHAM! noise | Urban noise (SF Bay Area) with intelligible speech removed; also a 48 kHz version | (size not verified) | **CC BY-NC 4.0** | Direct S3 links (`wham_noise.zip`, `high_res_wham.zip`); HF `philgzl/wham` (CC BY-NC 4.0) | Yes | [WHAM! site](http://wham.whisper.ai/), [HF](https://huggingface.co/datasets/philgzl/wham) |
| ESC-50 | Environmental clips extracted from Freesound, 50 classes, 5 folds | small (< 1 GB) | **CC BY-NC 3.0** (ESC-10 subset CC BY) | GitHub; HF `ashraq/esc50` | Yes | [GitHub](https://github.com/karolpiczak/ESC-50), [HF](https://huggingface.co/datasets/ashraq/esc50) |
| FSD50K | Freesound sound events (AudioSet ontology) | dev ≈18.4 GB + eval ≈6.3 GB (split zips) | Zenodo record licence shown as "other-at"; HF mirror tagged CC BY 4.0. Per-clip licences vary (not re-verified) | Zenodo 4060432; HF `Fhrozen/FSD50k` | Yes (large) | [Zenodo API](https://zenodo.org/api/records/4060432), [HF](https://huggingface.co/datasets/Fhrozen/FSD50k) |
| BUT Speech@FIT ReverbDB | Real RIRs from 9 rooms (offices, stairs, hotel, conference, lecture rooms), 31 mics, environmental noise; LibriSpeech re-transmissions | RIR-only tgz 8.7 GB; LibriSpeech re-transmission 117 GB | CC BY 4.0 | Direct download | Yes (RIR-only) | [BUT page](https://speech.fit.vut.cz/software/but-speech-fit-reverb-database) |
| MIT environmental IR survey | Real-world IRs | small | CC BY 4.0 on `benjamin-paine/mit-impulse-response-survey`; "unknown" on `davidscripka/...` | HF | Yes | [HF benjamin-paine](https://huggingface.co/datasets/benjamin-paine/mit-impulse-response-survey) |
| DEMAND | Multichannel environmental noise | small | CC BY-SA 3.0 (per DNS README); HF mirror tagged CC BY 4.0 (conflict) | Zenodo; HF `voice-biomarkers/DEMAND-acoustic-noise` | Yes | [DNS README](https://github.com/microsoft/DNS-Challenge), [HF](https://huggingface.co/datasets/voice-biomarkers/DEMAND-acoustic-noise) |

**Table C2 — telephone / codec / channel simulation tools**

| Tool | Relevant capabilities | Licence / version | Setup | Source |
|---|---|---|---|---|
| ffmpeg (CLI or subprocess) | Encoders in a standard Debian/Ubuntu build (checked locally): `libgsm`, `libgsm_ms`, `pcm_mulaw`, `pcm_alaw` (G.711), `g722`, `g726`, `g723_1`, `libopus`/`opus`, `libspeex`, `libcodec2`. The docs also cover `libopencore-amrnb` and `libvo-amrwbenc`, but no AMR encoder was present in this build | LGPL/GPL | Usually preinstalled. AMR-NB needs a build with `--enable-libopencore-amrnb` | [ffmpeg codecs docs](https://ffmpeg.org/ffmpeg-codecs.html); local `ffmpeg -encoders` |
| pedalboard (Spotify) | `GSMFullRateCompressor` ("emulate the sound of a 2G cellular phone connection", internal 8 kHz resample), `MP3Compressor`, `Resample` (selectable quality), plus filters | v0.9.26, GPL-3.0 | `pip install pedalboard` (wheels) | [docs](https://spotify.github.io/pedalboard/reference/pedalboard.html), [PyPI](https://pypi.org/project/pedalboard/) |
| audiomentations | AddBackgroundNoise, AddShortNoises, AddColorNoise/Gaussian SNR, ApplyImpulseResponse, RoomSimulator, AirAbsorption, BandPass/HighPass/LowPass/shelf/peaking filters, SevenBandParametricEQ, Mp3Compression, Aliasing, BitCrush, ClippingDistortion, Resample, Gain/GainTransition, Limiter, TimeMask... No built-in GSM/AMR/Opus codec transform | v0.43.1, MIT | `pip install audiomentations` | [README](https://github.com/iver56/audiomentations), [PyPI](https://pypi.org/project/audiomentations/) |
| lhotse | `Compress(codec in {"opus","mp3","vorbis","gsm"})` (GSM via soundfile GSM610 at 8 kHz); `Narrowband` (resample to 8 kHz, then µ-law or LPC10 codec, then optional resample back); `ReverbWithImpulseResponse`; `FastRandomRIRGenerator` | v1.33.0, Apache-2.0 | `pip install lhotse` (LPC10 needs SpanDSP) | [compress.py](https://github.com/lhotse-speech/lhotse/blob/master/lhotse/augmentation/compress.py), [torchaudio.py](https://github.com/lhotse-speech/lhotse/blob/master/lhotse/augmentation/torchaudio.py) |
| torchaudio | `functional.apply_codec` deprecated in 2.8 (pointed to `io.AudioEffector`). 2.9 removed the APIs deprecated in 2.8, and the 2.9 doc pages for both `apply_codec` and `AudioEffector` return 404. I/O moved to TorchCodec. Current PyPI version is 2.11.0. µ-law transforms still exist (used by lhotse) | BSD | Avoid for codecs on torch ≥ 2.9 | [2.8 apply_codec](https://docs.pytorch.org/audio/2.8/generated/torchaudio.functional.apply_codec.html), [2.9 notes](https://docs.pytorch.org/audio/2.9.0/torchaudio.html), [migration issue](https://github.com/pytorch/audio/issues/3902) |
| torch-audiomentations | GPU batch augmentations (noise, IR, filters) | v0.12.0 | pip | [PyPI](https://pypi.org/project/torch-audiomentations/) |

### Inferences
- **Fastest setup (under 30 minutes):** download SLR28 (1.3 GB; it has RIRs and noises in one zip) and SLR26 16 kHz (178 MB). For more noise variety, add MUSAN noise (the HF `bilguun/musan-noise` mirror, or `musan/noise` and `musan/music` from the 11 GB tar). Then build a single augmentation function:
  1. RIR convolution, p≈0.3.
  2. Additive noise at SNR 0–20 dB, p≈0.5.
  3. For "phone" variants (p≈0.3–0.4): band-pass 300–3400 Hz, resample to 8 kHz, then one random codec (GSM via pedalboard or ffmpeg libgsm, G.711 µ/A-law, Opus at 6–12 kbps, or Speex/Codec2), then resample back to 16 kHz.
  4. A little clipping or gain change.
- **Licences:** if the model card must stay clean for any commercial reading, prefer the CC BY / Apache / CC0 resources (MUSAN, SLR26/28, BUT ReverbDB, the CC0 parts of FSD50K) over WHAM! and ESC-50 (NC). A competition model card listing NC noise is probably acceptable for research, but flag it.
- **FSD50K, MUSAN-speech and AudioSet-derived noise contain human speech or babble.** Exclude speech-labelled clips, or use them only as low-SNR babble, so the model does not learn "speech in another language = other".
- **Simplest single dependency for telephone simulation:** `ffmpeg` via subprocess (already installed, covers GSM, G.711, G.722 and Opus). pedalboard is the in-memory alternative but is GPL-3.0; that matters only if the augmentation code ships with the model (it normally does not).

### Gaps
- WHAM! noise download size and the ESC-50 clip count were not re-verified this session.
- FSD50K's per-clip licence breakdown: the Zenodo API shows "other-at", which conflicts with the CC BY 4.0 tag on the HF mirror; the per-clip licences are listed in its metadata, which was not checked.
- Whether a static ffmpeg build with an AMR-NB encoder (libopencore-amrnb) is available in the user's training environment was not checked.

---

## Q4: Gated, scripted or removed datasets; current Common Voice access route

### Takeaway
Common Voice is **no longer distributed on HF**. Official access is only through MDC: account, API key, and per-dataset terms accepted in the web UI. CV 27.0 (2026-09-17) includes ha, ig, yo, pcm, ibb, tw and dag. FLEURS has been converted to parquet with no loading script, so `trust_remote_code` is not needed. NaijaVoices and WaZoBiaSpeech are auto-gated; AfriSpeech-200 is manually gated and script-based, and banned anyway.

### Cited Findings
- mozilla-foundation/common_voice_17_0 on HF now holds only a notice: "Effective October 2025, Mozilla Common Voice datasets are now exclusively available through Mozilla Data Collective" — [HF card](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0). mozilla-foundation/common_voice_16_1 returns an auth error, i.e. it has been removed or made private — [HF API](https://huggingface.co/api/datasets/mozilla-foundation/common_voice_16_1)
- MDC REST API: catalogue and dataset-detail endpoints need no credentials. Download needs an API key in the Authorization header, and the user must first have accepted the dataset's terms in the web interface; the API returns a signed download URL. Rate limits return 429 with Retry-After — [MDC API docs](https://mozilladatacollective.com/api-reference/docs); [search summary of docs](https://datacollective.mozillafoundation.org/api-reference/docs)
- Third-party (non-Mozilla) MDC clients exist on PyPI: `mavdt` and `mdc-ds` (unverified) — [mavdt](https://pypi.org/project/mavdt/), [mdc-ds](https://pypi.org/project/mdc-ds/)
- MDC Common Voice 24.0 Hausa terms prohibit speaker identification and re-hosting or re-sharing — [MDC ha 24.0](https://datacollective.mozillafoundation.org/datasets/cmj8u3p6w00alnxxbby9yqbms)
- The `datacollective.mozillafoundation.org` domain now 301-redirects to `mozilladatacollective.com` — [redirect observed](https://mozilladatacollective.com/datasets/cmj8u3p6w00alnxxbby9yqbms)
- google/fleurs: ungated; repo contents are `data/`, `parquet-data/` and README, with no `.py` script; lastModified 2026-05-15 — [HF API](https://huggingface.co/api/datasets/google/fleurs)
- google/WaxalNLP: ungated parquet, v2.0.0 (updated 03/2026); a speaker-disjoint ASR_v2 split map is available — [card](https://huggingface.co/datasets/google/WaxalNLP)
- naijavoices/naijavoices-dataset: `gated: auto`; you must register at naijavoices.com/membership and agree to the T&Cs; CC BY-NC-SA 4.0 — [HF API](https://huggingface.co/api/datasets/naijavoices/naijavoices-dataset)
- Africanvoice/*: `gated: auto`, CC BY 4.0 — [HF API](https://huggingface.co/api/datasets?author=Africanvoice)
- intronhealth/afrispeech-200: `gated: manual`, has the loading script `afrispeech-200.py` — [HF API](https://huggingface.co/api/datasets/intronhealth/afrispeech-200)
- TalTechNLP/voxlingua107_wds: ungated WebDataset tars (`train/`, `dev/`); the per-language zips on cs.taltech.ee are live — [HF](https://huggingface.co/datasets/TalTechNLP/voxlingua107_wds), [zips](https://cs.taltech.ee/staff/tanel.alumae/data/voxlingua107/)
- Ungated (checked via API): McGill-NLP/NaijaS2ST, asr-nigerian-pidgin/nigerian-pidgin-1.0, Data-Science-Nigeria/voice-of-care-health-dataset, CLEAR-Global/Gamayun-hausa-voice-minikit, ciatech-africa/central-kanuri-speech-dataset — [HF search results](https://huggingface.co/api/datasets?search=nigerian)

### Inferences
- **Fastest path for Common Voice:** create an MDC account, accept terms for ha, ig, yo, pcm (plus ibb for the parent task), generate an API key, then `curl` the download endpoint. About 1.4 GB total for the four, roughly 15–30 minutes including sign-up. Avoid HF CV mirrors so the model card stays clean.
- **Download plan for the deadline:** the large sets (NaijaVoices 84 GB+, NaijaS2ST about 78 GB, Waxal ASR 29–62 GB per language, BibleTTS Hausa 21 GB, VoxLingua107 about 10 GB per language, DNS about 550 GB) should be fetched as a few shards each (HF `hf_hub_download` of specific parquet or tar files), not in full.

### Gaps
- Not tested whether MDC download links are issued instantly for CC0 Common Voice datasets, or whether there is any approval lag.
- Not tested whether NaijaVoices' auto gate approves immediately after website registration.

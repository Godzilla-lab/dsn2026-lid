# External speech data for nine low-resource Nigerian languages (efi, ibb, igl, nup, anw, ego, idu, tiv, urh): DSN 2026 LID

Research date: 2026-10-20. Deadline context: about 27 h left. No audio was downloaded. Existence was checked through HF Hub API metadata, dataset cards, datasets-server first-rows (text columns only), HEAD requests, and web pages. Gated cards were read with the session's HF token. All parquet HEAD checks on gated repos returned **403**, so the token does not yet have access to any of the gated datasets below.

## Q1. Per-language inventory: what usable speech data exists for each of the nine languages?

### Takeaway
New, legitimate, non-prohibited audio is available today for the four weakest classes:
- **Efik**: offiongbassey/efik_speech_dataset, 8.79 h, CC-BY-4.0, ungated.
- **Igala, Nupe, Urhobo**: VoiceAfrica, about 12 h each, spontaneous speech, CC-BY-NC-SA-4.0, HF auto-gated.
- **Nupe**: NigBench-MAMAI, 50 h, CC-BY-4.0, but manual gating, so approval time is uncertain.
- **Ibibio**: Common Voice (18 h) carries a real overlap risk with Hypa-Voices.
- **Tiv**: the only safe extra audio is Global Recordings Network (about 3.9 h of religious audio). YouVersion and Faith Comes By Hearing Tiv Bible audio is almost certainly already in the prohibited AfriSpeech corpus.
- **Annang and Eggon**: essentially nothing beyond small GRN/MMS-ulab religious programs.

### Inventory table

Competition data per class (train / val / **test** hours): efi 1.42/0.15/**0.43**; ibb 1.76/0.28/**0.53**; igl 0.48/0.12/**0.15**; nup 0.50/0.00/**0.14**; tiv 4.27/0.63/**1.20**; urh 9.22/1.57/**1.21**; anw 8.12/1.83/**2.05**; ego 7.32/1.05/**2.15**; idu 7.16/2.12/**2.72** ([DSN hours_report.csv](https://huggingface.co/datasets/Data-Science-Nigeria/dsn-2026-lid-train)).

Overlap risk rating key:
- **Banned**: derived from or mirrors a prohibited corpus.
- **High**: the same upstream recordings likely feed a prohibited corpus.
- **Low/Med**: independent, but the provenance is not fully documented.
- **Low**: independent and well documented.

| Lang | Source | Hours / clips | Speakers | Speech type | Licence / terms | Access | Obtainable today? | Overlap risk |
|---|---|---|---|---|---|---|---|---|
| **efi** | [offiongbassey/efik_speech_dataset](https://huggingface.co/datasets/offiongbassey/efik_speech_dataset) | 8.79 h; 5,875 clips | 2 | Read (literary text, studio-quiet) | CC-BY-4.0 | HF, ungated | **Yes** | Low/Med (independent Interspeech 2026 corpus) |
| efi | [offiongbassey/efik_audio_dataset](https://huggingface.co/datasets/offiongbassey/efik_audio_dataset) | 2,632 clips (about 3 h, the original paper corpus) | 1 | Read | No licence tag | HF, ungated | Yes | Likely a subset of the 8.79 h set, so dedupe; licence unclear |
| efi | GRN "Words of Life" ([GRN efi](https://globalrecordings.net/en/language/efi)) | 28:20 min (+ "Noah" mp3) | Unknown | Religious, with music | CC BY-NC-SA | Zip download (24 MB MP3) | Yes (site has bot protection) | Low |
| efi | [espnet/mms_ulab_v2](https://huggingface.co/datasets/espnet/mms_ulab_v2) (GRN audio) | Per-language hours not retrievable | Unknown | Religious, unsegmented | CC-BY-NC-SA-4.0 | HF, ungated (115 mixed parquet shards, 637 GB) | Yes, but must filter by `iso3` | Low (same audio as GRN) |
| efi + ibb | [scubavoice/ibibio-efik-speech-corpus-sample](https://huggingface.co/datasets/scubavoice/ibibio-efik-speech-corpus-sample) | About 1 h total | Unknown in sample (500+ in full set) | Prompted conversational | "Evaluation purposes only… non-commercial research… Redistribution is not permitted" | HF, ungated | Yes, but terms are restrictive | Low; licence is the problem. [williamudo/…](https://huggingface.co/datasets/williamudo/ibibio-efik-speech-corpus-sample) is an identical re-upload (same 64,279,408 bytes) and should not be used |
| efi | Faith Comes By Hearing NT; Jesus Film in Efik (listed on [GRN efi](https://globalrecordings.net/en/language/efi)) | NT is typically about 20 h (not verified) | – | Bible | DBP terms forbid offline download (see Q2) | DBP API / streaming | **No** (terms) | Unknown |
| **ibb** | Common Voice Scripted Speech 25.0 – Ibibio ([MDC](https://datacollective.mozillafoundation.org/datasets/cmn2cp74s01hamm07s74uypfk)) | 18.18 h recorded / 7.76 h validated; 7,274 clips | 19 | Read (1,031 sentences) | CC0-1.0 plus MDC terms (no re-hosting, no re-identification) | Mozilla Data Collective download (login) | Yes | **High**: Hypa-Voices says it includes "Mozilla Common Voice-derived subsets" (see Q4) |
| ibb | GRN Words of Life (Ibibio) + Words of Life (Beebo variety) ([GRN ibb](https://globalrecordings.net/en/language/ibb)) | 43:14 + 46:28 min | Unknown | Religious, with music | CC BY-NC-SA | Zip (74.7 MB MP3) | Yes | Low |
| ibb | MMS-ulab v2 (GRN) | Unknown | – | Religious | CC-BY-NC-SA-4.0 | HF | Yes | Low |
| ibb | Scuba full corpus | About 200 h | 500+ | Prompted | Commercial licence | Email info@scubavoice.com | **No** (not within hours) | Low |
| **igl** | [naijavoices/voiceafrica-datasets](https://huggingface.co/datasets/naijavoices/voiceafrica-datasets), config `igl` | 12.1 h; 1,570 clips | 10 | **Spontaneous** (600 prompts, 1–2 min answers, chunked to ≤30 s) | CC-BY-NC-SA-4.0 + [NaijaVoices T&C](https://naijavoices.com/terms-and-conditions) | HF gated **auto**, plus NaijaVoices membership registration link | **Probably** (auto-approve; token currently gets 403) | Low/Med (Lanfrica–Meta collaboration, but `igl` is not in the omnilingual corpus) |
| igl | MMS-ulab v2 (GRN) | Unknown (GRN igl page blocked) | – | Religious | CC-BY-NC-SA-4.0 | HF | Yes | Low |
| igl | [okezieowen/afrispeech_igala](https://huggingface.co/datasets/okezieowen/afrispeech_igala) = [hypaai/Hypa-AfriSpeech](https://huggingface.co/datasets/hypaai/Hypa-AfriSpeech) `igala` split | 898 clips | – | **Igala-accented English** (transcripts are English) | None stated | HF | Yes | **Do not use as igl.** Wrong language; also part of the Hypa family |
| **nup** | VoiceAfrica `nup` ([card](https://huggingface.co/datasets/naijavoices/voiceafrica-datasets)) | 12.6 h; 1,604 clips | 10 | Spontaneous | CC-BY-NC-SA-4.0 + NaijaVoices T&C | HF gated auto | Probably | Low/Med |
| nup | [intronhealth/NigBench-MAMAI-Speech-QA](https://huggingface.co/datasets/intronhealth/NigBench-MAMAI-Speech-QA), config `Nupe` | **50.0 h** (45 train / 5 test); 1,479 recordings | Not stated (women, speaker-disjoint split) | Spontaneous maternal-health questions, rural noise | **CC-BY-4.0** | HF gated **manual** | **Uncertain** (needs human approval) | Low |
| nup | NaijaVoices Micro-Grant Nupe ([MDC](https://mozilladatacollective.com/datasets/cmihkoth10246md07dtojxehg)) | 2 h 40 m (metadata) / 3 h 17 m (description); 1,583–1,602 clips | 7–8 (Bida, Kutigi, Lapai accents) | Read with transcripts (not stated) | CC-BY-NC-SA-4.0; voice cloning banned; commercial waiver needed | MDC download (1.46 GB tar.gz) | Yes | Low |
| nup | [UmarBaba1/nupe_speech_corpus](https://huggingface.co/datasets/UmarBaba1/nupe_speech_corpus) | 1,905 clips (2.5 GB); hours unknown | ≥1 | Read/conversational Nupe text (mis-tagged `ha`) | CC-BY-NC-4.0 | HF, ungated | **Yes** | Low; likely overlaps the NaijaVoices Nupe set (same collector, Umar Baba Umar), so dedupe |
| nup | GRN Words of Life (Nupe) + Words of Life (Ibara / Nupe Tako) ([GRN nup](https://globalrecordings.net/en/language/nup)) | 50:22 + 29:01 min | Unknown | Religious | CC BY-NC-SA | Zip (55.9 MB MP3) | Yes | Low |
| **tiv** | GRN: Good News 33:53, Portrait of Jesus 1:17:42, Becoming a Friend of God 32:25, Words of Life 1 41:27, Words of Life 2 49:02 ([GRN tiv](https://globalrecordings.net/en/language/tiv)) | **About 3 h 54 min** (+35:34 Uive variety) | Unknown | Religious, with songs | CC BY-NC-SA | Zip (239.6 MB MP3) | Yes | Low (GRN is not upstream of the AfriSpeech Tiv data; see Q4) |
| tiv | MMS-ulab v2 (GRN) | Unknown | – | Religious | CC-BY-NC-SA-4.0 | HF | Yes | Low |
| tiv | [AfriSpeech/youversion-african-speech](https://huggingface.co/datasets/AfriSpeech/youversion-african-speech) `Tiv_tiv`; FCBH Tiv NT | 4.59 h; 1,982 segments | – | Bible | No licence tag; DBP terms forbid download | HF | – | **High / effectively banned** (Q4) |
| tiv | [SLANGUAGE/SLANGUAGE_TIV_001](https://huggingface.co/datasets/SLANGUAGE/SLANGUAGE_TIV_001) | Text only (1,169 sentences); "Audio recordings are in progress" | – | – | CC-BY-NC-4.0 | Gated manual | No audio | n/a |
| tiv | okezieowen/afrispeech_tiv / Hypa-AfriSpeech `tiv` | 75 clips | – | Tiv-accented **English** | – | HF | – | Do not use |
| **urh** | VoiceAfrica `urh` | 11.9 h; 1,511 clips | 10 | Spontaneous | CC-BY-NC-SA-4.0 | HF gated auto | Probably | Low/Med |
| urh | GRN Words of Life ([GRN urh](https://globalrecordings.net/en/language/urh)) | 46:04 min | Unknown | Religious | CC BY-NC-SA | Zip (40 MB) | Yes | Low |
| urh | FCBH NT Urhobo; Jesus Film in Urhobo (listed on [GRN urh](https://globalrecordings.net/en/language/urh)) | – | – | Bible / film | DBP terms forbid download | – | No | **Med/High** (the 25.67 h of AfriSpeech Urhobo has an undocumented, likely religious source) |
| **anw** | GRN Words of Life ([GRN anw](https://globalrecordings.net/en/language/anw)) | 34:57 min | Unknown | Religious | CC BY-NC-SA | Zip (29.7 MB) | Yes | Low |
| anw | [AfriSpeech/grn-african-speech](https://huggingface.co/datasets/AfriSpeech/grn-african-speech) `anw_annang`; also `4546_beebo` (an Ibibio variety) | 0.44 h / 106 clips; Beebo 0.42 h / 101 clips | – | GRN religious, music-filtered 15 s clips | "Audio (c) Global Recordings Network, used for research purposes" | HF, ungated | Yes | Low (GRN derivative, not upstream of a prohibited set) |
| anw | MMS-ulab v2 (GRN); Jesus Film in Anaang | Unknown | – | Religious | – | – | MMS: yes | Low |
| **ego** | MMS-ulab v2 (tagged `ego`); GRN page not reachable | Unknown | – | Religious | CC-BY-NC-SA-4.0 | HF | Yes | Low |
| **idu** | [mrheartng/idoma-tts-multiple-speakers](https://huggingface.co/datasets/mrheartng/idoma-tts-multiple-speakers); [idoma-tts-speaker1](https://huggingface.co/datasets/mrheartng/idoma-tts-speaker1); [adah-idoma](https://huggingface.co/datasets/mrheartng/adah-idoma) | 18,391 / 7,317 / 998 clips (9.8 GB / 3.7 GB / 0.1 GB) | Multi / 1 / ? | Collection method undocumented (thesis project, ASR/TTS) | Apache-2.0 | Gated manual / manual / auto | Uncertain | Unknown provenance; could be read or TTS-oriented |
| idu | MMS-ulab v2 (GRN) | Unknown | – | Religious | CC-BY-NC-SA-4.0 | HF | Yes | Low |

### Cited findings
- The HF Hub `language=` query parameter is silently ignored (it returned unrelated trending datasets). The working form is `filter=language:xxx`. Counts were: efi 44, ibb 29, igl 28, nup 27, anw 21, ego 23, idu 35, tiv 44, urh 35 datasets (most are text) — [HF API](https://huggingface.co/api/datasets?filter=language:efi)
- Offiongbassey Efik set: "this updated release contains two speakers and **8.79 hours** of speech, recorded in a quiet environment… manually transcribed and reviewed by linguists and native speakers". CC-BY-4.0, 5,875 rows — [HF card](https://huggingface.co/datasets/offiongbassey/efik_speech_dataset). The original paper corpus was single-speaker, 2,632 utterances, about 3 h — [arXiv 2607.04515](https://arxiv.org/abs/2607.04515)
- VoiceAfrica: "a Lanfrica–Meta collaboration (code-named *VoiceAfrica 1*)… ~125 hours… 16,383 audio samples from 118 speakers". igl 1,570 clips / 12.1 h / 10 speakers; nup 1,604 / 12.6 h / 10; urh 1,511 / 11.9 h / 10. "native speakers were given prompts… and spoke freely for 1–2 minutes per prompt… split into… chunks (≤30s)". "The **same 600 prompts** were used across all 11 languages", and prompts were shown in English for Nigeria — [HF card](https://huggingface.co/datasets/naijavoices/voiceafrica-datasets)
- VoiceAfrica gate prompt: "register on NaijaVoices for the tier that corresponds to your use case". Gating mode is `auto` — [HF API](https://huggingface.co/api/datasets/naijavoices/voiceafrica-datasets?full=true)
- NigBench-MAMAI ("Voices for Smart Care"): 600.16 h across 7 languages, of which Nupe is 45.00 h train + 5.00 h test (1,326 + 153 recordings). "participants described authentic maternal health concerns in their own words". "Released under the **CC BY 4.0** license". Gating mode is `manual` — [HF card](https://huggingface.co/datasets/intronhealth/NigBench-MAMAI-Speech-QA)
- NaijaVoices Nupe on MDC: CC-BY-NC-SA-4.0. "you must obtain a proper commercial waiver from NaijaVoices". "Voice cloning… is explicitly prohibited". Collected by Umar Baba Umar under the 2025 NaijaVoices Micro-Grants. Released 2025-11-27 — [MDC](https://mozilladatacollective.com/datasets/cmihkoth10246md07dtojxehg)
- Common Voice Ibibio 25.0 (March 2026): 7,274 clips, 18.18 h recorded (7.76 h validated), 19 speakers, 1,031 sentences, CC0-1.0, but datasheets forbid re-hosting — [MDC CV Ibibio 25.0](https://datacollective.mozillafoundation.org/datasets/cmn2cp74s01hamm07s74uypfk) (via search snippet; the page itself did not render for the fetcher)
- In the Common Voice language API, only `ibb` is contributable. Efik is registered as `efk` with `is_contributable: 0`, and Tiv as `tiv` with `is_contributable: 0`. igl, nup, anw, ego, idu and urh are absent — [CV API](https://commonvoice.mozilla.org/api/v1/languages)
- MMS ulab v2: "8900 hours of unlabeled speech across 4023 languages… raw unsegmented audio". CC-BY-NC-SA-4.0. Users are asked to "reference [The Global Recordings Network]… the original source of the data". All nine codes appear in its language list — [HF card](https://huggingface.co/datasets/espnet/mms_ulab_v2). A segmented version is at [coml/mmsulab](https://huggingface.co/datasets/coml/mmsulab) (pyannote-segmented, same licence)
- Scuba sample: "provided for **evaluation purposes only**… Free for evaluation and non-commercial research… Redistribution is not permitted". The full set is about 200 h and 500+ speakers via info@scubavoice.com — [HF card](https://huggingface.co/datasets/scubavoice/ibibio-efik-speech-corpus-sample)
- Hypa-AfriSpeech's split sizes (igala 898, tiv 75, idoma 1,760) and first rows match okezieowen/afrispeech_*. The transcripts are English sentences (e.g. "Today is day 3 of a spell countdown.") — [hypaai/Hypa-AfriSpeech](https://huggingface.co/datasets/hypaai/Hypa-AfriSpeech), [okezieowen/afrispeech_igala](https://huggingface.co/datasets/okezieowen/afrispeech_igala)
- SLANGUAGE Tiv v1 is text only, with 1,169 sentences: "Audio recordings are in progress and will be released separately". "not licensed for commercial use" — [HF card](https://huggingface.co/datasets/SLANGUAGE/SLANGUAGE_TIV_001)
- The Idoma mrheartng sets are a thesis project (wav2vec2 XLS-R ASR, MMS-TTS, NLLB). The cards do not describe how the audio was recorded. Apache-2.0; two are manually gated — [HF card](https://huggingface.co/datasets/mrheartng/idoma-tts-multiple-speakers)
- Google WAXAL and the other cached cards checked (FLEURS-R, NaijaS2ST, VoxLingua107) contain none of the nine language names or codes. Source: grep of cached HF cards (e.g. [google/WaxalNLP](https://huggingface.co/datasets/google/WaxalNLP))
- NaijaVoices/Lanfrica's MDC organisation lists Atyap, Ekpeye, Ehugbo, Ibọnọ/Obolo, Gbagyi and Nupe only, all CC-BY-NC-SA-4.0 — [MDC NaijaVoices org](https://datacollective.mozillafoundation.org/organization/cmidd9v3l002nmh07kmij7fq5)

### Inferences
- **Biggest wins by class:**
  - **nup**: VoiceAfrica + NigBench + NaijaVoices/UmarBaba1 add up to about 60+ h, against 0.50 h provided.
  - **igl**: VoiceAfrica adds 12.1 h, against 0.48 h provided.
  - **efi**: the Efik TTS set adds 8.79 h, against 1.42 h provided.
  - **ibb**: the only sizeable source is Common Voice, which is risky (Q4). The fallback is GRN (about 1.5 h incl. Beebo), plus the AfriSpeech GRN Beebo clips.
- Speaker diversity is low in several sources (Efik 2 speakers; VoiceAfrica 10 per language). Speaker-level grouping during training and validation is needed to avoid overfitting to voices.
- Domain match. The test audio for efi/ibb/igl/nup must come from Hypa-Voices, since it is the only prohibited corpus covering those codes. Hypa-Voices is read or prompted speech (CV-style sentences and translations), so read sources such as the Efik TTS set and the NaijaVoices Nupe set match the domain better than spontaneous VoiceAfrica or NigBench. Both styles probably help robustness.
- GRN and MMS-ulab audio contain songs and music. The program durations are upper bounds on usable speech; AfriSpeech's music-filtered GRN cut kept only about 0.44 h of the 35 min Annang program.
- MMS-ulab v2 per-language content most likely equals the GRN programs listed above. Its rows are whole GRN recordings tagged by `iso3`, so filtering by `iso3` would yield roughly the GRN minutes.

### Gaps
- MMS-ulab v2 per-language row counts and hours could not be retrieved. The datasets-server `/filter` endpoint kept returning "the dataset index is loading" for about 5 minutes.
- GRN pages for igl, ego and idu returned 403 (bunny.net bot protection), so their GRN program durations are unknown.
- Hours for UmarBaba1/nupe_speech_corpus, and whether it duplicates the MDC Nupe set, are unknown.
- Speaker count for NigBench Nupe is not stated.
- Provenance of the mrheartng Idoma audio is unknown.
- No evidence was found of any openly licensed speech data for **Eggon** beyond GRN/MMS-ulab, or for **Annang** beyond GRN, AfriSpeech's GRN cut and the Jesus Film.
- Not checked: OpenSLR, Zenodo, ELRA/LDC, Wikitongues, Masakhane. Budget ran out; web searches through Lanfrica/MDC/arXiv surfaced nothing for these languages from those catalogues. The "Voice of a Continent" survey PDF ([arXiv 2505.18436](https://arxiv.org/pdf/2505.18436)) could not be parsed. A search snippet said it lists an OlongoAfrica anthology that includes Tiv; its licence and hours are unverified.

## Q2. Religious-audio sources: coverage and what their terms permit for ML training

### Takeaway
GRN is the only religious source whose terms plausibly allow this use. It is CC BY-NC-SA for audio, allows adaptation under identical terms, and says nothing about ML. GRN covers all nine languages, directly or through MMS-ulab. Faith Comes By Hearing / Digital Bible Platform (Bible.is) terms forbid offline download and any non-app use, so its NT audio (efi, tiv, urh) is not usable. API.Bible explicitly forbids AI training without written consent. Jesus Film terms were not verified, and the GRN-hosted Jesus Film derivatives forbid modification.

### Cited findings
- GRN audio and video are "available to be copied and used under the Creative Commons Attribution-NonCommercial-ShareAlike license". "You may also adapt or build upon the audio or video as long as you license your new creations under identical terms". "do not benefit financially from them". "Permission may be given for copying and use of GRN materials beyond what is granted above". The page does not mention ML/AI or bulk download — [GRN copyright](https://globalrecordings.net/en/copyright)
- GRN pages list "Audio/Video from other sources":
  - Efik: Jesus Film; FCBH New Testament
  - Ibibio: Jesus Film only
  - Nupe: Jesus Film; "The Promise" (StoryRunners)
  - Tiv: Jesus Film; FCBH NT
  - Urhobo: Jesus Film; FCBH NT
  - Annang: Jesus Film in Anaang

  Sources: [efi](https://globalrecordings.net/en/language/efi), [ibb](https://globalrecordings.net/en/language/ibb), [nup](https://globalrecordings.net/en/language/nup), [tiv](https://globalrecordings.net/en/language/tiv), [urh](https://globalrecordings.net/en/language/urh), [anw](https://globalrecordings.net/en/language/anw)
- GRN program lengths:
  - efi: Words of Life 28:20
  - ibb: Words of Life 43:14, plus Beebo variety 46:28
  - nup: 50:22, plus Nupe Tako/Ibara 29:01
  - tiv: five programs totalling about 3:54:29, plus Uive 35:34
  - urh: 46:04
  - anw: 34:57

  Sources: the GRN language pages above
- FCBH Bible Brain / DBP licence:
  - "access the DBP Content solely through DBP (and not separately through offline download)"
  - "personal, non-commercial use on a single device"
  - "No DBP Content may be downloaded or made available for offline use by any person or outside of DBP"
  - "Other uses are not permitted without further consent"
  - "We reserve all rights not expressly granted"
  - No clause mentions AI/ML

  Source: [FCBH Bible Brain licence](https://www.faithcomesbyhearing.com/bible-brain/license)
- API.Bible (a different platform) restricts using content "to train, fine-tune, or improve generative AI models without prior written consent". Consent must come from both API.Bible and the IP holder — [API.Bible help: training AI](https://care.api.bible/article/386-what-if-i-want-to-train-ai-with-a-copyright-bible) (via search summary)
- GRN program pages for Jesus Film–based audio dramas state: "may be freely copied for personal or local ministry use on condition that it is not modified, and it is not sold or bundled with other products which are sold" — [GRN program page](https://globalrecordings.net/en/program/36060) (via search snippet)
- Open.Bible audio on HF (AfriSpeech re-host of davidguzmanr/open-bible-resources) is CC BY-SA 4.0 but covers 19 African languages, **none of the nine** — [AfriSpeech/open-bible-speech-african](https://huggingface.co/datasets/AfriSpeech/open-bible-speech-african)
- AfriSpeech's GRN derivative says "Audio (c) Global Recordings Network, used for research purposes. Please consult GRN's terms for any other use." Among the nine languages it contains only Annang (0.44 h) and Beebo (0.42 h) — [AfriSpeech/grn-african-speech](https://huggingface.co/datasets/AfriSpeech/grn-african-speech)

### Inferences
- GRN/MMS-ulab is CC BY-NC-SA, so a model trained on it arguably needs to be released NC-SA-compatible if it counts as an "adaptation". Whether model weights are an adaptation is legally unsettled. The model card should list GRN with attribution and note the NC-SA licence.
- BibleTTS, FLEURS and Open.Bible offer nothing for these nine languages.

### Gaps
- The Jesus Film Project's own terms page was not retrieved.
- JW.org audio (which probably exists for Tiv, Urhobo, Efik, Ibibio, Idoma, Igala) and its terms were not checked.
- eBible.org audio was not checked.
- The GRN "Language Hotline" / permission process time is unknown.

## Q3. Permissioned radio, podcast and YouTube sources

### Takeaway
No permissioned, downloadable radio or podcast archive was found for any of the nine languages. With about 27 h left, getting broadcaster permission is not realistic. Treat radio/YouTube as out of scope for this deadline.

### Cited findings
- AKBC (Akwa Ibom State Broadcasting Corporation, Uyo) provides TV and radio; one source gives the radio frequency as 90.5 MHz. No AKBC programme archive or podcast in Ibibio, Efik or Annang was found — [AKBC wiki mirror](https://wiki.riteme.site/wiki/AKBC)
- "Ufok Nyin" is an Ibibio/Efik learning podcast with 12 episodes; the last is dated 2024-03-08 and it is listed as inactive. No licence was found — [PodEngine](https://www.podengine.ai/podcasts/ufok-nyin)

### Inferences
- The fastest legitimate route would be emailing a podcast creator or station for written permission, then citing that permission in the model card. Response within hours is unlikely.
- Radio audio also carries code-switching and English, so labels would be noisy.

### Gaps
- Not researched due to budget: Radio Benue (Tiv), Kogi State radio (Igala), Radio Niger / Bida (Nupe), Delta Broadcasting Service (Urhobo), and YouTube channels.

## Q4. Overlap with the prohibited corpora (Hypa-Voices, omnilingual-asr-corpus, african-speech-ipa)

### Takeaway
The competition's train, validation and hidden test sets were all split from the three prohibited corpora.
- **efi, ibb, igl and nup** test audio must come from **Hypa-Voices**. Hypa-Voices says it contains Common Voice-derived material, so **Common Voice Ibibio is high-risk**.
- **Tiv** in african-speech-ipa almost certainly comes from YouVersion Bible audio, so **YouVersion, Bible.is and FCBH Tiv audio is effectively banned**.
- **Urhobo** in african-speech-ipa (25.67 h) has an undocumented, probably religious upstream, so FCBH Urhobo is risky.
- **Banned mirrors and derivatives** found on HF:
  - afri2universal
  - african-speech-public_v1
  - Hypa-Voices-snac
  - Hypa-Speech-10k and its mirror
  - the Hypa benchmarks
  - several omnilingual mirrors
  - vnahata/OmnilingualASR-retrieval
- VoiceAfrica, NigBench, the Efik TTS set, NaijaVoices Nupe and GRN show no documented overlap.

### Cited findings
- DSN PROHIBITED_SOURCES.txt: "The hidden test set is drawn from the corpora below. Do not train on them directly; use the provided train set instead." It lists hypaai/Hypa-Voices, facebook/omnilingual-asr-corpus and AfriSpeech/african-speech-ipa. The train set's per-source licences are "afrispeech_ipa… hypa_voices… omnilingual_asr" — [DSN train repo](https://huggingface.co/datasets/Data-Science-Nigeria/dsn-2026-lid-train) (gated; read with token)
- Language coverage per HF tags:
  - Hypa-Voices: efi, ibb, igl, nup, anw, ego, tiv, urh (the card also names Idoma)
  - omnilingual-asr-corpus: anw, ego, idu only (348 languages)
  - african-speech-ipa: tiv (4.03 h) and urh (25.67 h)

  Sources: [HF API](https://huggingface.co/api/datasets?filter=language:igl), [omnilingual card](https://huggingface.co/datasets/facebook/omnilingual-asr-corpus), [african-speech-ipa card](https://huggingface.co/datasets/AfriSpeech/african-speech-ipa)
- Hypa-Voices data preparation: "Structured AfroVoices recordings, pivot-based translation anchored in English and French, and supplementary open speech material (including Mozilla Common Voice-derived subsets for some languages)". The full pool is 8,442 h; the public subset is 8,800 rows — [Hypa-Voices card](https://huggingface.co/datasets/hypaai/Hypa-Voices)
- Hypa-Speech-10k (no licence tag; the card's licence line is the unfilled placeholder "[Specify license — e.g., CC0 1.0 / CC-BY 4.0…]"): "source text and base audio… drawn from the Mozilla Common Voice Corpus 17". It lists efi, ibb, igl, nup, anw, ego, idu, tiv, urh — [Hypa-Speech-10k card](https://huggingface.co/datasets/hypaai/Hypa-Speech-10k)
- african-speech-ipa is "[AfriSpeech](https://huggingface.co/datasets/AfriSpeech/african-speech-public_v1) audio paired with IPA". african-speech-public_v1 has Tiv at 1,374 clips / 4.03 h and Urhobo at 8,944 clips / 25.67 h, and its card states no upstream source — [african-speech-ipa](https://huggingface.co/datasets/AfriSpeech/african-speech-ipa), [african-speech-public_v1](https://huggingface.co/datasets/AfriSpeech/african-speech-public_v1)
- AfriSpeech/youversion-african-speech (created 2026-06-06, the same day as african-speech-public_v1) has Tiv at 1,982 segments / 4.59 h, and no Urhobo — [youversion-african-speech](https://huggingface.co/datasets/AfriSpeech/youversion-african-speech)
- AfriSpeech/afri2universal (created 2026-10-18, CC-BY-NC-4.0) uses corpus "afripa". Its per-language hours are anw 0.37, ego 0.05, idu 0.03, tiv 3.41, urh 7.92 — [afri2universal language_distribution.tsv](https://huggingface.co/datasets/AfriSpeech/afri2universal)
- Omnilingual mirrors on HF: lindonghello/omnilingual-asr-corpus, KathleenKunLiu/omnilingual-asr-corpus, Sellopale/omnilingualpaleoi, vnahata/OmnilingualASR-retrieval (all CC-BY-4.0, tagged anw/ego/idu). Hypa copies: hypaai/Hypa-Voices-snac, hypaai/Hypa-Speech-10k, Kppwdfgu1/Hypa-Speech-10k, hypaai/Hypa-Voices-ASR-Benchmark, hypaai/Hypa-Voices-Translation-Benchmark (efi) — [HF API language filter](https://huggingface.co/api/datasets?filter=language:anw)
- VoiceAfrica's 11 languages (bcz, ble, bvb, fan, igl, knc, krx, nup, pov, srr, urh) are all **absent** from the omnilingual corpus's 348-language list, even though VoiceAfrica is described as a Lanfrica–Meta collaboration — [omnilingual API metadata](https://huggingface.co/api/datasets/facebook/omnilingual-asr-corpus?full=true), [VoiceAfrica card](https://huggingface.co/datasets/naijavoices/voiceafrica-datasets)
- The omnilingual corpus is "a collection of spontaneous speech recordings… for 348 under-served languages", with prompts like "What was the last thing you ate?" — [omnilingual card](https://huggingface.co/datasets/facebook/omnilingual-asr-corpus)

### Inferences
- **Tiv**: african-speech-public_v1 (4.03 h) closely matches YouVersion Tiv (4.59 h) after filtering, and both were created the same day. That makes it very likely that the prohibited Tiv audio *is* YouVersion (Bible.is/FCBH) Tiv audio. Any Bible.is, YouVersion or FCBH Tiv NT audio should be treated as banned. GRN Tiv programs are different recordings (GRN evangelistic dramas, not the FCBH NT) and are not in african-speech-public_v1, since AfriSpeech's GRN derivative has no Tiv. Risk is therefore low, though GRN and FCBH could share narrators.
- **Urhobo**: the 25.67 h upstream is not YouVersion, not Open.Bible, and not AfriSpeech-GRN (none has Urhobo). It is plausibly FCBH/Bible.is or JW audio. Avoid FCBH, Bible.is and JW Urhobo audio; VoiceAfrica and GRN are the safest Urhobo additions. The 0 "known_speakers_train" for urh in hours_report.csv is consistent with african-speech-ipa being the main source.
- **Ibibio**: Hypa-Voices includes Common Voice-derived material, and Common Voice Ibibio is the only CV language among the nine. The Hypa-Voices ibb test clips could therefore be CV Ibibio clips. Under the rule "anything overlapping them is also banned", CV Ibibio should be avoided unless organisers confirm in writing. If it is used anyway, the only partial safeguard is near-duplicate filtering against the provided ibb train/val clips, which cannot catch hidden test clips.
- **Common Voice in general**: Hypa-Speech-10k shows the Hypa family re-uses CV 17 text and audio for all nine codes. Any "Common Voice" or "AfroVoices" data for these languages is suspect.
- **VoiceAfrica**: there is no direct overlap with the released omnilingual corpus; its language set looks deliberately complementary. Residual risk is that Hypa/AfroVoices sourced from it, which is undocumented. The spontaneous-prompt style is similar to omnilingual, which is useful for domain match.
- **Other mis-labelled data**: AfriSpeech-200-style "accent" sets (Hypa-AfriSpeech, okezieowen/afrispeech_*, intronhealth/afrispeech-dialog) are Nigerian-accented *English*. They must not be labelled as igl/tiv/idu. At most they could serve as "other" data.

### Gaps
- Hypa-Voices' full per-language source breakdown (which languages use CV-derived audio) is not public.
- Whether the DSN organisers consider GRN, VoiceAfrica or the Efik TTS set acceptable is unconfirmed. A quick organiser or forum query would settle it.
- african-speech-public_v1's upstream for Urhobo is undocumented.

## Q5. HF datasets that carry these ISO 639-3 tags: sources and licences

### Takeaway
Of roughly 300 tagged HF datasets, only about 25 contain audio. Most of those are either the prohibited corpora, their mirrors or derivatives, or GRN-derived MMS-ulab. The genuinely new, non-prohibited audio is limited to the sets below.

### Cited findings
Audio datasets by tag, from the [HF API](https://huggingface.co/api/datasets?filter=language:tiv) `filter=language:<code>` queries:
- **All nine codes**:
  - espnet/mms_ulab_v2 and its copies (sundram1996/…, amine-khelif/…) and coml/mmsulab: CC-BY-NC-SA-4.0, GRN source, ungated
  - Data-Science-Nigeria/dsn-2026-lid-train: the competition data
- **Hypa family (prohibited or derived)**:
  - hypaai/Hypa-Voices and -snac: all except idu
  - Hypa-Speech-10k: all except anw
- **AfriSpeech derivatives (prohibited, or text only)**:
  - african-speech-ipa: tiv, urh
  - afri2universal: anw, ego, idu, tiv, urh
  - afri-speech-pause-aligned: text only
- **omnilingual corpus and mirrors**: anw, ego, idu
- **naijavoices/voiceafrica-datasets**: igl, nup, urh
- **scubavoice / williamudo Ibibio–Efik sample**: efi, ibb
- **intronhealth/NigBench-MAMAI-Speech-QA**: nup
- **mrheartng Idoma sets**: idu

Untagged audio sets found by name search ([HF search](https://huggingface.co/api/datasets?search=nupe)):
- offiongbassey/efik_speech_dataset (CC-BY-4.0)
- offiongbassey/efik_audio_dataset
- UmarBaba1/nupe_speech_corpus (CC-BY-NC-4.0, mis-tagged `ha`)
- yacksweb001/nupe-dataset (156 MB, no card, unknown content)
- okezieowen/afrispeech_igala, afrispeech_tiv and afrispeech_idoma (accented English)

No HF audio datasets were found by name for Annang, Eggon or Urhobo beyond those above.

### Inferences
- Recommended priority to request or download now:
  1. offiongbassey/efik_speech_dataset (efi, instant)
  2. VoiceAfrica igl/nup/urh (click-through; also register on NaijaVoices)
  3. NigBench-MAMAI Nupe (request now; manual approval)
  4. MDC NaijaVoices Nupe and/or UmarBaba1 Nupe (dedupe)
  5. GRN zips or MMS-ulab rows for tiv, ibb, efi, anw, nup, urh (music-filter them)
- Everything should be listed in the model card with licences, attribution and NC flags. NaijaVoices defines non-commercial use as "academic, educational, or personal purposes", and a bootcamp competition arguably fits that. Prize money makes it a grey area worth noting.

### Gaps
- yacksweb001/nupe-dataset has no card, so its content and licence are unknown.
- Time to approve gated requests (NigBench, mrheartng) is unknown.
- MDC dataset pages do not render for automated fetchers, so some MDC metadata came from search snippets only.

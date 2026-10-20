# Prohibited corpora provenance: Hypa-Voices, Omnilingual ASR Corpus, AfriSpeech african-speech-ipa (DSN 2026 LID, Codabench 18424)

Research date: 2026-10-20. Method: HF dataset/model cards, HF Hub API, HF datasets-server `/size` endpoint (row counts only), arXiv 2511.09690 HTML, intron.io. No audio was downloaded. "Inference" means my reasoning from cited facts, not a stated fact.

## Q1. What languages, hours and speakers does each corpus contain, and which of the 13 DSN classes does each supply?

### Takeaway
Each prohibited corpus covers a distinct slice of the 12 classes:
- **efi, ibb, igl and nup** exist only in **Hypa-Voices**.
- **ego, anw and idu** are in **Omnilingual**; Hypa-Voices also tags all three, though it labels Idoma `idc`.
- **tiv and urh**, plus **yor, ibo and hau**, are in **AfriSpeech-IPA**; Hypa-Voices tags all five as well.
- **"other"** is fed by all three: English and Pidgin from Hypa, Cameroonian Pidgin and many Nigerian minority languages from AfriSpeech-IPA, and Nigerian minority languages from Omnilingual.

So any external Efik, Ibibio, Igala or Nupe audio that traces back to Hypa/AfroVoices is the highest-risk overlap.

### Cited findings

**DSN training set (the reference point)**
- Per-class hours as train h / val h (train clips / val clips):

  | Class | Hours (train / val) | Clips (train / val) |
  |---|---|---|
  | yor | 9.05 / 0.88 | 3160 / 330 |
  | ibo | 7.96 / 2.08 | 2592 / 704 |
  | hau | 8.31 / 1.42 | 2755 / 495 |
  | ego | 7.32 / 1.05 | 1631 / 222 |
  | anw | 8.12 / 1.83 | 1912 / 387 |
  | efi | 1.42 / 0.15 | 483 / 64 |
  | ibb | 1.76 / 0.28 | 607 / 107 |
  | idu | 7.16 / 2.12 | 1649 / 450 |
  | igl | 0.48 / 0.12 | 170 / 49 |
  | nup | 0.50 / 0.00 | 176 / 0 |
  | tiv | 4.27 / 0.63 | 1474 / 213 |
  | urh | 9.22 / 1.57 | 3122 / 550 |
  | other | 12.98 / 1.85 | 5043 / 781 |

  — [DSN dataset card](https://huggingface.co/datasets/Data-Science-Nigeria/dsn-2026-lid-train)
- DSN processing: clips are 16 kHz mono FLAC of 2–30 s. "Recordings longer than 30 seconds were cut into pieces at quiet points." "other covers English, Nigerian Pidgin, other Nigerian languages and further languages." Exact duplicates were removed, and "splits were rebuilt by speaker". Source tags are `afrispeech_ipa`, `hypa_voices` and `omnilingual_asr`. — [DSN dataset card](https://huggingface.co/datasets/Data-Science-Nigeria/dsn-2026-lid-train)
- The DSN repo is gated (`gated: auto`). It contains `PROHIBITED_SOURCES.txt`, `ATTRIBUTION.md`, `LICENSE.md` and `hours_report.csv`; these files could not be read without login. — [HF API](https://huggingface.co/api/datasets/Data-Science-Nigeria/dsn-2026-lid-train)

**Hypa-Voices (hypaai/Hypa-Voices)**
- The public release is **8,800 paired examples**, a single train split of 22 parquet shards (~1.45 GB). — [Hypa-Voices card](https://huggingface.co/datasets/hypaai/Hypa-Voices); [datasets-server size](https://datasets-server.huggingface.co/size?dataset=hypaai/Hypa-Voices)
- It is a "curated public subset" of a full pool of "roughly **2.98 million records, 8,442 hours, and 22 source languages**". — [Hypa-Voices card](https://huggingface.co/datasets/hypaai/Hypa-Voices)
- Language tags: en, fr, es, pt, ar, sw, **pcm, yo, ha, ig, ibb, tiv, efi, anw, idc, igl, nup, ego**, fon, **urh**, tw, ibr. The prose names "Ibibio, Tiv, Efik, Yoruba, Hausa, Igbo, Annang, Idoma, Igala, Nupe, Eggon, Fongbe, Urhobo, Twi, and Ebira" plus English, French, Spanish, Portuguese, Arabic, Swahili and Pidgin. — [Hypa-Voices card](https://huggingface.co/datasets/hypaai/Hypa-Voices)
- Fields: `text`, `src_lang`, `tgt_lang`, `duration_seconds`, `mode` (`transcribe` / `translate`) and `speaker`. In translate records, "speech is in `src_lang` and text is in `tgt_lang`". — [Hypa-Voices card](https://huggingface.co/datasets/hypaai/Hypa-Voices)
- Speakers are anonymised as `SPK-001`, `SPK-002`, … and "the same ID refers to the same voice across records". The card's example IDs go up to at least SPK-111. No total speaker count is published. — [Hypa-Voices card](https://huggingface.co/datasets/hypaai/Hypa-Voices)

**Omnilingual ASR Corpus (facebook/omnilingual-asr-corpus)**
- 348 languages, 547,546 rows, ~983 GB of parquet. Each language config has train/dev/test splits. — [card](https://huggingface.co/datasets/facebook/omnilingual-asr-corpus); [datasets-server size](https://datasets-server.huggingface.co/size?dataset=facebook/omnilingual-asr-corpus)
- The paper puts the training split at **3,350 h across 348 languages**. — [arXiv 2511.09690, Table 3](https://arxiv.org/html/2511.09690)
- Target-class configs, rows (parquet size): **ego_Latn 456 (1.06 GB), anw_Latn 603 (1.58 GB), idu_Latn 535 (1.22 GB)**. There are **no** configs for yor, ibo, hau, efi, ibb, igl, nup, tiv, urh, pcm or eng. — [datasets-server size](https://datasets-server.huggingface.co/size?dataset=facebook/omnilingual-asr-corpus); [HF API file list](https://huggingface.co/api/datasets/facebook/omnilingual-asr-corpus)
- Nigerian "other"-class configs, rows (parquet size): ijc_Latn Izon 466 (1.53 GB), ijn_Latn Kalabari 535 (1.65 GB), ikw_Latn Ikwere 504 (1.50 GB), ish_Latn Esan 481 (1.39 GB), iso_Latn Isoko 503 (1.50 GB), ogo_Latn Khana 509 (1.49 GB), fuv_Latn Nigerian Fulfulde 496 (0.66 GB), bcs_Latn Kohumono 486 (1.50 GB). bbu_Latn is also present and the paper's language table names it "Kulung (Nigeria)". — [datasets-server size](https://datasets-server.huggingface.co/size?dataset=facebook/omnilingual-asr-corpus); [arXiv 2511.09690 language table](https://arxiv.org/html/2511.09690)
- Speakers: the collection targeted "10 native speakers per language" at about 1 hour per speaker, and some languages fell short. — [arXiv 2511.09690 §3 (via fetch)](https://arxiv.org/html/2511.09690)
- The schema has `speaker_id` (e.g. "spk02"), `segment_id` and `prompt_id`. — [card](https://huggingface.co/datasets/facebook/omnilingual-asr-corpus)

**AfriSpeech african-speech-ipa (AfriSpeech/african-speech-ipa)**
- 141 language configs and 775,799 rows (~33.8 GB). Splits are "AfriSpeech's own, which are speaker-disjoint". Clips run 0.4–39 s. — [card](https://huggingface.co/datasets/AfriSpeech/african-speech-ipa); [datasets-server size](https://datasets-server.huggingface.co/size?dataset=AfriSpeech/african-speech-ipa)
- Target-class hours (clips): **yoruba_yor 14.19 h (4,945); igbo_ibo 38.07 h (12,619); hausa_hau 28.37 h (9,350); tiv_tiv 4.03 h (1,374); urhobo_urh 25.67 h (8,866)**. There is **no** efi, ibb, igl, nup, anw, ego or idu. — [card per-language table](https://huggingface.co/datasets/AfriSpeech/african-speech-ipa)
- Nigerian/"other"-relevant configs, hours (clips):

  | Config | Hours | Clips |
  |---|---|---|
  | edo_bin | 26.04 | 8,995 |
  | esan_ish | 24.19 | 8,474 |
  | isoko_iso | 26.84 | 9,474 |
  | itsekiri_its | 6.78 | 2,384 |
  | gokana_gkn | 7.98 | 2,801 |
  | khana_ogo | 3.62 | 1,210 |
  | ijaw_ijc | 0.88 | 306 |
  | pidgin_west_africa_wes | 21.09 | 7,394 |
  | liberian_english_lir | 22.06 | 7,715 |
  | krio_kri | 26.16 | 9,218 |

  `pidgin_west_africa_wes` is Cameroonian Pidgin, region CM. — [card](https://huggingface.co/datasets/AfriSpeech/african-speech-ipa); [public_v1 card](https://huggingface.co/datasets/AfriSpeech/african-speech-public_v1)
- No speaker counts are published for any language. — [card](https://huggingface.co/datasets/AfriSpeech/african-speech-ipa)

### Inferences
- **efi/ibb/igl/nup.** Neither Omnilingual nor AfriSpeech-IPA has these, so DSN's efi (547 clips), ibb (714), igl (219) and nup (176) almost certainly come entirely from Hypa-Voices. The hidden-test clips for these classes must come from Hypa-Voices too, either the public 8,800 subset or possibly non-public Hypa data (unknown). Only one or a few Hypa speakers per language may exist, which raises speaker leakage risk if any Hypa-derived audio is used.
- **ego/anw/idu.** Omnilingual has ~456–603 long recordings per language (avg ~2–2.6 MB FLAC per row, likely minutes long). Cutting them into ≤30 s chunks plausibly yields DSN's ~1,850–2,300 clips and ~8–9 h, so these classes are probably dominated by Omnilingual's ~10 speakers each, with some Hypa audio.
- **tiv.** DSN tiv totals ~4.9 h, more than AfriSpeech-IPA's 4.03 h, which implies all of AfriSpeech-IPA tiv plus some Hypa tiv. For urh/yor/ibo/hau, DSN used only a fraction of AfriSpeech-IPA hours.
- **Tag mismatches.** Hypa-Voices tags Idoma as `idc`; the ISO 639-3 code for Idoma is `idu`. To my knowledge `idc` is a different language (Idon), but that is unverified. Ebira is tagged `ibr`; ISO Ebira is `igb`. Anyone filtering Hypa-derived data by ISO code could miss Idoma/Ebira overlap.
- **Language mismatches.** Translate-mode Hypa records pair speech in one language with text in another. Any third-party dataset built from Hypa text fields could carry the wrong language label for the audio.

### Gaps
- Speaker counts per language for Hypa-Voices and AfriSpeech-IPA are not published. Omnilingual per-language hours and speaker counts are also not on the card; only row counts and bytes are.
- Hypa-Voices per-language record counts are not on the card. The parquet `src_lang` column would answer this, but I did not download data.
- How DSN sampled the hidden test set (which source splits or speakers) is undisclosed. `PROHIBITED_SOURCES.txt` and `hours_report.csv` are behind the HF gate.

## Q2. How was each corpus collected (protocol, text source, device), and by whom?

### Takeaway
The three corpora were collected in very different ways:
- **Omnilingual**: commissioned **spontaneous** speech (answers to open prompts) gathered by unnamed vendors and local organisations. No Nigerian partner is named for the corpus itself.
- **Hypa-Voices**: AfroVoices (a Hypa AI subsidiary) recordings with English/French pivot translations, plus Common Voice-derived material.
- **AfriSpeech-IPA**: a G2P-phonemised repackaging of `AfriSpeech/african-speech-public_v1`. The upstream source is undocumented but very likely **religious read narration** (strong inference, possibly JW.org publication audio).

### Cited findings

**Omnilingual ASR Corpus**
- "a collection of **spontaneous speech** recordings and their transcriptions for 348 under-served languages … for the purposes of training automatic speech recognition (ASR) and **spoken language identification** models." — [card](https://huggingface.co/datasets/facebook/omnilingual-asr-corpus)
- Protocol:
  - There was a pool of >1,500 prompts in English plus six pivot languages (French, Indonesian, Italian, Mandarin, Portuguese, Spanish).
  - Prompts were read aloud or given in a second language, and answered in the native language "naturally and at a normal pace".
  - Recording guidelines asked for a quiet environment and no touching or blowing into the microphone.
  - Transcripts were produced by speakers, on-site typists, or by digitising handwritten notes.
  - Example prompt: "What was the last thing you ate? Can you describe how it is made?"

  — [arXiv 2511.09690 §3 and App. C](https://arxiv.org/html/2511.09690); [card](https://huggingface.co/datasets/facebook/omnilingual-asr-corpus)
- Collectors are described only as "vendors" and "local organizations" that "recruited and compensated native speakers". For QA, a different vendor cross-checked a sample per language. Recording devices are not specified. — [arXiv 2511.09690](https://arxiv.org/html/2511.09690)
- Transcripts keep disfluencies and tags `<laugh>`, `<hesitation>`, `<unintelligible>` and `<noise>`. — [card](https://huggingface.co/datasets/facebook/omnilingual-asr-corpus)

**Hypa-Voices**
- Built by Hypa Intelligence and AfroVoices: "AfroVoices, a subsidiary of Hypa AI". — [card](https://huggingface.co/datasets/hypaai/Hypa-Voices)
- Collection: "Structured AfroVoices recordings, pivot-based translation anchored in English and French, and supplementary open speech material (including **Mozilla Common Voice-derived subsets** for some languages)." — [card](https://huggingface.co/datasets/hypaai/Hypa-Voices)
- Encoding: the full corpus is stored as SNAC tokens, and the public FLAC was "decoded from SNAC back to FLAC". The public audio has therefore passed through a 24 kHz neural codec, which may leave codec artefacts (inference). — [card](https://huggingface.co/datasets/hypaai/Hypa-Voices)
- Example content (Igbo): "Onye nlekọta na-enwe nsogbu nọ n'okpuru nrụgide imelite ọzụzụ." — [card](https://huggingface.co/datasets/hypaai/Hypa-Voices)
- Example translate record: English speech paired with Efik text about "Hope Medrano Elementary School". This suggests translated general/web-domain sentences rather than religious text (inference). — [card](https://huggingface.co/datasets/hypaai/Hypa-Voices)

**AfriSpeech-IPA**
- "AfriSpeech audio paired with IPA phoneme transcriptions, for 141 languages." The audio is from `AfriSpeech/african-speech-public_v1`; the IPA is produced with `africa-g2p` 0.2.0. The build code is GitHub `AfriSpeech/afrispeech-ipa-trainer`. — [card](https://huggingface.co/datasets/AfriSpeech/african-speech-ipa)
- Upstream `african-speech-public_v1` has 142 languages, 778,520 clips and **2,267.9 h**. Fields are `audio`, `text`, `duration` and `iso639_3`. The card has no provenance, recording-protocol or licence prose. — [public_v1 card](https://huggingface.co/datasets/AfriSpeech/african-speech-public_v1); [HF API](https://huggingface.co/api/datasets/AfriSpeech/african-speech-public_v1)
- The DSN card marks `afrispeech_ipa` as "CC-BY-4.0 (stated; upstream audio terms not verified)". It warns that "read recordings dominate some sources" and that "audio from some sources may be **read religious text**". — [DSN card](https://huggingface.co/datasets/Data-Science-Nigeria/dsn-2026-lid-train)
- The `AfriSpeech` HF org ("Speech processing for Africa") has **one member, Mich-Seth Owusu (michsethowusu)**. Its other speech releases are religious-domain scrapes:
  - GRN audio
  - YouVersion Bible audio
  - an Open.Bible / Faith Comes By Hearing mirror
  - "publicly accessible audio in the religious domain" (africa-female-speech)

  Its LID model was trained on "GRN … YouVersion Bible audio … **JW publication audio**". — [org members API](https://huggingface.co/api/organizations/AfriSpeech/members); [org datasets](https://huggingface.co/api/datasets?author=AfriSpeech); [african-speech-id card](https://huggingface.co/AfriSpeech/african-speech-id)

### Inferences
- **public_v1 is probably JW.org audio (strong inference, not confirmed).** Several language labels match JW.org naming exactly: "Pidgin (West Africa)" for Cameroonian Pidgin `wes`, "Chitonga (Malawi)", "Kikongo ya Leta", "Swahili (Congo)", "Makhuwa-Marrevone", "Ngbandi (Northern)". The language set (Isoko, Urhobo, Edo, Esan, Gokana, Khana, Kabuverdianu, Mauritian Creole) also matches JW.org's publication languages. Together with ~10 s average clip length and the org's JW-audio pipeline, this points to studio-narrated JW publications or Bible readings. If true, **any other JW.org-derived Nigerian audio could overlap** with the yor/ibo/hau/tiv/urh and Nigerian "other" test clips.
- **Domain cue.** AfriSpeech-IPA test clips are likely clean, single-narrator read speech. Omnilingual clips are spontaneous and on unknown devices. Hypa clips are codec-decoded read/translated sentences. A classifier may learn source/channel cues; this matters for validation but is not a provenance issue.

### Gaps
- No primary statement of where `african-speech-public_v1` audio comes from. The GitHub repos `AfriSpeech/afrispeech-ipa-trainer` and `AfriSpeech/africa-g2p` could not be read: this session's GitHub API is restricted to configured repos.
- No named Nigerian vendor or partner for the Omnilingual ASR Corpus itself. The paper names partners only for *other* training data (see Q5).
- Recording devices (phone vs studio) are not documented for any of the three.

## Q3. Sibling or overlapping datasets from the same organisations, and overlap risk

### Takeaway
The highest-risk siblings fall into three groups:
1. **Direct repackagings**: Hypa-Voices-snac, the Hypa-Voices benchmarks, `AfriSpeech/african-speech-public_v1`, and Omnilingual mirrors and subsets.
2. **Same-pipeline siblings**: `hypaai/Hypa-Speech-10k`, which covers all 12 target codes plus pcm, and AfriSpeech religious-audio sets for Hausa, Igbo, Yoruba, Tiv and Annang.
3. **Same-source data**: Common Voice for Hypa, and JW/Bible audio for AfriSpeech-IPA.

Treat groups 1 and 2 as **avoid**, and group 3 as **ask organisers first**.

### Cited findings

**Hypa-Voices family (hypaai)**
- `hypaai/Hypa-Voices-snac` has the "same curated subset and metadata" as Hypa-Voices, stored as SNAC tokens. Decoding the tokens reproduces the audio. — [Hypa-Voices card](https://huggingface.co/datasets/hypaai/Hypa-Voices)
- `hypaai/Hypa-Voices-ASR-Benchmark` is "a compact, 20-sample benchmark drawn from the larger hypaai/Hypa-Voices corpus". Its tags are en, fr, es, ig, yo, ha, ti, ib, ann, efi, ful. — [HF API](https://huggingface.co/api/datasets/hypaai/Hypa-Voices-ASR-Benchmark)
- `hypaai/Hypa-Voices-Translation-Benchmark` and `hypaai/Hypa-Voices-TTS-Benchmark` were created 2026-08-25 and 2026-10-19. The TTS benchmark is text-only ("1,010 sentences … Do not train on it"). — [HF API list](https://huggingface.co/api/datasets?author=hypaai); [TTS benchmark API](https://huggingface.co/api/datasets/hypaai/Hypa-Voices-TTS-Benchmark)
- `hypaai/Hypa-Speech-10k` (2026-06-12) has "10,000 audio-text pairs spanning 18 languages". Its language tags are en, fr, es, ar, sw, ha, ig, yo, **ann, efi, ego, ibb, idu, igl, nup, pcm, tiv, urh**. "The source text and base audio for this dataset were drawn from the Mozilla Common…" (card truncated in the API). It covers translation, transcription and **language detection**. — [HF API](https://huggingface.co/api/datasets/hypaai/Hypa-Speech-10k)
- Other hypaai datasets: Hypa_Fleurs and Hypa_Fleurs_v2 (FLEURS English with Igbo, Yoruba and Hausa translations), Hypa-LibreSpeech, Hypa-Text-10k, Hypa-Keyboard-*, and Hypa-AfriSpeech. — [HF API](https://huggingface.co/api/datasets?author=hypaai); [Hypa_Fleurs_v2](https://huggingface.co/datasets/hypaai/Hypa_Fleurs_v2)
- `hypaai/Hypa-AfriSpeech` and `okezieowen/afrispeech` (both 2025-10-12) have per-accent shards (afrikaans, ebira, idoma, …). — [HF API](https://huggingface.co/api/datasets/hypaai/Hypa-AfriSpeech); [HF API](https://huggingface.co/api/datasets/okezieowen/afrispeech)
- HF dataset search for "hypa-voices" returns only the 5 hypaai repos; no third-party mirrors were found. — [HF search](https://huggingface.co/api/datasets?search=hypa-voices)

**Omnilingual family**
- Full mirrors with 1,473 identical files: `lindonghello/omnilingual-asr-corpus` (2026-04-25) and `KathleenKunLiu/omnilingual-asr-corpus` (2026-06-23). — [HF API](https://huggingface.co/api/datasets/lindonghello/omnilingual-asr-corpus); [HF API](https://huggingface.co/api/datasets/KathleenKunLiu/omnilingual-asr-corpus)
- Subsets and derivatives:
  - `evie-8/omnilingual` ("a subset of the original Omnilingual ASR Corpus"; cgg/koo shown)
  - `14kwonss/omnilingual_slid` (name suggests a spoken-LID derivative; content not inspected)
  - `34data/v13-omnilingual-asr-corpus-b/-c`
  - `guruawe/omnilingual-finetune-data` (17,238 files)
  - `Chiz/omniASR-igbo-blindspots`, `vysakh25/fb-omnilingual-arabic-khaleeji` and `jbdc1/omnilingual-asr-en-vi-lo`

  — [HF search](https://huggingface.co/api/datasets?search=omnilingual); [evie-8 API](https://huggingface.co/api/datasets/evie-8/omnilingual)

**AfriSpeech (michsethowusu) family**
- `AfriSpeech/african-speech-public_v1` is the direct upstream of AfriSpeech-IPA and must be avoided. — [IPA card](https://huggingface.co/datasets/AfriSpeech/african-speech-ipa)
- `AfriSpeech/youversion-african-speech` includes Hausa 3.13 h, Igbo 4.18 h, **Tiv 4.59 h** and Yoruba 3.64 h of YouVersion Bible audio. — [card](https://huggingface.co/datasets/AfriSpeech/youversion-african-speech)
- `AfriSpeech/open-bible-speech-african` is the African subset of `davidguzmanr/open-bible-resources` (Open.Bible / Faith Comes By Hearing). It has Hausa 99.7 h, Igbo 94.2 h and Yoruba 89.9 h; "audio and text are unchanged from the source". — [card](https://huggingface.co/datasets/AfriSpeech/open-bible-speech-african)
- `AfriSpeech/grn-african-speech` has 3,077.9 h of Global Recordings Network clips, 15 s each. It includes **Annang 0.44 h**, Hausa 10.55 h, Yoruba 4.28 h, "English: Nigeria" 0.47 h and Hausa dialects. — [card](https://huggingface.co/datasets/AfriSpeech/grn-african-speech)
- `AfriSpeech/africa-female-speech` (and v2, v2-best60) is "extracted from publicly accessible audio in the religious domain". It has Hausa 18.59 h, Igbo 15.52 h and Yoruba 13.62 h. — [card](https://huggingface.co/datasets/AfriSpeech/africa-female-speech)
- `AfriSpeech/african-transcribed-speech-monolingual` is text-only, from LDS General Conference talks, and includes **Efik 38,437 sentences**, Yoruba and Igbo. — [card](https://huggingface.co/datasets/AfriSpeech/african-transcribed-speech-monolingual)
- Other sets not inspected: `african-corpus-jw`, `afri-speech-pause-aligned` (2026-10-18), `multivoice-synthetic-speech`, `afri2universal`, `africa-corpus`. — [org datasets](https://huggingface.co/api/datasets?author=AfriSpeech)

**Meta MMS**
- MMS is described as "primarily restricted to the religious domain". MMS-lab is one of the AllASR training sources alongside the Omnilingual ASR Corpus. — [arXiv 2511.09690](https://arxiv.org/html/2511.09690)

### Inferences
- **Hypa-Speech-10k is the single most dangerous sibling.** It comes from the same organisation, has the same 12 target codes plus pcm, includes "language detection" tasks, and shares Common Voice-derived base audio with Hypa-Voices. Speaker and recording overlap with Hypa-Voices, and so with the efi/ibb/igl/nup test clips, is plausible. **Avoid.**
- `Hypa-Voices-snac` and the Hypa benchmarks are verbatim subsets. **Avoid.**
- **Common Voice risk.** Hypa-Voices includes "Mozilla Common Voice-derived subsets for some languages", so Common Voice clips (e.g. Yoruba, Igbo, Hausa, Pidgin, and any CV Efik/Ibibio/Igala/Nupe/Tiv/Urhobo) could be the very recordings in Hypa-Voices. **Ask organisers first** before using Common Voice for target languages.
- **Hypa-AfriSpeech / okezieowen/afrispeech*.** Their accent-named shards match AfriSpeech-200 accent configs, so they are probably repackagings of `intronhealth/afrispeech-200` (accented English) rather than the prohibited AfriSpeech-IPA. They come from Hypa, though, and might have been folded into Hypa-Voices' English portion (unverified). Ask first.
- **AfriSpeech religious siblings.** youversion, open-bible, africa-female-speech, african-corpus-jw and GRN come from the same org and pipeline as the prohibited set, and are religious narration. If public_v1 is JW-sourced (Q2 inference), then:
  - JW-derived sets are **avoid**;
  - YouVersion, Open.Bible and GRN are different upstream publishers but the same scraper and domain, so **ask organisers first**.

  GRN Annang is especially sensitive because anw is a target class.
- **MMS-lab** (Bible.is / Faith Comes By Hearing New Testament readings) is a different source from JW publications. Recording overlap with AfriSpeech-IPA is unlikely but not ruled out. It does **not** overlap with Omnilingual or Hypa.

### Gaps
- I could not confirm that Hypa-Speech-10k and Hypa-Voices share speakers; that would need audio or speaker-ID comparison.
- Content of `14kwonss/omnilingual_slid`, `african-corpus-jw` and `afri-speech-pause-aligned` was not inspected.
- Whether any Omnilingual mirror differs from the original (e.g. added splits) is unknown; the file counts are identical.

## Intron / AfriSpeech family

### Takeaway
The **name collides, but the publisher does not appear to be the same**. The prohibited `AfriSpeech/african-speech-ipa` sits in an HF org whose only member is Mich-Seth Owusu. Intron Health's speech data lives under `intronhealth` (Tobi Olatunji et al.), was collected through different pipelines (crowd-sourced accented-English reading, in-the-wild conversation), and is mostly English. I found **no evidence** of shared recordings or pipeline with african-speech-ipa.

Verdicts:
- Intron's English sets are **safe w.r.t. overlap** but CC-BY-NC-SA and gated.
- AfriSwitch and AfriVox-v2 are **ask organisers first**, because of Common Voice / BibleTTS / NaijaVoices sub-sources and code-switching.
- Sahara v2/v2.5 is **avoid**: a closed API with undisclosed training data, unusable as a verifiable backbone.

### Cited findings

**Organisation and benchmark**
- The `intronhealth` HF org ("Intron Innovation", "NLP, ASR") has 5 members: Gabrial Zencha, Tobi Olatunji, Ezinwanne Aka, Aïmérou Ndiaye and Busayo Awobade. — [intronhealth members API](https://huggingface.co/api/organizations/intronhealth/members); [org overview](https://huggingface.co/api/organizations/intronhealth/overview)
- The `AfriSpeech` HF org hosting the prohibited set has 1 member, Mich-Seth Owusu. — [AfriSpeech members API](https://huggingface.co/api/organizations/AfriSpeech/members)
- The intron.io benchmark lists Methodology: "IJCNLP-AACL 2025 AfriSpeech MultiBench Paper • Evaluation datasets: African Code-Switching Dataset • Accented English • African Languages". It covers code-switched ASR, monolingual African-language ASR, TTS (MOS) and accented-English ASR for **Sahara v2.5**. — [intron.io](https://www.intron.io)
- AfriSpeech-MultiBench (Ashungafac, Sanni, Awobade, Gichamba, Olatunji; Intron Health; IJCNLP-AACL 2025) covers 100+ African English accents across 10+ countries and 7 domains. It is released on HF under CC BY-NC-SA 4.0. — [arXiv 2511.14255](https://arxiv.org/abs/2511.14255); [ACL Anthology](https://preview.aclanthology.org/setup/2025.ijcnlp-long.190)

**Intron datasets**

| Dataset | Content | Licence / access | Verdict |
|---|---|---|---|
| `intronhealth/afrispeech-200` | 200 h of clinical and general-domain **English** accented speech, 120 accents, 13 countries, 2,463 speakers. Accent configs include Nigerian ones such as igala, idoma, tiv, ebira. | cc-by-nc-sa-4.0, gated manual | **Safe w.r.t. overlap** (English, "other" class). Accent labels are not language labels. Ask if commercial licence matters. |
| `intronhealth/afrispeech-dialog` | 6 h of conversational dialogue, accents from Nigeria, Kenya and South Africa. | cc-by-nc-sa-4.0, not gated | **Safe** |
| `intronhealth/afrispeech-countries` | 21,581 clips / 67.73 h of African-accented English, 7 regions, read and conversational. | cc-by-nc-sa-4.0 | **Safe** |
| `intronhealth/afrispeech-parliament` | 35.86 h of parliamentary YouTube sessions from Nigeria, Ghana, South Africa and Kenya. | cc-by-nc-sa-4.0 | **Safe** |
| `intronhealth/med-convo-nig` | ~4.2 h of Nigerian-accented English simulated consultations. | cc-by-nc-sa-4.0 | **Safe** |
| `intronhealth/AfriSwitch` (likely the "African Code-Switching Dataset"; inference) | 54.41 h, test-only, in-the-wild code-switched speech in 14 languages including **Pidgin, Yoruba, Hausa, Igbo**, each switching with English. | cc-by-nc-sa-4.0, gated manual | **Ask organisers first.** No overlap evidence, but it is mixed-language and an evaluation-only benchmark. |
| `intronhealth/AfriSwitchCare` | 12.11 h of code-switched clinical conversations in 9 languages including Pidgin, Hausa, Igbo, Yoruba. | cc-by-nc-sa-4.0 | **Ask organisers first** |
| `intronhealth/afrivox-v2` (likely the "African Languages" benchmark; inference) | 24 languages including **pcm 12,546, yor 10,911, ibo 7,229, hau** utterances. Sources: Africa Next Voices (~100 h), Waxal (~69.5 h) and AfriVox v1, which aggregates "NCHLT, **Common Voice**, FLEURS, OpenSLR, **BibleTTS**, **NaijaVoices**, FISD" plus AfriVox-Medical. | cc-by-4.0, gated manual | **Ask organisers first** (see inferences) |
| `intronhealth/nig-bench` | Health QA collected in Nigeria in English, Hausa, Igbo, Pidgin, Yoruba. | cc-by-nc-sa-4.0 | **Safe w.r.t. overlap** but NC |
| `intronhealth/afrivox` (v1) | 18 languages | gated manual | **Ask organisers first** (same reasoning as afrivox-v2) |
| `intronhealth/afrivox_in_the_wild` | No card text | — | **Ask organisers first** |

— [afrispeech-200](https://huggingface.co/datasets/intronhealth/afrispeech-200); [afrispeech-dialog](https://huggingface.co/datasets/intronhealth/afrispeech-dialog); [afrispeech-countries](https://huggingface.co/datasets/intronhealth/afrispeech-countries); [afrispeech-parliament](https://huggingface.co/datasets/intronhealth/afrispeech-parliament); [med-convo-nig](https://huggingface.co/datasets/intronhealth/med-convo-nig); [AfriSwitch](https://huggingface.co/datasets/intronhealth/AfriSwitch); [AfriSwitchCare](https://huggingface.co/datasets/intronhealth/AfriSwitchCare); [afrivox-v2](https://huggingface.co/datasets/intronhealth/afrivox-v2); [nig-bench](https://huggingface.co/datasets/intronhealth/nig-bench); [afrivox](https://huggingface.co/datasets/intronhealth/afrivox); [HF API](https://huggingface.co/api/datasets?author=intronhealth)

**Intron models and Sahara**
- Intron's public HF models are `intronhealth/afrispeech-whisper-medium-all`, `intronhealth/Seyfelislem-afrispeech-whisper-large-v2-A100` and `intronhealth/afro-tts`, all from 2024. Models tagged with `intronhealth/afrispeech-200` include `Professor/whisper-small-afrispeech`, `Professor/whisper-medium-afrispeech`, `ogunlao/SBPN_multilingual_*`, `DivineAngel317/whisper-small-nigerian-accented-english` and others. — [intronhealth models](https://huggingface.co/api/models?author=intronhealth); [model filter](https://huggingface.co/api/models?filter=dataset:intronhealth/afrispeech-200)
- **Sahara v2 / v2.5** are Intron's ASR product (API). Reported training: ~50,000 h from 40,000+ speakers, 57 languages (v2), and 63 with v2.5. The data was described as "busy Nigerian clinics, Kenyan call centres, and South African courtrooms". These claims come from the company's own announcements. — [WeeTracker](https://weetracker.com/2026/03/06/intron-sahara-v2-african-voice-ai-big-tech/); [intron.io](https://www.intron.io); [intron.io/sahara](https://www.intron.io/sahara)

### Inferences
- **Mapping to intron.io names.** "Accented English" likely corresponds to the AfriSpeech-MultiBench suite (afrispeech-200/countries/dialog/parliament/med-convo-nig). "African Code-Switching Dataset" likely corresponds to AfriSwitch. "African Languages" likely corresponds to AfriVox-v2. The site gives no HF ids, so this mapping is inferred.
- **No shared pipeline with african-speech-ipa.** Intron data is crowd-sourced or in-the-wild with Intron's own transcription. african-speech-ipa is scraped religious narration with rule-based G2P. I found no evidence of shared speakers, recordings or pipeline. The "AfriSpeech" brand overlap, however, could lead organisers to treat Intron sets as related. Disclose use of any "AfriSpeech"-named set, or ask first.
- **Why AfriVox v1/v2 is "ask first".** BibleTTS (Open.Bible-derived Yoruba/Hausa read Bible) and Common Voice (a Hypa-Voices sub-source) sit inside it. NaijaVoices (Igala, Nupe, Urhobo per the Omnilingual paper) is not prohibited but is a target-language source.
- **Sahara.** The weights are not public, the training corpus is undisclosed and may include anything. Using API transcripts or labels as a teacher is unverifiable and probably outside competition rules. **Avoid.**

### Gaps
- I could not read Intron's dataset READMEs fully (gated). The NaijaVoices / BibleTTS share in AfriVox per language is unknown.
- I found no statement by Intron or by the AfriSpeech (michsethowusu) org about any relationship between them.

## Q4. Which public pretrained speech models were trained on the three corpora (avoid as backbones or teachers)?

### Takeaway
**Avoid the whole Omnilingual ASR family**: omniASR W2V, CTC and LLM at 300M/1B/3B/7B, LLM-7B-ZS, the v2 builds, and all mirrors, conversions, ONNX/sherpa exports and fine-tunes. They are trained on the prohibited corpus, which was built "for … spoken language identification".

Also avoid:
- **AfriSpeech/african-speech-id**: uses an omniASR recogniser, and its head was trained on religious audio including JW publications.
- **The Hypa-Orpheus family**: trained on the full Hypa-Voices.
- **Hypa ASR/LLM speech models**, whose training data is undisclosed but very likely Hypa-Voices.

No HF model lists `AfriSpeech/african-speech-ipa` or `african-speech-public_v1` as training data.

### Cited findings

**Omnilingual ASR family**
- The HF model filter `dataset:facebook/omnilingual-asr-corpus` returns 59 models:
  - official `facebook/omniASR-W2V-300M/1B/3B/7B`, `facebook/omniASR-CTC-300M/1B/3B/7B`, `facebook/omniASR-LLM-300M/1B/3B/7B`, `facebook/omniASR-LLM-7B-ZS`
  - mirrors and conversions: `bezzam/*`, `ylacombe/omniASR_W2V_*_SSL`, `Steveeeeeeen/*`, `gclose19/omnilingual-asr-ssl-hf`, `indicnode/*`, `ziywang50/*` (incl. `-hf`), `giangndm/*`, `OpenVoiceOS/omnilingual-asr-ctc-300m-onnx` and `-1b-onnx`, `ChipCracker/omniASR-CTC-300M-pruned-21L`, `RadeAI/Rade-ASR-CTC-3B-fa`, `omarSorour123/omniASR-LLM-7B`
  - other tagged models: `SWivid/Habibi-TTS`, `Nahim321/ni-01`, `AnthropicallyHuggingFaces/Owl`, `sarh21/whisper-small-ar`

  — [HF model filter](https://huggingface.co/api/models?filter=dataset:facebook/omnilingual-asr-corpus)
- An HF model search for "omniASR" returns ≥100 repos, including `ilhanemirhan/omniASR_*`, `hr16/omniASR_CTC_3B_fp16` and `EmreAkgul/omniASR-LLM-3B-v2-ONNX`. — [HF model search](https://huggingface.co/api/models?search=omniASR&limit=100)
- The Omnilingual ASR Corpus is a component of AllASR, "used to train our final models". The SSL encoders (W2V 300M–7B) are pretrained on 4.3M h, and the CTC and LLM-ASR models are fine-tuned from them. — [arXiv 2511.09690](https://arxiv.org/html/2511.09690)
- A "February 2026 v2" omniASR build exists. — [african-speech-id card](https://huggingface.co/AfriSpeech/african-speech-id)

**AfriSpeech models**
- `AfriSpeech/african-speech-id` is "language identification for 1,386 African languages". It is an omniASR CTC recogniser, "Meta's Omnilingual ASR … mirrored unmodified", plus a char n-gram head. The head was trained on GRN (235,212 clips), "YouVersion Bible audio" (76,202) and "JW publication audio" (32,054). Its card evaluates on the "omniASR corpus (test split)". — [card](https://huggingface.co/AfriSpeech/african-speech-id)
- `AfriSpeech/afrispeech-gender-id` was trained on google/WaxalNLP (CAM++ embeddings plus an MLP head). It is not trained on the prohibited sets. — [card](https://huggingface.co/AfriSpeech/afrispeech-gender-id)

**Hypa models**
- `hypaai/Hypa-Orpheus-3b-TTS-VC` was fine-tuned from `unsloth/orpheus-3b-0.1-ft` "on the full **Hypa-Voices** corpus": 3.73M examples, 8,442 h, 22 languages, SNAC at 24 kHz. Models tagged `dataset:hypaai/Hypa-Voices-snac`:
  - `hypaai/Hypa-Orpheus-3b-TTS-VC`
  - `hypaai/Hypa-Orpheus-3b-TTS-VC-LoRAs`
  - `hypaai/Hypa-Orpheus-3b-TTS-VC-ext`
  - `hypaai/Hypa-Orpheus-3b-TTS-VC-LoRAs-ext`
  - `mradermacher/Hypa-Orpheus-3b-TTS-VC-GGUF`

  — [Hypa-Orpheus card](https://huggingface.co/hypaai/Hypa-Orpheus-3b-TTS-VC); [HF model filter](https://huggingface.co/api/models?filter=dataset:hypaai/Hypa-Voices-snac)
- The card describes a roadmap "User Speech → Hypa-Whisper → Hypa-Llama → … → Hypa-Orpheus". — [Hypa-Orpheus card](https://huggingface.co/hypaai/Hypa-Orpheus-3b-TTS-VC)
- Hypa speech models whose cards do not state training data:
  - `hypaai/Hypa-Whisper-Small-v2` (from `hypaai/wspr_small_2025-11-11_12-12-17`, a whisper-small fine-tune tagged ig, yo, en, ha)
  - `hypaai/Hypa-Llama-SNAC-ASR-run1-runpod-256(-16bit)`
  - `hypaai/Hypa-Whispering-Llama-3.1-8B`
  - `hypaai/hypaai_orpheus_v5` and `hypaai/Hypa-Orpheus-Step-28821-16bit`
  - third-party `okezieowen/hypaai_orpheus`, `okezieowen/hypa_orpheus_v2`, `okezieowen/Hypa-Whisper-small-*`, `kumrandy123/hypa_ai_orpheus`, `Danieljava/Hypa-Whisper-small-2026-03-31-translate-check`

  — [hypaai models](https://huggingface.co/api/models?author=hypaai); [Hypa-Whisper-Small-v2](https://huggingface.co/hypaai/Hypa-Whisper-Small-v2); [wspr_small](https://huggingface.co/hypaai/wspr_small_2025-11-11_12-12-17); [HF model search](https://huggingface.co/api/models?search=hypa&limit=100)

**AfriSpeech-IPA models**
- No HF model lists `AfriSpeech/african-speech-ipa` or `AfriSpeech/african-speech-public_v1` in its `datasets` metadata. — [model filter IPA](https://huggingface.co/api/models?filter=dataset:AfriSpeech/african-speech-ipa); [model filter public_v1](https://huggingface.co/api/models?filter=dataset:AfriSpeech/african-speech-public_v1)

### Inferences
- **Omnilingual models.** Any omniASR-derived encoder has seen the ego/anw/idu (and Nigerian "other") recordings that the hidden test was drawn from. That includes SSL W2V, CTC and LLM models, the v2 builds, sherpa-onnx/ONNX exports and pruned or fine-tuned variants. **Avoid all of them as backbones, feature extractors or pseudo-label teachers.** This applies even if Meta held out the HF test split, which is not stated.
- **Hypa models.** Hypa-Orpheus TTS output (synthetic augmentation) would reproduce Hypa-Voices voices and text, which is derived contamination. **Avoid.** Hypa-Whisper and Hypa-Llama-SNAC-ASR very likely used Hypa-Voices. **Avoid.**
- **AfriSpeech-IPA models.** The `AfriSpeech/afrispeech-ipa-trainer` repo "builds this dataset and trains on it" (per the card), so any phoneme-CTC model it produced is contaminated. None was found on HF under the AfriSpeech org's 8 models, but an unpublished or GitHub release may exist (unverified).
- **Not contaminated by these three corpora on available evidence:** facebook/mms-* (MMS-1B-all, mms-lid-*; 2023, MMS-lab Bible data), Whisper, XLS-R and w2v-BERT. MMS-lab's religious read speech could overlap with AfriSpeech-IPA only if public_v1 were Bible.is audio, which is less likely than JW (Q2). Treat MMS as **safe to ask**: low risk, but disclose it.

### Gaps
- Exact Omnilingual SSL pretraining data composition, i.e. whether the corpus audio was in the 4.3M-h unlabeled set, is not explicit in the text I read. It is moot, because the CTC/LLM models are explicitly fine-tuned on AllASR including the corpus.
- I could not confirm whether Meta released a standalone omni LID model. None appeared in the HF search.
- Training data for Hypa-Whisper-Small-v2 and Hypa-Llama-SNAC-ASR is undisclosed.

## Q5. What do the DSN card and arXiv 2511.09690 say about the Omnilingual corpus's Nigerian coverage (and related Nigerian data in omniASR training)?

### Takeaway
The DSN card cites arXiv 2511.09690 but gives no per-language Omnilingual detail. The paper names no Nigerian corpus collector and lists no per-language hours for Eggon, Annang or Idoma. It does, however, show that omniASR training also used **African Next Voices (co-led by Data Science Nigeria, the competition organiser)** and **Lanfrica/NaijaVoices (incl. Igala, Nupe, Urhobo)**, so omniASR models carry target-language data beyond the prohibited corpus.

### Cited findings

**DSN card**
- It requests the Omnilingual citation (arXiv 2511.09690) for the `omnilingual_asr` portion under CC-BY-4.0. It gives no per-language Omnilingual breakdown. — [DSN card](https://huggingface.co/datasets/Data-Science-Nigeria/dsn-2026-lid-train)

**arXiv 2511.09690**
- AllASR training split, Table 3:

  | Source | Hours | Languages |
  |---|---|---|
  | Open source datasets | 15,000 | 200 |
  | LTPP, internal & licensed | 150,100 | 1,100 |
  | **African Next Voices** | 7,200 | 13 |
  | Open Multilingual Speech Fund (Common Voice) | 1,940 | 177 |
  | **Lanfrica/Naijavoices** | 110 | 11 |
  | **Omnilingual ASR Corpus** | 3,350 | 348 |
  | Total | 120,710 | 1,690 |

  — [arXiv 2511.09690](https://arxiv.org/html/2511.09690)
- "African Next Voices project, a consortium led by Maseno University in Kenya, University of Pretoria in South Africa and **Data Science Nigeria** … supported by the Gates Foundation". — [arXiv 2511.09690](https://arxiv.org/html/2511.09690)
- Lanfrica/Naijavoices "resulted in the creation of new datasets for 11 African languages (Bainouk-Gunyaamolo, Balanta-Kentohe, Bube, Fang, **Igala**, Central Kanuri, Karon, **Nupe-Nupe-Tako**, Upper Guinea Crioulo, Serer and **Urhobo**)". — [arXiv 2511.09690](https://arxiv.org/html/2511.09690)
- The paper's language table lists codes such as `bbu_Latn` "Kulung (Nigeria)" and `bcs_Latn` "Kohumono" with resource levels. A Nigerian deployment is mentioned: "health practitioners in Nigeria are using the system to facilitate Hausa transcriptions". The paper notes QA issues with language-code confusion (e.g. Hausa `hau` vs Hausa Sign Language `hsl`). — [arXiv 2511.09690](https://arxiv.org/html/2511.09690)
- In the low-resource fine-tuning study, `fuv` (Nigerian Fulfulde) had 6.5 training hours and best CER 15.1 (Table 20). — [arXiv 2511.09690](https://arxiv.org/html/2511.09690)
- AfriVox-v2 separately describes Africa Next Voices as "a multi-domain decentralized African conversational speech corpus funded by the Gates Foundation". — [afrivox-v2 card](https://huggingface.co/datasets/intronhealth/afrivox-v2)

### Inferences
- **omniASR has target-language training data beyond the prohibited corpus.** Nigerian ANV languages (likely Hausa/Yoruba/Igbo/Pidgin and others; the list was not given in the text I read) and NaijaVoices Igala/Nupe/Urhobo mean omniASR has trained on target-language speech from DSN-linked and other Nigerian projects. This reinforces avoiding omniASR. It also suggests that **African Next Voices and NaijaVoices data themselves are not prohibited**, but because DSN co-leads ANV, participants should **ask organisers first** before using ANV Nigerian subsets. Any DSN-collected audio might share speakers with Hypa/Omnilingual collections (unverified).
- **Two resource tiers for Eggon, Annang and Idoma.** With ~10 speakers × ~1 h targeted per language, ego/anw/idu test clips likely come from a few Omnilingual speakers (possibly the HF dev/test speakers). Any spontaneous Eggon/Annang/Idoma audio from the same vendor network could share speakers.

### Gaps
- The paper text I read does not list which 13 languages African Next Voices contributed. Whether ego, anw or idu appear in other omniASR training sources (LTPP/internal) is unknown.
- No primary-source Nigerian vendor for the corpus could be identified.

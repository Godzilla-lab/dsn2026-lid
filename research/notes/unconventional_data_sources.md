# Unconventional, legitimate speech-audio sources for the 12 DSN 2026 LID languages

Scope: sources that most teams will miss. Out of scope (other researchers cover them): HF datasets, OpenSLR, FLEURS, Common Voice, NaijaVoices, BibleTTS, Global Recordings Network, Bible.is and generic radio. Checked on 2026-10-20 through catalogue pages and APIs only. No audio was downloaded. "Obtainable today" means it can realistically be fetched and processed before 2026-10-21 23:59 UTC.

## Q1. Which unconventional sources are worth using, and what is the ranked shortlist?

### Takeaway
The best unconventional sources are four. First, **consented WhatsApp/phone voice notes from native speakers**: new speakers, a phone channel that matches the test, and collectable today. Second, **Wikimedia Nigeria's CC BY-SA 4.0 oral-history videos on Commons**: about 18 distinct native speakers across ibb, anw, efi and urh, plus Lower-Cross "confuser" languages. Third, **YouTube videos that uploaders marked CC-BY**, especially Inikpi Tv's full-length **Igala** films (multi-speaker, about 8 h on the channel) and Urhobo, Ibibio and Tiv channels. Fourth, **Wikitongues**: 3 Ibibio speakers under CC BY-SA 4.0. The language-documentation archives (ELAR, PARADISEC, TLA, UCLA) hold almost nothing usable for the priority classes. One Nupe dataset (NaijaVoices on Mozilla Data Collective, 7–8 speakers) is high-value but should be cleared with organisers first, because NaijaVoices supplied Nupe/Igala/Urhobo data to Meta's Omnilingual project.

### Cited Findings

**Master table** (languages limited to the 12 targets plus "other"; hrs = hours)

| # | Source | Target langs covered | Approx. size / speakers | Licence or terms (quoted where possible) | Access method / URL | Obtainable today? | Overlap risk with prohibited corpora | Verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | DIY consented voice notes (WhatsApp/phone) from bootcamp peers, friends, students' associations | any; prioritise nup, igl, ibb, efi, tiv | whatever you collect; 10–20 speakers × 2–5 min per class is realistic (inference) | Your own consent form (see Q8) | WhatsApp voice notes or phone recorder, plus a Google Form for consent and metadata | **Yes** (hours) | Low. A speaker could coincidentally also be a Hypa/Omnilingual contributor; unknowable | **Use**; disclose in model card |
| 2 | Wikimedia Nigeria "Nigerian Language Oral History Documentation Project" videos on Commons | ibb (4 files: Ini, Ibesikpo Asutan, Nsit Ubium, "Ibibio live"), anw (2: Ukanafun, Ika), efi (2), urh-cluster (about 10: Ughelli, Uwheru, Orogun, Olomu, Evwreni, Agbarha, Ughievwen, Idjerhe, Effurun-Otor, Agbon). Also confusers: Oro, Ibeno, Eket, Esit Eket, Itsekiri, and about 200 other Nigerian lects for "other" | about 3–8 min per file (sampled 6:07, 6:40, 2:58, 7:43); one named speaker per file; about 230 files in the gallery | "This file is licensed under the Creative Commons Attribution-Share Alike 4.0 International license." — [Efik file](https://commons.wikimedia.org/wiki/File:The_audiovisual_documentary_of_Efik_language_by_its_native_speaker.webm) | [Gallery](https://meta.wikimedia.org/wiki/Nigerian_Language_Oral_History_Documentation_Project/Gallery); [MediaSearch](https://commons.wikimedia.org/w/index.php?search=%22audiovisual+documentary%22+dialect+native+speaker&title=Special:MediaSearch&type=video&limit=200); direct .webm download | **Yes**. The Commons API rate-limits shared IPs; use a descriptive User-Agent and fetch slowly | Low (self-published by WMNG, 2022–2024) | **Use** (attribute WMNG; strip English segments with VAD/LID) |
| 3 | YouTube videos marked "Creative Commons Attribution licence (reuse allowed)" | igl (Inikpi Tv films and music; Uncle Wowo skits), urh (King Omo Urhobo Tv films), ibb/efi (KRACHOUSE ENTERPRISE talk, comedy and music), tiv (Akaa a Tiv radio and talk; TIV CATHOLIC LINK) | From my sampled searches: Inikpi Tv 25 videos / 8.3 h; KRACHOUSE 22 / 5.1 h; King Omo Urhobo 4 / 2.0 h; Akaa a Tiv 4 / 1.4 h; Uncle Wowo 12 / 0.4 h. Films contain many speakers | CC BY shown on each watch page I checked (e.g. [Igala film AMEMARA 1](https://www.youtube.com/watch?v=iVCPAMhX2Oo), [Tiv Radio](https://www.youtube.com/watch?v=nP9od03UaWA), [Ibibio hotel review](https://www.youtube.com/watch?v=nZw2Z2N-Wrw)) | YouTube search with the CC filter (`sp=EgIwAQ==`), then yt-dlp | **Yes** (a few hours including VAD and segmentation) | Low (not known to be in Hypa/Omnilingual) | **Use, with care**: uploader-declared licence; drop music-only videos; list every video ID; consider telling organisers |
| 4 | Wikitongues oral-history videos | ibb ×3 speakers (Esther 2018, Itoro 2024, Mary 2025 "Ibibio, Anaang, English"); anw (Mary, mixed); pcm ×4 (other) | about 1 short video per speaker | "Video license: CC BY-SA 4.0" — [Esther](https://wikitongues.org/videos/esther_20180728_ibb/), [Itoro](https://wikitongues.org/videos/itoro_20241218_ibb/), [Mary](https://wikitongues.org/videos/mary_20250409_ibb-anw-eng/) | [wikitongues.org/languages/ibb](https://wikitongues.org/languages/ibb/); each page offers Dropbox .mp4 and Commons .webm | **Yes** | Low | **Use** (trim English parts) |
| 5 | Mozilla Data Collective "Speech Data Collection for The Nupe Language" (NaijaVoices / Lanfrica Labs, 2025 Micro-Grants, collector Umar Baba Umar) | nup | 1,583–1,602 clips, 2 h 40 min–3 h 17 min, 7–8 speakers; accents Bida 49%, Kutigi 31%, Lapai 20% | "CC-BY-NC-SA-4.0"; voice cloning and speaker identification prohibited; no re-hosting — [MDC](https://mozilladatacollective.com/datasets/cmihkoth10246md07dtojxehg) | MDC download (account likely required) | Probably (1.46 GB) | **Medium–High.** NaijaVoices created Nupe/Igala/Urhobo datasets for Meta's Omnilingual project ([arXiv 2511.09690](https://arxiv.org/html/2511.09690)) | **Ask organisers first** (biggest Nupe win if cleared) |
| 6 | Bloom Library talking books (SIL) | ego ×1 ("Ori O̱há Atrà", © 2024 SIL Nigeria); tiv ×1 ("Kwav u Maria", © 2024 Emmanuel Orakaaga) | 2 books, 1 narrator each, minutes of audio | `cc-by` (API `license` field) | [Bloom Parse API](https://server.bloomlibrary.org/parse/classes/books) (public app id in the site JS); bloomlibrary.org book pages | **Yes** | Low | **Use** (tiny, but a new Eggon speaker) |
| 7 | Lingua Libre word recordings (Commons) | ibb 50 files / 7 speakers; idu 46 / 1 speaker; urh 23 / 1 speaker; none for efi, tiv, nup, igl, anw, ego | single words, mostly personal names (≈1 s each) | CC BY-SA (Lingua Libre default; per-file check advised) | [Category ibb](https://commons.wikimedia.org/wiki/Category:Lingua_Libre_pronunciation-ibb), [idu](https://commons.wikimedia.org/wiki/Category:Lingua_Libre_pronunciation-idu), [urh](https://commons.wikimedia.org/wiki/Category:Lingua_Libre_pronunciation-urh) | **Yes** | Low | **Use** (low value: shorter than 2 s; concatenate per speaker) |
| 8 | SautiDB-Naija (Nigerian L2 English) | other (English read by L1 speakers of Yoruba, Igbo, Edo, **Efik-Ibibio, Igala**) | 919 clips, 59 min | "flexible Creative Commons License" on Zenodo (variant unverified) — [arXiv 2112.06199](https://arxiv.org/abs/2112.06199) | Zenodo | Yes | Low | **Use** as hard negatives for "other" (check CC variant) |
| 9 | Speech Accent Archive (GMU), Nigeria page | other (English by L1 Hausa 9, Yoruba 7, Igbo 4, **Ibibio 3**, Ebira, Edo, Fulfulde, Kanuri) | 28 short read paragraphs | "BY-NC-SA 4.0" — [accent.gmu.edu](https://accent.gmu.edu/places/nigeria/) | Website | Yes | Low | **Use** for "other" (NC) |
| 10 | PARADISEC RB5 "Bantu Tivoid Tiv" (Roger Blench collection; compiler Raymond Agber, 2010) | tiv | about 76 WAV files (wordlists 0001–1500+, a few texts such as "hare and Kerenyongoga"); likely 1 speaker | "Open (subject to agreeing to PDSC access conditions)" — [OLAC/PARADISEC record](http://catalog.paradisec.org.au/repository/RB5/Bantu_Tivoid_Tiv); [DOI](https://dx.doi.org/10.4225/72/570295FA4A7FC) | PARADISEC account and agree to conditions | Probably (registration) | Low | **Use** if registration is quick (low priority: 1 speaker, wordlists) |
| 11 | PARADISEC AC1-546/547 (1930s Linguaphone Efik discs, E. K. Ekpenyon and Ida C. Ward) | efi | 4 disc sides, 1 speaker, historical audio | "Open (subject to agreeing to PDSC access conditions)" — [AC1-546](https://dx.doi.org/10.4225/72/57D18B7CE0395) | PARADISEC | Probably | Low | Low value (78 rpm channel); optional |
| 12 | ELAR deposits: Eno-Abasi Urua "Dirge songs among the Uruan people" and Bassey Okon "Drums and drum language in Ibibio ceremonies"; Eyo Mensah Kiong deposit (Efik as contact language) | ibb/efi (mostly sung dirges, drum language, performances) | about 24 sessions; up to about 10 named speakers per session | "Media files available to users of the Endangered Languages Archive"; handle redirects to Preservica sign-up — [example handle](https://hdl.handle.net/2196/00-0000-0000-0002-AC69-7) | ELAR registration (Preservica) | **No** (registration and approval) | Low | **Avoid** for this deadline (wrong speech style, slow access) |
| 13 | TLA/MPI "DoBeS Isubu and Wovia" items tagged `ibb` | **Not Ibibio.** Isubu is a Cameroon language mis-coded as `ibb` | 13 sessions | — [example](https://hdl.handle.net/1839/00-0000-0000-0020-7227-1) | — | — | — | **Avoid** (label noise trap) |
| 14 | UCLA Phonetics Lab Archive | efi (1960 and 1964 wordlists, Uyo speaker; story, Akirika Obu speaker); tiv (1962 wordlist, E. A. Akiga) | minutes; 1–3 speakers | "Creative Commons License" (variant not verified) — [efi index](https://archive.phonetics.ucla.edu/Language/EFI/efi.html) | Site returned 503 and an expired TLS certificate on 2026-10-20 | **No** (site down) | Low | Optional later; tiny |
| 15 | LLSTI Ibibio TTS voice and "uyo_ibibio_eno" speech database (Univ. of Uyo / Bielefeld, 2004) | ibb | single speaker read TTS DB (10 × ~3.5 MB parts) | Only "All Rights Reserved" on page — [LLSTI](https://outsideecho.com/llsti/downloads-languages-ibibio.htm) | File links not reachable (404 at guessed paths) | **No** | Low | **Ask authors**; not for this deadline |
| 16 | Efik TTS corpus (arXiv 2607.04515) | efi | 3.08 h, 2,632 utterances, 1 speaker | "will be released under a non-commercial license" (not yet released) — [paper](https://huggingface.co/papers/2607.04515.md) | Not released | **No** | Low | Watch; not usable now |
| 17 | JESUS Film (official, and unofficial Internet Archive mirrors in Efik, Ibibio, Nupe, Igala, Idoma, Urhobo, Eggon) | efi, ibb, nup, igl, idu, urh, ego | about 2 h film per language, many voice actors | "You agree you will not distribute, publish, transmit, modify, display or create derivative works from…" — [Terms](https://www.jesusfilm.org/terms/); FAQ bars new products "using scenes or audio from the films" — [FAQ](https://www.jesusfilm.org/about/faq/) | IA items e.g. [jesus-film-efik-language](https://archive.org/details/jesus-film-efik-language) (no licence field) | Technically yes | Unknown (religious audio, like the banned JW.org upstream) | **Avoid** (terms; IA copies are unauthorised mirrors) |
| 18 | Forvo | possibly efi/ibb/tiv/… (counts unverifiable: 403) | unknown | Removed CC terms by end of 2019 in favour of an "ad hoc" licence that restricts copying and redistribution — [Wikipedia](https://en.wikipedia.org/wiki/Forvo) | API is paid | No | Low | **Avoid** |
| 19 | Living Dictionaries (Living Tongues), Tiv dictionary | tiv | entry/audio counts not shown | Reuse notice: ask first; "attribution doesn't replace permission" — [livingdictionaries.app/tiv](https://livingdictionaries.app/tiv) | Web only | No | Low | **Avoid** unless permission |
| 20 | Internet Archive CC/PD-marked Hausa sermons and tafsir | hau | dozens of items (88 hits for hau audio with a CC/PD licence URL) | e.g. "publicdomain/mark/1.0" set by uploader — [IA search API](https://archive.org/advancedsearch.php?q=language%3A(hau)+AND+mediatype%3Aaudio+AND+licenseurl%3A*creativecommons*&fl[]=identifier&rows=50&output=json) | IA download | Yes | Low | Use only if hau needs more speakers (uploader-asserted licences) |
| 21 | VOA Hausa archive | hau | large archive; service shut down in 2025 | VOA-produced material is public domain; wire content (AP/AFP/Reuters) is not — [Commons:VOA](https://download.osmand.net/wiki/Commons:VOA); [VOA Learning English usage](https://learningenglish.voanews.com/p/6861.html) | voahausa.com still returns HTTP 200 | Yes | Low | **Use** if hau or "other" needs broadcast audio |
| 22 | Kaggle: nkowaokwu/ibo-dict (Igbo API by Nkọwa okwu) | ibo | 4.6 GB; 25k words with "pronunciations" | "CC BY-NC 4.0" — [Kaggle](https://www.kaggle.com/datasets/nkowaokwu/ibo-dict) | Kaggle API | Yes | Low | Use if ibo needs speaker variety |
| 23 | Kaggle: aiolapo/africannigerian-accent-speech-data-aasd | other (Nigerian-accented speech; region/language metadata) | 125 MB of short MP3s | Licence "Unknown" — [Kaggle](https://www.kaggle.com/datasets/aiolapo/africannigerian-accent-speech-data-aasd) | Kaggle API | Yes | Unknown | **Ask organisers first** (no licence) |
| 24 | Common Voice Ibibio on Mozilla Data Collective (v23–25) | ibb | 7,274 clips / 18.18 h (as reported) | CC0, but the datasheet forbids re-hosting — [MDC CV 25.0 Ibibio](https://datacollective.mozillafoundation.org/datasets/cmn2cp74s01hamm07s74uypfk) | MDC | Yes | **High** (Hypa-Voices derives partly from CV) | **Ask organisers first** (flag for the CV researcher) |
| 25 | Paid micro-collection (Indigenius AI; FundrAI "Platform Speaker"; Fiverr native-speaker gigs; GeoPoll / Appen off-the-shelf) | Mostly hau/yor/ibo/pcm; minority languages not listed | Indigenius pays contributors "₦80–₦220 per accepted utterance" with a "+50%" rare-language bonus — [Indigenius](https://indigenius.ai/collective) | Platform-owned; buyer terms not published | Sales contact / job boards | **No** (days–weeks), except possibly a 24 h Fiverr gig | Low | Not for this deadline |

**Other checks that found nothing usable**
- **OLAC aggregate (ELAR, PARADISEC, TLA, Pangloss, Kaipuleohone and others)** returned only Glottolog/WALS/PHOIBLE records for **anw, igl, idu, urh, ego**: no audio deposits. Nupe has one 1930s prayer segment inside PARADISEC AC1/301. — [OLAC search API](https://search.language-archives.org/elastic/olac-search/_search) (public key embedded in [language-archives.org](https://www.language-archives.org/language/nup))
- **Tatoeba**: 0 sentences with audio for yor, ibo, hau and urh; the API rejects efi, tiv, ibb, nup, igl, idu and pcm as language codes. — [Tatoeba API](https://api.tatoeba.org/unstable/sentences?lang=yor&has_audio=yes&sort=random)
- **Spoken Wikipedia**: no Yoruba, Igbo, Hausa or other Nigerian-language subcategory is visible. — [Commons](https://commons.wikimedia.org/wiki/Category:Spoken_Wikipedia)
- **Zenodo**: only a tiny Nupe pronunciation clip ("Dzana", elders and youths, CC BY 4.0) — [Zenodo 16787095](https://zenodo.org/records/16787095). Other hits were text, e.g. [Nupe-English Parallel Corpus](https://zenodo.org/records/10369218).
- **Figshare, Harvard Dataverse, OSF**: no target-language audio found (Dataverse returned 0 for Efik, Ibibio, Igala, Urhobo and Idoma).

### Inferences

**Ranked shortlist (highest value first)**
1. **Consented phone voice notes from native speakers (DIY).** This is the only route to many *new* speakers for nup and igl before the deadline, and it matches the test's phone condition. Target ≥10 speakers per priority class.
2. **Wikimedia Nigeria Commons videos (CC BY-SA 4.0).** About 18 distinct native speakers for ibb (4), anw (2), efi (2) and urh (about 10 clan dialects), downloadable today. The Lower-Cross files (Oro, Ibeno, Eket, Esit Eket) and about 200 other Nigerian lects make realistic "other" negatives. Caveat: some of those are close to Ibibio/Efik, so decide label policy deliberately.
3. **YouTube CC-BY Igala films (Inikpi Tv).** This is the most hours of multi-speaker igl speech found anywhere (several 55–62 min films). Then King Omo Urhobo Tv (urh films), KRACHOUSE (ibb talk and comedy), and Akaa a Tiv / Tiv Catholic Link (tiv).
4. **Wikitongues ibb ×3 (+anw).** Small, but each video is a new Ibibio speaker under a clean licence.
5. **NaijaVoices Nupe on MDC.** 7–8 speakers in three accents would transform nup (1 training speaker). Ask organisers first because of the Omnilingual link.
6. **Bloom talking books** (1 Eggon, 1 Tiv; CC BY). Minutes only, but an Eggon narrator from SIL Nigeria is rare.
7. **Hard negatives for "other":** SautiDB-Naija (English by Efik-Ibibio and Igala L1 speakers) and the Speech Accent Archive (Ibibio L1 English). These teach the model that L1-accented English is "other".
8. **Lingua Libre ibb (7 speakers).** Negligible duration, but cheap speaker diversity for augmentation or contrastive use.

**Metadata traps found (useful for anyone mining catalogues)**
- TLA "ibb" items are Isubu (Cameroon).
- Bloom "Nupe" books are Javanese text.
- ELAR "efi/ibb" items are dirges and drum language.
- Commons "Tiv (avk).wav" carries a wrong code.

### Gaps
- I could not count Forvo words per language (403 to automated access) or Webonary dictionaries (403).
- The UCLA licence variant is unverified: the site was down (503 / expired certificate).
- Whether DSN treats uploader-declared YouTube CC-BY as "collected with permission" is untested. Recommend a one-line message to organisers.
- Exact total duration of the Wikimedia Nigeria target-language files: I sampled 4 files only.

## Q2. Language-documentation archives (ELAR, PARADISEC, TLA/DoBeS, Pangloss, Kaipuleohone, SIL, UCLA, KWEF, Uyo)

### Takeaway
Archives are nearly empty for these languages. The only real holdings are four. ELAR has Ibibio/Efik dirge and drum-language performance projects (registration-gated). PARADISEC has one Tiv wordlist collection and 1930s Efik Linguaphone discs. UCLA has a few 1960s Efik/Tiv wordlists (site down). LLSTI has an "All Rights Reserved" single-speaker Ibibio TTS database. None is a fast, high-value win.

### Cited Findings
- **OLAC totals.** efi 32 records (20 ELAR, 3 PARADISEC), ibb 41 (24 ELAR, 13 TLA), tiv 6, nup 7, idu 4, urh 3, igl 1, anw 1, ego 2. The non-ELAR/PARADISEC ones are Glottolog/WALS/PHOIBLE reference records. — [OLAC search API](https://search.language-archives.org/elastic/olac-search/_search)
- **ELAR "Afagha" session** (creator Eno-Abasi Urua, 2016-01-29): formats audio/x-wav and ELAN .eaf; speakers Mrs. Elizabeth Etim and Mrs. Comfort Okon Akpan; rights "Media files available to users of the Endangered Languages Archive". — [hdl 2196/…AC69-7](https://hdl.handle.net/2196/00-0000-0000-0002-AC69-7)
- **ELAR "Abon" session ("Performance")** lists 10 speakers (Chief Ekanem, Usen Inyang, Nsikak Effiong, …). "Drum language" (creator Bassey Okon) lists 6. — [Abon](https://hdl.handle.net/2196/00-0000-0000-0002-A676-2), [Drum language](https://hdl.handle.net/2196/00-0000-0000-0002-A68B-B)
- **The underlying ELDP projects** are "Documentation of Dirge songs among the Uruan people" (PI Eno-Abasi Urua, Univ. of Uyo, from 2007) and "Documenting drums and drum language in Ibibio traditional ceremonies" (PI Bassey Okon, Univ. of Calabar, 2008). — [Dirge project](https://cultureincrisis.org/projects/documentation-of-dirge-songs-among-the-uruan-people-in-nigeria), [Drum project](https://cultureincrisis.org/projects/documenting-drums-and-drum-language-in-ibibio-traditional-ceremonies)
- **ELAR access.** Handles now redirect to an access.preservica.com sign-up page. — [redirect target](https://eldp.access.preservica.com/uncategorized/SO_99891b87-7410-4810-8397-781e5f546c07/)
- **ELAR also holds a Kiong (kkm) "Village Meeting" session** tagged Efik (libation ritual, 2016), contributors including Eyo Mensah. — [hdl 2196/…B487-B](https://hdl.handle.net/2196/00-0000-0000-0012-B487-B)
- **TLA "lion and antelope"** belongs to "DoBeS archive : Isubu and Wovia", recorded in Limbe, Cameroon, but is coded `ibb`. — [TLA](https://hdl.handle.net/1839/00-0000-0000-0020-7227-1)
- **PARADISEC RB5 Tiv** (Raymond Agber, compiler, 2010) holds 152 files (MP3/WAV pairs): wordlists WLs_0001…, texts "hare and Kerenyongoga", "how to trap a hare", "slow and fast speech"; access "Open (subject to agreeing to PDSC access conditions)". — [RB5 Tiv](http://catalog.paradisec.org.au/repository/RB5/Bantu_Tivoid_Tiv)
- **PARADISEC AC1-546/547** are "LINGUAPHONE (Made in England) Efik Record 1–4 … E. K. Ekpenyon, Miss I. C. Ward", including "The tortoise and the fishes". — [AC1-546](https://dx.doi.org/10.4225/72/57D18B7CE0395), [AC1-547](https://dx.doi.org/10.4225/72/57D18B81F3C53)
- **PARADISEC AC1/301** contains, at 00:43:06, "Prayer in the language Nupe as spoken at the Gunu ritual" (Nadel source). — [AC1/301](http://catalog.paradisec.org.au/repository/AC1/301)
- **KWEF material appears in PARADISEC** via Roger Blench collections (e.g. Yiwom, Gbiri). I found no dedicated KWEF audio site. — [PARADISEC RB3 Ywom](https://catalog.paradisec.org.au/repository/RB3/Ywom)
- **UCLA Phonetics Archive** lists Efik wordlists (1960, University College Ibadan, speaker from Uyo; 1964 at UCLA) and a story (speaker from Akirika Obu), plus a Tiv 1962 wordlist (E. A. Akiga, Mkar). — [Efik](https://archive.phonetics.ucla.edu/Language/EFI/efi.html), [Tiv](https://archive.phonetics.ucla.edu/Language/TIV/tiv_record_details.html)
- **LLSTI (Uyo/Bielefeld)** offers `uyo_ibibio_eno_v1.0.tar.gz`, `uyo_ibibio_eno_database.tar.gz` (or ten parts of about 3.5 MB) and a 2004 report. The only rights text is "All Rights Reserved"; "first prototype". — [LLSTI Ibibio](https://outsideecho.com/llsti/downloads-languages-ibibio.htm)
- **Ekpenyong et al.** built Ibibio statistical-parametric TTS with Edinburgh CSTR (2013) and unit-selection TTS with Urua and Gibbon. I found no newer public corpus. — [CSTR PDF](https://www.cstr.ed.ac.uk/downloads/publications/2013/Moses_Ibibio.pdf), [Bielefeld record](https://pub.uni-bielefeld.de/record/2955644)

### Inferences
- The ELAR sessions are useful for later, permission-based work. But sung dirges and drum language are a poor fit for spoken LID, and registration and possible depositor approval rule them out for 21 Oct.
- PARADISEC registration is usually self-service, so RB5 Tiv may be same-day. It adds one wordlist speaker; Tiv already has 4.27 h, so it is low priority.

### Gaps
- I did not verify ELAR's current terms text on ML/AI use; the search found no ELAR statement.
- I did not query the SIL Language & Culture Archives directly. OLAC returned no SIL audio for these codes, which suggests none is harvested.
- Pangloss and Kaipuleohone returned nothing via OLAC for these codes. I did not search them natively.

## Q3. Wikimedia sources (Lingua Libre, Commons, Wikitongues partnership, Spoken Wikipedia)

### Takeaway
Commons hides the single best unconventional find: Wikimedia Nigeria's 2022–2024 CC BY-SA 4.0 oral-history videos. Lingua Libre is tiny for the targets: ibb has 7 speakers and 50 words; idu and urh have 1 speaker each; there is nothing for efi, tiv, nup, igl, anw or ego.

### Cited Findings
- **Lingua Libre file counts** (Commons categoryinfo): ibo 7,669; yor 1,109; hau 375; pcm 81; ibb 50; idu 46; urh 23. No category exists for ego, anw, efi, igl, nup or tiv. — [Commons API](https://commons.wikimedia.org/w/api.php?action=query&titles=Category:Lingua_Libre_pronunciation-ibb&prop=categoryinfo)
- **Lingua Libre speakers.** ibb has 7 (Confidence24 15, Nwa Mmeme 12, GodswillSE 8, Andikan Efiok Eduok 6, Victorpeter001 3, Honeydear 3, Phina001 3). idu is a single speaker, "Adini Anna", 46 personal names. urh is a single speaker, "Bamjos", 23 personal names. — [ibb](https://commons.wikimedia.org/wiki/Category:Lingua_Libre_pronunciation-ibb), [idu](https://commons.wikimedia.org/wiki/Category:Lingua_Libre_pronunciation-idu), [urh](https://commons.wikimedia.org/wiki/Category:Lingua_Libre_pronunciation-urh)
- **The Wikimedia Nigeria project** aims at "freely licensed audiovisual files of spoken Nigerian languages", hosted on Commons, Wikitongues, the Library of Congress and the Internet Archive. — [Meta project page](https://meta.wikimedia.org/wiki/Nigerian_Language_Oral_History_Documentation_Project)
- **The gallery lists about 230 files**, including Efik ×2, Ibibio ×3–4, Annang ×2 and Delta lects (Effurun-Otor, Ughelli, Uwheru, Olomu, Orogun, Ughievwen, Agbarha, Evwreni, Idjerhe, Agbon). It has none for Tiv, Nupe, Igala, Idoma, Eggon or Hausa. — [Gallery](https://meta.wikimedia.org/wiki/Nigerian_Language_Oral_History_Documentation_Project/Gallery)
- **MediaSearch confirms 22 matching titles**, including "Ibibio (Nsit Ubium)", "Esit Eket", "Ibeno", "Oro" and "Itshekiri". — [MediaSearch](https://commons.wikimedia.org/w/index.php?search=%22audiovisual+documentary%22+dialect+native+speaker&title=Special:MediaSearch&type=video&limit=200)
- **Efik file:** 6 min 7 s; speaker Prof. Effiong Ani; recorded 23 April 2022, Calabar; CC BY-SA 4.0. — [file](https://commons.wikimedia.org/wiki/File:The_audiovisual_documentary_of_Efik_language_by_its_native_speaker.webm)
- **Ibibio (Ini) file:** 6 min 40 s; Dr. Alfred Udo; Ini, Akwa Ibom, 25 Aug 2024; CC BY-SA 4.0. — [file](https://commons.wikimedia.org/wiki/File:The_audiovisual_documentary_of_the_Ibibio_(Ini)_dialect_by_a_native_speaker.webm)
- **Ughelli file:** 2 min 58 s; Hon. Ese Agbobore; Delta, 28 Aug 2024; CC BY-SA 4.0. The page names the language "Ughelli", not "Urhobo". — [file](https://commons.wikimedia.org/wiki/File:The_audiovisual_documentary_of_the_Ughelli_dialect_by_native_speaker.webm)
- **Annang file:** 7 min 43 s; Offong Kevin Nkanang; Ukanafun, Akwa Ibom; CC BY-SA 4.0. — [file](https://commons.wikimedia.org/wiki/File:The_audiovisual_documentary_of_Annang_dialect_by_a_Native_speaker.webm)
- **Commons audio search** finds only scattered single files: "Our Lord's prayer in Igala language.ogg", "Igala proverbs…ogg", "Praise and Worship - Ibibio Language…ogg". — [audio MediaSearch](https://commons.wikimedia.org/w/index.php?search=Tiv+OR+Igala+OR+Idoma+OR+Nupe+OR+Eggon+OR+Efik+OR+Ibibio+OR+Urhobo&title=Special:MediaSearch&type=audio&limit=100)
- **Wikitongues** has 3 Ibibio videos, 1 tagged Anaang (mixed with Ibibio and English) and 4 Pidgin, all "CC BY-SA 4.0" with Dropbox and Commons downloads. Its pages for efi, tiv, nup, igl, idu, urh and ego list no videos. — [ibb page](https://wikitongues.org/languages/ibb/), [anw page](https://wikitongues.org/languages/anw/)
- **Spoken Wikipedia:** no Nigerian-language subcategory is visible. — [Commons](https://commons.wikimedia.org/wiki/Category:Spoken_Wikipedia)

### Inferences
- Urhobo labelling: Ughelli, Uwheru, Orogun, Olomu, Evwreni, Agbarha, Ughievwen, Idjerhe, Effurun-Otor and Agbon are, to my knowledge, Urhobo clans/kingdoms in Delta State. This is not stated on the pages, so verify by listening or with a native speaker before labelling them `urh`. This gives about 10 new urh speakers.
- "Ibibio live" and "Esit Eket/Eket/Ibeno/Oro" are Lower-Cross neighbours. Eket, Ibeno and Oro are distinct languages; treat them as "other" or exclude them, because labelling them `ibb` risks noise.
- Each video probably contains an English title or intro, possibly English Q&A. Run an off-the-shelf LID or VAD and keep only non-English stretches.

### Gaps
- No per-file total durations for all ~22 target-language files (4 sampled, 3–8 min).
- Whether WMNG ran a 2025–26 North-Central round (Tiv/Idoma/Igala/Nupe/Eggon): none found on Commons.

## Q4. Pronunciation and dictionary platforms (Forvo, Webonary, Living Dictionaries, Kaikki/Wiktionary, Tatoeba)

### Takeaway
None is usable for this deadline. Forvo's licence is now proprietary. Living Dictionaries (Tiv) requires permission. Tatoeba has no audio. Wiktionary audio is just Commons/Lingua Libre (covered above). Webonary was not verifiable.

### Cited Findings
- **Forvo** "had removed the 'Creative Commons' terms and conditions from their website, replacing them with an 'ad hoc' license that severely restricts user's rights to copy, modify and redistribute the audio files" (by end of 2019). The API is paid. — [Wikipedia: Forvo](https://en.wikipedia.org/wiki/Forvo)
- forvo.com/languages/{efi,ibb,tiv,nup,igl,urh,idu,anw,ego} all returned HTTP 403 to automated requests on 2026-10-20, so the word counts are unverified. — [forvo.com/languages/efi](https://forvo.com/languages/efi/)
- **Living Dictionaries Tiv:** no licence; the reuse notice says ask first, "attribution doesn't replace permission". Contributor pages elsewhere say "All rights reserved. Do not distribute or reproduce without permission." — [Tiv dictionary](https://livingdictionaries.app/tiv), [example contributors page](https://livingdictionaries.app/zwpqtvcaoirkvphgyjskrxbpd/contributors)
- **Webonary** supports audio in entries (SIL, FLEx-based). webonary.org returned 403 to automated access, so I found no Nigerian target dictionary. — [Webonary brochure](https://www.webonary.work/wp-content/blogs.dir/1/files/Webonary-brochure.pdf)
- **Tatoeba API** returned total 0 audio sentences for yor, ibo, hau and urh, and 400 (unknown language) for efi, tiv, ibb, nup, igl, idu and pcm. — [API](https://api.tatoeba.org/unstable/sentences?lang=urh&has_audio=yes&sort=random)

### Inferences
- Wiktionary/Kaikki audio for these languages is a subset of the Lingua Libre files counted in Q3, so it adds nothing new.

### Gaps
- Forvo per-language counts. Webonary's catalogue.

## Q5. Creative-Commons video (YouTube CC-BY filter, Vimeo, PeerTube)

### Takeaway
The YouTube CC filter works from this machine. It surfaces target-language channels whose videos display "Creative Commons Attribution licence (reuse allowed)": Igala (Inikpi Tv films and music; Uncle Wowo skits), Urhobo (King Omo Urhobo Tv films), Ibibio/Efik (KRACHOUSE ENTERPRISE) and Tiv (Akaa a Tiv; TIV CATHOLIC LINK). There is little or nothing for Nupe, Idoma, Annang and Eggon.

### Cited Findings
- **Confirmed CC BY text on these watch pages:**
  - Igala: [AMEMARA 1 full movie (59:26)](https://www.youtube.com/watch?v=iVCPAMhX2Oo)
  - Tiv: [Tiv Radio (Akaa a Tiv) 50:01](https://www.youtube.com/watch?v=nP9od03UaWA); [Back in the days in Tiv land 20:42](https://www.youtube.com/watch?v=yuD8EexhM5M); [Tiv Situations 11:07](https://www.youtube.com/watch?v=OOIFjGZVFZ8); [Tiv Catholic Link: Who Are We?](https://www.youtube.com/watch?v=1Iiw2Q0gxJ8)
  - Ibibio: [Review of Watbridge Hotel in Ibibio 10:21](https://www.youtube.com/watch?v=nZw2Z2N-Wrw); [MBAKARA 8:16](https://www.youtube.com/watch?v=aU-kMUCUMxw); [Made in Akwa Ibom 27:45](https://www.youtube.com/watch?v=fxag9IRrqb0)
  - Idoma channel: [Idoma okoho soup 13:37](https://www.youtube.com/watch?v=3SQiJzag3Sw) (spoken language unverified)
  - King Omo Urhobo Tv and Uncle Wowo items (IDs Fcdl2WYhCc4, BED2sEPFVQ4)
  - Source for all: YouTube watch pages fetched 2026-10-20
- **Sampled CC-filter search totals** (not exhaustive; durations from search results): Inikpi Tv 25 videos / 8.3 h, including "LATEST IGALA MOVIES | AMEMARA 1/2", "IGALA CHRISTIAN MOVIE | ARIDA OMEMELE (53:43)" and "OMABIBI AKPUNYI (1:02:43)", plus many music videos. KRACHOUSE 22 / 5.1 h (incl. "Dauda Yangiriwa Full Dvd 58:11" and music). King Omo Urhobo Tv 001: 4 / 2.0 h ("Urhobo Films EmmaBrown Records" 1:08:08, 39:06). Akaa a Tiv 4 / 1.4 h. — [YouTube CC search example](https://www.youtube.com/results?search_query=Igala+movie&sp=EgIwAQ%3D%3D)
- **Weak queries:** "Nupe", "Eggon" and "Annang" CC searches returned mostly unrelated or English/Hausa content (e.g. ADBN TV Nasarawa news). — [example](https://www.youtube.com/results?search_query=Eggon+Nasarawa&sp=EgIwAQ%3D%3D)

### Inferences
- **Igala:** Inikpi Tv's films are the most valuable igl find in this whole study. Several hours of multi-actor dialogue against 0.48 h / 2 speakers in training. Background music is present, so use VAD plus music/speech separation or keep segments with low music energy.
- **Licence caveats:** "CC BY" on YouTube is self-declared, and compilation channels (KRACHOUSE's "Oldschool Naija hits") may re-upload others' music. Prefer original talk, drama and radio content, and record video IDs and channel names for attribution in the model card. YouTube's ToS restricts downloading; teams commonly rely on the CC grant (precedent: CC-YouTube speech corpora). Flag this to organisers if unsure.
- Vimeo CC and PeerTube were not searched (time). They are unlikely to beat YouTube for these languages.

### Gaps
- Exhaustive CC-video inventory per channel (only first-page search results were parsed).
- Spoken-language verification of each video (titles only).

## Q6. Competition and data platforms (Kaggle, Zindi, Zenodo, Figshare, Mendeley, Dataverse, OSF, Lanfrica/MDC, GitHub orgs)

### Takeaway
Kaggle and the research repositories hold almost nothing for the priority languages. The notable platform find is Mozilla Data Collective: NaijaVoices' Nupe dataset (7–8 speakers) and Common Voice Ibibio (18 h). Both need organiser clearance because of provenance links to the prohibited corpora.

### Cited Findings
- **Kaggle API searches:**
  - "ibibio" and "eggon" returned 0 datasets.
  - "efik" returned only unrelated "traefik" logs.
  - "nupe" returned text datasets (davidvictor297/nupe-dataset, MIT).
  - "tiv language" returned a text Q&A set (CC BY 4.0).
  - "igbo audio" returned jameskaile/igbo-audio-dataset (MIT, 4.6 GB, no description) and nkowaokwu/ibo-dict (CC BY-NC 4.0).
  - "hausa audio" returned victorolufemi/pericles-video-dataset ("fongbe-hausa-youtube-speech-corpus", CC BY-NC-SA 4.0, 9.4 GB).
  - Source: [Kaggle API](https://www.kaggle.com/api/v1/datasets/list?search=ibibio)
- **Kaggle MD3-en** (Multi-Dialect Dataset of Dialogues, English, Eisenstein et al., CC BY-SA 4.0, 7.3 GB) is a possible "other" English source; Nigerian English share not checked. — [Kaggle](https://www.kaggle.com/datasets/jacobeis99/md3en)
- **MDC Nupe:** "collected by Umar Baba Umar as part of the 2025 NaijaVoices Micro-Grants Heritage project"; CC-BY-NC-SA-4.0; released 11/27/2025; WAV + TXT + Metadata.csv. — [MDC](https://mozilladatacollective.com/datasets/cmihkoth10246md07dtojxehg)
- **Lanfrica/NaijaVoices** "resulted in the creation of new datasets for 11 African languages (… Igala, … Nupe-Nupe-Tako, … Urhobo)" in Meta's Omnilingual ASR paper. — [arXiv 2511.09690](https://arxiv.org/html/2511.09690) (via sibling provenance notes)
- **The NaijaVoices org page on MDC** lists Atyap, Ekpeye, Ehugbo, Ibọnọ/Obolo, Gbagyi and Nupe. — [NaijaVoices org on MDC](https://datacollective.mozillafoundation.org/organization/cmidd9v3l002nmh07kmij7fq5)
- **Common Voice Ibibio** (MDC) is CC0 but "forbids re-hosting or re-sharing"; v25.0 released 3/22/2026, with Ekpenyong among the datasheet authors. — [CV 25.0 Ibibio](https://datacollective.mozillafoundation.org/datasets/cmn2cp74s01hamm07s74uypfk)
- **Zenodo** "Audio: Pronunciation of 'Dzana' in Nupe language by Elders and Youths" (CC BY 4.0) is a very small item. — [Zenodo 16787095](https://zenodo.org/records/16787095)
- **Figshare** search returned only an accent-embeddings preprint on SautiDB-Naija. — [CISPA/Figshare](https://publications.cispa.de/articles/preprint/Learning_Nigerian_accent_embeddings_from_speech_preliminary_results_based_on_SautiDB-Naija_corpus/32802845)
- **Mendeley Data** "Multimedia Digital Resources for African Tone Languages" (OAU): 175 WAVs of food words in English, Yoruba, Hausa and Igbo; CC BY 4.0; 28 Aug 2026. — [Mendeley](https://data.mendeley.com/datasets/dt3wthx5b4)
- **NaijaS2ST (2026)** covers ig, ha, yo and pcm with about 50 h each; no dataset URL found. — [arXiv 2604.16287](https://arxiv.org/abs/2604.16287)
- **Zindi:** no Nigerian-language audio competition dataset found in search. Appen and GeoPoll sell Hausa phone/telephony data (e.g. "Hausa (Nigeria) conversational telephony set of 33 hours"). — [Appen OTS](https://datasets.appen.com/country-laos-or-nigeria-or-poland), [GeoPoll Hausa](https://www.geopoll.com/?p=25179)

### Inferences
- The MDC Nupe set and the 2025 NaijaVoices micro-grants are separate from the Omnilingual-era NaijaVoices Nupe collection. But speaker overlap with the prohibited facebook/omnilingual-asr-corpus Nupe portion is plausible (same community and organisation). Hence "ask organisers first".
- jameskaile/igbo-audio-dataset has no provenance, so avoid it or ask; it could be a re-upload of NaijaVoices or Common Voice.

### Gaps
- GitHub org search (Masakhane, HausaNLP, IgboNLP) failed: this session's `gh` is repo-scoped and the search endpoint was blocked. Not verified.
- No Zindi competition data page was retrieved.

## Q7. Public-domain government audio, Internet Archive and Library of Congress, educational/religious and broadcaster sources

### Takeaway
VOA Hausa (public-domain VOA-produced audio) and Internet Archive PD/CC Hausa sermons are legitimate but only help Hausa. The Library of Congress could not be queried (Cloudflare). The JESUS Film exists in seven target languages, but its terms forbid derivative works, so avoid it. Bloom Library has two CC-BY talking books (Eggon, Tiv). I found no openly licensed state-broadcaster podcasts. A Catholic Tiv channel publishes under CC BY on YouTube.

### Cited Findings
- **VOA:** Commons summarises that VOA-produced text, audio and video are public domain, but third-party material "such as material from Agence France-Presse, Associated Press and Reuters" is not. — [Commons:VOA (mirror)](https://download.osmand.net/wiki/Commons:VOA). VOA Learning English states its texts and MP3s are public domain, with AP/Reuters/AFP excluded. — [Learning English](https://learningenglish.voanews.com/p/6861.html)
- **VOA Hausa was shut down in 2025** as part of the administration's dismantling of VOA; a court injunction was under appeal. — [Guardian NG](https://guardian.ng/news/world/africa/voa-goes-off-air-in-rural-nigeria-ghana-others/), [RSF](https://rsf.org/en/sub-saharan-africa-community-radio-programmes-shut-down-access-information-jeopardy-after-voice). voahausa.com still returned HTTP 200 on 2026-10-20 (own check).
- **Internet Archive:** 88 Hausa audio items carry a CC/PD licence URL (e.g. "Sheikh JAAFAR MAHMUD COMPLETE TAFSEER", publicdomain/mark/1.0). Target-language hits are mostly music plus "JESUS Film {Efik, Ibibio, Nupe, Igala, Idoma, Urhobo, Eggon} Language" with no licence. — [IA advancedsearch](https://archive.org/advancedsearch.php?q=title%3Aigala+AND+mediatype%3A(audio+OR+movies)&fl[]=identifier&fl[]=licenseurl&rows=20&output=json)
- **JESUS Film terms:** "You agree you will not distribute, publish, transmit, modify, display or create derivative works from…"; the FAQ prohibits new products "using scenes or audio from the films". — [Terms](https://www.jesusfilm.org/terms/), [FAQ](https://www.jesusfilm.org/about/faq/)
- **Bloom Library API:** ego 1 book (talking book, cc-by, © 2024 SIL Nigeria); tiv 12 books, of which 1 is a talking book ("Kwav u Maria", cc-by). The ibb, idu, igl and urh books are Bible for Children text-only (cc-by-nc-nd). The "nup" books are Javanese text. — [Bloom Parse API](https://server.bloomlibrary.org/parse/classes/books)
- **MegaVoice** sells a dramatized Eggon New Testament (Bible Society of Nigeria 1974 text); no reuse terms stated. — [MegaVoice](https://megavoice.com/media-cloud/m0b5085-new-testament-dramatized-ego-eggon-audio-bible/)
- **Radio Benue** broadcasts in English, Tiv and Idoma. I found no podcast archive for Radio Benue, AKBC, Kogi State Radio or Radio Niger. — [Wikipedia: Radio Benue](https://en.wikipedia.org/wiki/Radio_Benue)
- **Library of Congress:** loc.gov/audio JSON API returned a Cloudflare challenge (403) for all queries; not verified. — [loc.gov](https://www.loc.gov/audio/?q=efik&fo=json)

### Inferences
- State broadcasters (BRTV/Radio Benue, AKBC, Prime FM, Radio Niger) would require written permission. That is not achievable in about 30 hours unless a team member has a personal contact at the station.
- The MegaVoice/Bible Society Eggon NT is most likely the same recording family as Bible.is/FCBH (covered elsewhere). Treat it as religious read speech with unclear terms.

### Gaps
- The VOA official Terms-of-Use page text was not retrieved (secondary sources only).
- LOC / American Folklife Center holdings are unknown.

## Q8. Do-it-yourself consented collection and paid micro-collection

### Takeaway
DIY collection is the highest-value move for the priority classes. Many short phone recordings from many native speakers directly address the test set's speaker-disjoint, phone-heavy design. Use a one-page consent form (purpose, ML training for this competition, retention, withdrawal, no voice cloning), a 10-field metadata sheet, and mostly spontaneous prompts. Paid platforms cannot deliver minority-language data within hours.

### Cited Findings
- **Speaker mismatch is the core LID failure mode.** SIGTYP 2021 notes that low-resource training corpora are often "single-speaker or have different domains". Systems "can perform above 95% accuracy in-domain, but domain and speaker mismatch is very challenging". — [SIGTYP 2021 Robust SLID](https://arxiv.org/abs/2106.03895)
- **Consent form elements:**
  - A standard speech-corpus template (BAS Munich) covers use of recordings for a corpus, anonymisation after a period, a right to revoke, and a participant copy. — [BAS template](https://www.phonetik.uni-muenchen.de/Bas/BasTemplateInformedConsent_en.pdf)
  - Kenyan corpora (Kencorpus) used consent forms stating objectives, processing, access and use. AfriVoices-KE required consent in-app before profile creation. — [Kencorpus](https://arxiv.org/pdf/2208.12081), [AfriVoices-KE](https://arxiv.org/pdf/2604.08448)
  - Industry guidance (vendor blog, lower authority) recommends consent in local languages and oral consent where culturally appropriate, and warns that jargon like "machine learning model training" needs explanation. — [Way With Words](https://waywithwords.net/?p=15903)
- **Nigerian law:** under the NDPA 2023 "the researcher bears the burden of proof for establishing that the consent of a human subject had been sought for and obtained" (preprint analysis). Sensitive data includes data revealing ethnic origin or religion. — [Research Square preprint](https://researchsquare.com/article/rs-3293485/v1), [Mondaq NDPA overview](https://www.mondaq.com/nigeria/data-protection/1366934/the-nigerias-data-protection-act-2023-a-look-at-key-provisions)
- **Paid platforms:**
  - Indigenius AI pays contributors "₦80–₦220 per accepted utterance", "+50% for under-represented languages". Its languages named are Yoruba, Swahili, Hausa, Pidgin and Igbo (coming soon). The sales contact link is a placeholder. — [Indigenius](https://indigenius.ai/collective)
  - A Jobberman listing recruits "Platform Speaker" native speakers for FundrAI recordings. — [Jobberman](https://www.jobberman.com/listings/platform-speaker-455wd2)
  - A Fiverr gig advertises "native voice recordings for AI projects" in Nigerian languages (page 403; details unverified). — [Fiverr](https://fiverr.com/ehiefeanthony/teach-nigerian-languages-or-provide-native-voice-recordings-for-ai-projects)

### Inferences
**Recommended minimal protocol** (synthesis, not a sourced standard)

- **Who:** native speakers of nup, igl, ibb, efi and tiv first, then anw, ego, idu and urh. Recruit via bootcamp peers, their parents and relatives, and campus associations: Nupe students' associations (e.g. at IBB University Lapai, FUT Minna), Igala (Kogi State University, Prince Abubakar Audu University), Efik/Ibibio/Annang (UNICAL, UniUyo, AKSU), Tiv/Idoma (Benue State University, Federal University of Agriculture Makurdi), Eggon (Nasarawa State University). Institution names are suggestions, not verified contacts.
- **How many:** speaker count beats hours (SIGTYP). Aim for ≥10 speakers per priority class at 2–5 min each, i.e. about 20–40 clips of 5–15 s. Ten speakers × 3 min = 30 min of new speaker-diverse audio per class, comparable to the entire current nup/igl training sets but with 5–10× the speakers.
- **Channel:** WhatsApp voice notes are deliberately fine. They mimic the test's phone condition (WhatsApp voice notes are Opus-compressed: my knowledge, unverified here). Ask for some clips recorded indoors and some outdoors or with background noise. Keep the original files; do not re-encode before archiving.
- **Prompts (mix):**
  - (a) Spontaneous, about 60%: "Introduce yourself and your town", "Describe how to cook your favourite soup", "Tell a short folktale or proverb and explain it", "Give directions to your village market", "What did you do yesterday?"
  - (b) Read, about 40%: 10 short sentences in the language (from local textbooks or community-written sentences, not JW.org text).
  - Ask speakers to avoid English and Pidgin code-switching in target clips. Separately collect a few English/Pidgin clips from the same speakers as "other" so the model learns language, not speaker.
- **Consent text** (short, in English plus the local language if possible). "I agree that my voice recordings and the details below may be used by [team] to train and evaluate a language-identification model for the DSN 2026 competition, and may be described (not shared publicly) in the model card. My name will not be published. The recordings will not be used to clone my voice. I can withdraw by messaging [contact] before [date]. I am 18 or older." Capture a typed name and date, a tick-box, or a recorded oral "I agree" in the first voice note.
- **Metadata sheet (one row per speaker):** pseudonymous speaker_id; language + dialect/town; gender; age band; other languages spoken; phone model; app used (WhatsApp/recorder); environment (quiet/noisy/outdoor); prompt type (spontaneous/read); consent version and date.
- **Speaker-disjoint validation:** hold out whole speakers for validation, mirroring the test.
- **Turnaround:** a WhatsApp broadcast with a Google Form can return dozens of notes within hours. Paid platforms (Indigenius, Appen, GeoPoll) take days to weeks and list no target minority languages, so they are not viable for 21 Oct.

### Gaps
- No published NaijaVoices consent form was retrieved for reuse.
- Whether DSN accepts team-collected recordings as "ethically sourced": the rules' examples suggest yes, but confirm with organisers.
- Evidence quantifying LID gain per extra speaker (vs per hour) for these specific languages was not found.

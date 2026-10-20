# Win DSN 2026 with voices, not model size

Four classes decide this competition: Efik, Ibibio, Igala and Nupe. Together they carry **35% of the F1**. Every team is near zero on the first three: the leader, datapsalm, scores **F1 0.14, 0.03 and 0.16** on efi, ibb and igl, while its other nine classes already average about **0.77** ([Codabench 18424](https://www.codabench.org/competitions/18424/)). Each of these classes rests on **one to three recorded speakers** from a single corpus. The provided data is also full of recording fingerprints that let a model guess the label from silence and loudness instead of language. A bigger backbone does not fix this: the second-placed model, apparently a full 12-layer AfriHuBERT, still scores **0.0 on Ibibio**.

The winning plan is to make the model hear many more voices for those classes, legitimately:
- team-collected, consented voice notes;
- a clean 8.8-hour Efik corpus;
- Wikimedia Nigeria's CC BY-SA oral-history videos;
- once the organisers say yes, VoiceAfrica's 10-speaker Igala, Nupe and Urhobo sets.

On top of the data come augmentation that destroys the shortcuts, voice conversion to a shared pool of voices, an MMS-LID teacher if it passes a zero-shot test, and per-class decision offsets baked into the classifier bias.

Efficiency hardly separates a 4-layer from a 12-layer AfriHuBERT: the score multiplier is about 0.94 versus 0.91. Latency above 10 ms per clip, however, costs as much as that entire depth difference, so it must be measured.

Your training ladder suits this problem. The plan uses six dev submissions, each testing one change that has passed speaker-grouped cross-validation. The final model is retrained on all data, in an architecture already proven on the organisers' hardware. The main risks are not about modelling:
- data questions still unanswered (the email should go out today);
- packaging errors that the organisers' checker rejects;
- a final phase that allows only about three submissions in eight hours.

## What this research changes in this morning's plan

**In plain terms:** most of the plan holds. Seven points change: what limits the score, how to validate, which data and teacher to trust, how to package, and how many final shots there really are.

| Topic | Plan this morning | What the research now says |
|---|---|---|
| Efficiency | "Model size is the only real efficiency lever"; speed is saturated for small models | Speed earns full marks only at **≤10 ms per clip**. At 20 ms it loses about 0.03 of the multiplier, as much as going from 4 to 12 layers. Measure latency, not just size |
| Where the points are | Bake-off of backbones | Leaders average about 0.77 F1 on nine classes and ≤0.17 on efi/ibb/igl. A full AfriHuBERT still scores ibb 0.0. **The bottleneck is speakers, not the backbone** |
| Validation | K-fold grouped by `speaker_id` | **3 folds with forced speaker rotation.** K=5 leaves folds with no igl or ibb speaker. Group Hypa speakers across labels. Build pseudo-speakers for the 62.5% of clips with no id. Nupe cannot be validated on the provided data |
| Shortcuts | Check label × source | **Channel statistics alone reach F1 0.34**. The silence and loudness fingerprints must be destroyed by augmentation |
| External data | FLEURS; Bible/GRN after a licence check | Use the Efik corpus and Commons/Wikitongues/Bloom now. Ask first about VoiceAfrica/NaijaVoices, Common Voice, YouTube CC-BY and GRN. Avoid Bible.is/YouVersion/JW/JESUS Film. TTS is a dead end |
| Teacher | MMS-LID-4017 | Same choice, but **run a zero-shot test first**: it learned from New Testament readings and drops to about 80% accuracy on telephone speech |
| Packaging and final phase | fp16; submit the top 1–3 | fp16 is fine (the leader's fp16 model runs at 0.005 s/clip). Save with transformers 4.57.6 and use a **new HF repo per submission**. Only **about 3 final submissions fit in the 8-hour window**, and the final model must share its architecture with a dev-tested one |

## Efik, Ibibio and Igala decide the competition

**In plain terms:** the score rewards getting the small languages right far more than shrinking the model. The leaders are already decent on the big languages and almost blind on Efik, Ibibio and Igala. That gap is where this is won, and nobody has closed it yet.

The final score is language-weighted F1 × (0.5 + 0.5 × efficiency). Yoruba, Igbo and Hausa count **1/23 (4.3%)** each. The other nine languages and `other` count **2/23 (8.7%)** each, so 87% of the F1 sits in the low-resource classes ([Codabench 18424](https://www.codabench.org/competitions/18424/)).

No one is ranked yet, because the ranking floor is a weighted F1 of 0.60. datapsalm leads with a final score of 0.539 (wF1 0.586) and ray_nacido follows at 0.467 (wF1 0.531).

A short calculation from their published per-class numbers shows where the points are:
- **datapsalm.** efi, ibb, igl and nup contribute only 0.082 of its 0.586. Its other nine classes therefore average about **0.77**.
- **ray_nacido.** The same arithmetic gives about 0.72, with Ibibio at **0.0**.
- **Headroom.** Match 0.77 on the nine and lift the four scarce classes to an average of 0.5, and the weighted F1 is about **0.68**. Lift them to 0.7 and it is about **0.75**. Lifting efi, ibb and igl alone from about 0.1 to 0.5 adds roughly 0.10, more than the whole gap to the ranking floor.

The likely reason for the near-zeros is linguistic, and our out-of-fold confusion matrix should confirm or refute it. Efik, Ibibio and Annang are closely related Lower Cross languages. Annang has **8.1 h from 23 speakers**, against 1.4 h for Efik and 1.8 h for Ibibio. Igala is Yoruboid, a relative of Yoruba, which has 9 h. A model that has heard few voices falls back on the better-resourced relative. Closely related languages were also the hardest pair in NIST LRE22 (Xhosa and Zulu) ([NIST LRE22](https://arxiv.org/abs/2302.14624)).

Two parameter counts suggest what the leaders run:
- ray_nacido's 94.6M parameters match a public full 12-layer AfriHuBERT fine-tune of 94,571,917 parameters ([HF](https://huggingface.co/EmmanuelHardey/afrihubert)).
- datapsalm's 87.6M matches a Whisper-small encoder (88.2M).

Both are our inferences from parameter counts. If right, the best Africa-pretrained backbone has not fixed Ibibio on its own.

The efficiency term is the average of three parts, each between 0 and 1:
- **Size:** the mean of 1 − params/350M and 1 − repo/1.5 GB.
- **Memory:** 1 − peak/4 GB.
- **Speed:** 1 at 10 ms per clip or faster, falling with the logarithm of latency to 0 at 0.5 s.

Our local `src/score.py` reproduces the organisers' worked example (0.76) and a leaderboard row exactly ([Codabench 18424](https://www.codabench.org/competitions/18424/)). The table below assumes about 1 GB of fixed memory plus fp32 weights, and fp16 repo files. The leader's 20M model used 1.19 GB, which fits that assumption.

| Model | Params | Multiplier at ≤10 ms | at 20 ms | at 50 ms |
|---|---|---|---|---|
| Whisper-base encoder | 20.6M | 0.948 | 0.918 | 0.879 |
| AfriHuBERT, 4 layers | 37.7M | **0.939** | 0.909 | 0.870 |
| AfriHuBERT, 6 layers | 51.8M | 0.932 | 0.902 | 0.863 |
| AfriHuBERT, 8 layers | 66.0M | 0.924 | 0.895 | 0.856 |
| AfriHuBERT, 12 layers | 94.4M | 0.910 | 0.880 | 0.841 |
| Large model, 316M | 316M | 0.795 | 0.766 | 0.727 |

Two conclusions follow:
- **Depth is cheap.** A 12-layer model needs only about 3% more relative F1 than a 4-layer one to pay for itself.
- **Latency is the hidden lever.** Going from 10 ms to 20 ms costs as much as eight extra layers.

The leader's own history confirms that F1 dominates. Moving from 20.2M to 87.6M parameters lowered its multiplier by 2.5% but raised its F1 by 6.7%, a net gain of about 4%.

### What a top professional would do that most competitors won't

A top professional treats this as a **speaker-generalisation problem**, not a model-selection problem.

**Before training.** They build the exact scorer, latency included, and a speaker-grouped validation that holds out each small-class speaker in turn. Alongside it they keep a "Hypa-only" F1, the one number that the recording channel cannot inflate.

**Fixing the data.** They destroy the corpus fingerprints in the training audio, because channel statistics alone already reach F1 0.34. They also manufacture voice diversity uniformly across all classes. The cautionary result: when voice conversion uses class-specific target voices, accuracy collapses to chance ([Abdullah et al.](https://arxiv.org/html/2505.24713v1)).

**Getting new voices.** They recruit real new speakers instead of adding hours from the same ones. The lesson comes from SIGTYP 2021: a model trained on mostly single-speaker Bible audio scored 0.96 F1 in-domain and **13.7%** on new speakers, and adding multi-speaker data lifted the best team to 53.1% ([SIGTYP 2021](https://arxiv.org/html/2106.03895v1)).

**Data sources.** They mine openly licensed media that keyword searches on Hugging Face miss, such as Wikimedia Nigeria's oral-history videos. Before using any grey-area source, they get **written clearance** from the organisers.

**Inside the model.** They write the decision rule into `classifier.bias`, so that a standard model makes F1-optimal choices.

**Reading competitors.** They study competitors' public configs. datapsalm's config revealed a 10-second Whisper window set purely through config ([config](https://huggingface.co/datapsalm/dsn2026-lid-w03/blob/main/config.json)).

**Submissions.** They spend each submission on one question. They pick the final by pooled cross-validation, not by small public-leaderboard gaps. Across 120 Kaggle competitions there was little evidence of overfitting to the public board, but small, speaker-sparse test sets are noisy ([Roelofs et al.](https://papers.nips.cc/paper/9117-a-meta-analysis-of-overfitting-in-machine-learning)).

## The training data teaches recording sessions and voices, not languages

**In plain terms:** in the provided data, each language mostly comes from one recording source with its own "sound". A model can therefore score well by recognising the microphone and the speaker. That trick fails on new speakers and on phone audio. The small languages have so few speakers that the given validation set cannot tell good models from lucky ones.

We decoded all **29,126 clips (92.5 h)** of the [DSN training data](https://huggingface.co/datasets/Data-Science-Nigeria/dsn-2026-lid-train). Source and label are tightly coupled:
- **One source each.** Urhobo comes only from AfriSpeech-IPA; Efik, Ibibio, Igala and Nupe come only from Hypa-Voices.
- **Partial coverage.** Omnilingual supplies only Eggon, Annang and Idoma, so recognising the omnilingual channel narrows the answer to three classes. Recognising the AfriSpeech channel narrows it to six.
- **Missing cells.** The training split contains no Hypa-channel Yoruba: all 90 such clips sit in validation.

Fifteen scalar channel statistics, with no speech content at all, predict a clip's source with **99.5%** accuracy and its label at **fixed-share F1 0.34**. That is four times chance; Annang reaches F1 0.87 from channel alone.

The fingerprints are concrete:
- **AfriSpeech silence.** 23.7% of its samples are exact digital zeros, the noise floor sits near **−92 dBFS**, and the median clip ends with 0.86 s of silence.
- **Tiv inside AfriSpeech.** Tiv clips from that corpus differ from its other languages on silence alone (2% zeros, −73 dBFS floor).
- **Omnilingual loudness.** Within omnilingual, loudness separates the languages: Idoma has a median of −42.8 dBFS against −26.6 for Annang.
- **No phone audio.** Only **0.7% of clips are narrowband**, so the test set's phone-quality condition has almost no counterpart in training.

Duration and source together give a weaker shortcut (F1 0.17) that random cropping removes. One good sign: inside Hypa, the channel says almost nothing about the scarce classes (channel-only F1 is 0.02 for Efik and 0.00 for Igala). The Hypa-only F1 is therefore our most honest number.

The speaker picture is starker:

| Class | Hours (train + val) | Real speakers | Source | Test hours |
|---|---|---|---|---|
| nup | 0.50 | **1** (SPK-147), **no validation clips** | Hypa only | 0.14 |
| igl | 0.60 | 3 | Hypa only | 0.15 |
| ibb | 2.05 | 3 (two hold 87% of hours) | Hypa only | 0.53 |
| efi | 1.57 | 6 ids, one holds 72% | Hypa only | 0.43 |
| tiv | 4.89 | 4 Hypa speakers + unknown AfriSpeech speakers | Mixed | 1.20 |
| urh | 10.79 | Unknown (no ids) | AfriSpeech only | 1.21 |

Test hours are from the dataset's hours report ([DSN card](https://huggingface.co/datasets/Data-Science-Nigeria/dsn-2026-lid-train)). With 0.14–0.15 h of Nupe and Igala test audio, each class's leaderboard F1 probably reflects one or two test speakers. One voice can therefore swing 8.7% of the score.

**62.5% of clips (all of AfriSpeech, 53 h) have no speaker id.** An acoustic nearest-neighbour test found no sign of speaker leakage across the organisers' AfriSpeech split, but it cannot prove the split is disjoint. Speaker ids behave in two different ways:
- **Omnilingual collisions.** The ids `spk01`–`spk15` repeat across Annang, Eggon and Idoma but belong to different people.
- **Real bilingual speakers.** 42 Hypa speakers recorded a target language and also `other`, which is mostly Nigerian-accented English. These are genuinely the same voice, which is useful because it forces the model to learn language rather than voice. It also means each such person must sit in a single fold.

Gaps in the numbering (omnilingual skips spk04, 06, 09, 11 and 18; Hypa uses 86 of 150 ids) point to the held-out test speakers.

The organisers' validation split is a poor judge:
- it holds one Igala speaker, one Ibibio speaker and effectively one Efik speaker (98% of Efik validation hours);
- it has no Nupe at all;
- it has no Hypa-channel Igbo, Hausa, Eggon, Annang or Idoma.

Housekeeping before training:
- **Duplicates.** Drop one clip from each of the **36 near-duplicate pairs** (re-encoded copies within a speaker).
- **Silent clip.** Drop clip `69cd0258ddc28532`.
- **Label noise.** Whisper flagged English in some Hypa target-language clips (Hausa 4/6, Nupe 3/6 in a small sample). Check them with MMS-LID, starting with SPK-147 (Nupe). A 20% mislabel rate in Nupe would directly cap that class.

### Three folds, one held-out speaker per small class

Pool train and validation, then group clips:
- Hypa clips by `speaker_id`, pooled across labels, so a person's target-language and `other` clips stay together;
- omnilingual clips by the raw `spkNN` string;
- AfriSpeech clips by pseudo-speaker clusters from a speaker-verification embedding.

Use **three folds**, not five. With five, two folds would contain no Igala or Ibibio speaker.

| Class | Fold A holds out | Fold B holds out | Fold C holds out |
|---|---|---|---|
| igl | SPK-059 (49 clips) | SPK-086 (73) | SPK-031 (97) |
| ibb | SPK-071 (107) | SPK-004 (329) | SPK-104 (278) |
| efi | SPK-011 (363) | SPK-040 + SPK-052 (83) | SPK-140 + SPK-065 + SPK-129 (101) |
| tiv (Hypa) | SPK-120 + SPK-041 (89) | SPK-061 (184) | SPK-096 (159) |
| nup | never held out: SPK-147 trains in every fold | | |

Reading the folds correctly matters as much as building them:
- Compute F1 on the **pooled out-of-fold predictions**, never as an average of per-fold scores.
- Report recall per held-out speaker.
- For Nupe, report only its precision, which is measured fairly on every other class's held-out speakers.
- The first honest Nupe recall will come from external speakers: held-out voice-note speakers, or VoiceAfrica if it is cleared.

### Start from the raw material, not the summaries

The rawest useful material is already in hand:
- **Per-clip audit table.** Speaker, source, duration, channel features, hashes and near-duplicate links: `/tmp/claude-0/-home-user/a55f94d8-bdde-575d-8d9b-69b0f597d9e3/scratchpad/audit/metadata.parquet`. Copy the fold assignments as CSV into the private team repo, since the scratchpad is not permanent.
- **Listening.** Ten clips per Hypa (speaker, label) pair, plus the shortest and longest clips.
- **The leaderboard.** Its per-class F1 and its clean, noisy and phone accuracy columns.
- **Organiser tools.** The checker `check_model_folder.py` and our verified scorer.
- **Test hours** per class from the dataset's hours report.
- **Competitors' public configs.**
- **The teacher's view.** MMS-LID's zero-shot opinion of every training clip.
- **People.** The native speakers on and around the team, the one source of truth no model can replace.

## New voices beat more hours, and the legitimate ones are small and scattered

**In plain terms:** what the small languages need most is new people speaking them, not more minutes from the same few people. Some safe sources exist today. The biggest one (VoiceAfrica) needs an organiser OK first. Anything that might contain the test recordings, such as Bible-app audio or anything from the Hypa family, must be avoided entirely.

The evidence is consistent across challenges:
- **SIGTYP 2021.** Adding multi-speaker data quadrupled accuracy on new speakers ([SIGTYP 2021](https://arxiv.org/html/2106.03895v1)).
- **NIST LRE22.** A backend that never saw target-domain data was **5× worse** ([ABC LRE22](https://www.isca-archive.org/interspeech_2023/matejka23_interspeech.pdf)).

### Sources by class

| Class | Best sources | Status |
|---|---|---|
| **nup** | VoiceAfrica: **12.6 h, 10 speakers**, spontaneous ([VoiceAfrica](https://huggingface.co/datasets/naijavoices/voiceafrica-datasets)) | Download works; ask the organisers first |
| | NigBench-MAMAI: **50 h** from rural women, CC BY 4.0 ([NigBench](https://huggingface.co/datasets/intronhealth/NigBench-MAMAI-Speech-QA)) | Gate approval still pending |
| | NaijaVoices Nupe set on the Mozilla Data Collective: 7–8 speakers, three accents ([MDC](https://mozilladatacollective.com/datasets/cmihkoth10246md07dtojxehg)) | Ask the organisers first |
| **igl** | VoiceAfrica: 12.1 h, 10 speakers | Ask the organisers first |
| | Inikpi Tv: about 8.3 h of multi-actor Igala films, marked CC-BY by the uploader ([example](https://www.youtube.com/watch?v=iVCPAMhX2Oo)) | Ask the organisers first |
| **efi** | Efik corpus: **8.79 h**, 2 speakers, CC BY 4.0, ungated ([HF](https://huggingface.co/datasets/offiongbassey/efik_speech_dataset)) | Use now |
| **ibb** | Wikimedia Nigeria: 4 videos ([gallery](https://meta.wikimedia.org/wiki/Nigerian_Language_Oral_History_Documentation_Project/Gallery)) | Use now |
| | Wikitongues: 3 speakers ([Wikitongues](https://wikitongues.org/languages/ibb/)) | Use now |
| | Lingua Libre: 7 word-level speakers | Use now (low value) |
| | Common Voice Ibibio: 18 h, 19 speakers ([MDC](https://datacollective.mozillafoundation.org/datasets/cmn2cp74s01hamm07s74uypfk)) | Real overlap risk with Hypa-Voices |
| **tiv** | Small CC-BY sets: one Bloom talking book and a few YouTube CC-BY radio/talk videos | Tiny |
| | Voice notes | The real hope |
| **urh** | Wikimedia Nigeria: about 10 Urhobo-clan videos. These add new *channels*, which Urhobo lacks because it is 100% AfriSpeech | Use now; a native speaker should confirm the clan labels |
| | VoiceAfrica `urh` | Ask the organisers first |
| **anw, ego, idu** | Almost nothing beyond religious recordings: two Wikimedia Annang files and one Bloom Eggon book | Little value |
| **other** | Nigerian-accented English and Pidgin: NaijaS2ST, nigerian-pidgin-1.0, WAXAL `pcm_tts` | Use now |
| | FLEURS: the exact non-Nigerian languages in Hypa-Voices (en, fr, es, pt, ar, sw) | Use now |
| | Hard negatives: Lower Cross neighbours (Oro, Ibeno, Eket), Itsekiri and about 200 other Nigerian lects from Wikimedia Nigeria | Use now |

### The team's own voice notes

The team's consented WhatsApp voice notes are the highest-value source of all. They add new speakers on a phone channel that matches the test, and nobody else has them. Collection guidance:
- **Speakers before minutes.** At least 10 speakers per priority class, 2–5 minutes each, mostly spontaneous prompts.
- **Same voice, two languages.** Have each speaker also record one English or Pidgin clip for `other`, so the model learns language rather than voice.

### Keep new channels from becoming new shortcuts

New data can create new shortcuts. The Efik corpus is studio-quiet read speech from two people. Added only to `efi`, it teaches "quiet studio voice = Efik", the same failure as class-specific voice conversion. Two rules prevent this:
- Add external data to several classes at once (FLEURS, NaijaS2ST and FLEURS-other cover the big ones and `other`).
- Pass every external clip through the same channel augmentation.

The Hypa-Voices audio was decoded from the SNAC neural codec ([Hypa-Voices card](https://huggingface.co/datasets/hypaai/Hypa-Voices)), so the scarce-class test clips carry that codec's sound. Passing external Efik, Igala and Nupe audio through SNAC is a cheap way to match the test channel. This is **an untested idea of ours**: SNAC's licence is unchecked, and it would have to be disclosed as a generator model.

### What to avoid, and why

The avoid list is driven by provenance:
- **Hypa family.** `hypaai/Hypa-Speech-10k` comes from the same organisation, covers all twelve target codes plus Pidgin, and was built from Common Voice 17 ([card](https://huggingface.co/datasets/hypaai/Hypa-Speech-10k)). It is the most dangerous sibling.
- **AfriSpeech's Bible-app sets.** AfriSpeech's 4.59 h of YouVersion Tiv was created the same day as the upstream of the prohibited corpus and nearly matches its 4.03 h of Tiv ([youversion](https://huggingface.co/datasets/AfriSpeech/youversion-african-speech)). Bible.is, FCBH and YouVersion Tiv audio should be treated as banned. FCBH's licence also forbids offline download ([FCBH licence](https://www.faithcomesbyhearing.com/bible-brain/license)).
- **JESUS Film.** Its terms forbid derivative works ([terms](https://www.jesusfilm.org/terms/)).
- **Synthetic speech.** MMS-TTS has no voice for 10 of the 12 targets. OmniVoice, the one multilingual TTS covering Annang, Eggon, Ibibio and Idoma, lists the prohibited Omnilingual corpus in its training data ([OmniVoice](https://arxiv.org/html/2604.00688v1)).

### Data to use

**USE** means low risk: go ahead and disclose it in the model card. **ASK ORGANISERS FIRST** means download it if useful, but do not train on it until a written yes arrives. **AVOID** means do not use it.

| Source | Helps | Size / speakers | Licence | Verdict | Reason |
|---|---|---|---|---|---|
| DSN train + validation, pooled | all | 92.5 h | DSN terms | **USE** | Drop 36 near-duplicates and 1 silent clip. Move the 90 Hypa Yoruba validation clips into the final fit |
| [Efik speech dataset](https://huggingface.co/datasets/offiongbassey/efik_speech_dataset) | efi | 8.79 h, 2 speakers | CC BY 4.0 | **USE** | Independent Interspeech 2026 corpus. Skip its unlicensed sibling `efik_audio_dataset` |
| Team voice notes | nup, igl, ibb, efi, tiv; anw, ego, idu, urh; other | What you collect | Signed consent | **USE** (confirm Q5) | New speakers on the phone channel; the only honest Nupe test |
| [Wikimedia Nigeria oral-history videos](https://meta.wikimedia.org/wiki/Nigerian_Language_Oral_History_Documentation_Project/Gallery) | ibb, anw, efi, urh clans; other | About 18 target speakers, 3–8 min each, about 230 files | CC BY-SA 4.0 ([example](https://commons.wikimedia.org/wiki/File:The_audiovisual_documentary_of_Efik_language_by_its_native_speaker.webm)) | **USE** (confirm Q6b) | Published openly by the rights holder. Strip English segments. A native speaker should verify the Urhobo clan labels. Label Oro, Ibeno and Eket as `other` or leave them out |
| [Wikitongues](https://wikitongues.org/languages/ibb/) | ibb ×3, anw, pcm ×4 | 1 video per speaker | CC BY-SA 4.0 | **USE** | New Ibibio voices |
| [Bloom Library](https://server.bloomlibrary.org/parse/classes/books) talking books | ego 1, tiv 1 | Minutes | CC BY | **USE** | A rare Eggon narrator |
| Lingua Libre (Commons) | ibb (7 speakers), idu, urh | Single words | CC BY-SA | **USE**, low value | Clips under 2 s; concatenate them per speaker |
| [FLEURS](https://huggingface.co/datasets/google/fleurs) | yo, ig, ha; en, fr, es, pt, ar, sw and more for other | 10–14 h per language | CC BY 4.0 | **USE** | Covers Hypa-Voices' non-Nigerian languages, speaker-disjoint |
| [NaijaS2ST](https://huggingface.co/datasets/McGill-NLP/NaijaS2ST) | Nigerian English and pcm for other; yo, ig, ha | Very large; take slices | CC BY 4.0 | **USE** | Teaches that Nigerian-accented English is `other` |
| [nigerian-pidgin-1.0](https://huggingface.co/datasets/asr-nigerian-pidgin/nigerian-pidgin-1.0) | pcm (other) | 4,277 utterances, 10 speakers, recorded on Android | CC BY 4.0 | **USE** | |
| [WAXAL](https://huggingface.co/datasets/google/WaxalNLP) | `pcm_tts`; non-target West African languages | Take shards | CC BY / CC BY-SA | **USE** | |
| [VoxLingua107](https://huggingface.co/datasets/TalTechNLP/voxlingua107_wds) | yo, ha; broad other | Take a few shards | CC BY 4.0 | **USE** | Real-world YouTube channels |
| [Central Kanuri](https://huggingface.co/datasets/ciatech-africa/central-kanuri-speech-dataset), TWB Kanuri | other (a Nigerian language) | 12 h, 24 contributors | CC BY-NC 4.0 | **USE** (NC) | |
| [SautiDB-Naija](https://arxiv.org/abs/2112.06199), [Speech Accent Archive](https://accent.gmu.edu/places/nigeria/) | other | About 1 h; 28 paragraphs | CC (variant to check) / BY-NC-SA | **USE** | English spoken by Efik-Ibibio, Igala and Ibibio first-language speakers: hard negatives |
| MUSAN, OpenSLR 26/28 RIRs, BUT ReverbDB | Augmentation | | CC BY / Apache 2.0 | **USE** | Skip the non-commercial noise sets (WHAM!, ESC-50) |
| [NigBench-MAMAI Nupe](https://huggingface.co/datasets/intronhealth/NigBench-MAMAI-Speech-QA) | nup | 50 h, noisy rural speech | CC BY 4.0, manual gate | **USE once approved**, flagged in Q9 | Intron's own collection, with no link to the prohibited corpora (see below) |
| Intron accented-English sets (afrispeech-dialog, -countries, -parliament, med-convo-nig) | other | 4–68 h | CC BY-NC-SA | **USE**, low priority | Not the prohibited publisher |
| DSN voice-of-care Hausa; Gamayun Hausa (recorded on phones) | hau | 52 h; 6.7 h | CC BY 4.0; BY-NC | **USE**, low priority | Hausa is only 4.3% of the score |
| [VoiceAfrica](https://huggingface.co/datasets/naijavoices/voiceafrica-datasets) | igl, nup, urh (+ knc and others for other) | About 12 h and 10 speakers per language | CC BY-NC-SA + [NaijaVoices T&C](https://naijavoices.com/terms-and-conditions) | **ASK FIRST** (Q4) | The biggest single win. Download now; train only after a yes |
| NaijaVoices Nupe on MDC; `UmarBaba1/nupe_speech_corpus` | nup | 2.7–3.3 h, 7–8 speakers | CC BY-NC-SA (voice cloning banned); CC BY-NC | **ASK FIRST** (Q4) | Same collector and family. Never use these voices as voice-conversion targets |
| naijavoices-dataset; African Next Voices uploads | yo, ig, ha, pcm | Huge | NC-SA / not stated | **ASK FIRST**, low priority | DSN co-leads African Next Voices |
| Common Voice via MDC, for any language Hypa-Voices covers (ibb, yo, ig, ha, pcm, en…) | ibb 18 h, 19 speakers; others | | CC0 + MDC terms | **ASK FIRST** (Q3) | Hypa-Voices includes "Common Voice-derived subsets" |
| YouTube CC-BY channels (Inikpi Tv, King Omo Urhobo, KRACHOUSE, Akaa a Tiv) | igl, urh, ibb/efi, tiv | 8.3 h; 2.0 h; 5.1 h; 1.4 h | CC BY declared by uploader | **ASK FIRST** (Q6a) | The best multi-speaker Igala anywhere. But the licence is self-declared and YouTube's terms restrict downloading |
| [GRN](https://globalrecordings.net/en/copyright) / MMS-ulab v2 | All nine targets | Minutes per language, with music | CC BY-NC-SA | **ASK FIRST** (Q7), low priority | Religious recordings by single narrators |
| Intron `afrispeech-200`, AfriVox v1/v2, AfriSwitch | other; yo/ig/ha/pcm | | NC-SA / CC BY | **ASK FIRST** | Hypa re-hosted afrispeech-200; AfriVox bundles Common Voice, BibleTTS and NaijaVoices |
| The three prohibited corpora and every mirror or derivative: Hypa-Voices-snac, the Hypa benchmarks, Hypa-Speech-10k and its mirror, Hypa_Fleurs, Hypa-AfriSpeech, `okezieowen/afrispeech_*`, the omnilingual mirrors and subsets, `african-speech-public_v1`, `afri2universal` | | | | **AVOID** | The test set was drawn from these |
| AfriSpeech-org siblings: youversion, open-bible-speech-african, grn-african-speech, africa-female-speech, african-corpus-jw | | | | **AVOID** | Same org and scraper; the YouVersion Tiv is likely the prohibited Tiv |
| Bible.is / FCBH / YouVersion / DBP, API.Bible, JW.org, JESUS Film (including Internet Archive copies), MegaVoice | | | | **AVOID** | Terms forbid download, derivative works or AI training. JW audio is the likely upstream of AfriSpeech-IPA |
| OmniVoice, Hypa-Orpheus or MMS-TTS synthetic speech | | | | **AVOID** | Trained on prohibited corpora, or a single-voice shortcut |
| scubavoice/williamudo Ibibio-Efik sample; Common Voice mirrors on HF; Africanvoice/WaZoBiaSpeech; Kaggle AASD; jameskaile Igbo; mrheartng Idoma; BibleTTS | | | | **AVOID** | Evaluation-only terms, re-hosting bans, unknown provenance or Bible-overlap risk |
| Mislabelled traps: "igala/tiv/idoma" accent sets (these are English), TLA "ibb" (actually Isubu, from Cameroon), Bloom "Nupe" (actually Javanese) | | | | **AVOID** | Wrong language |

### How the conflicting verdicts were reconciled

**Intron Health datasets.** One note banned every `intronhealth` dataset as "AfriSpeech brand", to be conservative. The provenance note found that the prohibited `AfriSpeech/african-speech-ipa` sits in an HF organisation with **one member, Mich-Seth Owusu** ([members](https://huggingface.co/api/organizations/AfriSpeech/members)). `intronhealth` has **five different members** (Tobi Olatunji and others) ([members](https://huggingface.co/api/organizations/intronhealth/members)). Intron collects data through different pipelines, mostly crowd-sourced accented English and in-the-wild conversation, and no shared recordings were found.

The ban is defined by corpus and derivative, not by brand name. Intron's data is therefore **not banned by association**, and NigBench-MAMAI and the accented-English sets are usable. There is one specific exception. Hypa re-hosted `afrispeech-200` as `hypaai/Hypa-AfriSpeech`, and Hypa-Voices' English portion could contain it, which would put it inside the test set's `other` class. That set is ASK FIRST for that reason, not for its name. The AfriVox aggregates are ASK FIRST because they bundle Common Voice, BibleTTS and NaijaVoices ([afrivox-v2](https://huggingface.co/datasets/intronhealth/afrivox-v2)). Question 9 of the organiser message puts this reasoning on record.

**GRN (Global Recordings Network).** One note rated GRN low-risk; another said "ask", because the prohibited corpus's publisher also scrapes GRN. The sizes argue against overlap:
- the prohibited upstream's 25.67 h of Urhobo dwarfs GRN's 46 minutes;
- its Tiv matches YouVersion, not GRN.

So the risk of overlap is low. The value is low too: minutes of single narrators with music, the kind of Bible-style data that SIGTYP showed generalises poorly. Since the licence is NC-SA and the gain is small, it is "ask, low priority". AfriHuBERT already heard GRN audio during pretraining, through MMS-ulab v2 ([AfriHuBERT](https://huggingface.co/ajesujoba/AfriHuBERT)).

**VoiceAfrica and NaijaVoices.** One note rated these low-to-medium risk; another rated the NaijaVoices Nupe set medium-to-high, because NaijaVoices supplied Igala, Nupe and Urhobo to Meta's omniASR training ([arXiv 2511.09690](https://arxiv.org/html/2511.09690)). But the prohibited item is the **Omnilingual corpus**, not omniASR's full training mix. That corpus has no Igala, Nupe or Urhobo configs ([corpus card](https://huggingface.co/datasets/facebook/omnilingual-asr-corpus)). The Igala and Nupe test audio must come from Hypa-Voices. Overlap with the test is therefore unlikely. The reasons to ask are the NC-SA licence, DSN's co-leadership of African Next Voices, and the organisers' right to rule.

**Common Voice.** One note rated Common Voice for Yoruba, Igbo, Hausa and Pidgin low-risk; another rated Common Voice Ibibio high-risk. Hypa-Voices states that it contains Common Voice-derived subsets, and Hypa-Speech-10k is built from Common Voice 17. Any language that Hypa covers is therefore exposed, **English and Pidgin included**. FLEURS supplies the same languages without that link.

**If no answer arrives in time:** train the final model only on USE sources. Keep a parallel variant that adds the ASK sources, so it can be switched in within one retrain the moment a yes arrives.

## AfriHuBERT is the backbone; MMS-LID is the teacher only if it passes a test

**In plain terms:** start from AfriHuBERT, the one small model that has already "heard" almost all of these languages. Shrink it to four layers so it stays fast. Use Meta's large MMS-LID model, which knows all twelve languages, as a tutor, but only after checking that it is actually right on our clips.

**AfriHuBERT** is a continued pretraining of mHuBERT-147 on more than 10,000 hours in 1,226 African languages. Its language metadata lists **11 of the 12 targets** (all except Efik). On FLEURS African language ID it scores F1 **91.6**, against 88.0 for mHuBERT-147 ([AfriHuBERT paper](https://arxiv.org/html/2409.20201)).

Whisper's language set has Yoruba and Hausa but **no Igbo** and no minority target ([whisper-base config](https://huggingface.co/openai/whisper-base/blob/main/generation_config.json)). That fits the Whisper-based leader's near-zeros.

Truncation is supported by the literature:
- Pruning up to two-thirds of a 24-block wav2vec 2.0 encoder (keeping 8 blocks) maintained good language-ID accuracy ([Tjandra et al.](https://arxiv.org/abs/2107.04082)).
- A 9-of-18-layer sub-model beat the full model ([Bartley et al.](https://arxiv.org/abs/2211.05103)).

A 4-layer AfriHuBERT (about 37.7M parameters) is therefore a sound first rung. Set `num_hidden_layers` and `use_weighted_layer_sum` when loading the pretrained checkpoint.

**MMS-LID-4017** (970M parameters) is the only off-the-shelf language-ID model with classes for **all 12 targets plus Pidgin and English** ([config](https://huggingface.co/facebook/mms-lid-4017/resolve/main/config.json)). It reaches 97% on FLEURS but **80.5% on out-of-domain telephone speech**. It was trained largely on single-speaker New Testament readings ([MMS paper](https://arxiv.org/abs/2305.13516)). No accuracy on these nine minority languages has been published, so measure its zero-shot per-class F1 on our folds before distilling from it.

| Model | Role | Size used | Licence | Evidence | Verdict |
|---|---|---|---|---|---|
| [AfriHuBERT](https://huggingface.co/ajesujoba/AfriHuBERT) (-n) | Student backbone | 4 layers ≈37.7M → 6 ≈51.8M → 8 ≈66.0M | "cc"; inherits CC BY-NC-SA from mHuBERT-147 | 11/12 targets in pretraining; SLID F1 91.6 | **USE** (ask Q1 about the NC-SA licence) |
| AfriHuBERT-s | Alternative backbone | Same | "cc", unspecified | Best small model on FLEURS-African SLID: **93.2** | Parallel cross-validation check at rung 1, not a separate rung |
| [AfriHuBERT-large](https://huggingface.co/ajesujoba/AfriHuBERT-large); w2v-BERT 2.0 | Top rung only | 6 layers ≈88.7M; 4–6 layers ≈97–145M | Apache-2.0 label; MIT | No LID numbers (large); 92.7 at full depth (w2v-BERT) | **Only on your explicit go** |
| Whisper-base encoder | What the leader started with | 20.6M | Apache-2.0 | No Igbo or minority-language exposure | Skip |
| MMS-LID-4017 (or -1024) | Teacher, label auditor, filter for external data | 970M, never submitted | CC BY-NC 4.0 | All 14 codes; 80.5% on telephone speech | **USE after a zero-shot test** (Q2) |
| [kNN-VC](https://github.com/bshall/knn-vc) (WavLM-Large + HiFi-GAN) | Voice-conversion generator | Not submitted | MIT | +34% relative in dialect ID | **USE**; disclose in the model card |
| omniASR family, AfriSpeech `african-speech-id`, Hypa-Whisper / Orpheus, OmniVoice, Sahara | — | — | — | Trained on prohibited or undisclosed data | **AVOID** |
| AfriHuBERTo | — | — | CC BY-NC-ND | Derivatives cannot be shared | **AVOID** |

AfriHuBERT has two licence caveats:
- It inherits mHuBERT-147's non-commercial share-alike licence, so our public model must carry **CC BY-NC-SA 4.0**. If the organisers reject NC licences, the fallbacks are AfriHuBERT-large (labelled Apache-2.0, although its data includes NC-SA sources), w2v-BERT 2.0 (MIT) or Whisper (Apache-2.0).
- Its pretraining used Common Voice and religious audio. This is unlabelled pretraining only, but it is worth one line in the model card.

## Five techniques give a real edge over class weights and noise

**In plain terms:** most teams will add class weights and some background noise. The edge comes from five things:
1. erasing each corpus's "sound";
2. making every language heard through many voices;
3. writing the right decision rule into the model;
4. borrowing a tutor's knowledge;
5. treating `other` as a real class with tricky look-alikes.

**1. Destroy the corpus fingerprints first.** Apply these to every class:
- trim leading and trailing silence;
- add a random noise floor of −75 to −45 dBFS, so the AfriSpeech digital zeros vanish;
- randomise gain by ±15 dB on top of loudness normalisation;
- train on random 2–8 s crops and infer on the full clip.

Then add the test conditions the data lacks:
- a phone channel on about a quarter of samples: band-limit to 300–3,400 Hz, then apply a GSM, G.711, Opus (6–12 kbps) or MP3 (32 kbps) codec;
- room echo (p≈0.3);
- MUSAN noise at 0–15 dB SNR (p≈0.5);
- random EQ.

Augmentation of this kind cut x-vector LID error (Cavg) from 0.166 to 0.140 ([Snyder et al.](https://danielpovey.com/files/2018_odyssey_xvector_lid.pdf)), and the ML-SUPERB 2.0 winner used noise and reverb at p=0.5 each ([TalTech](https://arxiv.org/html/2506.01458v1)). The success test is ours: a channel-statistics classifier retrained on augmented audio should fall towards chance, while the Hypa-only F1 holds.

**2. Add voices uniformly across every class.** Apply speed (0.9–1.1), formant scaling and F0 scaling (×0.7–1.4), which keeps relative tone contours, together with random EQ, to all classes with p≈0.5–0.75 ([Yang et al.](https://arxiv.org/html/2306.06524v1)). Then add kNN-VC conversion to a shared pool of 8–20 class-independent target voices from CC BY corpora. In dialect ID, four such voices raised cross-domain accuracy from **60.2% to 80.7%**, while class-specific voices collapsed it to **24–27%** ([Abdullah et al.](https://arxiv.org/html/2505.24713v1)).

How to apply it:
- Convert scarce-class clips 4–8 times and other classes 1–2 times, so that every class contains converted audio.
- Keep all originals.
- Check for leakage with a probe that predicts "is converted".
- Never use NaijaVoices voices as targets (its terms ban voice cloning), and never use volunteers' voices as targets (the consent form does not cover it).

Cross-language voice conversion is untested for language ID, which is why this is a gated rung and not a default.

**3. Move the decision rule into the classifier bias.** Replace class-weighted cross-entropy with a logit-adjusted loss (τ=1, equivalent to Balanced Softmax). On CIFAR-100-LT, Balanced Softmax scored **50.3** against 45.3 for plain cross-entropy and 42.3 for class weighting ([Ren et al.](https://ar5iv.labs.arxiv.org/html/2007.10740)). If a class-balanced sampler is also used, set τ≈0 so the correction is not applied twice ([Menon et al.](https://arxiv.org/abs/2007.07314)).

Then tune the bias:
1. Search per-class offsets by coordinate ascent on pooled out-of-fold logits, scored against the exact fixed-share F1. Regularise by averaging over speaker bootstrap samples.
2. Leave Nupe's offset at its loss-adjusted value, because its recall cannot be measured.
3. Add the offsets to `classifier.bias`. The model stays a standard HF model, and the change costs nothing at inference.
4. Re-tune after any weight averaging or fp16 cast.

The search will concentrate on the Lower Cross triplet (anw, ibb, efi) and on Igala versus Yoruba.

**4. Borrow a teacher's ears, then distil an ensemble.**
1. Restrict MMS-LID's 4,017 outputs to the 12 targets plus `other` (the logsumexp of the rest, or of only English and Pidgin; keep whichever scores better).
2. Correct its priors with the same bias search.
3. If it beats the student on the scarce classes, distil from it: temperature ≈2, weight α≈0.5–0.7, teacher logits cached on fixed windows so the student sees exactly what the teacher scored.

The fold models, optionally together with MMS-LID, form an ensemble teacher. An ensemble cannot be submitted, but it can be distilled into one model.

**5. Make `other` a real class with hard negatives.** Outlier exposure gives the class a diverse mix:
- English, Pidgin, French, Arabic and Swahili;
- Nigerian-accented English;
- close-but-different Nigerian lects: Lower Cross neighbours, Itsekiri and Kanuri.

The 42 bilingual Hypa speakers already teach "same voice, different language". Tune the `other` bias jointly with the rest. One gap remains: Edoid languages and Ijaw appear only in banned AfriSpeech data.

**Skip under this deadline:**
- TTS (no voices exist for most targets);
- a speaker-adversarial head (with one Nupe speaker, speaker equals label);
- cross-class mixup;
- continued pretraining;
- head pruning (unsupported in HF);
- attentive-pooling heads, which `AutoModelForAudioClassification` cannot load ([transformers source](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/wav2vec2/modeling_wav2vec2.py)).

**Finish:**
- Average the last checkpoints; model soups improve on the best single fine-tune at no inference cost ([Wortsman et al.](https://arxiv.org/abs/2203.05482)).
- Optionally refit the classifier as LDA in closed form, which still folds into the linear head. A Gaussian backend beat direct softmax out of domain, Cavg 0.149 against 0.256 ([Snyder et al.](https://danielpovey.com/files/2018_odyssey_xvector_lid.pdf)).
- Export in fp16.

## Twelve packaging mistakes that would sink a submission

**In plain terms:** the organisers download and run our model themselves. One stray file, a private repo or a slow load means a zero, however good the model is. Each item below takes minutes to prevent.

| Mistake | What happens | Prevention |
|---|---|---|
| Stray files in the repo: `training_args.bin`, `optimizer.pt`, `generation_config.json`, Whisper tokenizer/vocab files, `checkpoint-*/`, notebooks, `.py` | `check_model_folder.py` fails: not scored | Push only `config.json`, `preprocessor_config.json`, `model.safetensors` and `README.md`. Run the checker on the exact folder you push |
| No `preprocessor_config.json` (Trainer omits it unless given the feature extractor) | Fails | `feature_extractor.save_pretrained(out)`, copied from the base model's settings |
| bf16 weights | Fails: only F32/F16 are allowed | Cast before saving |
| Custom heads or `auto_map` (attentive pooling, x-vector heads, adversary heads) | Fails or cannot load | Use stock `HubertForSequenceClassification`; strip training-only heads before `save_pretrained` |
| Repo still private at upload, or pushed to after submission | Organisers cannot download it; breaks the "unchanged" rule; exposes stronger weights early | **One new HF repo per submission**, private until the minute of upload, never touched again. Pass the 40-character hash to `make_submission.py` |
| Re-truncating a fine-tuned model that uses `use_weighted_layer_sum` | `layer_weights` size mismatch ([source](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/wav2vec2/modeling_wav2vec2.py)) | Set `num_hidden_layers` when loading the pretrained base |
| Version drift: the organisers' image may be Python 3.9, which caps transformers at 4.57.6 ([PyPI](https://pypi.org/pypi/transformers/json)) | Files saved only by v5 may not load | Train and save with 4.57.6. Test an offline load in `python:3.9` and in the latest transformers |
| fp16 on a CPU: transformers v5 loads stored dtype by default ([v5 source](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/modeling_utils.py)), and fp16 ran about 20× slower on CPU in our local test | Could break the 0.5 s/clip limit | The leader's fp16 model ran at 0.005 s/clip, which suggests GPU scoring. Confirm with D1's latency column and Q11 |
| Latency above 10 ms per clip | The speed term falls to 0.82 at 20 ms and 0.59 at 50 ms | Measure on a GPU with 2–30 s clips; weigh depth against measured latency |
| Whisper's fixed 30-s window ([source](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/whisper/modeling_whisper.py)) | Error unless the frame count matches `max_source_positions`; long clips truncated; padding dominates the average | Only relevant if Whisper is used: keep `max_source_positions` and `chunk_length` consistent |
| Group-norm front end (AfriHuBERT) with batched padding | Outputs shift slightly when clips are batched with padding (0.007–0.018 logit in our local test) | Keep the base preprocessor setting (no attention mask); include padded batches in training |
| Model card missing any dataset, teacher or generator | Disqualification | Checklist: every dataset with its licence; MMS-LID; kNN-VC (WavLM-Large, HiFi-GAN); noise corpora; "volunteer recordings: N speakers per language"; licence field CC BY-NC-SA 4.0 |

Keeping the recipe private fits these rules. The model card lists datasets, teacher and generator models and gives a one-line approach. Hyperparameters, sampling, augmentation settings and bias values stay off the card. A public repo still reveals the config and weights, exactly as datapsalm's revealed its 10-second window. That is the practical reason to keep the strongest version for the final phase.

## Outside evidence, not our own cross-validation, decides whether we are right

**In plain terms:** our own tests can fool us. The proof comes from things we do not control: the organisers' leaderboard columns, their checker, their written answers, native speakers, and recordings from people the model has never heard.

| Evidence | What it proves | When |
|---|---|---|
| Leaderboard per-class F1 and clean/noisy/phone accuracy within about 0.03 of pooled cross-validation | Our validation predicts the hidden speakers, so the private split should behave the same | After D1, then every submission |
| Leaderboard params, memory and latency against our local measurement | The efficiency estimate and the fp16 decision are right | D1 |
| `check_model_folder.py` PASS, plus an offline load in a clean Python 3.9 / transformers 4.57.6 environment | It will not be rejected | Every submission |
| Written organiser answers | Every external source is legitimate | Before any ASK source is used |
| MMS-LID agreement with training labels; native speakers spot-check confusions, voice notes and the Urhobo clan videos | The labels are right | Tue (MMS), Wed (native speakers) |
| Recall on held-out voice-note speakers (phone channel, never heard) | The first true speaker-disjoint Nupe and Igala estimate | Wed afternoon |
| Weighted F1 above 0.60 and above datapsalm's 0.586, with efi, ibb and igl above about 0.4 | We are competitive where it matters | D2–D6 |
| The private leaderboard | The result | Thu 22 Oct |

Treat leaderboard per-class F1 for Nupe and Igala as single-speaker measurements. With about 0.14 h of test audio each, they are too noisy to tune against. Trust the pooled cross-validation for decisions, and use the leaderboard to confirm calibration.

## Six dev submissions, one question each, then three final shots

**In plain terms:** each submission tests exactly one improvement, and only if our own speaker-grouped test says it helped first. The public score climbs step by step. The strongest model is trained last on all the data and submitted only in the final phase, using an architecture we have already seen run on the organisers' machines.

Times are UTC with Nigeria time (WAT = UTC+1) in brackets. The three cross-validation folds run in parallel on the three official members' **own** Kaggle accounts. The full-data fit then runs on whichever account frees up first. This container (4 CPUs, no GPU) does the data preparation, scoring and packaging. The plan assumes the 3-per-day limit resets at 00:00 UTC (unconfirmed), so **Tuesday's third slot must be used before 23:59 UTC (00:59 WAT)**. Unused slots are lost.

| When, UTC (WAT) | Rung | Work | Gate before moving on | Submission and the question it answers |
|---|---|---|---|---|
| Tue 10:00–11:00 (11:00–12:00) | Prep | Send the organiser email. Start downloads of USE sources; download VoiceAfrica but quarantine it. Kaggle phone verification. Voice-note call goes out | — | — |
| Tue 10:00–12:30 (11:00–13:30) | Prep | Drop duplicates and the silent clip. Build groups, including AfriSpeech pseudo-speaker clusters. Build the 3 rotation folds, the Hypa-only metric, and noisy and phone copies of each fold | The folds reproduce the rotation table | — |
| Tue 10:30–12:30 (11:30–13:30), GPU | Audit | MMS-LID-4017 zero-shot on train + validation: per-class F1, and flags on Hypa clips that sound English (SPK-147, SPK-132, SPK-024) | — | — |
| Tue 12:30–13:30 (13:30–14:30) | **0** Smoke test | AfriHuBERT 4 layers on about 200 clips per class: train, save fp16, checker, offline load, local score, latency and memory harness | Everything runs end to end | None |
| Tue 13:30–16:15 (14:30–17:15) | **1** Small baseline | AfriHuBERT 4 layers, provided data only, speaker-balanced sampling, 3–6 s crops. Optionally, a teammate runs AfriHuBERT-s in the same cross-validation | Cross-validation computed; inside every limit | **D1 ≈16:30 (17:30):** does local CV match the leaderboard within 0.03 per class and per condition? What latency and memory do the organisers measure for a 38M fp16 HuBERT? |
| Tue 16:30–20:15 (17:30–21:15) | **2** Loss and bias | Logit-adjusted loss, then the out-of-fold bias search with speaker bootstrap, written into `classifier.bias` | CV final score improves | **D2 ≈20:30 (21:30):** do the tuned biases transfer to unseen test speakers, so that efi, ibb and igl rise as predicted without Yoruba, Igbo and Hausa collapsing? |
| Tue 20:30–23:30 (21:30–00:30) | **3** Robustness | Shortcut removal, phone, noise and reverb channel, speed, formant and EQ, uniform across classes. Overnight on a spare account: kNN-VC conversion and caching of teacher logits | CV improves, and the local noisy/phone copies improve | **D3 by 23:45 (00:45 Wed):** do the noisy and phone accuracy columns rise while clean accuracy holds? |
| Wed 00:00–07:30 (01:00–08:30) | **4** More data | USE sources, plus any ASK sources approved by then (as variant 4b). Per-speaker caps, so new data cannot swamp a class | CV improves, and the Hypa-only F1 improves | **D4 ≈08:00 (09:00):** does external data lift efi, igl and nup on test speakers we have never had? |
| Wed 11:00 (12:00) | Voice notes in | Quality check: consent present, a native speaker confirms the language. Split by speaker: about half for training, half held out as the phone-channel test | — | — |
| Wed 11:00–16:15 (12:00–17:15) | **5** New voices and teacher | kNN-VC shared pool and/or MMS-LID distillation, whichever wins in CV, with the voice-note training speakers. Each change is still gated separately in local CV; only the submission bundles them, because the notes arrive too late for their own slot | CV improves, and recall on held-out voice notes improves | **D5 ≈16:30 (17:30):** do new voices and teacher knowledge lift the scarce classes on the leaderboard? |
| Wed 14:00 (15:00) | **Decision** | **You decide: top rung yes or no** | — | — |
| Wed 16:30–21:15 (17:30–22:15) | **6** Depth | 6, then 8 layers, chosen by CV F1 × the multiplier at measured latency | CV final score improves after efficiency | **D6 ≈21:30 (22:30):** does depth pay on the organisers' hardware? If you approved the top rung, submit its early checkpoint instead, to measure its latency and memory |
| Wed 21:30–23:59 (22:30–00:59) | Final build | Retrain the best gated recipe on 100% of the data (train, validation, cleared external data, all voice notes). Re-tune the biases on out-of-fold predictions from the identical config, average checkpoints, and package in a new private repo | Checker, offline load, local latency | — |
| Thu 00:00–00:30 (01:00–01:30) | Final | Make the F1 repo public and upload it | — | **F1** |
| Thu ≈05:00 (06:00) | Final | F2: the top rung if approved and checked; otherwise a hedge (the F1 recipe with only loss-adjusted biases) | — | **F2** |
| Thu by 07:30 (08:30) | Final | F3 is held in reserve for a failed run | — | **F3** |

Rules that keep the ladder honest:
- **A failed gate means the change is dropped**, and its slot asks a cheap fallback question instead: AfriHuBERT-s against -n, fp16 against fp32 latency, or bias-regularisation strength.
- **The top rung runs only on your explicit word**, never by default.
- **F1 must share its model type, depth, preprocessor and dtype** with a dev submission that already ran on the organisers' hardware. Only its weights and data are new.
- **Final picks are ranked by pooled cross-validation F1 × the measured multiplier**, not by small public-leaderboard differences.

## Message to the organisers, ready to paste

**In plain terms:** send this by email now, because it names our specific data sources. If there is no reply by 17:00 UTC (18:00 WAT), post questions 1, 2, 5, 8, 10 and 11, which reveal nothing about our sources, on the Codabench forum.

> **To:** community@datascientistsfoundation.org
> **Subject:** DSN 2026 LID (Codabench 18424): data, licence and team questions from team "godzilla"
>
> Hello DSN team,
>
> We are team "godzilla" in the DSN 2026 "Find the Language I Am Speaking" competition. Before we train on anything beyond the provided data, we would like your written confirmation on the points below. We will list every dataset, teacher model and generator model in our model card. We will not use hypaai/Hypa-Voices, facebook/omnilingual-asr-corpus or AfriSpeech/african-speech-ipa, or any mirror or derivative of them. A yes or no for each question is enough.
>
> 1. **Licence of our submission.** Our model is fine-tuned from ajesujoba/AfriHuBERT, a continued pretraining of utter-project/mHuBERT-147 (CC BY-NC-SA 4.0). Some data we would like to use (e.g. VoiceAfrica) is also CC BY-NC-SA 4.0. May our public submission repo carry a CC BY-NC-SA 4.0 licence?
> 2. **MMS-LID as a teacher.** May we use facebook/mms-lid-4017 (CC BY-NC 4.0, released 2023, not trained on any of the three prohibited corpora) as a teacher, i.e. for soft labels in distillation and for filtering external audio? Its weights would not be part of our submission, and we would disclose it. (Its training data includes New Testament recordings; please tell us if you see an overlap concern with AfriSpeech-IPA.)
> 3. **Common Voice.** Hypa-Voices states that it includes "Mozilla Common Voice-derived subsets". May we use Common Voice from the Mozilla Data Collective for (a) target languages such as Ibibio, Yoruba, Igbo and Hausa, and (b) English and Nigerian Pidgin for the "other" class?
> 4. **NaijaVoices, VoiceAfrica and African Next Voices.** NaijaVoices/Lanfrica contributed Igala, Nupe and Urhobo data to Meta's Omnilingual ASR training, and DSN co-leads African Next Voices. None of these is on the prohibited list, and VoiceAfrica's languages are not in facebook/omnilingual-asr-corpus. May we use (a) naijavoices/voiceafrica-datasets, (b) the NaijaVoices Nupe dataset on the Mozilla Data Collective and UmarBaba1/nupe_speech_corpus, (c) naijavoices/naijavoices-dataset, and (d) African Next Voices Nigeria subsets? Can you confirm that the hidden test audio contains no recordings from these collections?
> 5. **Team-collected recordings.** May we train on voice notes our team collects from consenting adult native speakers? Consent is written, names are never published, and the recordings are not shared. The model card would describe them as "volunteer recordings, N speakers per language".
> 6. **Openly licensed media.** May we use (a) YouTube videos that their uploaders marked "Creative Commons Attribution (reuse allowed)", with every video ID listed in the model card, and (b) Wikimedia Commons and Wikitongues recordings under CC BY-SA 4.0, such as Wikimedia Nigeria's oral-history videos?
> 7. **Global Recordings Network.** May we use GRN audio (CC BY-NC-SA)? We plan to avoid Bible.is / Faith Comes By Hearing, YouVersion, JW.org and JESUS Film audio entirely; please tell us if that is unnecessary.
> 8. **A fourth helper.** Our group has four people and teams are limited to three. Our registered members are [NAME 1], [NAME 2] and [NAME 3]. May [NAME 4], who is not on our team or any other team, help only by recruiting volunteers and collecting consented recordings, with no access to our code, models or Codabench account?
> 9. **Intron Health data.** We read "AfriSpeech/african-speech-ipa" as that specific repository and its derivatives, and not as Intron Health's separate datasets (e.g. intronhealth/NigBench-MAMAI-Speech-QA). Is that correct?
> 10. **Final phase.** With the 3-per-day limit and the 00:00–08:00 UTC window on 22 October, can at most 3 of the 5 final submissions be used? Is the best final submission scored, rather than the last?
> 11. **Efficiency measurement.** Are speed and memory measured on a GPU? Is the model loaded in its stored dtype (fp16), and with which transformers version?
>
> Thank you very much,
> [YOUR NAME], team "godzilla"

## First things to do today

**In plain terms:** these seven actions unblock everything else. Most take a few minutes.

1. **Send the organiser email above now.** Fill in the names, and post the source-free questions on the forum if there is no reply by 18:00 WAT.
2. **Confirm the three official Codabench members.** Tell the fourth person they may only share the call for volunteers until the organisers answer question 8. Recordings and consent go straight to an official member, and nobody else submits.
3. **Ask each official member to phone-verify their own Kaggle account** for GPU access, so the three cross-validation folds can run in parallel.
4. **Register a NaijaVoices membership and a Mozilla Data Collective account.** VoiceAfrica and Common Voice can then be used within minutes of a yes. Keep checking the NigBench approval.
5. **Push the voice-note drive.** Nupe and Igala first. Each speaker gives consent before recording and also records one English or Pidgin clip. The deadline is Wed 12:00 WAT.
6. **Say "go" for rungs 0 and 1.** The first submission is due around 17:30 WAT today.
7. **Put two decisions in your calendar:** "top rung yes or no" on Wed 15:00 WAT, and "final model sign-off" on Wed 22:30 WAT.

## Conclusion

The leaderboard shows a competition that looks like a model-size contest but is really a speaker-scarcity problem. Efficiency is nearly flat between 4 and 12 layers. Latency matters more than size above 10 ms. Every team, including one already using AfriHuBERT, is near zero on the three classes with the fewest voices. The edge therefore comes from work most teams will not do:
- erasing the recording fingerprints;
- spreading synthetic and real voices evenly across classes;
- collecting consented native-speaker audio that no one else has;
- turning grey-area data into usable data through a written organiser answer.

The training ladder is more than caution here. With one Nupe speaker and about 0.14 h of Nupe test audio, the public leaderboard cannot reliably separate two good models. A gated, one-change-at-a-time ladder, scored on pooled speaker-held-out predictions and confirmed against the leaderboard's per-condition columns, is the only way to know which improvements are real. The ladder only works, though, if the final model is the one it validated: the same architecture, retrained on more voices, and submitted with the packaging already proven.

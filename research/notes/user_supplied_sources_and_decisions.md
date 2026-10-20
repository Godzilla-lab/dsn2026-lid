# User-supplied sources and decisions (recorded by the coordinator, 2026-10-20)

## Sources from the Nigeria AI Collective repository (github.com/aicollectiveng/Nigeria-AI-Repository, "Data & Open-Source Assets")

The user's instruction: "use whatever is good". The coordinator checked these on 2026-10-20.

- **google/WaxalNLP (WAXAL, Google Research Africa)**
  - Licence: cc-by-4.0 / cc-by-sa-4.0 (per HF card), not gated.
  - HF configs:
    - ASR: ach, aka (+v2), amh (+v2), dag, dga, ewe, ful, kpo, lin, lug, mas, mlg, nyn, orm (+v2), sid (+v2), sna, tir (+v2), sog, wal (+v2)
    - TTS: ach, bau, ewe, fat, ful, hau, ibo, kik, lug, luo, nyn, pcm, swa, twi, yor
  - Use:
    - `hau_tts`, `ibo_tts`, `yor_tts`: studio, few speakers, so low value except as clean-condition examples.
    - `pcm_tts`: Nigerian Pidgin for the "other" class.
    - The ASR and TTS sets for non-target African languages feed the "other" class, notably `bau` (Bada, a language of Nigeria) and `ful` (Fula), which are Nigerian or regional neighbours.
  - No overlap with the prohibited corpora found.
  - Verdict: USE.
- **"African Voices Nigeria" (2,500 h, 2,800+ speakers; yor/ibo/hau/pcm)**
  - The repo's link points to arXiv 2604.16287, which is actually the NaijaS2ST paper (~50 h per language for ibo/hau/yor/pcm paired with English; CC BY 4.0; already covered in major_languages_other_class_augmentation.md). Verdict for NaijaS2ST: USE.
  - The real African Next Voices Nigeria data appears on HF only as manually gated uploads (e.g. `chukypedro/africanvoices_hau`, licence not stated). Data Science Nigeria co-leads that consortium. Verdict: ASK ORGANISERS FIRST.
- **CLEAR-Global/TWB-voice-TTS-Kanuri-1.0-sampleset**: Kanuri (a Nigerian language) TTS sample. Use for the "other" class. Verdict: USE, small.
- **NaijaVoices (Lanfrica)**: yor/ibo/hau. Already covered; CC-BY-NC-SA, registration needed.
- **Common Voice**: already covered. Ask the organisers before using it for target languages (Hypa-Voices derives partly from Common Voice).
- **YoruLect**: Yoruba dialects. Low priority, since Yoruba is 4.3% of the F1. Not verified whether it includes speech.
- Everything else in that section is text-only (MT, NER, sentiment, news) and is irrelevant to speech LID.

## Access status checked by the coordinator

- The DSN training data (`Data-Science-Nigeria/dsn-2026-lid-train`) is downloadable with the session HF token (HTTP 200 after the user accepted the gate and fixed the token scope).
- **`naijavoices/voiceafrica-datasets`: download works (HTTP 200).**
  - Language folders: bcz, ble, bvb, fan, igl, knc, krx, nup, pov, srr, urh. That is target igl/nup/urh, plus knc (Central Kanuri, Nigerian) and others for the "other" class.
  - Licence: cc-by-nc-sa-4.0.
- `intronhealth/NigBench-MAMAI-Speech-QA`: requested by the user; manual approval still pending (HTTP 403).

## User decisions to reflect in the report's recommendations

1. **Training ladder: start small and step up gradually.**
   - Begin with the smallest model and data (a smoke test), then a small baseline, then add one change per rung, each gated on local speaker-grouped CV improving.
   - The rungs, in order: loss/bias → augmentation → external data → voice conversion/distillation → depth.
   - **Only go to the highest/largest configuration when the user explicitly says so.** The leaderboard score should climb gradually; the strongest model is held for the final phase.
2. **Keep the training recipe private.**
   - Disclose every external dataset (and any teacher or generator models used to create training data) in the model card, as the rules require.
   - Describe the approach only briefly. Training code, hyperparameters and augmentation recipes stay private.
   - Keep the HF model repo private until each submission.
3. **Collect consented native-speaker voice notes** for nup, igl, ibb, efi and tiv first, then anw, ego, idu and urh. A private GitHub team repo (`Godzilla-lab/dsn2026-lid`) holds the consent form and templates. Audio and personal data are never committed.
4. **Multiple Kaggle accounts per person are not used** (against Kaggle's terms). Teammates may each use their own account for parallel experiments.

# Status: what's running and what to upload next

_Updated automatically as runs finish. Times are UTC; Nigeria time (WAT) = UTC+1._

## Deadline we are working to

**Wednesday 21 Oct, 18:00 WAT (17:00 UTC)**, the deadline the team was given. Codabench itself shows a later close: the dev phase ends Thu 00:59 WAT and a final phase runs Thu 01:00–09:00 WAT. We plan for the earlier time, so the best model must be uploaded before 18:00 WAT Wednesday.

| When (WAT) | What |
|---|---|
| Tue ~17:30 | D1 ready (rung 1): upload it |
| Tue ~21:30 | D2 ready (rung 2): upload it |
| Tue before 00:59 | D3 (rung 3), if you're awake: today's slots are lost after 00:59 |
| Wed ~09:00 | D4 ready (rung 4, extra data) |
| Wed 12:00 | Voice notes deadline |
| Wed ~16:00 | Final model (best recipe + voice notes) ready |
| Wed by 17:30 | Upload the final model. Leave 30 min of margin before 18:00 |

## How we choose a model

The organisers' principle: *"We are building systems that work optimally… accurate across all languages (including the low-resourced ones) and also small, fast and efficient enough to run in production."*

So a model is chosen by its **predicted final score** (accuracy × efficiency) from the local leaderboard (`src/predict_lb.py`), never by accuracy alone:
- **Depth (4/6/8/12 layers):** we keep extra layers only if the accuracy they add beats the efficiency they cost.
- **Low-resource languages first:** Efik, Ibibio, Igala and Nupe each count 8.7%, twice Yoruba, Igbo or Hausa.
- **Measured efficiency so far:** speed is already perfect (D1: 0.004 s per clip). Memory is the weak part, at 1.72 GB, which is typical for HuBERT models. Size is fine: 38M parameters, 0.076 GB.

## Upload next

**D2 was uploaded but NOT SCORED: the organisers' run timed out after processing 0 of 6,659 clips.** It is a platform problem, not our model: D2's config is identical to D1's, which ran at 0.004 s per clip. See `results/d2/leaderboard.md`. Ask the organisers to re-run #1005591. **Hold today's last slot for D3** until the queue works again (watch for any team's new result on the leaderboard).

D1 result: final 0.3421 (F1 0.374, efficiency 0.830). See `results/d1/leaderboard.md`.

**How to upload a ready submission**
1. On Hugging Face, open the model repo listed below, go to **Settings → Change repo visibility → Public**. Or message Claude "submit D1" and it will do this for you.
2. On Codabench, go to **Participate → Submit** and upload that row's `submission.zip` (in `submissions/<name>/`).
3. Paste the leaderboard numbers into the chat, or tell Claude it's uploaded and it will read them.

**Repo visibility policy (from Wed 12:35 UTC):** every model repo is private. A repo is made public only for the minutes the organisers' runner needs to download it, then private again once scored. It must be public again for the final-phase uploads and if the organisers ask to verify.

**Submission limits:** 3 per day, and the day resets at 00:00 UTC (01:00 WAT). Today's unused slots are lost after **00:59 WAT**.

## Submissions

**Team upload (HF pull request #1, adenike123):** 4 Urhobo WhatsApp recordings, 62 min. D4 already labels all 377 ten-second windows as Urhobo, so training on them would not change the model; they stay a test set. Not used for training (no consent line yet).

**D5b scored 0.6222 at 12:15 UTC Wed: first place** (ray_nacido 0.6026 at 11:48, D5 0.6078). F1 0.6788, efficiency 0.791, clean / noisy / phone accuracy 0.792 / 0.793 / 0.736. Weak classes on the real test: ibb 0.06, efi 0.28, igl 0.37, tiv 0.53, nup 0.73, yor 0.75. Neither longer training nor stronger voice changes beat D4's recipe in CV.

**Final phase (Thu 00:00–08:00 UTC, 5 uploads, private test clips, scores shown):** upload **D7, then D5b, then D5c** as soon as Codabench works; all three repos are public for it. The best private-split score is the final pick; the others go private again afterwards.

**Codabench outage (Wed 14:22 UTC → at least Thu 07:30 UTC, i.e. the whole scheduled final phase):** no team could upload in the final phase. The dev leaderboard (us first at 0.6222) is the only result that exists; the organisers will have to extend or re-run the final phase. We keep checking every 5 minutes and every 4 hours.

**Earlier note:** the site returns HTTP 503 / connection resets for everyone, so no uploads or scores are possible. D5c and D7 are public and ready to upload the moment it returns; the final phase (Thu 00:00–08:00 UTC, 5 uploads) needs the best repo public at that time.

**Organisers' runner (Wed 11:15 UTC): working again.** simple0001 scored 0.4106 at 10:04 UTC with a 164.5M model at 0.025 s per clip (GPU speed), the first real score since Tue 13:15. Submissions at 09:04 and 09:31 still failed, so the fix landed about 10:00. **D5 made public and sent for upload.** The probe confirmed a healthy GPU runner (0.003 s per clip), so the CPU-fast Whisper backup was dropped unfinished.

| Name | HF repo (private until upload) | Rung | Question it tests | Local CV final | Ready? | Uploaded? |
|---|---|---|---|---|---|---|
| D1 | [godzilla1215/dsn2026-lid-d1](https://huggingface.co/godzilla1215/dsn2026-lid-d1) @ `a9cbed5f` | 1 | Does local CV match the leaderboard? What latency and memory do the organisers measure? | – | yes | **yes: 0.3421** |
| D2 | [godzilla1215/dsn2026-lid-d2](https://huggingface.co/godzilla1215/dsn2026-lid-d2) @ `3e073891` | 3 | Does augmentation fix noisy/phone accuracy and the Idoma dump? | – | yes | uploaded, **not scored** (platform timeout) |
| D4 | [godzilla1215/dsn2026-lid-d4](https://huggingface.co/godzilla1215/dsn2026-lid-d4) @ `a653142c` | final | everything + cleaned labels + offsets ×0.75 | CV recipe predicted 0.621 before offsets (+≈0.02 F1 with them) | **yes** (public, checker PASS) | uploaded Wed, **not scored** (platform timeout, 0 of 6659 clips) |
| D3 | [godzilla1215/dsn2026-lid-d3](https://huggingface.co/godzilla1215/dsn2026-lid-d3) @ `41db455c` | 3+4+6 | 12 layers + augmentation + 33 h external data. Real check of the new recipe; uses today's last slot | CV fold A (with VoiceAfrica): held-out F1 0.54 clean / 0.52 noisy / 0.57 phone (D1 recipe: 0.40 / 0.17 / 0.12) | **yes** (public, checker PASS) | sent to team lead |
| D5 | [godzilla1215/dsn2026-lid-d5](https://huggingface.co/godzilla1215/dsn2026-lid-d5) @ `e3390666` | final + averaging | D4 weight-averaged with a second D4 trained on the same data in a different order (one ordinary model, same size and speed) + D4's offsets | fold A: **0.5646** vs 0.5465 for D4's single-model equivalent (and 0.5640 for the other seed). See `results/d5/` | **yes** (public) | **yes: 0.6078, #1** (F1 0.6788, efficiency 0.791) |
| D5b | [godzilla1215/dsn2026-lid-d5b](https://huggingface.co/godzilla1215/dsn2026-lid-d5b) @ `4ea16f1e` | final + corrected offsets | D5's weights with per-language offsets corrected from D5's real confusion matrix (Yoruba and Tiv up, Efik and Ibibio down). See `results/d5/offsets_d5b.json` | neutral on local CV | **yes** (public) | **yes: 0.6222, #1** (F1 0.6951) |
| D7 | godzilla1215/dsn2026-lid-d7 (not created yet) | final + new data | D5's recipe retrained on `folds_final2.csv`: + Common Voice 27.0 Ibibio (CC0, MDC; 17 speakers, 2.9 h, 2 speakers dropped by the voice-overlap check) and, if they arrive in time, MDC Nupe (7 speakers, 3.3 h), Efik-TTS and Tiv-TTS. D5b offsets | – | **yes** (public, checker PASS) | sent 14:13 UTC |
| D6 | godzilla1215/dsn2026-lid-d6 (not created yet) | final + 3-way averaging | D5 plus a third same-recipe model (seed 2) in the average, with the better of D5/D5b's offsets. **Ships only if it beats on held-out fold A** | training (`dsn2026-d6-seed2`, done ≈14:05 UTC) | no | – |

## Ladder

| Rung | Change | Status | Result |
|---|---|---|---|
| 0 | Smoke test | cancelled: Kaggle's mounted disk was too slow for its many small reads; checks moved into rung 1 | – |
| 1 | AfriHuBERT 4 layers, provided data, speaker- and source-balanced sampling, 3–6 s crops | full model done (13 min train); CV queued after speaker embeddings | 37.9M params, 0.076 GB fp16, checker PASS, 38 ms/clip on a Kaggle T4 |
| 2 | Logit-adjusted loss + per-class bias tuning | queued | – |
| 3 | Shortcut removal + phone/noise/reverb augmentation, 8,000 steps | **done: D2** (29 min train) | checker PASS; upload to see the noisy/phone effect |
| 3+4+6 | **Go-for-#1 stack**: 12 layers + augmentation + external data (USE sources, capped at half of a class) | full model done (48 min); packaged as D3, held | 94.6M, 0.189 GB, checker PASS |
| Calibration | CV of D1's recipe with clean/noisy/phone held-out scoring + eval-only VoiceAfrica | **done** | predicted 0.307 vs real 0.342: slightly cautious, good for ranking (`results/calibration/`) |
| D3-recipe CV | 12 layers + aug + ext + VoiceAfrica (7 train / 3 held-out speakers per language), checkpoints at 4k and 8k steps | **done** | **predicted final 0.621** at 8k steps (4k: 0.585). Cross-fit offsets ×0.75 add about +0.02 F1. See `results/cv_r5b/` |
| Final (D4) | Best recipe on all data (cleaned competition data + ext + all VoiceAfrica + Wikimedia/Wikitongues), 8k steps, offsets ×0.75 | **done, packaged, public** | 94.6M, checker PASS; expected final about 0.61–0.64 (ray_nacido: 0.5685) |
| Longer training | Same recipe, 12,000 steps | **stopped after fold A: no gain** | fold A 12k: 0.540 / 0.501 / 0.555 (clean / noisy / phone) vs 8k: 0.545 / 0.501 / 0.558. 8k stays |
| Voice-change test | Wider speed/pitch shifts (0.85–1.15, 70% of clips) + feature masking, 8k steps, 3 folds | **done: no gain, not used** | predicted final 0.610 vs 0.621 for D4's recipe; efi/ibb/igl unchanged (0.17 / 0.03 / 0.15). See `results/cv_spkaug/` |
| 4 | External data marked USE in the report | queued | – |
| 5 | Voice conversion / MMS-LID teacher | queued | – |
| 6 | Depth 6 → 8 layers | queued | – |
| Top | Largest setup | **off until the team lead says so** | – |

## Compute

| Kaggle account | Owner | GPU left this week | Use |
|---|---|---|---|
| godzilla1215 | team lead | 0 h (resets Oct 24) | data storage only |
| adewumiadenike | official member | ~30 h | all runs for now |
| third account (`KAGGLE_API_TOKEN_2`) | official member | not visible to this session yet | parallel CV folds once it appears |

## Waiting on

- Organiser email: needs the 3 official names, the 4th helper's name and the sender's name. Until a reply arrives, only USE-verdict data is used.
- Voice notes: by Wed 21 Oct, 12:00 WAT.

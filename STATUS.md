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

**Submission limits:** 3 per day, and the day resets at 00:00 UTC (01:00 WAT). Today's unused slots are lost after **00:59 WAT**.

## Submissions

| Name | HF repo (private until upload) | Rung | Question it tests | Local CV final | Ready? | Uploaded? |
|---|---|---|---|---|---|---|
| D1 | [godzilla1215/dsn2026-lid-d1](https://huggingface.co/godzilla1215/dsn2026-lid-d1) @ `a9cbed5f` | 1 | Does local CV match the leaderboard? What latency and memory do the organisers measure? | – | yes | **yes: 0.3421** |
| D2 | [godzilla1215/dsn2026-lid-d2](https://huggingface.co/godzilla1215/dsn2026-lid-d2) @ `3e073891` | 3 | Does augmentation fix noisy/phone accuracy and the Idoma dump? | – | yes | uploaded, **not scored** (platform timeout) |
| D3 | [godzilla1215/dsn2026-lid-d3](https://huggingface.co/godzilla1215/dsn2026-lid-d3) @ `41db455c` | 3+4+6 | 12 layers + augmentation + 33 h external data. Real check of the new recipe; uses today's last slot | CV fold A (with VoiceAfrica): held-out F1 0.54 clean / 0.52 noisy / 0.57 phone (D1 recipe: 0.40 / 0.17 / 0.12) | **yes** (public, checker PASS) | sent to team lead |

## Ladder

| Rung | Change | Status | Result |
|---|---|---|---|
| 0 | Smoke test | cancelled: Kaggle's mounted disk was too slow for its many small reads; checks moved into rung 1 | – |
| 1 | AfriHuBERT 4 layers, provided data, speaker- and source-balanced sampling, 3–6 s crops | full model done (13 min train); CV queued after speaker embeddings | 37.9M params, 0.076 GB fp16, checker PASS, 38 ms/clip on a Kaggle T4 |
| 2 | Logit-adjusted loss + per-class bias tuning | queued | – |
| 3 | Shortcut removal + phone/noise/reverb augmentation, 8,000 steps | **done: D2** (29 min train) | checker PASS; upload to see the noisy/phone effect |
| 3+4+6 | **Go-for-#1 stack**: 12 layers + augmentation + external data (USE sources, capped at half of a class) | full model done (48 min); packaged as D3, held | 94.6M, 0.189 GB, checker PASS |
| Calibration | CV of D1's recipe with clean/noisy/phone held-out scoring + eval-only VoiceAfrica | **done** | predicted 0.307 vs real 0.342: slightly cautious, good for ranking (`results/calibration/`) |
| D3-recipe CV | 12 layers + aug + ext + VoiceAfrica (7 train / 3 held-out speakers per language), checkpoints at 4k and 8k steps | **running** (until about 20:15 UTC) | predicted score, offsets, best length |
| Final | Best recipe on all data (cleaned competition data + ext + all VoiceAfrica + Wikimedia/Wikitongues), tuned offsets | next, tonight | upload before 00:59 WAT if it is predicted best |
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

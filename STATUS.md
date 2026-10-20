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

## Upload next

**D2 is ready: upload `submissions/D2/submission.zip`.** Its repo is already public, and an anonymous download passes the organisers' checker. D2 = D1 plus augmentation and 2× training (rung 3). It tests whether noisy/phone accuracy rises and the Idoma dump goes away.

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
| D2 | [godzilla1215/dsn2026-lid-d2](https://huggingface.co/godzilla1215/dsn2026-lid-d2) @ `3e073891` | 3 | Does augmentation fix noisy/phone accuracy and the Idoma dump? | – | **yes** (public, checker PASS) | no |

## Ladder

| Rung | Change | Status | Result |
|---|---|---|---|
| 0 | Smoke test | cancelled: Kaggle's mounted disk was too slow for its many small reads; checks moved into rung 1 | – |
| 1 | AfriHuBERT 4 layers, provided data, speaker- and source-balanced sampling, 3–6 s crops | full model done (13 min train); CV queued after speaker embeddings | 37.9M params, 0.076 GB fp16, checker PASS, 38 ms/clip on a Kaggle T4 |
| 2 | Logit-adjusted loss + per-class bias tuning | queued | – |
| 3 | Shortcut removal + phone/noise/reverb augmentation, 8,000 steps | **done: D2** (29 min train) | checker PASS; upload to see the noisy/phone effect |
| 3+4+6 | **Go-for-#1 stack**: 12 layers + augmentation + external data (USE sources, capped at half of a class) | training now (D3) | – |
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

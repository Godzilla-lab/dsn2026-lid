# Status: what's running and what to upload next

_Updated automatically as runs finish. Times are UTC; Nigeria time (WAT) = UTC+1._

## Upload next

Nothing ready yet. The first submission (D1) comes from rung 1.

**How to upload a ready submission**
1. On Hugging Face, open the model repo listed below, go to **Settings → Change repo visibility → Public**. Or message Claude "submit D1" and it will do this for you.
2. On Codabench, go to **Participate → Submit** and upload that row's `submission.zip` (in `submissions/<name>/`).
3. Paste the leaderboard numbers into the chat, or tell Claude it's uploaded and it will read them.

**Submission limits:** 3 per day, and the day resets at 00:00 UTC (01:00 WAT). Today's unused slots are lost after **00:59 WAT**.

## Submissions

| Name | HF repo (private until upload) | Rung | Question it tests | Local CV final | Ready? | Uploaded? |
|---|---|---|---|---|---|---|
| D1 | [godzilla1215/dsn2026-lid-d1](https://huggingface.co/godzilla1215/dsn2026-lid-d1) | 1 | Does local CV match the leaderboard? What latency and memory do the organisers measure? | – | no | no |

## Ladder

| Rung | Change | Status | Result |
|---|---|---|---|
| 0 | Smoke test: train, save, checker, offline load, latency, speaker embeddings | running (Kaggle, Adewumi) | – |
| 1 | AfriHuBERT 4 layers, provided data, speaker-balanced sampling, 3–6 s crops | queued | – |
| 2 | Logit-adjusted loss + per-class bias tuning | queued | – |
| 3 | Shortcut removal + phone/noise/reverb augmentation | queued | – |
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

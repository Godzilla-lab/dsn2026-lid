# DSN 2026: Find the Language I Am Speaking

Private team workspace for the [DSN 2026 spoken language-ID competition](https://www.codabench.org/competitions/18424/) on Codabench. Ask the team for the secret link; don't paste it here.

**Task:** given a 2–30 s clip of speech, predict one of 12 Nigerian languages or `other`.

| Code | Language | Code | Language |
|---|---|---|---|
| `yor` | Yoruba | `ibb` | Ibibio |
| `ibo` | Igbo | `idu` | Idoma |
| `hau` | Hausa | `igl` | Igala |
| `ego` | Eggon | `nup` | Nupe |
| `anw` | Annang | `tiv` | Tiv |
| `efi` | Efik | `urh` | Urhobo |
| `other` | English, Nigerian Pidgin, anything else | | |

**Score:** language-weighted F1 × (0.5 + 0.5 × efficiency).
- In the F1, Yoruba, Igbo and Hausa count 4.3% each; every other class counts 8.7%.
- Efficiency rewards a small, fast, low-memory model.
- Limits: 350M parameters, 1.5 GB repo, 4 GB memory, 0.5 s per clip. A submission that breaks any limit is not scored.

## Rules we must not break

- **Never train on the three banned corpora** (the test set was drawn from them), or on any mirror or derivative of them:
  - `hypaai/Hypa-Voices`
  - `facebook/omnilingual-asr-corpus`
  - `AfriSpeech/african-speech-ipa`
- **List every external dataset in the model card.** A missing one gets the team disqualified.
- **The model must be a standard Hugging Face audio-classification model:**
  - no custom code: no `.py`, `.bin`/`.pt` or `auto_map`
  - weights as fp32/fp16 safetensors
  - `id2label` exactly `yor ibo hau ego anw efi ibb idu igl nup tiv urh other`
- **Once submitted, the Hugging Face repo must stay public and unchanged.** We submit by commit hash.
- **One Codabench account per team** (up to 3 people). Never share submissions with other teams.

## What goes in this repo, and what never does

| In this repo | Never in this repo |
|---|---|
| Plan, research report, notes | Tokens or keys (HF, Kaggle, Codabench). Keep them in Kaggle Secrets or env vars |
| Scripts and notebooks (scorer, audit, training, packaging checks) | Audio or datasets: licences forbid re-hosting |
| Model card template, submission log | Volunteers' voice notes, names or contact details |
| Blank consent form and metadata sheet | Model weights: they go to a **private** Hugging Face repo until we submit |

`.gitignore` already blocks audio, parquet, weights and common secret files. Check `git status` before every commit anyway.

## Layout

```
docs/plan.md          plan, training ladder and timeline
research/             research report and notes (added when finished)
src/                  scripts: local scorer, data audit, training, packaging checks
templates/            consent form, voice-note metadata sheet, message to send volunteers
submissions/LOG.md    one row per Codabench upload
```

## How we work

- **Training ladder:** start small and step up one change at a time, keeping a change only if the local speaker-grouped CV improves (see `docs/plan.md`). The top rung runs only when the team lead says so.
- **Submissions:** each one tests a single stated question and gets a row in `submissions/LOG.md`.
- **Visibility:** keep the Hugging Face model repo private until the moment of each upload.

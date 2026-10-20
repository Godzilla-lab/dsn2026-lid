# Local leaderboard calibration (D1 recipe)

CV of D1's exact recipe (4 layers, provided data only, 4,000 steps). Each fold's model scored its held-out speakers clean, noisy and phone. VoiceAfrica Nupe speakers (evaluation only) filled the Nupe pool, which has no held-out speaker in the competition data.

| | Local prediction | D1's real leaderboard |
|---|---|---|
| Final score | **0.307** (90%: 0.295–0.319) | **0.342** |
| Weighted F1 | 0.335 | 0.374 |
| clean / noisy / phone accuracy | 0.749 / 0.254 / 0.336 | 0.641 / 0.210 / 0.436 |
| yor ibo hau | 0.67 0.54 0.59 | 0.42 0.56 0.60 |
| ego anw efi ibb | 0.50 0.39 0.00 0.00 | 0.80 0.49 0.00 0.00 |
| idu igl nup tiv | 0.41 0.01 0.00 0.27 | 0.40 0.03 0.28 0.16 |
| urh other | 0.53 0.61 | 0.86 0.50 |

## Reading
- **Slightly pessimistic, by about 0.035 on the final score.** That is expected, because CV models train on about 2/3 of the data.
- The failure pattern matches: efi, ibb and igl are about 0, and idu, ibo and hau line up closely.
- **Use it to rank recipes against each other**, and add about +0.035 when comparing with real leaderboard numbers.
- **Clean is too optimistic and phone too pessimistic;** the two partly cancel. The phone simulation (μ-law plus 8 kHz) is harsher than the test's phone audio.
- Nupe from VoiceAfrica speakers scores 0.00 locally against 0.28 on the leaderboard. The VoiceAfrica recordings are a different domain from the test's Nupe (hypa).

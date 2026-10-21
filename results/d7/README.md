# D7: D5's recipe retrained with new Ibibio and Nupe speakers

Training set `splits/folds_final3.csv`: 46,720 clips, 159.0 h = `folds_final` (44,823) + Common Voice 27.0 Ibibio
(CC0, Mozilla Data Collective; 17 speakers, 2 dropped by the voice-overlap check; 1,204 clips, 2.9 h) + MDC "Speech
Data Collection for The Nupe Language" (CC BY-NC-SA 4.0; 7 speakers, 2 dropped as matching VoiceAfrica voices; 693
clips, 2.1 h). Same recipe and size as D5 (12-layer AfriHuBERT, augmentation, 8,000 steps, seed 0); D5b's offsets.
Kaggle run `dsn2026-d7-full`, 48 min on a T4. Single model (not averaged).

## Sanity checks (D5b vs D7)

| Test | D5b | D7 |
|---|---|---|
| CV Ibibio, 2 held-out speakers never trained on, 60 clips (true: ibb) | 0/60 (54 called Efik) | **60/60** |
| MDC Nupe, 2 held-out speakers never trained on, 60 clips (true: nup) | 44/60 | **51/60** |
| 130 competition clips seen in training | 128 | 127 |
| Team Urhobo recordings (63 windows) | 63 urh | 63 urh |

The Ibibio result is on the same recording style as the new training clips, so it overstates what the test's
Ibibio (hypa recordings) will do, but it shows the model now has an Ibibio concept instead of mapping Ibibio to Efik.

# CPU-fast models that still load with AutoModelForAudioClassification (transformers 4.57.6)

Why: on Wed 21 Oct the organisers said the runner is "capped on purpose". If the cap means CPU-only
inference, every model must finish 6,659 test clips (mean 11.4 s) in 1,800 s, i.e. 0.27 s per clip.
Timings below: random-weight models, one clip at a time through `fe(wave)` then `model(**inp)`, on this
container's CPU (AVX-512, no FP16/AMX), 6,659-clip totals in seconds at 1 / 2 / 4 threads (±15%).

| Model | Params | 1 thread | 2 threads | 4 threads |
|---|---|---|---|---|
| whisper-tiny, 8 s window | 7.9M | 706 | 461 | 354 |
| whisper-tiny, 10 s | 7.9M | 896 | 560 | 417 |
| whisper-tiny, 12 s | 8.0M | 1133 | 722 | 547 |
| whisper-base, 6 s | 20.1M | 1144 | 767 | 557 |
| whisper-base, 7 s | 20.1M | 1304 | – | 660 |
| whisper-base, 8 s, 4 layers | 13.9M | 1069 | 697 | 576 |
| whisper-base, 8 s | 20.2M | 1491–1717 | 996 | 770 |
| whisper-base, 10 s | 20.2M | 2172 | – | 1010 |
| whisper-small, 8 s, 4 layers | 30.8M | 1940 | – | 782 |
| whisper-base 8 s run in fp16 on CPU | 20.2M | 12960 | – | – |

Raw-waveform and spectrogram models, seconds per clip at 4 / 1 threads (11.4 s clip):
HuBERT 12 layers 0.81 / 2.16; HuBERT 4 layers 0.52 / 1.47; HuBERT 2 layers 0.44 / 1.17 (the conv front
end alone is about 0.4 s); SEW 0.30 / 0.78 (English-only checkpoints); w2v-BERT cut to 2 layers 0.60 / 1.36;
AST 1024 / 512 frames 1.15 / 2.78 and 0.63 / 1.23. CLAP is not in the mapping. None fits the budget.

## Whisper with a shortened window (source verified)

- The encoder requires mel length == `max_source_positions * 2` (`modeling_whisper.py:668-672`);
  `embed_positions` has exactly `max_source_positions` rows (sinusoids, length-independent).
- `WhisperFeatureExtractor` sets `n_samples = chunk_length * 16000` and defaults to `truncation=True`,
  `padding="max_length"`, `max_length=n_samples`, so by default it keeps the **first chunk_length
  seconds** of any clip (`feature_extraction_whisper.py:93-95, 195, 199, 298`). `chunk_length` is read
  from `preprocessor_config.json` and must be an integer.
- A model saved with `max_source_positions=400` and a sliced `embed_positions` reloads with
  `AutoModelForAudioClassification` with no missing or unexpected keys (fp16 and fp32 tested).
- Pitfalls: keep `return_attention_mask` false (a true value returns a sample-level mask that breaks the
  model call); `do_normalize` in the config is ignored; default loading gives fp32 even from fp16 files,
  but `dtype="auto"` would give fp16, which is 7.5x slower on CPU, so ship fp32; mean pooling includes
  padded frames, so train with the same extractor; `layer_weights` size follows `encoder_layers`.

## No config trick for HuBERT

`Wav2Vec2FeatureExtractor.__call__` defaults to `padding=False, max_length=None, truncation=False` and reads
only `do_normalize`, `return_attention_mask`, `padding_value` from the config. The other raw-waveform
extractors (SpeechT5, Encodec, Dac, Dia, Kyutai) do not truncate either. Only AST and Whisper truncate from
the config.

## Other levers

- fp16 is never faster on CPU here. Cutting layers and shortening the window both help; attention makes
  cost grow faster than linearly with the window.
- Thread risk: torch follows the CPU affinity mask, but under a docker `--cpus` quota it sees every host
  core; over-subscription cost about 20%. Whisper-base at 400–500 frames hit an MKL `addmm` slowdown
  (10–15 s per clip) in this torch 2.14 build; 300–375 frames were fine.

## Recommendation (if the runner is CPU-only)

1. whisper-base, 6 layers, **6 s window**: 1,144 s on 1 core, 767 s on 2.
2. whisper-tiny, 10 s window as the safe fallback: 896 s on 1 core.
3. Re-window an already trained model by slicing `embed_positions` and editing `chunk_length`, then
   fine-tune briefly.

Outcome: later on Wed 21 Oct the runner scored 94.6M and 164.5M models at 0.014–0.025 s per clip, i.e.
on a GPU, so the cap was not CPU-only. The Whisper model (`dsn2026-whisper-full`) stays a backup.
Scripts and logs: scratchpad `research_fast/` (bench.py, roundtrip.py, prof.py, bench_*.log).

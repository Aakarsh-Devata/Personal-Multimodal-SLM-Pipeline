# Lightweight CPU ASR smoke test

## Observed result (2026-10-04 UTC)

A genuine local model run completed on the official EPIC-KITCHENS `P02_05`
recording. This is a small execution/repeatability smoke test, **not an ASR
accuracy evaluation**. It used no reference annotation CSVs, personal media,
remote inference, synthetic transcript, or ground-truth text.

- Input: extracted 16 kHz mono audio, 18.56 seconds.
- Model: `Systran/faster-whisper-tiny`, revision
  `d90ca5fe260221311c53c58e660288d3deb8d356`.
- Runtime: `faster_whisper`, CPU, int8, four CPU threads, beam size 5,
  temperature 0, `condition_on_previous_text=false`, automatic language
  detection, local Silero VAD enabled.
- VAD retained 0.832 seconds. Whisper emitted empty text and zero segments.
- First successful cached load + inference + output write: 1.363 seconds.
- Independent-process repeat: 1.425 seconds; output bytes were identical.
  These timings exclude the initial package/model downloads and are specific
  to this environment; they are not a hardware benchmark.

The output is an **unvalidated model hypothesis**. Empty output does not
establish that the recording contains no intelligible speech. Automatic language
selection (`en`, probability 0.38736) is also a model estimate, especially weak
for this low-speech input. No listening-based accuracy assessment or word-error
rate measurement was performed.

Actual output at `data/processed/P02_05/transcript.json` includes:

```json
{
  "text": "",
  "segments": [],
  "language": "en",
  "language_probability": 0.3873627781867981,
  "duration_sec": 18.56,
  "duration_after_vad_sec": 0.832,
  "source_type": "model_transcript",
  "model": {
    "backend": "faster_whisper",
    "name": "tiny",
    "device": "cpu",
    "compute_type": "int8",
    "revision": "d90ca5fe260221311c53c58e660288d3deb8d356"
  },
  "accuracy_validated": false
}
```

The complete file also records inference options. Its SHA-256 is
`6be66cdb30a636f435ac2ece439aa863ead4ba34de0d887b0180c375621c06f0`.
Timing/check reports are local ignored outputs in
`data/reports/asr-smoke-cached.json` and `data/reports/asr-smoke-repeat.json`.

## Provenance and license

Source record: [`datasets/epic-kitchens-smoke.example.json`](../datasets/epic-kitchens-smoke.example.json).
Original video: [University of Bristol official EPIC-KITCHENS download](https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/train/P02/P02_05.MP4).
EPIC-KITCHENS P02_05 by Dima Damen and the EPIC-KITCHENS contributors;
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), noncommercial
research use, no endorsement implied. The original video bytes are unchanged;
the derived audio was decoded and resampled to 16 kHz mono, and the transcript
is machine-generated. Dataset action and sound annotations remain
`evaluation_reference_only` and were not opened or loaded for inference.

- Original video SHA-256:
  `75bddc7634447d52502fd596ba4494d11fdb7a7e316d51035282c90a846c5efa`
- Extracted audio SHA-256:
  `8e7819b98b5e7c92f7e0a430bccb51961f332468336f039e2ad974b70865d4a1`
- Official model: [Systran/faster-whisper-tiny at the pinned revision](https://huggingface.co/Systran/faster-whisper-tiny/tree/d90ca5fe260221311c53c58e660288d3deb8d356), MIT license.
- `model.bin`: 75,538,270 bytes; SHA-256
  `dcb76c6586fc06cbdac6dd21f14cfd129cc4cdd9dce19bf4ffa62e59cbe6e6d1`.
- All four downloaded model assets total 78,203,619 bytes (under 100 MB).
- `config.json`: SHA-256
  `a73a28cdfe1c43ccc7202fa333d1f89c202477271407ae9a7f19afa52039cac8`.
- `tokenizer.json`: SHA-256
  `fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab`.
- `vocabulary.txt`: SHA-256
  `34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913`.

## Reproduce

Run from the repository root after the public sample has been acquired and
preprocessed. Only this one video is selected. Package installation and the
first explicitly authorized model download require internet access; inference
uses local audio and model files.

```bash
python3 -m venv .venv                    # if not already created
.venv/bin/python -m pip install -r requirements.txt -r requirements-asr.txt
mkdir -p data/models/faster-whisper
cat > data/asr_smoke_config.yaml <<YAML
models:
  whisper:
    backend: faster_whisper
    name: tiny
    device: cpu
    compute_type: int8
    cpu_threads: 4
    revision: d90ca5fe260221311c53c58e660288d3deb8d356
    download_root: "$PWD/data/models/faster-whisper"
    vad_filter: true
pipeline:
  skip_existing: false
  allow_model_downloads: false
YAML

# One explicitly opted-in model acquisition/inference invocation.
SLM_CONFIG="$PWD/data/asr_smoke_config.yaml" \
  SLM_ALLOW_DOWNLOADS=1 HF_HUB_DISABLE_TELEMETRY=1 \
  .venv/bin/python -m slm_pipeline.ingestion.phase_2r_transcribe --video P02_05

# Subsequent local-only invocation; remove any inherited download opt-in.
env -u SLM_ALLOW_DOWNLOADS \
  SLM_CONFIG="$PWD/data/asr_smoke_config.yaml" \
  HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
  .venv/bin/python -m slm_pipeline.ingestion.phase_2r_transcribe --video P02_05
```

The measured environment had an existing `.venv` with system packages readable;
new ASR packages were installed into `.venv`, not globally. No PyTorch or BLIP
installation was needed. The packages relevant to the actual run were:

| Component | Observed version |
| --- | --- |
| Python / platform | 3.12.14 / Linux x86_64 |
| faster-whisper | 1.2.1 |
| ctranslate2 | 4.8.2 |
| av | 18.1.0 |
| onnxruntime | 1.30.0 |
| tokenizers | 0.23.2 |
| huggingface-hub | 1.33.0 |
| numpy | 2.3.5 |
| PyYAML | 6.0.3 |
| socksio (environment proxy helper) | 1.0.0 |

Two setup issues were found and resolved before successful inference:

1. This environment's SOCKS proxy required `socksio`; installed with
   `.venv/bin/python -m pip install socksio==1.0.0`. This helper is only needed
   in an environment configured to use a SOCKS proxy.
2. PyAV 19.0.1 was incompatible with faster-whisper 1.2.1's `metadata_errors`
   argument. `requirements-asr.txt` now pins PyAV 18.1.0, the actual
   successful version, plus the observed primary ASR runtime versions. See the [upstream PyAV changelog](https://github.com/PyAV-Org/PyAV/blob/main/CHANGELOG.rst).

The environment also emits optional native-runtime warnings about UCX VFS
and ONNX telemetry persistence. `UCX_VFS_ENABLE=n` suppressed the UCX warning
for the measured runs. The ASR loader explicitly disables ONNX Runtime telemetry
collection; it does not require a writable user home or global package changes.

## Offline and regression checks

- Defaults remain OpenAI Whisper `base` on CPU. Selecting `faster_whisper` is
  optional. For that backend, CPU defaults to int8; no reference text is used.
- `pipeline.allow_model_downloads=false` is the default. The existing
  `SLM_ALLOW_DOWNLOADS=1` override or an explicit configuration change is needed
  for model acquisition.
- OpenAI Whisper requires an existing local weights file in offline mode.
- faster-whisper resolves the configured revision with `local_files_only=true`
  in offline mode. The loader then requires local weights, config, and tokenizer
  assets before constructing the model, preventing an implicit tokenizer fetch.
- The actual repeat run used `HF_HUB_OFFLINE=1`, disabled Hugging Face telemetry,
  and rejected Python DNS/socket connection calls. It finished with zero such
  attempts and byte-identical JSON. This is an offline behavior check, not a
  process-wide firewall/packet audit of native libraries.
- `tests/test_transcription.py`: 11 model-free contract tests cover default
  offline loading, explicit opt-in, local assets, missing tokenizers, legacy
  Whisper support, lazy segment consumption, timestamp/provenance preservation,
  empty hypotheses, no-audio handling, and the environment override.

An input with **no audio stream** is separately represented as
`source_type=no_audio_stream`; an empty ASR hypothesis keeps
`source_type=model_transcript`. Neither unit-test fixtures nor dataset reference
annotations are promoted into model output.

# Personal Multimodal SLM Pipeline

Local video → audio/frames → transcript/captions → timestamped context → structured semantics → searchable FAISS memory. The existing seven-stage design is retained; inference runs locally and does not use paid APIs.

Current status: [paused reproducible checkpoint](docs/CURRENT_CHECKPOINT.md).
The latest local macOS run passed 99 tests with 4 warnings. Real Qwen3-VL
MLX/Metal inference runs, but action claims remain unreliable, including in the
final longer-context experiment. This is not a production action-memory system.

## First reproducible milestone

The lightweight profile validates one-video scope, source timestamps, alignment, deterministic indexing, and evidence retrieval using synthetic tests plus licensed EPIC-KITCHENS media preprocessing. It **does not establish end-to-end model or question-answering accuracy**. The optional SmolVLM2 frame-observation adapter, semantic extraction via Ollama, and the sentence-transformer model need separately provisioned weights and sufficient compute.

Human action/sound annotations are held separately as evaluation references. They are never represented as transcripts, captions, model predictions, or inferred answers. EPIC is a small plumbing fixture; the provider-neutral manifest can later describe other licensed sources such as Ego4D once access is obtained. See [public-data preparation](docs/PUBLIC_DATA.md).

For a bounded local Qwen3-VL public-video temporal/control experiment, see the
[pre-registered evaluation protocol and label-blind results](docs/next-visual-evaluation.md).

The CPU-only candidate selector can prepare an auditable bounded set of
temporal windows without loading any model:

```bash
python -m slm_pipeline.pipelines.temporal_candidates \
  --video-dir data/processed/P04_19 --window-size 4 --max-windows 3 \
  --max-selected-frames 12 --minimum-new-frames 2
```

It records reduced-luma change measurements, every candidate interval, skipped
intervals, frame hashes, and quality exclusions. These are selection
measurements—not confidence or action scores—and any later model events remain
unvalidated until independently reviewed.

## Install and test

The original lightweight milestone used Python 3.12, Linux x86_64, and FFmpeg/ffprobe 7.1. A later local macOS checkpoint passed the 99-test suite described above. Python 3.12 is recommended for the pinned wheel versions. FFmpeg must be installed separately via the system package manager. Verify that the pinned wheels install for the actual architecture, and select CPU/MPS/CUDA only after inspecting available hardware. The setup script does not assume CUDA or silently install a GPU stack.

```sh
git clone https://github.com/Aakarsh-Devata/Personal-Multimodal-SLM-Pipeline.git
cd Personal-Multimodal-SLM-Pipeline
./setup.sh
source .venv/bin/activate
python -m pytest -q
```

`setup.sh` uses a virtual environment, stops on failure, and installs the pinned lightweight/test profile. It does not launch services or download model weights. Do not run historical generated indexes bundled in the upstream repository; the new default paths use a separate `data/` tree.

For an optional small CPU ASR run, see [ASR smoke instructions](docs/ASR_SMOKE.md).
The complete tested scope is recorded in [the milestone report](docs/MILESTONE_REPORT.md).

## Input and configuration

Put one authorized video under `data/raw/<video-id>.mp4` (also `.mov`, `.mkv`, `.avi`, case insensitive). Its filename stem is the video ID. Duplicate stems across extensions are rejected.

```sh
# Real media preprocessing only; requires no models.
python -m slm_pipeline.orchestrator --video P02_05 --start-phase 1 --end-phase 2
python -m slm_pipeline.orchestrator --video P02_05 --start-phase 3.5 --end-phase 3.5

# Entire pipeline, after separately provisioning the required models.
python -m slm_pipeline.orchestrator --video P02_05

# Start/end phase selection also works with --video.
python -m slm_pipeline.orchestrator --video P02_05 --start-phase 5 --end-phase 7
```

Every stage uses that exact video ID. Failures stop execution with a nonzero exit; missing files and empty input selections are errors. Default runs recompute selected stages. `--resume` explicitly trusts existing artifacts and is only appropriate when inputs, configuration and models are unchanged. Downstream stages are not automatically invalidated if you run only an earlier phase; rerun the affected suffix after changes.

Configuration is in `slm_pipeline/config/settings.yaml`. Paths resolve relative to the repository rather than the current working directory. Overrides:

- `SLM_CONFIG=/absolute/custom.yaml`: merge a custom YAML file over defaults
- `SLM_DATA_ROOT=/absolute/scratch/data`: relocate all input/output directories
- `SLM_RAW_DIR`, `SLM_PROCESSED_DIR`, `SLM_MEMORY_DIR`, `SLM_EMBEDDINGS_DIR`: override individual paths
- `SLM_ALLOW_DOWNLOADS=1`: explicit model-download opt-in; otherwise cached/local weights only

All media, model weights, generated transcripts, indexes and reports belong under ignored `data/` or `artifacts/`. `.env` is ignored; never commit credentials. Existing tracked upstream generated records are not removed or history-rewritten by this patch; `.gitignore` does not untrack them.

## Stages and evidence

1. Validate exact raw input
2. FFmpeg mono 16 kHz audio, fixed/scene frames, SHA256, and source frame timestamps
3. Local Whisper transcription; an absent audio stream is explicitly recorded
3.5. Existing image-quality and perceptual-dedup filtering, retaining source timestamps
4. Optional local SmolVLM2 (or legacy BLIP-2) observations from the usable-frame manifest
5. Speech/vision alignment, including visual-only context when speech is absent
6. Existing Ollama structured extraction with invalid/empty responses treated as failures
7. Existing Sentence-Transformers embeddings + FAISS, replacing that video's old rows on every run

`frame_timestamps.json` records seconds from the first decoded video frame. `frames_manifest.json` carries those times through image renaming. Captions, aligned contexts and retrieved index records retain frame IDs, timestamps and source-relative paths. Scene frame numbers are **never** used as timestamps. If processing a temporal excerpt, retain the manifest clip-start offset and add it when mapping clip-relative evidence back to the full source. The current media preprocessor does not automatically trim clips and rejects nonzero audio/video stream start timestamps rather than silently losing an A/V offset. Normalize unsupported media to one common zero origin first.

Vector replacement handles repeated, changed, reduced and empty block sets. Searches return up to the available neighbors; empty stores and FAISS missing-neighbor IDs cannot return phantom results. Video/time filters are applied before top-k selection. Storage is designed for one writer; concurrent writes and crash-atomic two-file persistence are not yet supported. Distances/similarity are retrieval heuristics, not calibrated answer confidence.

## Full inference (separate, not covered by the lightweight check)

Install a suitable [PyTorch build](https://pytorch.org/) and then `pip install -r requirements-inference.txt`. This optional profile has compatibility bounds, not a validated full lock. Provision the configured SmolVLM2 (or legacy BLIP-2) and embedding weights, local Whisper weights, and an [Ollama](https://ollama.com/) model yourself or explicitly opt in to downloads. No API keys are needed.

The default `HuggingFaceTB/SmolVLM2-256M-Video-Instruct` produces per-frame observations with original decoded timestamps and `source_type: model`; it is still model output, not verified truth. Its weights are never fetched unless `SLM_ALLOW_DOWNLOADS=1` is set. Ollama must already be running at localhost:11434 with the configured model. Semantic output may be inaccurate even if valid JSON; no model's self-reported confidence has been calibrated here.

```sh
python -m slm_pipeline.cli query "What happened?" --video P02_05
python -m slm_pipeline.cli summarize --meeting P02_05
```

Queries retrieve timestamped evidence. A grounded generative QA answerer and benchmark scoring against held-out labels remain subsequent work. Never evaluate a model against labels supplied to it as input.

## Bounded Apple Silicon smoke profile

`config/epic_cpu.yaml` samples one frame every four seconds and caps visual generation at 24 tokens. It is intended for a responsive CPU fallback when MPS is unavailable; it does not change source timestamps. Keep model caches under the ignored data tree and use an explicit download opt-in:

```sh
PATH=/opt/homebrew/bin:$PATH \
HF_HOME=data/models/hf SLM_CONFIG=config/epic_cpu.yaml SLM_ALLOW_DOWNLOADS=1 \
python -m slm_pipeline.orchestrator --video P02_05 --start-phase 4 --end-phase 4
```

Once the weights are present, omit `SLM_ALLOW_DOWNLOADS` to run offline. Inspect `vision_captions.json` for each observation's original timestamp, model identifier, device, latency, and run-level memory metric. Do not use held-out human annotations as prompt input or model-generated evidence.

### Optional isolated MLX video runtime

`slm_pipeline.pipelines.mlx_video.MLXVideoRunner` is an optional adapter for a
separate Apple-Silicon MLX environment. It invokes a local MLX-VLM executable
for an ordered video window, records the decoder's actual frame indices and
times before mapping them to source time, and does not pass
`--trust-remote-code`. Keep MLX and its model weights in a
separate ignored virtual environment/data directory so its newer Transformers
dependency does not alter the validated pipeline environment. Its output is
still unverified model evidence; evaluate it against held-out references only
after inference.

For strict temporal claims, call `decode_video_trace` in that isolated runtime
with the same `fps` and `video_num_frames` passed to the runner, then record
the model preprocessor's actual `video_grid_thw` through `video_input_trace`.
The trace must contain every decoded frame and their mapped source times;
neither a separately generated preprocessing manifest nor a requested frame
rate is evidence of what reached the model.

For a persistent MLX video pilot, `PersistentMLXVideoRunner` loads one local
model once, decodes and preprocesses each window once, and passes those exact
native-video tensors into generation. It freezes a label-free question by ID,
records decoder PTS separately from any model-relative time markers, and
rejects a window whose actual merged visual-token count exceeds its configured
cap before generation. In MLX-VLM 0.7.4, set the Qwen3-VL video processor's
`max_pixels` control directly and verify `video_grid_thw`; merely passing an
image-processor `max_pixels` keyword does not bound video resolution.

### Evidence and abstention policy

`slm_pipeline.pipelines.evidence_policy` keeps raw visual model text as an
`unvalidated_model_hypothesis`. Caption retrieval remains available, but it
labels returned text as unvalidated and never turns similarity, model
confidence, or nonidentical frames into an action claim. Temporal/action
answers abstain unless independently accepted evidence overlaps the requested
source-time range. Identical frame hashes fail closed as repeated static input;
different hashes still do not prove an action. Phase 5 retains source-frame
provenance, and Phases 6/7 keep semantic model output out of factual
decisions/tasks/topics unless it is explicitly marked
`accepted_independent_evidence`. Calibration thresholds are experimental.

See [the next visual-evaluation design](docs/next-visual-evaluation.md) before
running a broader held-out public-clip study.

## Caption-only retrieval

For a small retrieval-only check, index Phase 4's actual model observations with the configured local MiniLM embedding model. This path does not run ASR, semantic extraction, or answer generation; every search result is an unvalidated caption with its source timestamp.

```sh
python -m slm_pipeline.cli visual-index --captions data/processed/P02_05/vision_captions.json
python -m slm_pipeline.cli visual-search "washing machine" --video P02_05
```

Treat FAISS distance as a ranking signal only. Abstain when the available captions do not establish the requested fact, action, or time interval.

## Uniform temporal comparison

`config/epic_dense_cpu.yaml` and `config/epic_1fps_cpu.yaml` write to independent experiment trees, preserving the sparse baseline. `temporal_experiment` groups consecutive quality-approved frames in manifest order; it does not select windows from annotations. Each temporal observation records every source timestamp in its input window.

```sh
SLM_CONFIG=config/epic_1fps_cpu.yaml python -m slm_pipeline.pipelines.temporal_experiment \
  --video-dir data/experiments/onefps/processed/P02_05 --window-size 4
```

Higher source sampling increases the chance of covering a short action interval, but does not by itself validate a model's action claim. Evaluate coverage, observed text, and abstentions separately from held-out action labels.

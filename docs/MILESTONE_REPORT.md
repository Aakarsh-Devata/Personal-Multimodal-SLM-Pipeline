# Reproducible one-video milestone

> Historical first milestone. The test count, environment, and unpublished
> status below describe that earlier stage. See the
> [current paused checkpoint](CURRENT_CHECKPOINT.md) for the latest scope.

Date: 2026-10-04 UTC. Upstream:
https://github.com/Aakarsh-Devata/Personal-Multimodal-SLM-Pipeline

Base commit: `aa98182b8141526ea73a7ca213615f1692258744`.
Local branch: `milestone/reproducible-one-video`.
Changes are local and unpublished. No remote pushes, PRs, registrations, paid
APIs, deployments, history rewriting or modifications of upstream generated
memory were performed. The source-only package excludes Git history, model
weights, media, generated indexes and all source-derived observation files.

## Verified results

- **74 tests pass** in a newly created isolated lightweight virtualenv, and in the
  optional-ASR environment. Tests run offline with temporary synthetic media,
  mock neural outputs where needed, and real FAISS operations.
- Clean install through `VENV=.venv-clean ./setup.sh` succeeds.
- `pip check`: no broken requirements.
- `compileall`, shell syntax, direct/module entrypoint help, and `git diff --check`
  succeed. One known FAISS NumPy-internal deprecation warning remains.
- Genuine EPIC-KITCHENS P02_05 preprocessing and quality filtering succeed:
  18.551867 seconds of video, 19 fixed sampled frames, no scene-cut frames at the
  configured 0.2 threshold, 19 usable frames. Source times run from 0 to 18.018 s.
- Actual official tiny CPU ASR runs successfully; see [ASR run details](ASR_SMOKE.md).
  Its empty hypothesis is not validated evidence of speech absence.
- Independent review verified the fixed audio-offset rejection, evidence-field
  protection, source provenance, input scoping and synthetic regressions.

A full synthetic-media integration test runs all seven stages twice, with neural
stages explicitly stubbed, then queries real FAISS memory. It verifies retained
source-frame evidence, stable index cardinality, and that a second corrupt,
unselected input is never processed. This establishes plumbing correctness, not
perception, reasoning, retrieval relevance or question-answering accuracy.

## Repairs and retained architecture

The original phase modules, image-quality/dedup logic, BLIP-2 processor, alignment
class, semantic extraction, FAISS store and retrieval agent remain in use.

- Portable config and environment overrides replace machine-specific paths.
- Syntax-broken orchestration becomes a fail-fast exact-video dispatcher, honoring
  both start/end phase bounds and returning nonzero on failure.
- Fixed/scene frames store decoded source times; quality filtering and captions
  preserve them rather than deriving times from frame filenames.
- Nonzero A/V stream starts are rejected before extraction to prevent silent
  misalignment; frame/audio extraction explicitly selects verified first streams.
- Visual-only context remains available when transcripts are empty.
- Invalid/empty semantic responses fail; generated fields cannot replace source
  block IDs or timestamp intervals.
- Indexing replaces one video's complete row set, including reduced/empty sets,
  preserving other videos. Search handles empty stores, missing neighbors, and
  video/time filtering before top-k.
- A bounded provider-neutral manifest records source, license, attribution,
  duration, interval, size and SHA256. The EPIC reference adapter remains separate
  from observations and model output.

## Reproduction commands

From the source root:

```sh
VENV=.venv-clean ./setup.sh
.venv-clean/bin/python -m pip check
.venv-clean/bin/python -m pytest -q
.venv-clean/bin/python -m compileall -q slm_pipeline tests
bash -n setup.sh
```

After obtaining the licensed file at the official URL specified in the manifest:

```sh
.venv-clean/bin/python -m slm_pipeline.datasets verify \
  --manifest datasets/epic-kitchens-smoke.example.json --video P02_05 \
  --source data/raw/P02_05.MP4
.venv-clean/bin/python -m slm_pipeline.orchestrator \
  --video P02_05 --start-phase 1 --end-phase 2
.venv-clean/bin/python -m slm_pipeline.orchestrator \
  --video P02_05 --start-phase 3.5 --end-phase 3.5
```

Dataset verification returned 70,130,936 bytes and SHA256
`75bddc7634447d52502fd596ba4494d11fdb7a7e316d51035282c90a846c5efa`.
The full official annotation files remain excluded; separate reference selection
found five action and six sound annotations. Human narrations were not used as
ASR input or ground-truth speech.

## Environment and remaining limits

Validated environment: Python 3.12.14, Linux x86_64, FFmpeg 7.1.5, CPU only. Runtime
reported nine logical CPUs and approximately 9.7 GiB RAM. The full BLIP-2 vision,
Ollama semantics and sentence-transformer inference stages were **not run on
EPIC media**; their model weights/runtime were not provisioned. No end-to-end QA
benchmark or accuracy score is claimed.

Remaining work: appropriately sized vision/semantic models, model-specific
reproducibility locks, held-out evidence/QA evaluation, confidence/abstention
calibration, and an authorized Ego4D adapter after access is obtained. The current
manifest handles bounded full videos; temporal trimming and general A/V-offset
normalization are deliberately unsupported. Index persistence assumes a single
writer and is not crash-atomic across its two files. Explicit `--resume` trusts
existing artifacts; rerun the affected stage suffix after input/model changes.

# Bounded public-data smoke fixture

The dataset adapter is offline. It neither downloads videos/annotations nor loads
models. It accepts an explicitly supplied, licensed local video. Human annotation
CSVs are exported separately as **evaluation references only**. They never become
transcripts, visual captions, semantic findings, or retrieval corpus content.

## Chosen fixture and provenance

`datasets/epic-kitchens-smoke.example.json` records one full, short public video:

- Dataset: EPIC-KITCHENS, original EK55 video with EK100 training annotations
- Video ID: `P02_05`; split: `train`
- Interval: `[0, 18.551867]` seconds from the start of the video stream
- Original size: 70,130,936 bytes (about 66.9 MiB)
- SHA256: `75bddc7634447d52502fd596ba4494d11fdb7a7e316d51035282c90a846c5efa`
- Video: 1920 × 1080, 60000/1001 fps; audio: stereo, 48 kHz
- Video duration: 18.551867 s; container/audio duration: 18.560000 s
- Original bytes downloaded over verified HTTPS and checked on 2026-10-04

Source: [official video](https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/train/P02/P02_05.MP4).
The [official download and copyright page](https://epic-kitchens.github.io/2025#download)
links the video releases and annotations. The page places the datasets and
benchmarks under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/):
retain credit, license link, and a description of changes, and do not imply
endorsement. This fixture is for noncommercial research; availability is not a
commercial-use grant. Review the actual license for the intended use.

Attribution: EPIC-KITCHENS P02_05 by Dima Damen and the EPIC-KITCHENS contributors,
original EK55 video used with EK100 annotations. Original media bytes are
unchanged. Frame sampling, rescaling, audio conversion and model processing are
derivations and should be described in each run's provenance. No endorsement is
implied. The official page supplies citations for the dataset publications.

No media or annotation CSV is part of the source deliverable. Local media,
references and derived artifacts belong under ignored `data/` (or an explicitly
configured external data root). The JSON example contains metadata and checksums,
not annotations or invented model output. Its `verified` status describes the
observed original; it does not establish that any caller's local file is valid.
Always run `verify` against that file.

## Offline workflow

Run these commands from the repository with the project's environment activated.
Acquire the file through the official source after checking that the intended use
complies with its license. Supply a local path; there is no downloader command.

```bash
python -m slm_pipeline.datasets validate \
  --manifest datasets/epic-kitchens-smoke.example.json

python -m slm_pipeline.datasets verify \
  --manifest datasets/epic-kitchens-smoke.example.json \
  --video P02_05 --source /path/to/licensed/P02_05.MP4

SLM_DATA_ROOT=data python -m slm_pipeline.datasets prepare \
  --manifest datasets/epic-kitchens-smoke.example.json \
  --video P02_05 --source /path/to/licensed/P02_05.MP4 \
  --acknowledge-license --output-manifest data/manifests/P02_05.verified.json
```

`prepare` defaults to the configured raw directory, respecting `SLM_DATA_ROOT`
and `SLM_RAW_DIR`; `--raw-dir` overrides it explicitly. It copies original bytes,
checks the copy, and produces a manifest containing the observed digest. An input
already at the intended destination is checked in place. Existing different
files or output manifests are not overwritten. `verify` requires a declared
checksum, compares size and SHA256, and checks the first video stream's duration
with `ffprobe` (1 ms tolerance). Neither operation trusts the filename or the
manifest's status in place of verification.

For a new approved local source, create a manifest with accurate provenance,
interval and duration, `verification_status: "pending"`, `sha256: null`, and
`size_bytes: null` until measured. `prepare` records a digest; this establishes
local-byte identity, not proof of authorship or a grant of rights.

The initial preparation adapter supports **full short videos only**. It rejects
partial intervals instead of silently processing the wrong content. The manifest
and reference reader retain explicit source offsets for a future clip-extraction
adapter. Bounds: at most 10 records, at most 256 MiB per source, and at most 120
seconds per interval. It rejects nonfinite, negative, reversed, out-of-duration,
wrong-unit and ambiguous-origin timestamps, unsafe/duplicate video IDs, malformed
digests and missing license/attribution metadata.

## Evaluation references, not observations

The separate official sources are:

- [EK100 action annotations](https://github.com/epic-kitchens/epic-kitchens-100-annotations),
  `EPIC_100_train.csv`: five selected action segments for P02_05
- [EPIC-SOUNDS annotations](https://github.com/epic-kitchens/epic-sounds-annotations),
  `EPIC_Sounds_train.csv`: six selected sound segments for P02_05

Those counts were measured from the checked files. The manifest pins each CSV's
actual SHA256 from 2026-10-04; mutable upstream URLs may later serve different
bytes. A changed digest fails closed and requires deliberate provenance review.
The source files' human narrations and sound descriptions are labels, not ASR
speech. Audio and visual annotations need not describe the same event. Do not
score speech recognition against action narrations or present these labels as
model predictions. This single training video supports plumbing/debugging only,
not held-out accuracy or generalization claims.

After obtaining the local licensed CSVs:

```bash
python -m slm_pipeline.datasets references \
  --manifest datasets/epic-kitchens-smoke.example.json --video P02_05 \
  --kind action --csv data/references/EPIC_100_train.csv \
  --output data/references/P02_05.actions.evaluation.json

python -m slm_pipeline.datasets references \
  --manifest datasets/epic-kitchens-smoke.example.json --video P02_05 \
  --kind sound --csv data/references/EPIC_Sounds_train.csv \
  --output data/references/P02_05.sounds.evaluation.json
```

Reference exports preserve license, attribution, source URL, source CSV digest,
original CSV row and timestamps, source interval and clipped/rebased interval.
Timestamps must be `HH:MM:SS[.fraction]`; intervals must fit the video. Overlapping
annotations are allowed because actions and sounds can overlap. Each export is
marked `evaluation_reference_only`. The CLI rejects configured pipeline roots (including custom raw, processed,
embedding and memory directories), conventional artifact directories, actual
pipeline artifact filenames and symlinks resolving inside those roots. Keep
references under a separate references directory. No pipeline phase imports the
reference reader, and it does not emit any pipeline-observation schema.

## Verification

```bash
python -m unittest discover -s tests -p test_dataset_manifest.py -v
```

Unit tests use temporary synthetic bytes and CSVs, with media probing mocked;
they do not download anything or claim to run ASR/vision/semantic models. The real
fixture verification and reference selection are separate CLI checks.

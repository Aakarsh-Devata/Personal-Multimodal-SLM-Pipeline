# Current paused checkpoint

Date: 2026-10-04 UTC.

This source checkpoint freezes the tested local pipeline so work can pause.
It is a reproducible research/prototype checkpoint, not a production action-
recognition or autonomous-memory system.

## What is verified

- The final Linux cloud run passed **130 tests with 3 third-party warnings**
  after a fresh isolated install of `requirements-test.txt`; `pip check` found
  no broken requirements. Tests use synthetic inputs and mocked neural outputs,
  with real FAISS operations. No neural models were loaded for this run.
- Earlier macOS runs passed 99 tests with 4 warnings for the baseline and
  107 tests with 5 warnings after the initial question interface. Those runs
  preceded the laptop execution pause and final cloud-only corrections. None
  of these test counts establishes model accuracy or native runtime stability;
  they are local test results, not GitHub CI results.
- Public EPIC-KITCHENS dataset verification, bounded preprocessing, decoded
  source-frame/PTS provenance, exact-video scope, repeatable index replacement,
  CPU/Apple-Silicon adapters, and evidence/hypothesis boundaries are covered by
  the implementation and tests.
- The installed local Qwen3-VL 4B MLX/Metal path ran real inference. The final
  fresh longer-context experiment still produced unsupported action claims;
  local action understanding remains unreliable. More context did not establish
  the required capability. See the [experiment record](next-visual-evaluation.md).
- The earlier real tiny CPU ASR run produced an empty hypothesis. That is not
  proof that speech was absent; see [the ASR record](ASR_SMOKE.md).
- A thin CLI/API/page question interface retrieves timestamped evidence cards.
  Source-video/time filtering precedes top-k ranking, model observations remain
  unvalidated, and raw captions return `insufficient_evidence`. The legacy API
  routes remain available with lazy backend loading and existing evidence guards.
  No ten-minute-video or general video-QA performance claim is established.

## Known macOS native-runtime risk

Four local Python processes crashed with SIGSEGV during MiniLM/FAISS attempts;
the reports included native PyTorch frames. A deferred-FAISS import allowed one
later real query to finish, returning raw caption evidence and
`insufficient_evidence`. That workaround does not prove the root cause is fixed
or establish stability. Laptop Python execution was stopped. The final changes
and tests ran only in the Linux cloud environment; no further Mac run is claimed.
See [the runtime warning](local-evidence-query.md#native-macos-stability-warning).

## Publication checks and scope

The publication preparation verified the source archive checksum, safe archive
paths, source-only file allowlist, and common secret patterns. Python compilation,
Python/JSON/YAML parsing, shell syntax, CLI help, and the complete 130-test suite
passed in the publication environment. The fresh profile includes the API and
HTTP test dependencies, so the documented test setup collects the query tests.
The three warnings are third-party Starlette/httpx, Starlette/AnyIO, and FAISS/
NumPy deprecations. API tests cover page serving, text-safe rendering source,
query edge cases, compatibility routes, and lazy startup. Interactive browser
UI validation was not completed.

Only source, configuration examples, synthetic tests, and documentation are added
or updated. Media, model weights, generated indexes, raw model outputs, and
label-bearing evaluation records are excluded. Previously tracked upstream
generated artifacts remain unchanged; do not use them as validated memory.

## Pause boundary

Stop model/prompt optimization and additional benchmarking at this checkpoint.
Keep model output as unvalidated hypotheses, with explicit abstention until
independent evidence supports a claim. Production readiness, reliable action
memory, and generalization are not established. No merge or deployment is part
of this checkpoint.

The [original milestone report](MILESTONE_REPORT.md) is historical: its 74-test
count, Linux environment, and unpublished status describe that earlier stage.
They do not describe the current publication or supersede this checkpoint.

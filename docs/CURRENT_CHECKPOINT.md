# Current paused checkpoint

Date: 2026-10-04 UTC.

This source checkpoint freezes the tested local pipeline so work can pause.
It is a reproducible research/prototype checkpoint, not a production action-
recognition or autonomous-memory system.

## What is verified

- The latest local macOS test run passed **99 tests with 4 warnings**. These
  synthetic/offline plumbing and regression tests do not measure model action
  accuracy. They are local test results, not GitHub CI results.
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

## Publication checks and scope

The publication preparation verified the source archive checksum, safe archive
paths, source-only file allowlist, and common secret patterns. Python compilation,
Python/JSON/YAML parsing, shell syntax, and 14 dataset-contract unit tests passed
in the publication environment. The full pytest suite was not rerun there
because its test/runtime dependencies were unavailable; the 99-test result above
belongs to the separate local macOS run.

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

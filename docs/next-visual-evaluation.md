# Next visual evaluation: pre-registered, claim-level, and fail-closed

This design is for public EPIC-KITCHENS clips only. It does not tune prompts,
sampling, thresholds, or model choice against known action labels.

1. Freeze the candidate model revision, local runtime version, prompt ID,
   temperature, frame count, pixel/token cap, and answer schema before viewing
   outputs. Select clip IDs and source intervals without placing action labels
   in the prompt or development fixture.
2. For every clip, record raw source SHA256, source PTS, decoded frame IDs and
   hashes, model revision, processor grid/tensor shape, and raw model text.
   Model-relative video markers remain distinct from source PTS.
3. Pair every temporal clip with a native repeated-still control made from one
   of its decoded public frames. A model family/configuration cannot advance to
   broader action evaluation if it describes a temporal change in this control.
   Forward/reverse order is different: it is a direction-correspondence probe,
   because an opening can visibly become closing in reverse. A nonempty reverse
   response is therefore not a failure by itself; its claimed action and
   direction still require blind visual support.
4. Conduct a blinded claim review before opening labels. Review each output
   claim independently as supported, unsupported, uncertain, or
   viewpoint-limited. Score actor, object, temporal order, time reference, and
   unsupported specificity separately. Nonidentical frames, model confidence,
   and retrieval distance are never proof of an action.
5. Only after the claim review is frozen, compare compatible claims with held-
   out labels. Report timestamp/window coverage, human observability in the
   actual sampled frames, and model recognition separately: interval overlap
   alone is not evidence that the action was visible in the supplied images.
   Report abstention, supported claims, unsupported claims, schema-format
   failures, and control outcomes alongside any label correspondence.

Scientific limits: first-person camera perspective may hide mouth contact,
off-camera state, or object identity; sparse frame sampling can miss onset; and
an annotation can describe an event not distinguishable from the supplied
visual evidence. These are abstention conditions, not zero-confidence proof of
model failure. Calibration thresholds remain experimental until a separate,
pre-registered validation set establishes them.

## Bounded two-clip batch (frozen before outputs)

The post-pilot batch uses two extra public EPIC-KITCHENS source videos selected
from the official `EPIC_100_video_info.csv` solely by technical metadata, before
opening any action rows: `P03_15` (10.477133 s) and `P06_02` (13.947267 s).
Both are 1920x1080 at 59.9400599400599 fps.  Official Bristol-host HEAD
responses measured 39,548,685 and 52,556,185 bytes respectively (92,104,870
bytes combined, below the 200 MB acquisition cap).  The selection is not a
claim of benchmark representativeness.

Every temporal input and each same-clip repeated-still negative control uses
the existing local `mlx-community/Qwen3-VL-4B-Instruct-4bit` revision
`2fd8dacbdb8f1e54b8c005f081ec5bf79c56376b`, 16 native-decoded frames at the
runner's 2 fps request, 3,211,264 video `max_pixels`, a hard cap of 2,304 merged
visual tokens, `wearer_hands_temporal_v2_no_action`, temperature 0, and 128 maximum
new tokens. The answer request explicitly permits `no action observed` and
uncertainty, and does not contain an action label. The model loads once for the batch; source PTS, actual tensor
grid, frame hashes, native-control construction, raw output and resource peak
are recorded. Ground truth stays in a separate evaluation-only location until
the non-label claim review is frozen.

The output review reports supported visual claims, unsupported claims,
abstentions, and coverage. An abstention is coverage information, not an
automatically correct answer. The repeated-still controls test a necessary
temporal discriminant; passing them does not establish action accuracy.

### Completed v2 batch (label-blind)

The local v2 run completed in one persistent-model batch (`run_id`
`qwen3vl4b_bounded_two_clip_v2_local_001`): cold load 3,794.13 ms and 70,867.37
ms end-to-end for two temporal inputs plus two controls. Every input reached
the model as 16 native frames and 1,440 merged visual tokens; both temporal
sequences had 16 distinct decoded-frame hashes, while every lossless
repeated-still control had one hash repeated 16 times. Both controls returned
exactly `no action observed`.

The non-label visual review counted three strictly supported temporal
verb/object claims (pot-lid lift; dishwasher opening; bowl-with-spoon placement)
and four unsupported or over-specific atomic claims (boiling state, pot lift,
pot return, and a sink-reaching detail). The two primary temporal responses did
not abstain. This is small-sample claim accounting, not an accuracy rate,
calibration result, or a license to promote any model text into memory facts.
Raw provenance and the review live in ignored local evaluation records. Labels
remained unopened through output generation and the blind review, then were
opened only by the frozen held-out procedure below.

### Frozen held-out comparison

After the v2 output and label-blind review were hash-frozen, the official
evaluation-only action rows were opened for the two clips. Two of nine action
intervals covered by the native decoder span had a strict semantic and
time-compatible match (0.2222 recall on this tiny covered-action set); seven
were unmatched. The metric is deliberately **not** a full-video accuracy,
precision, calibration, or generalization result. In particular, a visually
supported lid-motion claim did not match a reference action under strict
direction/time rules, while the dishwasher sequence matched broad opening and
bowl-placement events but missed several finer sequential events. The frozen
blind review still independently contains four unsupported or over-specific
atomic claims and no primary abstentions. Labels remain evaluation-only and do
not make those model claims accepted facts.

## Evidence-grounded next implementation increment

Do not broaden models or download a larger model from this checkpoint. The
measured failures instead motivate a small, testable local increment:

1. Require a structured observation schema containing `verb`, `object`,
   `source_pts_range`, `frame_ids`, and `uncertain` for each proposed event.
   Store the raw text too, but reject a claim from answer support when it lacks
   native frame/PTS provenance or contains an unsupported state completion.
2. Add a cheap, non-generative candidate-window stage: reserve uniformly
   distributed temporal anchors first, then allow frame-difference and
   object/hand persistence to use only the remaining window/frame budget. Use
   the existing VLM only on those windows. This aims to improve temporal
   localization without raising global frame rate or adding a new model.
3. Preserve the lossless repeated-still control and add a directional control
   (the same public frames in forward versus reverse order). A temporal verb
   must not be emitted identically for both orderings.
4. Evaluate this increment on a preregistered, separate short public set with
   a fixed compute budget. Keep claim-level visual review, covered-reference
   recall, missed actions, unsupported claims, abstentions, control outcomes,
   and frame/interval coverage as separate columns. Do not choose a confidence
   threshold from that test set.

The decision criterion is lower unsupported temporal specificity and better
covered-action recall at the same bounded VLM input budget—not a higher output
rate, and not a promise of factual memory acceptance.

The historical P04 registration is retained in
[`config/qwen3vl_candidate_evaluation_v1.example.json`](../config/qwen3vl_candidate_evaluation_v1.example.json).
It froze 16 fixed-rate CPU-decoded source frames, transparent 64x36 luma-change
selection, one four-frame primary VLM window, and forward, reverse, and
identical-frame controls. It has been executed and is not a future-run recipe.
The model event response uses integer frame indices (or safe digit-string
serializations) that the local parser materializes to immutable frame IDs and
source PTS. Unknown, repeated, fractional, boolean, or out-of-range indices are
retained as schema-format failures; schema validity remains unvalidated and
does not cross the accepted-evidence boundary.

The selector was exercised CPU-only on the existing development-only public
clips, without opening or using labels: it retained all eight candidate
intervals for P03_15 (three selected; ten unique frames) and all nine for
P06_02 (three selected; ten unique frames). The latter's raw luma ranking
favoured later motion, while the bounded novelty rule forced an early window
into the selection instead of repeatedly selecting one overlapping region.
Every skipped interval and each quality-excluded frame remains in the local
artifact for later coverage accounting. These are selector diagnostics, not
action-recognition results.

### P04 structured triad checkpoint

The preregistered P04 pilot used exactly four losslessly derived frames in
forward, reverse, and repeated-still order, with equal 1,554-token native
inputs. Lossless identity/order checks passed. Forward and static outputs both
returned an empty event list, while reverse output produced two raw hypotheses
that failed the frozen schema (numeric IDs) and the label-blind visual review.
The particular pouring/placing hypotheses were unsupported, so this directional
probe failed. That failure is not a rule that reverse input must abstain:
reverse video can legitimately show a changed or reversed visible action.

Post-review timestamp accounting found that the pure change-ranked three-window
selector dropped one of five full-span action intervals (20%), whereas equal-
budget uniform windows overlapped all five. The one VLM primary window
timestamp-overlapped two reference intervals and emitted no event. This is not
an accuracy result and does not show those references were human-observable in
the four sampled frames; observability was not scored separately in the old
pilot. It does show that pure change ranking is not an adequate action-coverage
policy here. Do not rerun or tune on this clip.

The structured prompt-ablation protocol is
[`config/structured_prompt_ablation_v1.example.json`](../config/structured_prompt_ablation_v1.example.json).
It fixes one fresh public clip, 16 fixed-PTS frames, two uniform anchors plus at
most one motion-fill window, a single preselected primary window, and identical
lossless forward/reverse/static control images for the original and visible-only
prompts. Before labels, blind review must separately record schema-format
failures, visual false claims, and abstention. After labels, it must separately
record timestamp coverage, human sampled-frame observability, and recognition.
It has been executed once on a fresh short official EPIC-KITCHENS source with
the already-installed local model, using six total generations in one retained
model-load batch. Both frozen prompt arms returned valid empty event lists for
the primary, forward/reverse, and repeated-still presentations. The primary
window had timestamp-overlapping annotations, but the frozen blind review found
none of those named transitions observable in the exact four displayed stills;
therefore action recall and precision were undefined rather than favorable.
This is a valid abstention/control observation, not an action-recognition
success and not evidence that either prompt wording achieved the required
visual inference. Earlier prompts already requested direct visibility and
uncertainty, so wording alone cannot be credited for a control pass or expected
to guarantee perception correctness. The source archive intentionally excludes
the local source video, raw outputs, blind review, and label-bearing evaluation
records.

### Preprocessing parity gate and evidence-first comparison

Before that one comparison, use the source-only
[Qwen3-VL preprocessing parity audit](qwen3vl_preprocessing_parity.md). The
installed MLX path is a local processor port, not the separate official
`qwen-vl-utils` workflow, so native tensors alone do not prove cross-runtime
preprocessing identity. Freeze package versions and record the normalization,
patch/grid, resize, frame order, PTS, hashes, tensor shape, and token budget.
Stop debugging preprocessing unless those recorded invariants expose a concrete
discrepancy.

The follow-up is one evidence-first comparison at the same prompt-arm token and
time budgets, not a model sweep. Blind review first records direct visible
states/contact or motion observations separately from optional action
hypotheses; frame IDs establish provenance, never independent proof. After
labels, report precision and recall only with their denominators, alongside
timestamp coverage, human observability in the exact model inputs, schema
failures, visual false claims, abstention, and temporal-reference correctness.
No arbitrary pass threshold, third-party hallucination guarantee, or reliable
milestone claim follows from this plan.

### Dense-versus-sparse source-observable diagnostic

A separate local development diagnostic used a public source sequence that was
first inspected as the exact MLX decoder inputs. The 16-frame 8-fps input and a
four-frame subset both visibly show the wearer pushing a dishwasher rack inward;
the exported contact sheets, unique native frame hashes, exact source PTS, and
source-pixel checks were frozen before generation. The existing local model
failed both presentations: dense emitted repeated outward-pull claims and
truncated before valid JSON; sparse emitted an outward-pull claim with an
invalid numeric event ID. These are unsupported against the directly visible
source transition. Dense had 1,440 native visual tokens and sparse 1,554, so
this is not an equal-token score or evidence that sparse is preferable.

This is a concrete capability limitation of this installed model on this
source-observable transition. Do not retry or tune the model on short
action-recognition diagnostics. The archived contact sheets and local review
are data artifacts, not source-package content, and no reference labels were
opened for this diagnostic.

For a primary system-level evaluation, use the design-only
[`config/long_context_event_sequence_v1.example.json`](../config/long_context_event_sequence_v1.example.json):
20–30 seconds of public video, a sparse global overview for context, and two to
four overlapping 2–4 second local windows at 4–8 fps for action evidence.
Select source-visible transitions before labels, preserve exact PTS/tensor
provenance, and report event sequence/order, unsupported claims, abstention,
coverage, and any reference recognition separately. A source whose labels were
previously opened is development-only rather than held out.

### Final fresh longer-context checkpoint

One fresh official CC BY-NC EPIC-KITCHENS public source (21.59 seconds) was
then tested once, with labels closed through source selection, exact-input
review, output generation, and blind visual review. Each of two presentations
contained four sparse full-video context frames before and after a unique,
full-resolution 16-frame 4-fps local burst; both retained 24 native decoded
frames, exact source PTS, and a 1,440-token visual budget. The source visibly
showed a clear container being moved toward/at a black bin and then handled
away from it. Both outputs instead repeatedly asserted unsupported throwing;
one also truncated despite a 384-token allowance. Later annotation overlap did
not establish sampled-frame observability of the named reference actions.

This is the stopping point for local model benchmarking. The tested system can
provide local, provenance-linked visual observations, frame/PTS traceability,
schema validation, and explicit abstention boundaries. It is **not** validated
as reliable autonomous action memory, event-sequence recognition, or a
production action-recognition service. Do not represent this checkpoint as a
completed reliable-memory milestone or continue prompt/model optimization from
these evaluation clips.

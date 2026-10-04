# Local evidence query

The local question layer is a small retrieval interface, not a chat product or
a generative video-QA system. It sends a question to the already-installed
MiniLM embedder, searches the local FAISS caption index, and returns
timestamped evidence cards. It does not send content, telemetry, or queries to
an external service.

Index model observations first (this step uses only a Phase 4 caption file):

```sh
python -m slm_pipeline.cli visual-index \
  --captions data/processed/P02_05/vision_captions.json
```

Then retrieve cards from the terminal:

```sh
python -m slm_pipeline.cli ask "What might be happening?" --video P02_05 --limit 5
python -m slm_pipeline.cli ask "What might be happening?" --video P02_05 \
  --start-sec 20 --end-sec 40
```

Or start the simple local page:

The API dependencies are in `requirements-api.txt` and are included in the
lightweight test setup. Real queries also need the separately provisioned local
embedding model/runtime and a caption index. Read the macOS stability warning
below before starting real inference on Apple Silicon.

```sh
python -m slm_pipeline.api.main
```

It binds only to `127.0.0.1:8000`; open `http://127.0.0.1:8000/ask` from the
same Mac. There is no permissive CORS configuration. The service loads the
local retriever only on its first request, so merely starting the page does not
load MiniLM.

The legacy `/query`, `/summary/{video_id}`, `/tasks`, `/decisions`, and `/topics`
routes retain their success response shapes. Their backends are also loaded
only when requested, existing fact-acceptance guards remain in force, and
backend errors do not expose local filesystem details.

## Interpreting an answer

Every card includes the source video ID, source-relative time, observation,
evidence status, and the FAISS distance used for ranking. That distance is **not
calibrated confidence**. Typical current cards are raw local-model observations
and are displayed as `unvalidated_model_hypothesis`.

The returned `answer_status` is intentional:

- `supported_by_accepted_evidence` means a retrieved record was explicitly
  independently accepted as fact. The interface still points to cards instead
  of inventing a free-form claim.
- `insufficient_evidence` means model observations may be useful leads, but do
  not establish an action, event, identity, or causal fact.

The separate `status` field is `index_unavailable` when retrieval reports
missing index files. An empty newly initialized index returns `no_matches`.

No cutoff is presented as a calibrated relevance or answer threshold. Empty
queries, invalid bounds, missing index files, no hits, missing source-video
matches, and time ranges outside the available observations have explicit safe
states.

Source-video and source-time filters are applied before top-k ranking, so a
closer out-of-range observation cannot hide an eligible in-range result.

## Native macOS stability warning

Four local Python processes crashed with SIGSEGV during MiniLM/FAISS attempts
in the tested Apple-Silicon environment; the crash reports included native
PyTorch frames. Deferring the FAISS import until after MiniLM initialization
allowed one subsequent real query to complete, but this is an import-order
workaround, not a demonstrated root-cause fix or stability guarantee.

Laptop Python execution was stopped after the crashes. The later packaging,
regression fixes, and tests ran in a separate Linux cloud environment with
synthetic inputs and no neural model loading. Passing those tests does not
establish that the native macOS runtime is safe or stable. Further real-model
execution on that environment needs a separate stability investigation.

## Scope

This has been exercised against the current small public P02_05 caption index,
not benchmarked on ten-minute videos or broad video question answering. The
batch/index design can conceptually process longer inputs, but duration,
latency, and retrieval/answer quality for ten-minute material are unknown and
must be measured before any product claim. Do not treat a retrieved caption as
a factual video answer without independent accepted evidence.

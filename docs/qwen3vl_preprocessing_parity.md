# Qwen3-VL local preprocessing parity audit

This is a source-only audit of the installed local stack. It did not load a
model, run Metal generation, download software, or inspect private media.

## Evidence inspected

- Local environment: `mlx-vlm 0.7.4`, `mlx 0.32.3`, and `transformers 5.18.0`.
  `qwen-vl-utils` is not installed, so this runner does not use a second
  Qwen utility resize path.
- The installed MLX Qwen3-VL processor is explicitly documented in its source
  as a NumPy/PIL port of the Hugging Face Qwen3-VL processor. It uses patch
  size 16, temporal patch size 2, merge size 2, RGB conversion, 1/255
  rescaling, [0.5, 0.5, 0.5] normalization, bicubic resizing, and dimensions
  aligned to 32 pixels.
- The runner decodes exactly its declared frame count once, passes those
  decoded arrays directly to `prepare_inputs`, and generates from the returned
  native tensors. It records decoded hashes, source PTS, `video_grid_thw`,
  tensor shape, the merged-token count, and the settings that affect the video
  tensor.
- The Qwen reference describes a video processor pixel budget as a budget over
  frames × height × width; the local Qwen3-VL processor's resize calculation
  likewise uses padded frame count × resized height × resized width. The
  runner sets `video_processor.max_pixels` directly before preparation and
  records the resulting grid/token count.

## Result and limit

The static call-path audit found no second `qwen-vl-utils` resize path or
obvious double resize in the local runner: prompt construction carries a video
marker, while preprocessing receives the already-decoded frame array once.
This is **not** a claim of numerical parity with the official Transformers
runtime. The official `qwen-vl-utils` example has a distinct workflow: it
performs resize itself, then requires `do_resize=False` in the processor to
avoid a second resize. That workflow is absent locally, so neither its
`do_resize=False` flag nor its video metadata API can be assumed to validate
the MLX port.

Before any future model run, retain an audit record with these exact fields:

1. installed package versions and local model revision;
2. preprocessor contract (patch/temporal/merge factors, RGB, rescale,
   normalization, min/max pixels, sampling settings);
3. decoder frame IDs, source PTS, decoded hashes, and the exact ordering;
4. output `video_grid_thw`, pixel tensor shape, merged-token count, and
   whether max-pixel/token caps passed.

If those invariants agree across every prompt arm and control, that establishes
input parity **within the local experiment**, not external-runtime numerical
equivalence. Stop preprocessing debugging unless a concrete invariant differs
or a later explicit tensor-comparison test is authorized.

The official Qwen reference documents Qwen3-VL's patch size 16,
video-processor total pixel budget, video FPS/frame controls, and the
`qwen-vl-utils` resize/double-resize rule. See the
[Qwen3-VL reference usage guide](https://github.com/QwenLM/Qwen3-VL#new-qwen-vl-utils-usage).

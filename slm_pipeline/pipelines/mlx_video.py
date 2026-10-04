"""Local MLX-VLM video runner with timestamp-preserving result metadata.

This module intentionally invokes the locally installed MLX-VLM CLI in a
separate environment.  Keeping MLX out of the main inference environment
avoids changing its Transformers dependency set.  The caller owns video
sampling and supplies the original source timestamps; model output can never
replace them.
"""

from __future__ import annotations

import hashlib
import subprocess
import time
from pathlib import Path


DEFAULT_PROMPT = (
    "Describe only changes directly visible across this ordered video sequence. "
    "Do not infer actions that are not visually established."
)

# These questions are frozen before a pilot run.  They deliberately contain no
# benchmark labels and ask the model to abstain rather than complete an action
# from prior knowledge.  A result stores the key and exact text together.
FROZEN_DEVELOPMENT_QUESTIONS = {
    "wearer_hands_temporal_v1": (
        "From this first-person video, list at most three chronological, directly "
        "visible wearer-hand and object changes. For each, state the visible evidence "
        "and an approximate relative time. If a change or object is not clear, say "
        "uncertain. Do not infer an unseen action or use a dataset action label."
    ),
    "wearer_hands_temporal_v2_no_action": (
        "From this first-person video, list at most three chronological, directly "
        "visible wearer-hand and object changes. For each, state the visible evidence "
        "and an approximate relative time. If no temporal change is directly visible, "
        "say exactly 'no action observed'. If a change or object is not clear, say "
        "uncertain. Do not infer an unseen action or use a dataset action label."
    ),
    "structured_temporal_events_v1": (
        "Return JSON only with an events list. Each event must contain event_id, verb, "
        "object, actor (wearer, other_person, or unknown), direction (unknown, none, up, "
        "down, in, out, open, close, toward, or away), uncertain (true or false), and "
        "evidence_frame_indices containing at least two ordered frame positions starting at 0. "
        "List at most two directly visible changes. If no temporal change is directly visible, "
        "return {\"events\": []}. Do not infer unseen actions, states, or dataset labels."
    ),
    "structured_temporal_events_v2_visible_only": (
        "Return JSON only with an events list. Each event must contain event_id, verb, "
        "object, actor (wearer, other_person, or unknown), direction (unknown, none, up, "
        "down, in, out, open, close, toward, or away), uncertain (true or false), and "
        "evidence_frame_indices containing at least two ordered frame positions starting at 0. "
        "List at most two changes. Name an action or object only when a temporal change is "
        "directly established by at least two shown frames; do not complete an action from "
        "context or infer an unseen state, cause, or destination. If no directly visible "
        "temporal change is established, return {\"events\": []}."
    ),
    "structured_temporal_action_evidence_v1": (
        "Return JSON only with an events list. Each event must contain event_id, verb, "
        "object, actor (wearer, other_person, or unknown), direction (unknown, none, up, "
        "down, in, out, open, close, toward, or away), uncertain (true or false), and "
        "evidence_frame_indices containing at least two ordered frame positions starting at 0. "
        "List at most one directly visible hand-object transition. Name an action or object "
        "only when the same shown hand/object establishes its motion or state change across "
        "at least two frames. Do not infer an unseen state, cause, destination, or dataset label. "
        "If no such transition is directly visible, return {\"events\": []}."
    ),
    "structured_temporal_sequence_evidence_v1": (
        "Return JSON only with an events list. Every event must contain verb, object, actor "
        "(wearer, other_person, or unknown), direction (unknown, none, up, down, in, out, open, "
        "close, toward, or away), uncertain (true or false), and evidence_frame_indices with at "
        "least two ordered integer positions starting at 0. Do not emit event_id; the application "
        "assigns IDs. The presentation contains sparse full-video context plus an ordered dense local "
        "burst. List at most three directly visible transitions, and do not infer omitted motion, "
        "an unseen state, cause, destination, or dataset label. If no transition is directly visible, "
        "return {\"events\": []}."
    ),
}


def map_clip_times_to_source(clip_relative_timestamps_sec, source_clip_start_sec):
    """Map actual decoder times from a derived clip back to the source video."""
    values = [float(value) for value in clip_relative_timestamps_sec]
    if not values or any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError("clip_relative_timestamps_sec must be increasing")
    start = float(source_clip_start_sec)
    if start < 0:
        raise ValueError("source_clip_start_sec must be non-negative")
    return [round(start + value, 6) for value in values]


def decode_video_trace(video_path, source_clip_start_sec, fps=1.0,
                       video_num_frames=4, load_video_fn=None):
    """Decode and record the exact frame sequence requested for an MLX run.

    This is deliberately a small, lazy MLX boundary: importing this module
    does not require the isolated MLX runtime or a Metal device.  In a real
    MLX run it calls ``mlx_vlm.utils.load_video`` using the same fixed-frame
    sampling contract as :class:`MLXVideoRunner`.  The returned trace is
    decoder provenance only; add ``video_grid_thw`` from the model's actual
    preprocessor with :func:`video_input_trace` before making any statement
    about visual tokens.

    ``load_video_fn`` is an injection point for the no-Metal test suite.
    """
    if float(fps) <= 0:
        raise ValueError("fps must be positive")
    if int(video_num_frames) < 2:
        raise ValueError("video_num_frames must be at least two")
    if load_video_fn is None:
        from mlx_vlm.utils import VideoSampling, load_video

        load_video_fn = lambda path: load_video(
            str(path), sampling=VideoSampling(fps=float(fps), nframes=int(video_num_frames))
        )
    _frames, metadata = load_video_fn(Path(video_path))
    relative = [float(value) for value in metadata.timestamps]
    if len(relative) != int(video_num_frames):
        raise ValueError("decoder did not return the requested number of video frames")
    return {
        "decoded_frame_count": len(relative),
        "decoded_frame_indices": [int(value) for value in metadata.frames_indices],
        "clip_relative_timestamps_sec": relative,
        "source_timestamps_sec": map_clip_times_to_source(relative, source_clip_start_sec),
        "source_fps": float(metadata.fps),
        "sampled_fps": float(metadata.sampled_fps),
    }


def video_input_trace(metadata, video_grid_thw, source_clip_start_sec):
    """Return provenance for the frames and tokens that actually reached MLX.

    ``metadata`` is the MLX-VLM decoder metadata, not a preprocessing manifest.
    This distinction prevents a caller from recording desired frame times in
    place of the independently decoded ones.
    """
    decoded = decode_video_trace(
        "unused-by-injected-decoder", source_clip_start_sec,
        video_num_frames=len(metadata.timestamps),
        load_video_fn=lambda _path: (None, metadata),
    )
    grid = [int(value) for value in video_grid_thw]
    if len(grid) != 3 or min(grid) < 1:
        raise ValueError("video_grid_thw must contain three positive dimensions")
    rows = grid[0] * grid[1] * grid[2]
    if rows % 4:
        raise ValueError("video patch rows must be divisible by Qwen merge area")
    return {
        **decoded,
        "video_grid_thw": grid,
        "visual_patch_rows": rows,
        "merged_visual_tokens": rows // 4,
    }


def tensor_shape(value):
    """Return a JSON-safe tensor shape without materialising the tensor."""
    return [int(dimension) for dimension in value.shape]


def qwen_video_preprocessor_contract(video_processor):
    """Record Qwen video-preprocessor settings that affect native tensors.

    This is provenance, not a claim that an MLX port is numerically identical
    to every upstream runtime.  A future parity check still compares the
    recorded grid/tensor shape and normalization contract for the exact
    installed versions before interpreting an experiment.
    """
    return {
        key: getattr(video_processor, key, None)
        for key in (
            "patch_size", "temporal_patch_size", "merge_size", "min_pixels",
            "max_pixels", "do_convert_rgb", "do_rescale", "rescale_factor",
            "do_normalize", "image_mean", "image_std", "fps", "min_frames", "max_frames",
        )
    }


def decoded_frame_hashes(frames):
    """Hash actual decoder frames without treating hashes as semantic evidence."""
    values = []
    for frame in frames:
        if not hasattr(frame, "tobytes"):
            return []
        # Include shape/mode where available so different pixel layouts cannot
        # collide by concatenation alone. This is provenance for the native
        # input, not proof of any depicted event.
        prefix = f"{getattr(frame, 'mode', '')}:{getattr(frame, 'size', '')}:".encode()
        values.append(hashlib.sha256(prefix + frame.tobytes()).hexdigest())
    return values


class PersistentMLXVideoRunner:
    """Load one local MLX model once and retain exact native-video tensors.

    This class is intentionally usable only from the isolated MLX environment.
    Unlike the CLI adapter above, it decodes a window once, preprocesses those
    frames once, and passes those *same* tensors to generation.  Its trace is
    therefore provenance for the tensors that reached the model, not a proxy
    sampling schedule.  The imports are lazy so CPU-only pipeline tests stay
    free of MLX/Metal requirements.
    """

    def __init__(self, model_path, fps=2.0, video_num_frames=16,
                 max_pixels=3_211_264, max_new_tokens=128, question_id="wearer_hands_temporal_v1",
                 max_merged_visual_tokens=2_304, implementations=None):
        self.model_path = Path(model_path)
        self.fps = float(fps)
        self.video_num_frames = int(video_num_frames)
        self.max_pixels = int(max_pixels)
        self.max_new_tokens = int(max_new_tokens)
        self.max_merged_visual_tokens = int(max_merged_visual_tokens)
        self.question_id = str(question_id)
        self.implementations = implementations
        self.model = None
        self.processor = None
        self.cold_load_ms = None
        if (self.fps <= 0 or self.video_num_frames < 2 or self.max_pixels < 1
                or self.max_new_tokens < 1 or self.max_merged_visual_tokens < 1):
            raise ValueError("invalid persistent MLX video runner settings")
        if self.question_id not in FROZEN_DEVELOPMENT_QUESTIONS:
            raise KeyError(f"unknown frozen question: {self.question_id}")

    @property
    def question(self):
        return FROZEN_DEVELOPMENT_QUESTIONS[self.question_id]

    def _implementations(self):
        if self.implementations is not None:
            return self.implementations
        from mlx_vlm import load
        from mlx_vlm.generate import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import VideoSampling, load_video, prepare_inputs
        return {
            "load": load, "generate": generate, "apply_chat_template": apply_chat_template,
            "VideoSampling": VideoSampling, "load_video": load_video,
            "prepare_inputs": prepare_inputs,
        }

    def load_once(self):
        """Load only a local directory, exactly once for this process."""
        if self.model is not None:
            return
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"Missing local MLX model: {self.model_path}")
        started = time.perf_counter()
        self.model, self.processor = self._implementations()["load"](str(self.model_path))
        self.cold_load_ms = round((time.perf_counter() - started) * 1000, 2)

    def prepare_window(self, video_path, source_clip_start_sec, source_frame_pts=None):
        """Decode and preprocess one window without generating text.

        Callers may inspect the returned trace before generation.  The hard
        token cap makes an accidentally huge visual input a failure rather than
        a silent memory/performance change.
        """
        self.load_once()
        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"Missing video window: {video_path}")
        api = self._implementations()
        started = time.perf_counter()
        frames, metadata = api["load_video"](
            str(video_path),
            sampling=api["VideoSampling"](fps=self.fps, nframes=self.video_num_frames),
        )
        if len(metadata.timestamps) != self.video_num_frames:
            raise ValueError("decoder did not return every requested video frame")
        if len(frames) != self.video_num_frames:
            raise ValueError("decoder frame payload does not match requested video frame count")
        prompt = api["apply_chat_template"](
            self.processor, self.model.config, self.question,
            video=str(video_path), fps=metadata.sampled_fps, max_pixels=self.max_pixels,
        )
        # Qwen3-VL 0.7.4's processor accepts image kwargs but does not forward
        # ``max_pixels`` into its video processor.  Set the documented video
        # processor control directly, then record/verify the resulting grid.
        video_processor = getattr(self.processor, "video_processor", None)
        if video_processor is None or not hasattr(video_processor, "max_pixels"):
            raise RuntimeError("native Qwen video processor lacks a max_pixels control")
        video_processor.max_pixels = self.max_pixels
        inputs = api["prepare_inputs"](
            self.processor, videos=[frames], prompts=prompt,
            image_token_index=getattr(self.model.config, "image_token_index", None),
            add_special_tokens=False, video_metadata=[metadata], fps=[metadata.sampled_fps],
            max_pixels=self.max_pixels,
        )
        if "pixel_values_videos" not in inputs or "video_grid_thw" not in inputs:
            raise RuntimeError("native video preprocessing did not produce video tensors")
        grids = inputs["video_grid_thw"].tolist()
        if len(grids) != 1:
            raise RuntimeError("persistent runner expects exactly one video per window")
        trace = video_input_trace(metadata, grids[0], source_clip_start_sec)
        if source_frame_pts is not None:
            exact_pts = [float(value) for value in source_frame_pts]
            if len(exact_pts) != self.video_num_frames or any(
                    right <= left for left, right in zip(exact_pts, exact_pts[1:])):
                raise ValueError("source_frame_pts must be strictly increasing and match decoded frame count")
            trace["source_timestamps_sec"] = exact_pts
            trace["source_timestamp_mapping"] = "caller_provided_exact_source_pts_for_ordered_decoded_frames"
        trace.update({
            "decoded_frame_hashes": decoded_frame_hashes(frames),
            "preprocessor_input_keys": sorted(inputs.keys()),
            "pixel_values_videos_shape": tensor_shape(inputs["pixel_values_videos"]),
            "model_relative_marker_fps": float(metadata.sampled_fps),
            "model_relative_markers_are_exact_source_pts": False,
            "configured_video_max_pixels": self.max_pixels,
            "video_preprocessor_contract": qwen_video_preprocessor_contract(video_processor),
            "preprocessing_parity_status": "recorded_local_contract_only; numerical parity with another runtime is not implied",
        })
        if trace["merged_visual_tokens"] > self.max_merged_visual_tokens:
            raise RuntimeError(
                f"visual token cap exceeded: {trace['merged_visual_tokens']} > "
                f"{self.max_merged_visual_tokens}"
            )
        return {
            "prompt": prompt, "inputs": inputs, "input_trace": trace,
            "preparation_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    def generate_prepared(self, prepared):
        """Generate from the exact tensors returned by :meth:`prepare_window`."""
        inputs = prepared["inputs"]
        generation_kwargs = {
            key: value for key, value in inputs.items()
            if key not in ("input_ids", "pixel_values", "attention_mask")
        }
        return self._implementations()["generate"](
            self.model, self.processor, prepared["prompt"],
            input_ids=inputs["input_ids"], mask=inputs.get("attention_mask"),
            temperature=0.0, max_tokens=self.max_new_tokens, verbose=False,
            **generation_kwargs,
        )

    def observe_window(self, video_path, source_clip_start_sec, source_frame_pts=None):
        """Decode, preprocess, trace, and generate from one public video window.

        The return value does not score labels.  A later evaluator must review
        visual support, unsupported claims, abstention, actor, object, and
        chronology separately from held-out action labels.
        """
        started = time.perf_counter()
        prepared = self.prepare_window(video_path, source_clip_start_sec, source_frame_pts=source_frame_pts)
        response = self.generate_prepared(prepared)
        return {
            "question_id": self.question_id,
            "question": self.question,
            "observation": response.text,
            "source_type": "model",
            "input_type": "native_video_tensors",
            "backend": "mlx_vlm",
            "model": self.model_path.name,
            "device": "metal",
            "cold_load_ms": self.cold_load_ms,
            "window_latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "preparation_ms": prepared["preparation_ms"],
            "mlx_peak_memory_gb": float(response.peak_memory),
            "input_trace": prepared["input_trace"],
            "evaluation_schema": {
                "visual_claim_support": "unreviewed",
                "unsupported_claims": "unreviewed",
                "actor_object_order": "unreviewed",
                "held_out_action_label": "not supplied to model; score separately",
                "viewpoint_limitations": "record separately when an action is not visually determinable",
            },
        }


class MLXVideoRunner:
    """Run a local MLX-VLM video model without remote-code or network flags."""

    def __init__(self, model_path, python_executable, max_new_tokens=24, fps=1.0,
                 video_num_frames=4):
        self.model_path = Path(model_path)
        self.python_executable = str(python_executable)
        self.max_new_tokens = int(max_new_tokens)
        self.fps = float(fps)
        self.video_num_frames = int(video_num_frames)
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.video_num_frames < 2:
            raise ValueError("video_num_frames must be at least two")

    def command(self, video_path, prompt=DEFAULT_PROMPT):
        """Build a local-only MLX-VLM command for a video window."""
        video_path = Path(video_path)
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"Missing local MLX model: {self.model_path}")
        if not video_path.is_file():
            raise FileNotFoundError(f"Missing video window: {video_path}")
        return [
            self.python_executable,
            "-m", "mlx_vlm.generate",
            "--model", str(self.model_path),
            "--video", str(video_path),
            "--fps", str(self.fps),
            "--video-num-frames", str(self.video_num_frames),
            "--prompt", prompt,
            "--max-tokens", str(self.max_new_tokens),
            "--temp", "0.0",
        ]

    def observe(self, video_path, input_trace, prompt=DEFAULT_PROMPT):
        """Run one pre-sampled window using actual decoder provenance.

        ``input_trace`` must come from :func:`video_input_trace`; raw manifest
        timestamps are deliberately not accepted because CLI video sampling can
        choose different frames.
        """
        timestamps = input_trace.get("source_timestamps_sec", [])
        if len(timestamps) != self.video_num_frames:
            raise ValueError("input_trace must describe every decoded video frame")
        started = time.perf_counter()
        completed = subprocess.run(
            self.command(video_path, prompt), check=True, text=True,
            capture_output=True,
        )
        return {
            "observation": completed.stdout.strip(),
            "source_timestamps_sec": timestamps,
            "source_time_range_sec": [timestamps[0], timestamps[-1]],
            "input_trace": input_trace,
            "source_type": "model",
            "input_type": "video",
            "backend": "mlx_vlm",
            "model": self.model_path.name,
            "device": "metal",
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

"""Run the frozen public two-clip Qwen3-VL temporal/control batch locally.

This executable deliberately knows no EPIC action annotations. It uses one
persistent local model instance, records native input provenance, and writes
raw output for a later label-blind visual review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import time
from pathlib import Path

from slm_pipeline.pipelines.mlx_video import PersistentMLXVideoRunner


MODEL_REVISION = "2fd8dacbdb8f1e54b8c005f081ec5bf79c56376b"
SOURCE_URL = "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/train/{participant}/{video_id}.MP4"
FROZEN_CONFIG = {
    "model_repository": "mlx-community/Qwen3-VL-4B-Instruct-4bit",
    "model_revision": MODEL_REVISION,
    "question_id": "wearer_hands_temporal_v2_no_action",
    "fps_request": 2.0,
    "decoded_frame_count": 16,
    "video_max_pixels": 3_211_264,
    "merged_visual_token_cap": 2_304,
    "temperature": 0.0,
    "max_new_tokens": 128,
}
CLIPS = (
    {"video_id": "P03_15", "participant": "P03", "official_metadata_duration_sec": 10.477133,
     "official_head_bytes": 39_548_685},
    {"video_id": "P06_02", "participant": "P06", "official_metadata_duration_sec": 13.947267,
     "official_head_bytes": 52_556_185},
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_repeated_still_control(source: Path, target: Path) -> dict:
    """Make an 8-second, 16-frame native MP4 from source frame zero."""
    image = target.with_suffix(".source-still.png")
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-ss", "0", "-i", str(source),
        "-frames:v", "1", str(image),
    ], check=True)
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-loop", "1", "-framerate", "2", "-i", str(image),
        # Lossless RGB avoids codec-introduced pixel drift between otherwise
        # identical control frames; the runner still verifies decoded hashes.
        "-t", "8", "-c:v", "libx264rgb", "-crf", "0", "-preset", "ultrafast",
        "-pix_fmt", "rgb24", str(target),
    ], check=True)
    return {
        "kind": "native repeated-still MP4 from source frame at 0.0 seconds",
        "source_still_sha256": sha256(image),
        "control_mp4_sha256": sha256(target),
        "control_duration_sec": 8.0,
    }


def observe_pair(runner: PersistentMLXVideoRunner, source: Path, control: Path) -> dict:
    temporal = runner.observe_window(source, 0.0)
    repeated = runner.observe_window(control, 0.0)
    temporal_trace, repeated_trace = temporal["input_trace"], repeated["input_trace"]
    if temporal_trace["decoded_frame_count"] != repeated_trace["decoded_frame_count"]:
        raise RuntimeError("control did not use the frozen decoded-frame budget")
    if temporal_trace["merged_visual_tokens"] != repeated_trace["merged_visual_tokens"]:
        raise RuntimeError("control did not use the same merged-token budget")
    hashes = repeated_trace.get("decoded_frame_hashes", [])
    if len(hashes) != FROZEN_CONFIG["decoded_frame_count"] or len(set(hashes)) != 1:
        raise RuntimeError("repeated-still control did not preserve identical native decoded frames")
    return {"temporal": temporal, "repeated_still_control": repeated}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--report", type=Path, default=Path("data/evaluation/qwen3vl4b_bounded_two_clip_raw.json"))
    parser.add_argument("--run-id", default="qwen3vl4b_bounded_two_clip_v2_local_001")
    args = parser.parse_args()
    root = args.root.resolve()
    model = root / "data/models/mlx/Qwen3-VL-4B-Instruct-4bit"
    raw_dir, experiment_dir = root / "data/raw", root / "data/experiments/qwen3vl4b_bounded_controls"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    runner = PersistentMLXVideoRunner(model, fps=FROZEN_CONFIG["fps_request"],
                                      video_num_frames=FROZEN_CONFIG["decoded_frame_count"],
                                      max_pixels=FROZEN_CONFIG["video_max_pixels"],
                                      max_new_tokens=FROZEN_CONFIG["max_new_tokens"],
                                      max_merged_visual_tokens=FROZEN_CONFIG["merged_visual_token_cap"],
                                      question_id=FROZEN_CONFIG["question_id"])
    started = time.perf_counter()
    results = []
    for clip in CLIPS:
        source = raw_dir / f"{clip['video_id']}.MP4"
        if not source.is_file() or source.stat().st_size != clip["official_head_bytes"]:
            raise FileNotFoundError(f"verified source file missing or wrong size: {source}")
        control = experiment_dir / f"{clip['video_id']}_repeated_still.mp4"
        print(json.dumps({"stage": "build_control", "video_id": clip["video_id"]}), flush=True)
        control_info = create_repeated_still_control(source, control)
        print(json.dumps({"stage": "run_temporal_then_control", "video_id": clip["video_id"]}), flush=True)
        pair = observe_pair(runner, source, control)
        print(json.dumps({"stage": "completed_pair", "video_id": clip["video_id"]}), flush=True)
        results.append({
            **clip,
            "official_source_url": SOURCE_URL.format(**clip),
            "local_source_sha256": sha256(source),
            "local_source_bytes": source.stat().st_size,
            "control": control_info,
            **pair,
        })
    report = {
        "run_id": args.run_id,
        "scope": "Bounded local visual test on two public EPIC-KITCHENS videos; no labels were read by this executable.",
        "selection": "Technical metadata and official HEAD byte size only, before model outputs.",
        "frozen_config": FROZEN_CONFIG,
        "model_loaded_once_within_batch": True,
        "cold_load_ms": runner.cold_load_ms,
        "batch_wall_ms": round((time.perf_counter() - started) * 1000, 2),
        "process_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "results": results,
        "review_status": "raw outputs only; visual claim review and held-out labels are deliberately separate",
    }
    report_path = args.report if args.report.is_absolute() else root / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": args.run_id, "report": str(report_path), "cold_load_ms": runner.cold_load_ms,
                      "clips": [item["video_id"] for item in results]}, indent=2))


if __name__ == "__main__":
    main()

"""Uniform temporal-window observations from Phase 3.5 frames, independent of labels."""

import argparse
import json
import platform
import resource
import time
from pathlib import Path

from slm_pipeline.pipelines.phase_4_vision import VisionProcessor


def uniform_windows(usable_frames, window_size=4):
    """Chunk manifest order uniformly, preserving each decoded source timestamp."""
    if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 2:
        raise ValueError("window_size must be an integer of at least two")
    ordered = sorted(usable_frames, key=lambda frame: float(frame["source_timestamp_sec"]))
    for left in range(0, len(ordered), window_size):
        window = ordered[left:left + window_size]
        if len(window) >= 2:
            yield window


def observe(video_dir, window_size=4):
    video_dir = Path(video_dir)
    manifest = json.loads((video_dir / "frames_manifest.json").read_text(encoding="utf-8"))
    frames = manifest.get("usable_frames", [])
    windows = list(uniform_windows(frames, window_size))
    if not windows:
        raise ValueError("Need at least two usable frames for a temporal observation")
    processor = VisionProcessor()
    started = time.perf_counter()
    observations = []
    for index, window in enumerate(windows):
        timestamps = [float(frame["source_timestamp_sec"]) for frame in window]
        result = processor.generate_video_observation(
            [video_dir / frame["path"] for frame in window], timestamps
        )
        if not result["success"]:
            raise RuntimeError(result.get("error", "Temporal observation failed"))
        observations.append({
            "window_id": f"uniform_{index:03d}",
            "selection": "uniform_manifest_order",
            "source_timestamps_sec": timestamps,
            "source_time_range_sec": [timestamps[0], timestamps[-1]],
            "frame_ids": [Path(frame["path"]).name for frame in window],
            **result,
        })
    output = {"video_id": manifest.get("video_id", video_dir.name), "source_type": "model",
              "input_type": "video", "window_size": window_size,
              "window_selection": "uniform_manifest_order", "observations": observations,
              "run_metrics": {"elapsed_sec": round(time.perf_counter() - started, 3),
                              "max_rss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                              "max_rss_unit": "bytes" if platform.system() == "Darwin" else "KiB"}}
    (video_dir / "temporal_observations.json").write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--window-size", type=int, default=4)
    args = parser.parse_args(argv)
    observe(args.video_dir, args.window_size)


if __name__ == "__main__":
    main()

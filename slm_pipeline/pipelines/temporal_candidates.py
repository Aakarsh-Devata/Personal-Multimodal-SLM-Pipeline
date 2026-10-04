"""Write a CPU-only, auditable temporal candidate-window selection artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_pipeline.pipelines.temporal_events import (
    select_fixed_video_windows,
    select_video_candidate_windows,
    select_video_uniform_windows,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--max-windows", type=int, default=3)
    parser.add_argument("--max-selected-frames", type=int, default=12)
    parser.add_argument("--minimum-new-frames", type=int)
    parser.add_argument("--mode", choices=("change", "uniform", "anchored_change"), default="change")
    parser.add_argument("--frame-source", choices=("manifest", "fixed"), default="manifest")
    parser.add_argument("--expected-frame-count", type=int)
    parser.add_argument("--uniform-anchor-count", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.frame_source == "fixed":
        result = select_fixed_video_windows(
            args.video_dir, window_size=args.window_size, max_windows=args.max_windows,
            max_selected_frames=args.max_selected_frames, expected_frame_count=args.expected_frame_count,
            mode=args.mode, minimum_new_frames_per_selected_window=args.minimum_new_frames,
            uniform_anchor_count=args.uniform_anchor_count,
        )
    elif args.mode == "change":
        result = select_video_candidate_windows(
            args.video_dir, window_size=args.window_size, max_windows=args.max_windows,
            max_selected_frames=args.max_selected_frames,
            minimum_new_frames_per_selected_window=args.minimum_new_frames,
        )
    else:
        result = select_video_uniform_windows(
            args.video_dir, window_size=args.window_size, max_windows=args.max_windows,
            max_selected_frames=args.max_selected_frames,
        )
    output = args.output or args.video_dir / f"temporal_{args.mode}_windows.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    main()

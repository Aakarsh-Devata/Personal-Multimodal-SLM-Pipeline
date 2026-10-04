"""Shared, deterministic input selection for every pipeline phase."""
from pathlib import Path

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi"}


def validate_video_id(video_id):
    if not video_id or video_id in {".", ".."} or Path(video_id).name != video_id or "\\" in video_id:
        raise ValueError("video ID must be a single directory/file stem")
    return video_id


def selected_raw_videos(config, video_id=None):
    root = Path(config["paths"]["raw_dir"])
    if video_id is not None:
        validate_video_id(video_id)
    if not root.is_dir():
        raise FileNotFoundError(f"Raw directory not found: {root}")
    videos = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
                    and (video_id is None or p.stem == video_id))
    if not videos:
        raise FileNotFoundError(f"No matching raw videos found in {root}")
    if len({p.stem for p in videos}) != len(videos):
        raise ValueError("Multiple input files have the same video ID; rename before processing")
    return videos


def selected_video_dirs(config, video_id=None):
    root = Path(config["paths"]["processed_dir"])
    if video_id is not None:
        validate_video_id(video_id)
        folder = root / video_id
        if not folder.is_dir():
            raise FileNotFoundError(f"Processed video not found: {video_id}")
        return [folder]
    if not root.is_dir():
        raise FileNotFoundError(f"Processed directory not found: {root}")
    folders = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    if not folders:
        raise FileNotFoundError(f"No processed videos found in {root}")
    return folders


def phase_cli(main):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="Only this exact video ID")
    args = parser.parse_args()
    main(video_id=args.video)

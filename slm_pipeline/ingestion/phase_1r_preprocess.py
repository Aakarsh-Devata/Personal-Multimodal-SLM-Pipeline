#!/usr/bin/env python3
"""Extract mono audio and frames while preserving decoded source timestamps."""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from slm_pipeline.config import config
from slm_pipeline.runtime import selected_raw_videos, phase_cli


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command):
    """Argument vectors avoid shell quoting bugs; a failed stage is a failure."""
    return subprocess.run(command, check=True, capture_output=True, text=True)


def ffprobe_info(path):
    return json.loads(run(["ffprobe", "-v", "error", "-print_format", "json",
                           "-show_format", "-show_streams", str(path)]).stdout)


def extract_frames(video, folder, prefix, selection):
    folder.mkdir(parents=True, exist_ok=True)
    # Reruns replace only this stage's generated files, never source media.
    for old in folder.glob(f"{prefix}_*.jpg"):
        old.unlink()
    result = run(["ffmpeg", "-hide_banner", "-y", "-i", str(video), "-map", "0:v:0", "-an",
                  "-vf", f"setpts=PTS-STARTPTS,select='{selection}',showinfo",
                  "-fps_mode", "vfr", "-pix_fmt", "yuvj420p", "-q:v", "2", str(folder / f"{prefix}_%06d.jpg")])
    timestamps = [float(value) for value in re.findall(r"\bn:\s*\d+\s+pts:\s*\S+\s+pts_time:([\d.eE+-]+)", result.stderr)]
    files = sorted(folder.glob(f"{prefix}_*.jpg"))
    if len(files) != len(timestamps):
        raise RuntimeError(f"Frame/timestamp count mismatch: {len(files)} frames, {len(timestamps)} timestamps")
    return {str(path.relative_to(folder.parent)): {"timestamp_sec": ts, "source_timestamp_sec": ts,
                                                  "sampling": prefix}
            for path, ts in zip(files, timestamps)}


def process_video(video):
    video = Path(video)
    out_dir = Path(config["paths"]["processed_dir"]) / video.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    info = ffprobe_info(video)
    if not any(s.get("codec_type") == "video" for s in info.get("streams", [])):
        raise ValueError(f"No video stream: {video.name}")
    video_stream = next(stream for stream in info['streams'] if stream.get('codec_type') == 'video')
    audio_stream = next((stream for stream in info['streams'] if stream.get('codec_type') == 'audio'), None)
    # Without this guard, extracting WAV resets its origin while video setpts resets
    # a different one. Reject unsupported offsets rather than misalign evidence.
    for stream in (video_stream, audio_stream):
        if stream is not None and abs(float(stream.get('start_time', 0))) > 0.000001:
            raise ValueError("Nonzero audio/video stream start is unsupported; normalize both streams to a common zero origin first")
    duration = float(video_stream.get('duration') or info.get("format", {}).get("duration", 0))
    if duration <= 0:
        raise ValueError("Video duration must be positive")
    has_audio = any(s.get("codec_type") == "audio" for s in info.get("streams", []))
    if has_audio:
        run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-map", "0:a:0", "-vn", "-ac", "1",
             "-ar", "16000", "-sample_fmt", "s16", str(out_dir / "audio_clean.wav")])
    fps = config["pipeline"]["frame_sampling_fps"]
    threshold = config["pipeline"]["scene_threshold"]
    frames = extract_frames(video, out_dir / "frames_fixed", "frame",
                            f"isnan(prev_selected_t)+gte(t-prev_selected_t,{1 / fps})")
    scene = extract_frames(video, out_dir / "frames_scene", "scene", f"gt(scene,{threshold})")
    if not frames:
        raise RuntimeError("Video produced no fixed sample frames")
    frames.update(scene)
    # Quality stage falls back to fixed frames directly; no renamed timestamps.
    (out_dir / "frame_timestamps.json").write_text(json.dumps({"schema_version": 1,
        "video_id": video.stem, "timebase": "seconds from first decoded video frame",
        "frames": frames}, indent=2), encoding="utf-8")
    meta = {"schema_version": 1, "video_id": video.stem, "source_filename": video.name,
            "source_sha256": sha256(video), "duration_sec": duration, "has_audio": has_audio,
            "video_format": info.get("format", {}), "counts": {
                "frames_fixed": len(frames) - len(scene), "frames_scene": len(scene)},
            "parameters": {"frame_sampling_fps": fps, "scene_threshold": threshold}}
    (out_dir / "processed_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main(video_id=None):
    for video in selected_raw_videos(config, video_id):
        process_video(video)


if __name__ == "__main__":
    phase_cli(main)

#!/usr/bin/env python3
"""
phase_b_preprocess.py
Run from ProjectRoot. Scans raw/ for videos and produces processed/<video-id>/...
"""

import os, subprocess, json, hashlib, shlex, shutil
from pathlib import Path
import sys

# Import config
sys.path.insert(0, str(Path.cwd()))
from slm_pipeline.config import config

# ====== CONFIG ======
RAW_DIR = Path(config['paths']['raw_dir'])
PROC_ROOT = Path(config['paths']['processed_dir'])
FIXED_FPS = 1        # extract one frame per second
SCENE_THRESH = 0.2   # detect subtle scene changes
# =====================

def sha256(path: Path):
    """Compute SHA256 checksum for a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def run(cmd):
    """Run shell command safely with logs."""
    print(f"\n▶️  RUN: {cmd}")
    try:
        subprocess.check_call(shlex.split(cmd))
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Command failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def ensure(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def ffprobe_info(path: Path):
    """Return ffprobe JSON info."""
    cmd = f'ffprobe -v error -print_format json -show_format -show_streams "{path}"'
    out = subprocess.check_output(shlex.split(cmd))
    return json.loads(out)

# =====================
# MAIN PIPELINE
# =====================

for video in RAW_DIR.iterdir():
    if not video.is_file() or video.suffix.lower() not in (".mp4", ".mov", ".mkv", ".avi"):
        continue

    vid_id = video.stem
    out_dir = PROC_ROOT / vid_id
    ensure(out_dir)
    ensure(out_dir / "frames_fixed")
    ensure(out_dir / "frames_scene")
    ensure(out_dir / "thumbs")
    ensure(out_dir / "audio_chunks")

    print(f"\n🎬 Processing video: {video.name}")
    print(f"Output folder: {out_dir}")

    # 1) Extract audio (16 kHz mono)
    audio_path = out_dir / "audio.wav"
    cmd_audio = f'ffmpeg -y -i "{video}" -vn -ac 1 -ar 16000 -sample_fmt s16 "{audio_path}"'
    run(cmd_audio)
    print("✅ Audio extracted successfully:", audio_path)

    # 2) Skip denoise (arnndn not available on macOS)
    audio_clean = out_dir / "audio_clean.wav"
    try:
        shutil.copy(audio_path, audio_clean)
        print("✅ Copied audio.wav → audio_clean.wav (skipped denoise)")
    except Exception as e:
        print("⚠️ Failed to copy audio:", e)

    # 3) Fixed-rate frames (1 fps)
    run(f'ffmpeg -y -i "{video}" -vf fps={FIXED_FPS} "{out_dir}/frames_fixed/frame_%06d.jpg"')
    print("✅ Extracted fixed-rate frames.")

    # 4) Scene-based frames (detect scene cuts)
    scene_dir = out_dir / "frames_scene"
    print("🔍 Detecting scene changes...")
    run(f'ffmpeg -y -i "{video}" -vf "select=\'gt(scene,{SCENE_THRESH})\',showinfo" -fps_mode vfr "{scene_dir}/scene_%06d.jpg"')

    # Fallback if no scene-change frames
    scene_frames = list(scene_dir.glob("*.jpg"))
    if not scene_frames:
        print("⚠️ No scene changes detected — copying fallback frames from frames_fixed/")
        fixed_frames = sorted((out_dir / "frames_fixed").glob("*.jpg"))
        if fixed_frames:
            sample_frames = fixed_frames[::max(1, len(fixed_frames)//5)]  # ~5 evenly spaced
            for f in sample_frames:
                dst = scene_dir / f"fallback_{f.name}"
                shutil.copy(f, dst)
            print(f"✅ Fallback: {len(sample_frames)} frames copied to frames_scene/")
        else:
            print("❌ No fallback frames available either.")
    else:
        print(f"✅ Scene-based frames extracted: {len(scene_frames)}")

    # 5) Thumbnails from fixed frames (scale width 320)
    run(f'ffmpeg -y -pattern_type glob -i "{out_dir}/frames_fixed/frame_*.jpg" -vf "scale=320:-1" "{out_dir}/thumbs/thumb_%04d.jpg"')
    print("✅ Thumbnails generated.")

    # 6) Chunk audio into 10-minute segments (600s)
    run(f'ffmpeg -y -i "{audio_clean}" -f segment -segment_time 600 -c copy "{out_dir}/audio_chunks/chunk_%03d.wav"')
    print("✅ Audio chunks created.")

    # 7) Gather metadata and checksums
    meta = {}
    try:
        info = ffprobe_info(video)
        meta['video_format'] = info.get('format', {})
    except Exception as e:
        meta['video_format'] = {'error': str(e)}

    meta['audio'] = {}
    try:
        ainfo = ffprobe_info(audio_clean)
        meta['audio']['streams'] = ainfo.get('streams', [])
        meta['audio']['format'] = ainfo.get('format', {})
    except Exception as e:
        meta['audio']['error'] = str(e)

    # Counts
    meta['counts'] = {
        'frames_fixed': len(list((out_dir / "frames_fixed").glob("*.jpg"))),
        'frames_scene': len(list((out_dir / "frames_scene").glob("*.jpg"))),
        'thumbs': len(list((out_dir / "thumbs").glob("*.jpg"))),
        'audio_chunks': len(list((out_dir / "audio_chunks").glob("*.wav"))),
    }

    # Checksums
    meta['checksums'] = {}
    for p in (out_dir / "frames_fixed").glob("*.jpg"):
        meta['checksums'][p.name] = sha256(p)
    meta['checksums']['audio_clean.wav'] = sha256(audio_clean)

    # Write processed_meta.json
    meta_path = out_dir / "processed_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Processed {video.name} -> {out_dir}")
    print(f"📄 Metadata saved to: {meta_path}")

print("\n🎉 All videos processed successfully.")

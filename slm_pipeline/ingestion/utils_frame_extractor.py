#!/usr/bin/env python3
"""
phase_c2_extract_frames.py
Re-extract frames (fixed FPS + scene-change) for videos listed in processed/.
Adds fallback: if no scene changes detected, copies representative frames.
"""

import subprocess, shlex, json, shutil
from pathlib import Path

# --- CONFIG ---
RAW_DIR = Path("/Users/aakarsh/Desktop/ProjectRoot/phone_videos/raw")
PROC_ROOT = Path("/Users/aakarsh/Desktop/ProjectRoot/phone_videos/processed")

FIXED_FPS = 1
SCENE_THRESH = 0.
FALLBACK_FRAME_COUNT = 3  # number of frames to copy if no scene detected


def run(cmd: str):
    """Run a shell command safely."""
    print(f"\n▶ Running: {cmd}")
    try:
        subprocess.check_call(shlex.split(cmd))
        print("✅ Done.")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Command failed: {cmd}\n{e}")


def ensure(p: Path):
    """Ensure directory exists."""
    p.mkdir(parents=True, exist_ok=True)


# --- MAIN LOOP ---
for video_dir in PROC_ROOT.iterdir():
    if not video_dir.is_dir():
        continue

    vid_name = video_dir.name
    print(f"\n🎞 Processing folder: {vid_name}")

    # Find corresponding raw video
    video = RAW_DIR / f"{vid_name}.mp4"
    if not video.exists():
        for ext in [".mov", ".mkv", ".avi"]:
            alt = RAW_DIR / f"{vid_name}{ext}"
            if alt.exists():
                video = alt
                break
        else:
            print(f"⚠️ No matching raw video found for {vid_name}, skipping.")
            continue

    print(f"🎬 Found raw video: {video}")

    # Prepare output directories
    frames_fixed = video_dir / "frames_fixed_v2"
    frames_scene = video_dir / "frames_scene_v2"
    ensure(frames_fixed)
    ensure(frames_scene)

    # --- 1) Fixed-rate frame extraction ---
    cmd_fixed = (
        f'ffmpeg -y -i "{video}" -vf fps={FIXED_FPS} "{frames_fixed}/frame_%06d.jpg"'
    )
    run(cmd_fixed)

    # --- 2) Scene-change frame extraction ---
    cmd_scene = (
        f'ffmpeg -y -i "{video}" '
        f'-vf "select=\'gt(scene,{SCENE_THRESH})\',showinfo" -vsync vfr '
        f'"{frames_scene}/scene_%06d.jpg"'
    )
    run(cmd_scene)

    # --- 3) Check scene frames and fallback if empty ---
    scene_frames = list(frames_scene.glob("*.jpg"))
    fixed_frames = list(frames_fixed.glob("*.jpg"))
    fallback_used = False

    if not scene_frames:
        fallback_used = True
        print(f"⚠️ No scene changes detected for {vid_name}. Creating fallback frames.")
        if fixed_frames:
            step = max(1, len(fixed_frames) // FALLBACK_FRAME_COUNT)
            for i, src in enumerate(fixed_frames[::step][:FALLBACK_FRAME_COUNT]):
                dst = frames_scene / f"fallback_{i+1:03d}.jpg"
                shutil.copy(src, dst)
            print(f"✅ Added {FALLBACK_FRAME_COUNT} fallback frames to frames_scene_v2/")
        else:
            print("⚠️ No fixed frames found either — skipping fallback.")

    # --- 4) Metadata summary ---
    meta_summary = {
        "video_name": vid_name,
        "frames_fixed_count": len(fixed_frames),
        "frames_scene_count": len(list(frames_scene.glob('*.jpg'))),
        "source_video": str(video),
        "fallback_used": fallback_used,
    }

    with open(video_dir / "frame_summary.json", "w", encoding="utf-8") as f:
        json.dump(meta_summary, f, indent=2)

    print(f"✅ Frame extraction complete for {vid_name}")
    print(json.dumps(meta_summary, indent=2))

print("\n🌟 All videos processed successfully!")

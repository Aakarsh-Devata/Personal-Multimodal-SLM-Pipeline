#!/usr/bin/env python3
"""
phase_c3_preprocess_frames.py

Phase C3 - Frame preprocessing & quality validation (production-ready).

Produces per-video frames_manifest.json and optional frames_ready/ cache.

Configurable options near the top.
"""

import os
import sys
import json
import shutil
import time
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image, ImageOps
import imagehash
import cv2
from tqdm import tqdm

# ----------------------
# CONFIG
# ----------------------
import sys
sys.path.insert(0, str(Path.cwd()))
from slm_pipeline.config import config

PROC_ROOT = Path(config['paths']['processed_dir'])

# Frame input folders (created by Phase B/C2)
FIXED_SUB = "frames_fixed"
SCENE_SUB = "frames_scene"

# Output ready folder (resized, RGB images)
READY_SUB = "frames_ready"

# Manifest filename
MANIFEST_FN = "frames_manifest.json"

# Resize / normalization settings
TARGET_SHORTER_SIDE = 512      # target smaller dimension; keeps aspect ratio
FORCE_SQUARE = False           # if True, center-crop/pad to exact TARGET x TARGET
SQUARE_SIZE = 512              # if FORCE_SQUARE True, final size

# Quality thresholds
BLUR_VARIANCE_THRESH = 60.0    # Laplacian variance below this => blurry
DARK_MEAN_THRESH = 40          # below -> too dark (0-255)
BRIGHT_MEAN_THRESH = 215       # above -> too bright

# Duplicate removal settings
HASH_SIZE = 16                 # perceptual hash size (imagehash); larger = more sensitive
HAMMING_THRESH = 6             # Hamming distance <= this considered duplicate

# Sampling / retention policy
KEEP_DUPLICATES = False        # if True, mark duplicates but keep; else remove duplicates from usable set

# Behavior toggles
WRITE_READY_IMAGES = True      # write processed images into frames_ready/
REMOVE_EXCLUDED_FILES = False  # if True, physically remove excluded files (use with care)
VERBOSE = True

# ----------------------
# helpers
# ----------------------

def log(*a, **k):
    if VERBOSE:
        print(*a, **k)


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    files = []
    for e in exts:
        files.extend(sorted(folder.glob(e)))
    return files


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def read_image_cv(path: Path):
    # read BGR via cv2 then convert to RGB PIL
    arr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return None
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(arr)


def pil_to_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img)


def is_blurry_pil(img: Image.Image, thresh: float = BLUR_VARIANCE_THRESH) -> Tuple[bool, float]:
    # compute Laplacian variance on grayscale
    arr = np.array(img.convert("L"))
    var = cv2.Laplacian(arr, cv2.CV_64F).var()
    return (var < thresh, float(var))


def mean_brightness(img: Image.Image) -> float:
    arr = np.array(img.convert("L"))
    return float(arr.mean())


def resize_and_normalize(img: Image.Image, target_short: int = TARGET_SHORTER_SIDE,
                         force_square: bool = FORCE_SQUARE, square_size: int = SQUARE_SIZE) -> Image.Image:
    w, h = img.size
    if force_square:
        # center-crop to square on smaller side, then resize
        img = ImageOps.fit(img, (square_size, square_size), Image.Resampling.LANCZOS)
        return img.convert("RGB")

    # maintain aspect ratio, scale so shorter side == target_short
    if min(w, h) == 0:
        return img.convert("RGB")
    if w < h:
        new_w = target_short
        new_h = int(h * (target_short / w))
    else:
        new_h = target_short
        new_w = int(w * (target_short / h))
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return img.convert("RGB")


def compute_hash(img: Image.Image, hash_size: int = HASH_SIZE):
    # use phash (perceptual hash)
    return imagehash.phash(img, hash_size=hash_size)


def hamming_distance(h1: imagehash.ImageHash, h2: imagehash.ImageHash) -> int:
    return (h1 - h2)


# ----------------------
# Core per-video processing
# ----------------------

def process_video_folder(video_dir: Path):
    """
    video_dir is processed/<video-id> and must contain fixed and scene folders from Phase C2
    """
    log("\n" + "=" * 60)
    vid = video_dir.name
    log(f"Processing folder: {video_dir}")

    fixed_dir = video_dir / FIXED_SUB
    scene_dir = video_dir / SCENE_SUB
    ready_dir = video_dir / READY_SUB

    fixed_imgs = list_images(fixed_dir)
    scene_imgs = list_images(scene_dir)

    ensure_dir(video_dir)  # just to be safe
    if WRITE_READY_IMAGES:
        ensure_dir(ready_dir)

    manifest = {
        "video_id": vid,
        "fixed_count": len(fixed_imgs),
        "scene_count": len(scene_imgs),
        "processed_at_unix": int(time.time()),
        "usable_frames": [],
        "excluded_frames": {"blurry": [], "dark": [], "bright": [], "duplicates": []},
        "stats": {}
    }

    # merge candidate list: prefer scene frames (more concise). If scene empty fallback to fixed.
    candidates = scene_imgs if scene_imgs else fixed_imgs
    if not candidates:
        log(f"⚠ No candidate frames found in {scene_dir} or {fixed_dir} for {vid}. Skipping.")
        manifest["stats"] = {"usable": 0}
        (video_dir / MANIFEST_FN).write_text(json.dumps(manifest, indent=2))
        return

    # process images in order
    hashes = []
    kept = []
    excluded = {"blurry": [], "dark": [], "bright": [], "duplicates": []}

    for p in tqdm(candidates, desc=f"Processing frames for {vid}", unit="img"):
        # safe-read image (cv2/PIL combo)
        try:
            img = read_image_cv(p)
            if img is None:
                log("⚠ Unable to read image:", p)
                excluded["blurry"].append(str(p.name))
                continue
        except Exception as e:
            log("⚠ Error reading image", p, e)
            excluded["blurry"].append(str(p.name))
            continue

        # convert/resize
        img_proc = resize_and_normalize(img)

        # quality checks
        blurry, blur_var = is_blurry_pil(img_proc)
        mean_b = mean_brightness(img_proc)
        if blurry:
            excluded["blurry"].append({"name": p.name, "laplacian_var": blur_var})
            if REMOVE_EXCLUDED_FILES:
                try:
                    p.unlink()
                except Exception:
                    pass
            continue

        if mean_b < DARK_MEAN_THRESH:
            excluded["dark"].append({"name": p.name, "mean": mean_b})
            if REMOVE_EXCLUDED_FILES:
                try:
                    p.unlink()
                except Exception:
                    pass
            continue

        if mean_b > BRIGHT_MEAN_THRESH:
            excluded["bright"].append({"name": p.name, "mean": mean_b})
            if REMOVE_EXCLUDED_FILES:
                try:
                    p.unlink()
                except Exception:
                    pass
            continue

        # compute perceptual hash & duplicate check
        h = compute_hash(img_proc, HASH_SIZE)
        is_dup = False
        for prev_h, prev_p in hashes:
            if hamming_distance(h, prev_h) <= HAMMING_THRESH:
                # duplicate detected
                is_dup = True
                excluded["duplicates"].append({"name": p.name, "duplicate_of": prev_p.name,
                                               "hamming": hamming_distance(h, prev_h)})
                break

        if is_dup:
            if KEEP_DUPLICATES:
                # keep but mark duplicate
                kept.append({"path": str(p.name), "hash": str(h)})
                hashes.append((h, p))
            else:
                # skip adding to usable set
                if REMOVE_EXCLUDED_FILES:
                    try:
                        p.unlink()
                    except Exception:
                        pass
                continue
        else:
            hashes.append((h, p))
            # write ready image if required
            if WRITE_READY_IMAGES:
                # create filename preserving prefix
                dest_name = f"ready_{p.name}"
                dest_path = ready_dir / dest_name
                try:
                    # save with PIL to avoid cv2 saving issues
                    img_proc.save(dest_path, format="JPEG", quality=90)
                except Exception as e:
                    # fallback: convert to RGB and save
                    img_proc.convert("RGB").save(dest_path, format="JPEG", quality=90)
                kept.append({"path": str(dest_path.name), "orig": str(p.name), "hash": str(h)})
            else:
                kept.append({"path": str(p.name), "hash": str(h)})

    # finalize manifest
    manifest["usable_frames"] = kept
    manifest["excluded_frames"] = excluded
    manifest["stats"] = {
        "total_candidates": len(candidates),
        "usable": len(kept),
        "excluded_blurry": len(excluded["blurry"]),
        "excluded_dark": len(excluded["dark"]),
        "excluded_bright": len(excluded["bright"]),
        "excluded_duplicates": len(excluded["duplicates"])
    }

    # write manifest
    manifest_path = video_dir / MANIFEST_FN
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    log(f"✅ Wrote manifest: {manifest_path}")

    # update processed_meta.json if present
    proc_meta_path = video_dir / "processed_meta.json"
    if proc_meta_path.exists():
        try:
            with proc_meta_path.open("r", encoding="utf-8") as fh:
                proc_meta = json.load(fh)
        except Exception:
            proc_meta = {}
        proc_meta["frames_manifest"] = manifest
        with proc_meta_path.open("w", encoding="utf-8") as fh:
            json.dump(proc_meta, fh, indent=2)
        log("✅ Updated processed_meta.json")

    log(f"Done {vid}: usable={len(kept)} excluded={sum(len(v) for v in excluded.values())}")


def main():
    if not PROC_ROOT.exists():
        log("ERROR: processed root not found:", PROC_ROOT)
        sys.exit(1)
    # process each processed/<video-id> folder
    folders = sorted([p for p in PROC_ROOT.iterdir() if p.is_dir()])
    if not folders:
        log("No video folders found in processed/. Nothing to do.")
        return

    for folder in folders:
        process_video_folder(folder)

    log("\nPhase C3 complete. Manifests created for each processed video.")


if __name__ == "__main__":
    main()

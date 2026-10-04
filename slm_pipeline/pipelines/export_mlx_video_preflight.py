"""Freeze and render the exact MLX video-decoder frames before generation.

This is intentionally decoder-only: it never loads a language model or
generates text.  A later diagnostic must compare its actual prepared-input
hashes against this record before it can call generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

from slm_pipeline.pipelines.mlx_video import decoded_frame_hashes


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_image(frame):
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    # mlx-vlm's video decoder currently returns uint8 CHW arrays, whereas
    # Pillow requires HWC.  Persisting the conversion makes the human contact
    # sheet inspect the same decoded pixel planes that will be preprocessed.
    shape = getattr(frame, "shape", ())
    if len(shape) == 3 and shape[0] in (1, 3, 4) and shape[-1] not in (1, 3, 4):
        frame = frame.transpose(1, 2, 0)
    return Image.fromarray(frame).convert("RGB")


def make_contact_sheet(frames, output):
    images = [as_image(frame) for frame in frames]
    thumb_width = 384
    thumbs = []
    for index, image in enumerate(images):
        thumb = image.copy()
        thumb.thumbnail((thumb_width, 216))
        canvas = Image.new("RGB", (thumb_width, 244), "black")
        canvas.paste(thumb, ((thumb_width - thumb.width) // 2, 24))
        ImageDraw.Draw(canvas).text((8, 5), f"input {index}", fill="white")
        thumbs.append(canvas)
    columns = 4
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_width, rows * 244), "black")
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % columns) * thumb_width, (index // columns) * 244))
    sheet.save(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--source-pts-json", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args(argv)
    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    source_pts = [float(value) for value in json.loads(args.source_pts_json)]
    if len(source_pts) != args.frames or any(right <= left for left, right in zip(source_pts, source_pts[1:])):
        raise ValueError("source PTS must be strictly increasing and match requested frame count")

    from mlx_vlm.utils import VideoSampling, load_video

    frames, metadata = load_video(str(args.video), sampling=VideoSampling(fps=args.fps, nframes=args.frames))
    if len(frames) != args.frames or len(metadata.timestamps) != args.frames:
        raise RuntimeError("MLX decoder did not return exactly the frozen frame budget")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = args.output_dir / f"native_inputs_{args.name}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for index, frame in enumerate(frames):
        path = frame_dir / f"frame_{index:02d}.png"
        as_image(frame).save(path)
        saved_paths.append(path)
    contact_sheet = args.output_dir / f"contact_native_{args.name}.png"
    make_contact_sheet(frames, contact_sheet)
    record = {
        "name": args.name,
        "video_path": str(args.video.resolve()),
        "video_sha256": sha256(args.video),
        "sampling": {"fps": args.fps, "nframes": args.frames},
        "source_pts_sec": source_pts,
        "decoder": {
            "decoded_frame_count": len(frames),
            "decoded_frame_indices": [int(value) for value in metadata.frames_indices],
            "clip_relative_timestamps_sec": [float(value) for value in metadata.timestamps],
            "sampled_fps": float(metadata.sampled_fps),
            "frame_hashes": decoded_frame_hashes(frames),
            "unique_frame_hash_count": len(set(decoded_frame_hashes(frames))),
        },
        "saved_png_paths": [str(path) for path in saved_paths],
        "saved_png_sha256": [sha256(path) for path in saved_paths],
        "contact_sheet": str(contact_sheet),
        "contact_sheet_sha256": sha256(contact_sheet),
        "status": "decoder_only_preflight_no_model_loaded_no_text_generated",
    }
    record_path = args.output_dir / f"native_input_preflight_{args.name}.json"
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"record": str(record_path), "contact_sheet": str(contact_sheet)}, indent=2))
    return record


if __name__ == "__main__":
    main()

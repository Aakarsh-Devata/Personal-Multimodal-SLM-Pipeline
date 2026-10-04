"""Run one or two frozen structured-event prompt arms on identical triad inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import time
from pathlib import Path

from slm_pipeline.pipelines.mlx_video import PersistentMLXVideoRunner
from slm_pipeline.pipelines.temporal_events import ingest_structured_event_hypotheses


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_lossless_video(output, source_paths):
    """Encode the exact supplied still files in presentation order at 1 fps."""
    output = Path(output)
    staging = output.with_suffix("")
    staging.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(source_paths):
        target = staging / f"frame_{index:06d}.jpg"
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(Path(source).resolve())
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-framerate", "1", "-i", str(staging / "frame_%06d.jpg"),
        "-frames:v", str(len(source_paths)), "-c:v", "libx264rgb", "-crf", "0", "-preset", "ultrafast",
        "-pix_fmt", "rgb24", str(output),
    ], check=True)
    return {"path": str(output), "sha256": sha256(output), "frame_count": len(source_paths),
            "encoding": "lossless libx264rgb from declared source still order"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--question-id", default="structured_temporal_events_v1")
    parser.add_argument("--compare-question-id")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    selection_path = args.selection if args.selection.is_absolute() else root / args.selection
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    candidates = selection["candidate_selection"]["all_candidate_intervals"]
    candidate = next((item for item in candidates if item["candidate_id"] == args.candidate_id and item.get("selected")), None)
    if candidate is None:
        raise ValueError("candidate ID must name a selected frozen candidate")
    source_paths = {item["frame_id"]: root / "data/processed" / selection["video_id"] / item["source_path"]
                    for item in selection["frame_change_records"]}
    frame_ids, source_pts = candidate["frame_ids"], candidate["source_pts_range_sec"]
    frame_records = [next(item for item in selection["frame_change_records"] if item["frame_id"] == frame_id)
                     for frame_id in frame_ids]
    source_pts = [item["source_pts_sec"] for item in frame_records]
    if not all(source_paths[item["frame_id"]].is_file() for item in frame_records):
        raise FileNotFoundError("selected source frame is missing")
    experiment = root / "data/experiments" / f"{selection['video_id']}_structured_triad"
    forward = create_lossless_video(experiment / "forward.mp4", [source_paths[item] for item in frame_ids])
    reverse = create_lossless_video(experiment / "reverse.mp4", [source_paths[item] for item in reversed(frame_ids)])
    static = create_lossless_video(experiment / "static.mp4", [source_paths[frame_ids[0]]] * len(frame_ids))
    model_path = root / "data/models/mlx/Qwen3-VL-4B-Instruct-4bit"
    question_ids = [args.question_id] + ([args.compare_question_id] if args.compare_question_id else [])
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("prompt arms must use distinct frozen question IDs")
    runner = PersistentMLXVideoRunner(model_path, fps=1.0, video_num_frames=len(frame_ids), max_pixels=3_211_264,
                                      max_new_tokens=256, max_merged_visual_tokens=2_304,
                                      question_id=question_ids[0])
    started = time.perf_counter()
    prompt_arms = {}
    baseline_contract = None
    for question_id in question_ids:
        runner.question_id = question_id
        results = {
            "forward": runner.observe_window(forward["path"], 0.0),
            "reverse": runner.observe_window(reverse["path"], 0.0),
            "static": runner.observe_window(static["path"], 0.0),
        }
        hashes = {key: item["input_trace"]["decoded_frame_hashes"] for key, item in results.items()}
        tokens = {item["input_trace"]["merged_visual_tokens"] for item in results.values()}
        if hashes["reverse"] != list(reversed(hashes["forward"])):
            raise RuntimeError("lossless reverse control did not preserve exact decoded frame identity/order")
        if len(set(hashes["static"])) != 1 or hashes["static"][0] != hashes["forward"][0]:
            raise RuntimeError("lossless static control did not preserve exact decoded source-frame identity")
        if len(tokens) != 1 or any(item["input_trace"]["decoded_frame_count"] != len(frame_ids) for item in results.values()):
            raise RuntimeError("triad did not keep one equal native input budget")
        contract = {
            key: {
                "decoded_frame_hashes": item["input_trace"]["decoded_frame_hashes"],
                "source_timestamps_sec": item["input_trace"]["source_timestamps_sec"],
                "video_grid_thw": item["input_trace"]["video_grid_thw"],
                "pixel_values_videos_shape": item["input_trace"]["pixel_values_videos_shape"],
                "merged_visual_tokens": item["input_trace"]["merged_visual_tokens"],
                "video_preprocessor_contract": item["input_trace"]["video_preprocessor_contract"],
            }
            for key, item in results.items()
        }
        if baseline_contract is None:
            baseline_contract = contract
        elif contract != baseline_contract:
            raise RuntimeError("prompt arms did not receive identical decoded/preprocessed control inputs")
        provenance = {"model": runner.model_path.name, "backend": "mlx_vlm", "run_id": args.run_id,
                      "question_id": question_id}
        prompt_arms[question_id] = {
            "question": runner.question, "results": results,
            "parsed_unvalidated_hypotheses": {
                "forward": ingest_structured_event_hypotheses(results["forward"]["observation"], frame_ids, source_pts, provenance),
                "reverse": ingest_structured_event_hypotheses(results["reverse"]["observation"], list(reversed(frame_ids)), list(reversed(source_pts)), provenance, input_order="source_reverse"),
                "static": ingest_structured_event_hypotheses(results["static"]["observation"], [frame_ids[0]] * len(frame_ids), [source_pts[0]] * len(frame_ids), provenance, input_order="same_source_frame_repeated"),
            },
            "input_contract": contract,
        }
    report = {
        "run_id": args.run_id, "status": "raw_outputs_frozen_before_labels", "selection_path": str(selection_path),
        "candidate": candidate, "source_presentation": {"frame_ids": frame_ids, "source_pts_sec": source_pts},
        "derived_inputs": {"forward": forward, "reverse": reverse, "static": static},
        "model_loaded_once_within_six_generation_batch": True, "cold_load_ms": runner.cold_load_ms,
        "triad_wall_ms": round((time.perf_counter() - started) * 1000, 2),
        "process_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "prompt_arm_ids": question_ids, "prompt_arms": prompt_arms,
        "identical_preprocessed_inputs_across_prompt_arms": True,
        "review_status": "awaiting frozen label-blind visual review; labels must remain unopened",
    }
    report_path = args.report if args.report.is_absolute() else root / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": args.run_id, "report": str(report_path), "cold_load_ms": runner.cold_load_ms}, indent=2))


if __name__ == "__main__":
    main()

"""Run one idempotent, two-generation dense-versus-sparse local diagnostic.

The lock and journal deliberately treat any interrupted or incomplete attempt
as non-rerunnable.  A recovery is a separate, explicit investigation; this
command never silently starts a second process for the same run id.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import tempfile
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


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(payload, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def append_journal(path, event, **fields):
    record = {"event": event, "wall_time_unix": time.time(), **fields}
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def acquire_run(lock_path, report_path, run_id, config_sha):
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock.get("run_id") != run_id or lock.get("config_sha256") != config_sha:
            raise RuntimeError("existing diagnostic lock does not match this frozen run")
        if lock.get("status") == "completed" and report_path.is_file():
            return lock, True
        raise RuntimeError(
            "an incomplete or failed diagnostic lock already exists; do not retry without a separate "
            "transport-failure investigation and a new explicit authorization"
        )
    lock = {"run_id": run_id, "config_sha256": config_sha, "pid": os.getpid(),
            "status": "running", "created_unix": time.time(), "generation_count": 0}
    atomic_json(lock_path, lock)
    return lock, False


def validate_prepared(name, prepared, presentation, root):
    preflight_path = root / presentation["preflight"]
    if sha256(preflight_path) != presentation["preflight_sha256"]:
        raise RuntimeError(f"{name} preflight record hash changed after registration")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    trace = prepared["input_trace"]
    expected = preflight["decoder"]["frame_hashes"]
    if trace["decoded_frame_hashes"] != expected:
        raise RuntimeError(f"{name} native decoder hashes do not match frozen preflight")
    if trace["source_timestamps_sec"] != [float(value) for value in presentation["source_pts_sec"]]:
        raise RuntimeError(f"{name} source PTS do not match frozen input order")
    if len(set(trace["decoded_frame_hashes"])) != presentation["frame_count"]:
        raise RuntimeError(f"{name} unexpectedly reuses a decoded frame")
    if trace["merged_visual_tokens"] > 2304:
        raise RuntimeError(f"{name} exceeds registered visual-token cap")
    return trace


def result_from_response(name, runner, prepared, response, presentation, run_id):
    trace = prepared["input_trace"]
    provenance = {"model": runner.model_path.name, "backend": "mlx_vlm", "run_id": run_id,
                  "question_id": runner.question_id}
    return {
        "presentation": name,
        "question_id": runner.question_id,
        "question": runner.question,
        "observation": response.text,
        "source_type": "model",
        "input_type": "native_video_tensors",
        "backend": "mlx_vlm",
        "model": runner.model_path.name,
        "device": "metal",
        "preparation_ms": prepared["preparation_ms"],
        "mlx_peak_memory_gb": float(response.peak_memory),
        "input_trace": trace,
        "parsed_unvalidated_hypotheses": ingest_structured_event_hypotheses(
            response.text, [f"{name}_{index:02d}" for index in range(presentation["frame_count"])],
            presentation["source_pts_sec"], provenance, assign_event_ids=True,
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    preregistration_path = args.preregistration if args.preregistration.is_absolute() else root / args.preregistration
    report_path = args.report if args.report.is_absolute() else root / args.report
    config_sha = sha256(preregistration_path)
    config = json.loads(preregistration_path.read_text(encoding="utf-8"))
    contract = config["inference_contract"]
    run_id = contract["run_id"]
    lock_path = report_path.with_name(f"{run_id}.lock.json")
    journal_path = report_path.with_name(f"{run_id}.journal.jsonl")
    runstate_path = report_path.with_name(f"{run_id}.runstate.json")
    lock, already_completed = acquire_run(lock_path, report_path, run_id, config_sha)
    if already_completed:
        print(json.dumps({"run_id": run_id, "status": "already_completed", "report": str(report_path)}, indent=2))
        return json.loads(report_path.read_text(encoding="utf-8"))

    append_journal(journal_path, "lock_acquired", run_id=run_id, config_sha256=config_sha)
    atomic_json(runstate_path, {"run_id": run_id, "status": "preparing_native_tensors", "generation_count": 0})
    model_path = root / contract["model_path"]
    presentation_order = config.get("presentation_order", list(config["presentations"]))
    if not isinstance(presentation_order, list) or not presentation_order:
        raise ValueError("presentation_order must list at least one frozen presentation")
    if any(name not in config["presentations"] for name in presentation_order):
        raise ValueError("presentation_order names an unknown presentation")
    first_presentation = config["presentations"][presentation_order[0]]
    runner = PersistentMLXVideoRunner(
        model_path, question_id=contract["question_id"], max_new_tokens=contract["max_new_tokens"],
        max_pixels=contract["max_pixels"], max_merged_visual_tokens=contract["max_merged_visual_tokens"],
        fps=first_presentation["fps"], video_num_frames=first_presentation["frame_count"],
    )
    started = time.perf_counter()
    prepared = {}
    try:
        for name in presentation_order:
            presentation = config["presentations"][name]
            runner.fps = float(presentation["fps"])
            runner.video_num_frames = int(presentation["frame_count"])
            prepared[name] = runner.prepare_window(
                root / presentation["video"], presentation["source_pts_sec"][0],
                source_frame_pts=presentation["source_pts_sec"],
            )
            validate_prepared(name, prepared[name], presentation, root)
            append_journal(journal_path, "prepared_and_verified", presentation=name,
                           native_frame_count=presentation["frame_count"],
                           visual_tokens=prepared[name]["input_trace"]["merged_visual_tokens"])
        atomic_json(runstate_path, {"run_id": run_id, "status": "native_tensors_verified_before_generation",
                                    "generation_count": 0})
        outputs = {}
        for name in presentation_order:
            append_journal(journal_path, "generation_started", presentation=name)
            response = runner.generate_prepared(prepared[name])
            outputs[name] = result_from_response(name, runner, prepared[name], response,
                                                 config["presentations"][name], run_id)
            lock["generation_count"] += 1
            atomic_json(lock_path, lock)
            atomic_json(runstate_path, {"run_id": run_id, "status": "partial_output_frozen",
                                        "generation_count": lock["generation_count"], "outputs": outputs})
            append_journal(journal_path, "generation_output_frozen", presentation=name,
                           generation_count=lock["generation_count"])
        report = {
            "run_id": run_id,
            "status": "raw_outputs_frozen_before_post_output_review",
            "preregistration_path": str(preregistration_path),
            "preregistration_sha256": config_sha,
            "model_loaded_once": True,
            "retained_generation_count": lock["generation_count"],
            "cold_load_ms": runner.cold_load_ms,
            "wall_ms": round((time.perf_counter() - started) * 1000, 2),
            "process_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "results": outputs,
            "review_status": "awaiting source-based claim review; do not read reference labels for this diagnostic",
        }
        atomic_json(report_path, report)
        lock.update({"status": "completed", "completed_unix": time.time()})
        atomic_json(lock_path, lock)
        atomic_json(runstate_path, {"run_id": run_id, "status": "completed", "generation_count": lock["generation_count"],
                                    "report": str(report_path)})
        append_journal(journal_path, "completed", report=str(report_path), generation_count=lock["generation_count"])
        print(json.dumps({"run_id": run_id, "report": str(report_path), "generation_count": lock["generation_count"]}, indent=2))
        return report
    except Exception as error:
        lock.update({"status": "failed_or_interrupted", "error": repr(error), "generation_count": lock["generation_count"]})
        atomic_json(lock_path, lock)
        atomic_json(runstate_path, {"run_id": run_id, "status": "failed_or_interrupted",
                                    "generation_count": lock["generation_count"], "error": repr(error)})
        append_journal(journal_path, "failed_or_interrupted", error=repr(error),
                       generation_count=lock["generation_count"])
        raise


if __name__ == "__main__":
    main()

"""CPU-only temporal candidate selection and unvalidated event schemas.

This module deliberately separates three things that are often conflated:
pixel-change selection, a model's structured hypothesis, and independently
accepted evidence.  A schema-valid event is still an unvalidated model
hypothesis and is never a pipeline fact merely because it names exact frames.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path


EVENT_SCHEMA_VERSION = "temporal_event_hypothesis_v1"
UNVALIDATED = "unvalidated_model_hypothesis"
_DIRECTIONS = {"unknown", "none", "up", "down", "in", "out", "open", "close", "toward", "away"}
_ACTORS = {"wearer", "other_person", "unknown"}
_INTEGER_TEXT = re.compile(r"^(?:0|[1-9][0-9]*)$")


def _finite_timestamp(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _ordered_frame_records(frames):
    if not isinstance(frames, list) or len(frames) < 2:
        raise ValueError("at least two frame records are required")
    records = []
    seen_ids, previous = set(), None
    for item in frames:
        if not isinstance(item, dict):
            raise ValueError("frame records must be objects")
        frame_id = item.get("frame_id") or Path(str(item.get("path", ""))).name
        timestamp = item.get("source_pts_sec", item.get("source_timestamp_sec", item.get("timestamp_sec")))
        if not isinstance(frame_id, str) or not frame_id:
            raise ValueError("each frame record needs a frame_id or path")
        if frame_id in seen_ids or not _finite_timestamp(timestamp):
            raise ValueError("frame IDs must be unique and source PTS finite")
        timestamp = float(timestamp)
        if previous is not None and timestamp <= previous:
            raise ValueError("frame source PTS must be strictly increasing")
        records.append({"frame_id": frame_id, "source_pts_sec": timestamp, **item})
        seen_ids.add(frame_id)
        previous = timestamp
    return records


def _safe_frame_path(video_dir, item):
    video_dir = Path(video_dir).resolve()
    relative = item.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError("candidate selection requires a manifest frame path")
    path = (video_dir / relative).resolve()
    if video_dir not in path.parents or not path.is_file():
        raise ValueError("manifest frame path must resolve to a file inside video_dir")
    return path


def _luma_bytes(path):
    from PIL import Image

    with Image.open(path) as image:
        reduced = image.convert("L").resize((64, 36))
        pixels = reduced.tobytes()
    return pixels


def _mean_absolute_delta(left, right):
    if len(left) != len(right):
        raise ValueError("luma frames must have the same reduced dimensions")
    return round(sum(abs(a - b) for a, b in zip(left, right)) / len(left), 6)


def frame_change_records(video_dir, usable_frames):
    """Compute auditable low-resolution luma deltas for decoded manifest frames.

    The result has no confidence field.  Its score is a stated image-space
    quantity (mean absolute 8-bit luma difference), and is only a window
    selection rationale—not an event/action score.
    """
    records, previous_luma = [], None
    for item in _ordered_frame_records(usable_frames):
        path = _safe_frame_path(video_dir, item)
        luma = _luma_bytes(path)
        records.append({
            "frame_id": item["frame_id"],
            "source_pts_sec": item["source_pts_sec"],
            "source_path": item["path"],
            "frame_hash": hashlib.sha256(luma).hexdigest(),
            "mean_abs_luma_delta_from_previous": None if previous_luma is None else _mean_absolute_delta(previous_luma, luma),
            "change_metric_units": "mean absolute 8-bit luma difference over 64x36 pixels",
        })
        previous_luma = luma
    return records


def fixed_sampling_records(video_dir, expected_frame_count=None):
    """Read every fixed-rate decoded frame with its recorded source PTS.

    This intentionally does not substitute dense scene-change frames for a
    frozen fixed-frame budget. If a frame cannot be read later, selection
    fails rather than silently shrinking coverage.
    """
    video_dir = Path(video_dir)
    payload = json.loads((video_dir / "frame_timestamps.json").read_text(encoding="utf-8"))
    items = []
    for source_path, timing in payload.get("frames", {}).items():
        if not source_path.startswith("frames_fixed/"):
            continue
        items.append({
            "frame_id": Path(source_path).name, "path": source_path,
            "source_timestamp_sec": timing.get("source_timestamp_sec", timing.get("timestamp_sec")),
        })
    items = _ordered_frame_records(sorted(items, key=lambda item: item["source_timestamp_sec"]))
    if expected_frame_count is not None and len(items) != expected_frame_count:
        raise ValueError(f"fixed sampling produced {len(items)} frames, expected {expected_frame_count}")
    return frame_change_records(video_dir, items)


def select_change_candidate_windows(frame_records, window_size=4, max_windows=3, max_selected_frames=12,
                                    minimum_new_frames_per_selected_window=None):
    """Select bounded windows while retaining every candidate and its rationale."""
    if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 2:
        raise ValueError("window_size must be an integer of at least two")
    if isinstance(max_windows, bool) or not isinstance(max_windows, int) or max_windows < 1:
        raise ValueError("max_windows must be a positive integer")
    if isinstance(max_selected_frames, bool) or not isinstance(max_selected_frames, int) or max_selected_frames < window_size:
        raise ValueError("max_selected_frames must cover at least one complete window")
    if minimum_new_frames_per_selected_window is None:
        minimum_new_frames_per_selected_window = max(1, window_size // 2)
    if (isinstance(minimum_new_frames_per_selected_window, bool)
            or not isinstance(minimum_new_frames_per_selected_window, int)
            or not 1 <= minimum_new_frames_per_selected_window <= window_size):
        raise ValueError("minimum_new_frames_per_selected_window must be within the window size")
    frames = _ordered_frame_records(frame_records)
    if len(frames) < window_size:
        raise ValueError("not enough frames for requested candidate window size")
    candidates = []
    for start in range(0, len(frames) - window_size + 1):
        window = frames[start:start + window_size]
        deltas = [item.get("mean_abs_luma_delta_from_previous") for item in window[1:]]
        if not all(_finite_timestamp(delta) for delta in deltas):
            raise ValueError("candidate windows require explicit frame luma deltas")
        candidates.append({
            "candidate_id": f"change_{start:04d}_{start + window_size - 1:04d}",
            "source_pts_range_sec": [window[0]["source_pts_sec"], window[-1]["source_pts_sec"]],
            "frame_ids": [item["frame_id"] for item in window],
            "frame_hashes": [item.get("frame_hash") for item in window],
            "selection_metric": "mean_abs_luma_delta_from_previous",
            "selection_metric_value": round(sum(deltas) / len(deltas), 6),
        })
    ranked = sorted(candidates, key=lambda item: (-item["selection_metric_value"], item["source_pts_range_sec"][0]))
    selected_ids, selected_frame_ids = set(), set()
    for rank, candidate in enumerate(ranked, start=1):
        candidate["selection_rank"] = rank
        new_frame_ids = set(candidate["frame_ids"]) - selected_frame_ids
        if len(selected_ids) >= max_windows:
            candidate["selected"] = False
            candidate["selection_reason"] = "not selected: maximum candidate-window count reached"
        elif selected_ids and len(new_frame_ids) < minimum_new_frames_per_selected_window:
            candidate["selected"] = False
            candidate["selection_reason"] = "not selected: insufficient novel frames for temporal coverage diversity"
        elif len(selected_frame_ids) + len(new_frame_ids) > max_selected_frames:
            candidate["selected"] = False
            candidate["selection_reason"] = "not selected: adding unique frames would exceed frame budget"
        else:
            candidate["selected"] = True
            candidate["selection_reason"] = "selected by highest transparent luma-change metric within window and unique-frame budgets"
            selected_ids.add(candidate["candidate_id"])
            selected_frame_ids.update(candidate["frame_ids"])
    by_id = {candidate["candidate_id"]: candidate for candidate in ranked}
    all_candidates = [by_id[candidate["candidate_id"]] for candidate in candidates]
    return {
        "selection_method": "ranked mean absolute reduced-luma difference; not a confidence or action score",
        "source_frame_count": len(frames),
        "window_size": window_size,
        "max_windows": max_windows,
        "max_selected_unique_frames": max_selected_frames,
        "minimum_new_frames_per_selected_window": minimum_new_frames_per_selected_window,
        "selected_unique_frame_count": len(selected_frame_ids),
        "all_candidate_intervals": all_candidates,
        "selected_candidate_ids": [candidate["candidate_id"] for candidate in ranked if candidate["selected"]],
        "skipped_candidate_ids": [candidate["candidate_id"] for candidate in ranked if not candidate["selected"]],
    }


def uniform_candidate_windows(frame_records, window_size=4, max_windows=3, max_selected_frames=12):
    """Select evenly distributed windows under the same explicit frame budget."""
    frames = _ordered_frame_records(frame_records)
    if (isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 2
            or isinstance(max_windows, bool) or not isinstance(max_windows, int) or max_windows < 1
            or isinstance(max_selected_frames, bool) or not isinstance(max_selected_frames, int)
            or max_selected_frames < window_size or len(frames) < window_size):
        raise ValueError("invalid uniform candidate-window bounds")
    last_start = len(frames) - window_size
    count = min(max_windows, max_selected_frames // window_size, last_start + 1)
    starts = [0] if count == 1 else [round(index * last_start / (count - 1)) for index in range(count)]
    starts = list(dict.fromkeys(starts))
    candidates = []
    for start in range(last_start + 1):
        window = frames[start:start + window_size]
        candidates.append({
            "candidate_id": f"uniform_{start:04d}_{start + window_size - 1:04d}",
            "source_pts_range_sec": [window[0]["source_pts_sec"], window[-1]["source_pts_sec"]],
            "frame_ids": [item["frame_id"] for item in window],
            "frame_hashes": [item.get("frame_hash") for item in window],
            "selected": start in starts,
            "selection_reason": "selected by evenly distributed source-frame position under the same window/frame budget" if start in starts else "not selected by fixed uniform source-frame positions",
        })
    selected = [item for item in candidates if item["selected"]]
    return {
        "selection_method": "uniform source-frame positions; no motion/action/confidence score",
        "source_frame_count": len(frames), "window_size": window_size,
        "max_windows": max_windows, "max_selected_unique_frames": max_selected_frames,
        "selected_unique_frame_count": len({frame_id for item in selected for frame_id in item["frame_ids"]}),
        "all_candidate_intervals": candidates,
        "selected_candidate_ids": [item["candidate_id"] for item in selected],
        "skipped_candidate_ids": [item["candidate_id"] for item in candidates if not item["selected"]],
    }


def anchored_change_candidate_windows(frame_records, window_size=4, max_windows=3, max_selected_frames=12,
                                      uniform_anchor_count=2, minimum_new_frames_per_selected_window=None):
    """Reserve uniform temporal anchors before spending any budget on motion.

    Motion ranking only fills slots that remain after anchors. It can never
    replace source-span anchors or silently turn a coverage policy into a
    change-score policy.
    """
    frames = _ordered_frame_records(frame_records)
    if (isinstance(uniform_anchor_count, bool) or not isinstance(uniform_anchor_count, int)
            or not 1 <= uniform_anchor_count <= max_windows):
        raise ValueError("uniform_anchor_count must be between one and max_windows")
    anchors = uniform_candidate_windows(frames, window_size=window_size, max_windows=uniform_anchor_count,
                                        max_selected_frames=max_selected_frames)
    anchor_ranges = {tuple(item["source_pts_range_sec"]): item for item in anchors["all_candidate_intervals"] if item["selected"]}
    changes = select_change_candidate_windows(
        frames, window_size=window_size, max_windows=max_windows, max_selected_frames=max_selected_frames,
        minimum_new_frames_per_selected_window=minimum_new_frames_per_selected_window,
    )
    all_candidates = []
    selected_ids, selected_frame_ids = [], set()
    for candidate in changes["all_candidate_intervals"]:
        clone = dict(candidate)
        key = tuple(clone["source_pts_range_sec"])
        if key in anchor_ranges:
            clone["selected"] = True
            clone["selection_kind"] = "required_uniform_anchor"
            clone["selection_reason"] = "selected as required uniform temporal anchor before motion ranking"
            selected_ids.append(clone["candidate_id"])
            selected_frame_ids.update(clone["frame_ids"])
        else:
            clone["selected"] = False
            clone["selection_kind"] = "motion_candidate"
        all_candidates.append(clone)
    if minimum_new_frames_per_selected_window is None:
        minimum_new_frames_per_selected_window = max(1, window_size // 2)
    for candidate in sorted(all_candidates, key=lambda item: (-item["selection_metric_value"], item["source_pts_range_sec"][0])):
        if candidate["selected"]:
            continue
        new_ids = set(candidate["frame_ids"]) - selected_frame_ids
        if len(selected_ids) >= max_windows:
            candidate["selection_reason"] = "not selected: all remaining budget slots are exhausted after required anchors"
        elif len(new_ids) < minimum_new_frames_per_selected_window:
            candidate["selection_reason"] = "not selected: insufficient novel frames after required anchor coverage"
        elif len(selected_frame_ids) + len(new_ids) > max_selected_frames:
            candidate["selection_reason"] = "not selected: would exceed unique-frame budget after required anchors"
        else:
            candidate["selected"] = True
            candidate["selection_kind"] = "motion_fill"
            candidate["selection_reason"] = "selected by transparent motion metric only after required uniform anchors"
            selected_ids.append(candidate["candidate_id"])
            selected_frame_ids.update(candidate["frame_ids"])
    return {
        "selection_method": "required uniform temporal anchors plus transparent motion-ranked leftover slots; not a confidence/action score",
        "source_frame_count": len(frames), "window_size": window_size, "max_windows": max_windows,
        "uniform_anchor_count": uniform_anchor_count, "max_selected_unique_frames": max_selected_frames,
        "minimum_new_frames_per_motion_window": minimum_new_frames_per_selected_window,
        "selected_unique_frame_count": len(selected_frame_ids), "all_candidate_intervals": all_candidates,
        "selected_candidate_ids": selected_ids,
        "skipped_candidate_ids": [item["candidate_id"] for item in all_candidates if not item["selected"]],
    }


def _reference_interval(item):
    interval = item.get("source_interval_sec", item.get("source_time_range_sec")) if isinstance(item, dict) else None
    if (not isinstance(interval, list) or len(interval) != 2 or not all(_finite_timestamp(value) for value in interval)
            or interval[1] <= interval[0]):
        raise ValueError("reference intervals require increasing finite source times")
    return [float(value) for value in interval]


def _interval_overlap(left, right):
    return left[0] < right[1] and right[0] < left[1]


def compare_reference_coverage(reference_intervals, full_decoded_pts_sec, candidate_selection, uniform_selection):
    """Compare action-reference coverage only after outputs/claim review are frozen.

    This function does not read a CSV, choose windows, or score model output.
    Callers supply evaluation-only reference intervals after the frozen boundary.
    """
    full = _source_window([f"decoded_{index}" for index in range(len(full_decoded_pts_sec))], full_decoded_pts_sec)
    full_span = [full[0]["source_pts_sec"], full[-1]["source_pts_sec"]]
    for selection in (candidate_selection, uniform_selection):
        if not isinstance(selection, dict) or not isinstance(selection.get("all_candidate_intervals"), list):
            raise ValueError("selection artifacts must retain all candidate intervals")
    output = []
    for item in reference_intervals:
        interval = _reference_interval(item)
        candidate_hits = [candidate["candidate_id"] for candidate in candidate_selection["all_candidate_intervals"]
                          if candidate.get("selected") and _interval_overlap(interval, candidate["source_pts_range_sec"])]
        uniform_hits = [candidate["candidate_id"] for candidate in uniform_selection["all_candidate_intervals"]
                        if candidate.get("selected") and _interval_overlap(interval, candidate["source_pts_range_sec"])]
        candidate_full = [candidate["candidate_id"] for candidate in candidate_selection["all_candidate_intervals"]
                          if candidate.get("selected") and candidate["source_pts_range_sec"][0] <= interval[0]
                          and candidate["source_pts_range_sec"][1] >= interval[1]]
        uniform_full = [candidate["candidate_id"] for candidate in uniform_selection["all_candidate_intervals"]
                        if candidate.get("selected") and candidate["source_pts_range_sec"][0] <= interval[0]
                        and candidate["source_pts_range_sec"][1] >= interval[1]]
        output.append({
            "reference_id": item.get("reference_id") if isinstance(item, dict) else None,
            "source_interval_sec": interval,
            "within_full_decoded_span": _interval_overlap(interval, full_span),
            "candidate_selected_window_ids": candidate_hits,
            "uniform_selected_window_ids": uniform_hits,
            "candidate_fully_containing_window_ids": candidate_full,
            "uniform_fully_containing_window_ids": uniform_full,
            "dropped_by_candidate_selection": _interval_overlap(interval, full_span) and not candidate_hits,
            "dropped_by_uniform_selection": _interval_overlap(interval, full_span) and not uniform_hits,
            "sampled_frame_observability": "not assessed: timestamp/window overlap does not establish that a human can see the reference action in the sampled frames",
            "model_recognition": "not assessed by coverage bookkeeping",
        })
    covered = [item for item in output if item["within_full_decoded_span"]]
    return {
        "full_decoded_pts_span_sec": full_span,
        "reference_intervals": output,
        "covered_reference_interval_count": len(covered),
        "candidate_dropped_reference_interval_count": sum(item["dropped_by_candidate_selection"] for item in covered),
        "uniform_dropped_reference_interval_count": sum(item["dropped_by_uniform_selection"] for item in covered),
        "candidate_fully_contained_reference_interval_count": sum(bool(item["candidate_fully_containing_window_ids"]) for item in covered),
        "uniform_fully_contained_reference_interval_count": sum(bool(item["uniform_fully_containing_window_ids"]) for item in covered),
        "status": "evaluation-only timestamp/window coverage accounting; not sampled-frame human observability, recognition, inference confidence, or an action score",
    }


def select_video_candidate_windows(video_dir, window_size=4, max_windows=3, max_selected_frames=12,
                                   minimum_new_frames_per_selected_window=None):
    """Build a self-describing CPU selection artifact from a frame manifest."""
    video_dir = Path(video_dir)
    manifest_path = video_dir / "frames_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = frame_change_records(video_dir, manifest.get("usable_frames", []))
    return {
        "video_id": manifest.get("video_id", video_dir.name),
        "source_type": "instrumented_measurement",
        "accepted_as_fact": False,
        "verification_status": "selection_measurement_not_event_evidence",
        "selection_scope": "quality-filtered usable frames only; excluded manifest frames are retained below and are not silently treated as covered",
        "manifest_excluded_frames": manifest.get("excluded_frames", {}),
        "frame_change_records": records,
        "candidate_selection": select_change_candidate_windows(
            records, window_size=window_size, max_windows=max_windows,
            max_selected_frames=max_selected_frames,
            minimum_new_frames_per_selected_window=minimum_new_frames_per_selected_window,
        ),
    }


def select_video_uniform_windows(video_dir, window_size=4, max_windows=3, max_selected_frames=12):
    """Build a matching non-motion baseline from exactly the same usable frames."""
    video_dir = Path(video_dir)
    manifest = json.loads((video_dir / "frames_manifest.json").read_text(encoding="utf-8"))
    records = frame_change_records(video_dir, manifest.get("usable_frames", []))
    return {
        "video_id": manifest.get("video_id", video_dir.name),
        "source_type": "instrumented_measurement",
        "accepted_as_fact": False,
        "verification_status": "uniform_selection_measurement_not_event_evidence",
        "selection_scope": "quality-filtered usable frames only; excluded manifest frames are retained below and are not silently treated as covered",
        "manifest_excluded_frames": manifest.get("excluded_frames", {}),
        "frame_change_records": records,
        "candidate_selection": uniform_candidate_windows(
            records, window_size=window_size, max_windows=max_windows,
            max_selected_frames=max_selected_frames,
        ),
    }


def select_fixed_video_windows(video_dir, window_size=4, max_windows=3, max_selected_frames=12,
                               expected_frame_count=None, mode="change",
                               minimum_new_frames_per_selected_window=None, uniform_anchor_count=2):
    """Select from a frozen fixed-PTS frame budget, never the scene-frame pool."""
    records = fixed_sampling_records(video_dir, expected_frame_count=expected_frame_count)
    if mode == "change":
        selection = select_change_candidate_windows(
            records, window_size=window_size, max_windows=max_windows,
            max_selected_frames=max_selected_frames,
            minimum_new_frames_per_selected_window=minimum_new_frames_per_selected_window,
        )
    elif mode == "uniform":
        selection = uniform_candidate_windows(
            records, window_size=window_size, max_windows=max_windows,
            max_selected_frames=max_selected_frames,
        )
    elif mode == "anchored_change":
        selection = anchored_change_candidate_windows(
            records, window_size=window_size, max_windows=max_windows,
            max_selected_frames=max_selected_frames,
            minimum_new_frames_per_selected_window=minimum_new_frames_per_selected_window,
            uniform_anchor_count=uniform_anchor_count,
        )
    else:
        raise ValueError("fixed selection mode must be change, uniform, or anchored_change")
    return {
        "video_id": Path(video_dir).name,
        "source_type": "instrumented_measurement",
        "accepted_as_fact": False,
        "verification_status": "fixed_sampling_selection_measurement_not_event_evidence",
        "selection_scope": "all fixed-rate decoded frames under the declared budget; dense scene frames are not pilot inputs",
        "frame_change_records": records,
        "candidate_selection": selection,
    }


def _source_window(frame_ids, source_pts_sec):
    if not isinstance(frame_ids, list) or not isinstance(source_pts_sec, list) or len(frame_ids) != len(source_pts_sec):
        raise ValueError("source window needs equally sized frame IDs and PTS")
    return _ordered_frame_records([
        {"frame_id": frame_id, "source_pts_sec": timestamp}
        for frame_id, timestamp in zip(frame_ids, source_pts_sec)
    ])


def temporal_order_controls(frame_ids, source_pts_sec):
    """Return forward/reverse and repeated-still controls with declared derived order."""
    frames = _source_window(frame_ids, source_pts_sec)
    ids, pts = [item["frame_id"] for item in frames], [item["source_pts_sec"] for item in frames]
    return {
        "purpose": "test temporal-direction sensitivity; control outputs are unvalidated model hypotheses",
        "forward": {
            "input_order": "source_chronological", "frame_ids": ids,
            "presentation_source_pts_sec": pts, "source_pts_chronological_sec": pts,
        },
        "reverse": {
            "input_order": "source_reverse", "frame_ids": list(reversed(ids)),
            "presentation_source_pts_sec": list(reversed(pts)), "source_pts_chronological_sec": pts,
        },
        "identical_still": {
            "input_order": "same_source_frame_repeated", "frame_ids": [ids[0]] * len(ids),
            "presentation_source_pts_sec": [pts[0]] * len(ids), "source_still_frame_id": ids[0],
            "source_still_pts_sec": pts[0],
        },
        "pass_condition": "Review whether forward/reverse claims correspond to the changed presentation direction and visible action; reverse order may legitimately show a reversed action. The repeated-identical-still input must not receive a temporal action claim. Schema validity alone is not a pass.",
    }


def _window_index(frame_ids, source_pts_sec):
    frames = _source_window(frame_ids, source_pts_sec)
    return {item["frame_id"]: item["source_pts_sec"] for item in frames}


def _materialize_evidence(raw_event, frames):
    """Accept exact references or model frame indices and emit exact references."""
    evidence = raw_event.get("evidence") if isinstance(raw_event, dict) else None
    if evidence is not None:
        return evidence, []
    indices = raw_event.get("evidence_frame_indices") if isinstance(raw_event, dict) else None
    if not isinstance(indices, list) or len(indices) < 2:
        return None, ["temporal event evidence needs at least two frame references or frame indices"]
    normalized = []
    for item in indices:
        if isinstance(item, bool):
            return None, ["evidence_frame_indices must use nonnegative integer or digit-string serializations"]
        if isinstance(item, int):
            value = item
        elif isinstance(item, str) and _INTEGER_TEXT.fullmatch(item):
            value = int(item)
        else:
            return None, ["evidence_frame_indices must use nonnegative integer or digit-string serializations"]
        if value < 0 or value >= len(frames):
            return None, ["evidence_frame_indices must reference supplied presentation-frame positions"]
        normalized.append(value)
    if len(set(normalized)) != len(normalized):
        return None, ["evidence_frame_indices must not repeat a source frame for a temporal event"]
    if any(item < 0 or item >= len(frames) for item in normalized):
        return None, ["evidence_frame_indices must reference supplied presentation-frame positions"]
    return [
        {"frame_id": frames[item]["frame_id"], "source_pts_sec": frames[item]["source_pts_sec"], "frame_index": item}
        for item in normalized
    ], []


def _event_errors(raw_event, index, frames, require_chronological):
    if not isinstance(raw_event, dict):
        return ["event must be an object"]
    errors = []
    for key in ("event_id", "verb", "object", "actor", "direction"):
        if not isinstance(raw_event.get(key), str) or not raw_event[key].strip():
            errors.append(f"{key} must be nonempty text")
    if raw_event.get("actor") not in _ACTORS:
        errors.append("actor must be wearer, other_person, or unknown")
    if raw_event.get("direction") not in _DIRECTIONS:
        errors.append("direction must be a declared controlled value")
    if not isinstance(raw_event.get("uncertain"), bool):
        errors.append("uncertain must be boolean")
    evidence, evidence_errors = _materialize_evidence(raw_event, frames)
    errors.extend(evidence_errors)
    if evidence is not None:
        previous = None
        seen = set()
        for item in evidence:
            if not isinstance(item, dict) or not isinstance(item.get("frame_id"), str) or not _finite_timestamp(item.get("source_pts_sec")):
                errors.append("each evidence reference needs frame_id and finite source_pts_sec")
                continue
            frame_id, pts = item["frame_id"], float(item["source_pts_sec"])
            if frame_id in seen or frame_id not in index or not math.isclose(index[frame_id], pts, abs_tol=1e-6):
                errors.append("evidence references must exactly match distinct source-window frame IDs and PTS")
            if require_chronological and previous is not None and pts <= previous:
                errors.append("event evidence must be in chronological source PTS order")
            seen.add(frame_id)
            previous = pts
    return errors, evidence


def ingest_structured_event_hypotheses(raw_model_text, frame_ids, source_pts_sec, model_provenance,
                                       input_order="source_chronological", assign_event_ids=False):
    """Parse model JSON while retaining every invalid/raw hypothesis transparently."""
    if not isinstance(raw_model_text, str) or not raw_model_text.strip():
        raise ValueError("raw model text must be nonempty")
    if not isinstance(model_provenance, dict) or not all(isinstance(model_provenance.get(key), str) and model_provenance[key] for key in ("model", "backend")):
        raise ValueError("model provenance needs model and backend")
    if input_order not in {"source_chronological", "source_reverse", "same_source_frame_repeated"}:
        raise ValueError("input_order must declare chronological, reverse, or repeated presentation")
    if input_order == "source_chronological":
        frames = _source_window(frame_ids, source_pts_sec)
    else:
        if not isinstance(frame_ids, list) or not isinstance(source_pts_sec, list) or len(frame_ids) != len(source_pts_sec) or len(frame_ids) < 2:
            raise ValueError("derived control needs equally sized presentation frame IDs and PTS")
        frames = []
        for position, (frame_id, timestamp) in enumerate(zip(frame_ids, source_pts_sec)):
            if not isinstance(frame_id, str) or not frame_id or not _finite_timestamp(timestamp):
                raise ValueError("derived control needs nonempty frame IDs and finite PTS")
            frames.append({"frame_id": frame_id, "source_pts_sec": float(timestamp), "presentation_index": position})
    index = {item["frame_id"]: item["source_pts_sec"] for item in frames}
    record = {
        "schema_version": EVENT_SCHEMA_VERSION,
        "source_type": "model",
        "raw_model_text": raw_model_text,
        "model_provenance": dict(model_provenance),
        "source_window": {"frame_ids": list(frame_ids), "presentation_source_pts_sec": list(source_pts_sec),
                          "source_pts_chronological_sec": sorted(float(item) for item in source_pts_sec)},
        "input_order": input_order,
        "verification_status": UNVALIDATED,
        "accepted_as_fact": False,
        "acceptance_boundary": "Only independently accepted evidence may support an answer; schema validity, frame references, or model self-report do not accept this hypothesis.",
        "events": [],
        "invalid_events": [],
    }
    try:
        payload = json.loads(raw_model_text)
    except json.JSONDecodeError as error:
        record["invalid_events"].append({"raw_event": raw_model_text, "schema_errors": [f"invalid JSON: {error.msg}"]})
        return record
    if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
        record["invalid_events"].append({"raw_event": payload, "schema_errors": ["top-level object requires an events list"]})
        return record
    seen_ids = set()
    for position, raw_event in enumerate(payload["events"], start=1):
        if assign_event_ids and isinstance(raw_event, dict) and "event_id" not in raw_event:
            raw_event = {**raw_event, "event_id": f"model_event_{position:02d}"}
        errors, evidence = _event_errors(raw_event, index, frames, require_chronological=input_order == "source_chronological")
        if isinstance(raw_event, dict) and raw_event.get("event_id") in seen_ids:
            errors.append("event_id must be unique within a response")
        if errors:
            record["invalid_events"].append({"raw_event": raw_event, "schema_errors": errors})
            continue
        seen_ids.add(raw_event["event_id"])
        evidence = [{key: (float(item[key]) if key == "source_pts_sec" else item[key])
                     for key in ("frame_id", "source_pts_sec", "frame_index") if key in item} for item in evidence]
        record["events"].append({
            "event_id": raw_event["event_id"].strip(), "verb": raw_event["verb"].strip(),
            "object": raw_event["object"].strip(), "actor": raw_event["actor"],
            "direction": raw_event["direction"], "uncertain": raw_event["uncertain"],
            "evidence": evidence,
            "source_pts_range_sec": [min(item["source_pts_sec"] for item in evidence), max(item["source_pts_sec"] for item in evidence)],
            "verification_status": UNVALIDATED,
            "accepted_as_fact": False,
        })
    return record


def summarize_hypothesis_outcomes(hypothesis_record, visual_reviews=()):
    """Keep schema failures, visual support, and abstention as disjoint counts."""
    if not isinstance(hypothesis_record, dict):
        raise ValueError("hypothesis_record must be an object")
    events = hypothesis_record.get("events")
    invalid = hypothesis_record.get("invalid_events")
    if not isinstance(events, list) or not isinstance(invalid, list):
        raise ValueError("hypothesis record must retain valid and invalid event lists")
    statuses = {"supported", "unsupported", "uncertain"}
    reviews = list(visual_reviews)
    if any(not isinstance(item, dict) or item.get("status") not in statuses for item in reviews):
        raise ValueError("visual reviews must explicitly declare supported, unsupported, or uncertain")
    if len(reviews) > len(events) + len(invalid):
        raise ValueError("visual reviews cannot exceed retained valid and invalid hypotheses")
    return {
        "schema_valid_event_count": len(events),
        "schema_format_failure_count": len(invalid),
        "abstention": not events and not invalid,
        "visual_supported_count": sum(item["status"] == "supported" for item in reviews),
        "visual_unsupported_count": sum(item["status"] == "unsupported" for item in reviews),
        "visual_uncertain_count": sum(item["status"] == "uncertain" for item in reviews),
        "visual_unreviewed_hypothesis_count": len(events) + len(invalid) - len(reviews),
        "accepted_as_fact": False,
        "note": "Schema validity and a visual-review status are separate from accepted evidence.",
    }

"""EPIC human annotations for evaluation only, never generated observations."""
import csv
import re
from pathlib import Path

from .manifest import sha256_file

_TIMESTAMP = re.compile(r"^(\d{2,}):([0-5]\d):([0-5]\d)(?:\.(\d{1,9}))?$")
MAX_REFERENCE_BYTES = 32 * 1024 * 1024


def timestamp_seconds(value):
    """Require explicit HH:MM:SS[.fraction], disallow NaN/overflow/negatives."""
    if not isinstance(value, str):
        raise ValueError("Timestamp must be HH:MM:SS[.fraction] text")
    match = _TIMESTAMP.fullmatch(value)
    if not match:
        raise ValueError(f"Invalid timestamp: {value!r}")
    hours, minutes, seconds, fraction = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + (float("0." + fraction) if fraction else 0)


def read_epic_references(csv_path, record, kind):
    """Select overlapping rows; retain original rows and source/clip intervals."""
    metadata = next((ref for ref in record.get("reference_annotations", []) if ref["kind"] == kind), None)
    if metadata is None or metadata.get("role") != "evaluation_reference_only":
        raise ValueError(f"No evaluation-only {kind!r} reference declared in manifest")
    path = Path(csv_path)
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_REFERENCE_BYTES:
        raise ValueError("Reference CSV must be local, nonempty, and at most 32 MiB")
    digest = sha256_file(path)
    if metadata.get("sha256") is not None and metadata["sha256"] != digest:
        raise ValueError("Reference CSV SHA256 does not match manifest")
    id_column, label_column = {"action": ("narration_id", "narration"), "sound": ("annotation_id", "class")}[kind]
    required = {"video_id", "start_timestamp", "stop_timestamp", id_column, label_column}
    start_clip, end_clip = record["clip_interval_seconds"]
    rows, seen = [], set()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)) or not required.issubset(reader.fieldnames):
            raise ValueError(f"Reference CSV requires unique columns: {sorted(required)}")
        for line, row in enumerate(reader, start=2):
            if row.get("video_id") != record["video_id"]:
                continue
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed CSV row {line}")
            start = timestamp_seconds(row["start_timestamp"])
            end = timestamp_seconds(row["stop_timestamp"])
            if not 0 <= start < end <= record["source_duration_seconds"]:
                raise ValueError(f"Reference interval outside source duration on row {line}")
            annotation_id = row[id_column]
            if not annotation_id or annotation_id in seen or not row[label_column].strip():
                raise ValueError(f"Missing/duplicate annotation ID or empty label on row {line}")
            seen.add(annotation_id)
            if end <= start_clip or start >= end_clip:
                continue
            rows.append({
                "annotation_id": annotation_id, "source_csv_row": line,
                "source_interval_seconds": [start, end],
                "clip_interval_seconds": [max(start, start_clip) - start_clip, min(end, end_clip) - start_clip],
                "label": row[label_column], "original": row,
            })
    return {
        "schema_version": 1, "role": "evaluation_reference_only", "kind": kind,
        "video_id": record["video_id"], "timestamp_unit": "seconds",
        "source_timestamp_origin": "source_video_start", "clip_timestamp_origin": "clip_start",
        "source_clip_interval_seconds": [start_clip, end_clip],
        "source_csv_sha256": digest, "provenance": metadata, "annotations": rows,
    }

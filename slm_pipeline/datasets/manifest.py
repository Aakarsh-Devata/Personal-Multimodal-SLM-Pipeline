"""Small, bounded manifests for caller-supplied licensed media. No networking."""
import copy
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse

MAX_RECORDS = 10
MAX_SOURCE_BYTES = 256 * 1024 * 1024
MAX_CLIP_SECONDS = 120.0
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def text_field(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")


def https_url(value, name):
    text_field(value, name)
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError(f"{name} must be an HTTPS URL without credentials")


def checksum(value, name, required=False):
    if value is None and not required:
        return
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")


def validate_manifest(manifest):
    """Validate metadata only; a 'verified' claim is not proof of local bytes."""
    if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    for key in ("dataset_id", "dataset_version", "purpose"):
        text_field(manifest.get(key), key)
    records = manifest.get("records")
    if not isinstance(records, list) or not 1 <= len(records) <= MAX_RECORDS:
        raise ValueError(f"records must contain 1 to {MAX_RECORDS} items")
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("record must be an object")
        video_id = record.get("video_id")
        if not isinstance(video_id, str) or not _ID.fullmatch(video_id) or video_id in seen:
            raise ValueError("video_id must be a unique safe filename stem")
        seen.add(video_id)
        for key in ("split", "license_id", "attribution", "changes"):
            text_field(record.get(key), key)
        for key in ("source_url", "license_url"):
            https_url(record.get(key), key)
        if record.get("timestamp_unit") != "seconds" or record.get("timestamp_origin") != "source_video_start":
            raise ValueError("timestamps must use seconds from source_video_start")
        duration = number(record.get("source_duration_seconds"), "source_duration_seconds")
        interval = record.get("clip_interval_seconds")
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError("clip_interval_seconds must be [start, end]")
        start, end = [number(v, "clip interval") for v in interval]
        if not 0 <= start < end <= duration or end - start > MAX_CLIP_SECONDS:
            raise ValueError(f"clip interval must be within source duration and at most {MAX_CLIP_SECONDS:g} seconds")
        size = record.get("size_bytes")
        if size is not None and (type(size) is not int or not 0 < size <= MAX_SOURCE_BYTES):
            raise ValueError(f"size_bytes must be positive and at most {MAX_SOURCE_BYTES}")
        status = record.get("verification_status")
        if status not in {"pending", "verified"}:
            raise ValueError("verification_status must be pending or verified")
        checksum(record.get("sha256"), "sha256", required=status == "verified")
        if status == "verified" and size is None:
            raise ValueError("verified records require size_bytes")
        references = record.get("reference_annotations", [])
        if not isinstance(references, list):
            raise ValueError("reference_annotations must be a list")
        kinds = set()
        for ref in references:
            if not isinstance(ref, dict) or ref.get("kind") not in {"action", "sound"} or ref["kind"] in kinds:
                raise ValueError("references must have unique action/sound kinds")
            kinds.add(ref["kind"])
            if ref.get("role") != "evaluation_reference_only":
                raise ValueError("human annotations must be evaluation_reference_only")
            for key in ("source_url", "license_url"):
                https_url(ref.get(key), key)
            for key in ("license_id", "attribution"):
                text_field(ref.get(key), key)
            checksum(ref.get("sha256"), "reference sha256")
    return manifest


def load_manifest(path):
    with Path(path).open(encoding="utf-8") as stream:
        return validate_manifest(json.load(stream))


def select_record(manifest, video_id):
    validate_manifest(manifest)
    matches = [record for record in manifest["records"] if record["video_id"] == video_id]
    if not matches:
        raise ValueError(f"Video {video_id!r} is absent from manifest")
    return matches[0]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe_video(path):
    """Inspect local video duration, not audio padding/container duration."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=duration,start_time", "-of", "json", str(Path(path).resolve())],
        check=True, text=True, capture_output=True, timeout=30,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if not streams or "duration" not in streams[0]:
        raise ValueError("Input must have a video stream with a known duration")
    duration = number(float(streams[0]["duration"]), "probed video duration")
    start = number(float(streams[0].get("start_time", 0)), "probed video start")
    if duration <= 0 or abs(start) > 0.001:
        raise ValueError("Expected a positive-duration video starting at zero")
    return duration


def verify_source(record, source, *, require_checksum=True):
    """Verify local bytes; never fetch source_url or trust a manifest status."""
    source = Path(source).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"Local source file not found: {source}")
    size = source.stat().st_size
    if not 0 < size <= MAX_SOURCE_BYTES:
        raise ValueError(f"Source exceeds the {MAX_SOURCE_BYTES}-byte bound or is empty")
    if record.get("size_bytes") is not None and size != record["size_bytes"]:
        raise ValueError("Source size does not match manifest")
    expected = record.get("sha256")
    checksum(expected, "sha256", required=require_checksum)
    actual = sha256_file(source)
    if expected is not None and actual != expected:
        raise ValueError("Source SHA256 does not match manifest")
    duration = probe_video(source)
    if abs(duration - record["source_duration_seconds"]) > 0.001:
        raise ValueError("Video stream duration does not match manifest (1 ms tolerance)")
    return {"sha256": actual, "size_bytes": size, "video_duration_seconds": duration}


def prepare_local(manifest, video_id, source, raw_dir, *, acknowledge_license=False):
    """Copy one full, bounded source video; preserve original bytes and offsets.

    This initial adapter deliberately rejects partial clips rather than silently
    feeding a full video where clip-relative timestamps would be expected.
    """
    if not acknowledge_license:
        raise ValueError("Review the manifest's license and pass --acknowledge-license")
    record = select_record(manifest, video_id)
    if record["clip_interval_seconds"] != [0, record["source_duration_seconds"]]:
        raise ValueError("Preparation currently supports full short videos only; partial-clip extraction is not implemented")
    source = Path(source).expanduser().resolve()
    if source.suffix.lower() not in {".mp4", ".mov", ".mkv", ".avi"}:
        raise ValueError("Unsupported local video extension")
    evidence = verify_source(record, source, require_checksum=False)
    raw_dir = Path(raw_dir).expanduser().resolve()
    destination = raw_dir / (video_id + source.suffix.upper())
    raw_dir.mkdir(parents=True, exist_ok=True)
    alternatives = [p for p in raw_dir.iterdir() if p.stem == video_id and p.resolve() != destination]
    if alternatives:
        raise ValueError("Raw directory already contains another file with this video ID")
    if source != destination:
        # Stage and check first; atomic exclusive linking never exposes a partial
        # input and does not replace an existing file or follow a destination link.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=raw_dir, prefix=".dataset-", delete=False) as output:
                temporary = Path(output.name)
                with source.open("rb") as incoming:
                    shutil.copyfileobj(incoming, output, length=1024 * 1024)
            if sha256_file(temporary) != evidence["sha256"]:
                raise ValueError("Copied file failed checksum verification")
            os.link(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    updated = copy.deepcopy(manifest)
    prepared = select_record(updated, video_id)
    prepared.update(sha256=evidence["sha256"], size_bytes=evidence["size_bytes"], verification_status="verified")
    prepared["changes"] = "None; original media bytes retained."
    return updated, destination, evidence

"""python -m slm_pipeline.datasets: offline validate/verify/prepare/references."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from .manifest import load_manifest, prepare_local, select_record, verify_source
from .references import read_epic_references


def write_new_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def check_reference_destination(path, config):
    """Keep evaluation labels out of actual and conventional pipeline outputs."""
    # Keep the caller's absolute spelling for the resulting file path.  On
    # macOS, ``/var`` is a symlink to ``/private/var``; returning ``resolve()``
    # changes an otherwise valid destination even though checks must still
    # follow symlinks to prevent an artifact-directory bypass.
    output = Path(path).expanduser().absolute()
    resolved_output = output.resolve()
    artifact_names = {
        "transcript.json", "vision.json", "semantic.json", "aligned.json", "embeddings.json",
        "context_blocks.json", "vision_captions.json", "semantic_structure.json",
        "processed_meta.json", "frame_timestamps.json", "index.faiss", "metadata.pkl",
    }
    conventional_parts = {"raw", "processed", "aligned", "semantic", "embeddings", "memory"}
    configured_roots = [Path(value).expanduser().resolve() for value in config["paths"].values()]
    if (resolved_output.name in artifact_names or set(resolved_output.parts) & conventional_parts
            or any(resolved_output == root or root in resolved_output.parents for root in configured_roots)):
        raise ValueError("Evaluation references must be written outside pipeline artifact directories")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "verify", "prepare", "references"):
        sub = subs.add_parser(command)
        sub.add_argument("--manifest", required=True)
        if command != "validate":
            sub.add_argument("--video", required=True)
        if command in {"verify", "prepare"}:
            sub.add_argument("--source", required=True, help="Caller-supplied licensed local video; never a URL")
        if command == "prepare":
            sub.add_argument("--raw-dir", help="Defaults to configured SLM_RAW_DIR / SLM_DATA_ROOT/raw")
            sub.add_argument("--output-manifest", required=True)
            sub.add_argument("--acknowledge-license", action="store_true")
        if command == "references":
            sub.add_argument("--kind", choices=("action", "sound"), required=True)
            sub.add_argument("--csv", required=True, help="Caller-supplied local reference CSV")
            sub.add_argument("--output", required=True, help="Evaluation-only JSON, outside pipeline artifact directories")
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        if args.command == "validate":
            print(json.dumps({"metadata_valid": True, "records": len(manifest["records"]), "local_bytes_checked": False}))
        elif args.command == "verify":
            print(json.dumps(verify_source(select_record(manifest, args.video), args.source)))
        elif args.command == "prepare":
            if Path(args.output_manifest).exists():
                raise ValueError("Output manifest already exists; choose a new path")
            if args.raw_dir:
                raw_dir = args.raw_dir
            else:
                from slm_pipeline.config import load_config
                raw_dir = load_config()["paths"]["raw_dir"]
            updated, destination, evidence = prepare_local(manifest, args.video, args.source, raw_dir, acknowledge_license=args.acknowledge_license)
            write_new_json(args.output_manifest, updated)
            print(json.dumps({"prepared": str(destination), "manifest": args.output_manifest, **evidence}))
        else:
            from slm_pipeline.config import load_config
            output = check_reference_destination(args.output, load_config())
            data = read_epic_references(args.csv, select_record(manifest, args.video), args.kind)
            write_new_json(output, data)
            print(json.dumps({"output": str(output), "role": data["role"], "annotations": len(data["annotations"])}))
        return 0
    except (ValueError, OSError, subprocess.SubprocessError, KeyError, TypeError) as error:
        print(f"Dataset error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

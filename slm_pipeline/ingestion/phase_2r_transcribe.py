#!/usr/bin/env python3
"""Local ASR with explicit download opt-in and auditable model hypotheses.

OpenAI Whisper remains the default. The optional faster_whisper backend supports
small CPU/int8 runs without installing PyTorch. Neither backend reads reference
annotations or treats an empty model hypothesis as verified absence of speech.
"""
from dataclasses import asdict
import json
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from slm_pipeline.config import config
from slm_pipeline.runtime import selected_video_dirs, phase_cli


def load_asr_model(settings, allow_downloads=False):
    """Return a local model and provenance; default loading cannot use the network."""
    backend = settings.get("backend", "whisper")
    name = settings.get("name", "base")
    device = settings.get("device", "cpu")
    metadata = {"backend": backend, "name": name, "device": device}
    if backend == "whisper":
        cache_root = Path(settings.get("download_root") or
                          Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "whisper").expanduser()
        model_path = Path(name).expanduser()
        if not model_path.is_file() and not allow_downloads:
            model_path = cache_root / f"{name}.pt"
            if not model_path.is_file():
                raise FileNotFoundError(
                    f"Whisper weights not cached: {name}; explicitly opt in with "
                    "pipeline.allow_model_downloads or SLM_ALLOW_DOWNLOADS=1")
        import whisper
        model = whisper.load_model(str(model_path) if model_path.is_file() else name,
                                   device=device, download_root=str(cache_root))
        metadata["compute_type"] = "float32"
        return model, metadata
    if backend != "faster_whisper":
        raise ValueError(f"Unknown ASR backend {backend!r}; use whisper or faster_whisper")

    from faster_whisper import WhisperModel, download_model
    import onnxruntime
    # Silero VAD runs locally through ONNX Runtime; disable its optional telemetry.
    onnxruntime.disable_telemetry_events()
    model_path = Path(name).expanduser()
    if not model_path.is_dir():
        # Even a named cached model goes through the library's local-only path.
        # Resolve first so an incomplete cache cannot trigger tokenizer fallback.
        model_path = Path(download_model(
            name, local_files_only=not allow_downloads,
            cache_dir=settings.get("download_root"), revision=settings.get("revision")))
    required = ("model.bin", "config.json", "tokenizer.json")
    missing = [asset for asset in required if not (model_path / asset).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete faster_whisper model at {model_path}: {', '.join(missing)}")
    compute_type = settings.get("compute_type", "int8" if device == "cpu" else "default")
    model = WhisperModel(str(model_path), device=device, compute_type=compute_type,
                         cpu_threads=settings.get("cpu_threads", 4), local_files_only=True)
    metadata.update(compute_type=compute_type, revision=settings.get("revision"))
    return model, metadata


def transcribe_audio(model, metadata, audio, settings):
    """Normalize both backends to timestamped segments with honest provenance."""
    if metadata["backend"] == "whisper":
        options = {"fp16": False, "temperature": 0}
        if settings.get("language"):
            options["language"] = settings["language"]
        result = model.transcribe(str(audio), **options)
    else:
        options = {"temperature": 0, "beam_size": settings.get("beam_size", 5),
                   "vad_filter": settings.get("vad_filter", True),
                   "condition_on_previous_text": False,
                   "language": settings.get("language")}
        segments, info = model.transcribe(str(audio), **options)
        # Iterating is necessary: faster_whisper inference is lazy.
        segments = [asdict(segment) for segment in segments]
        result = {"text": "".join(segment["text"] for segment in segments).strip(),
                  "segments": segments, "language": info.language,
                  "language_probability": info.language_probability,
                  "duration_sec": info.duration,
                  "duration_after_vad_sec": info.duration_after_vad}
    result.update(source_type="model_transcript", model=dict(metadata),
                  inference_options=options, accuracy_validated=False)
    return result


def main(video_id=None):
    model = None
    metadata = None
    settings = config["models"]["whisper"]
    for folder in selected_video_dirs(config, video_id):
        output = folder / "transcript.json"
        if output.exists() and config["pipeline"]["skip_existing"]:
            continue
        meta = json.loads((folder / "processed_meta.json").read_text())
        if not meta["has_audio"]:
            result = {"text": "", "segments": [], "source_type": "no_audio_stream"}
        else:
            audio = folder / "audio_clean.wav"
            if not audio.is_file():
                raise FileNotFoundError(f"Missing extracted audio for {folder.name}")
            if model is None:
                model, metadata = load_asr_model(settings, config["pipeline"]["allow_model_downloads"])
            result = transcribe_audio(model, metadata, audio, settings)
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(output)


if __name__ == "__main__":
    phase_cli(main)

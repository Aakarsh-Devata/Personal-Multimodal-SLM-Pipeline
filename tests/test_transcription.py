"""ASR backend contracts without model downloads or inference-quality claims."""
from dataclasses import dataclass
import json
import sys
from types import SimpleNamespace

import pytest

from slm_pipeline.ingestion import phase_2r_transcribe as phase


@pytest.fixture
def model_assets(tmp_path):
    for asset in ("model.bin", "config.json", "tokenizer.json"):
        (tmp_path / asset).touch()
    return tmp_path


def fake_faster_whisper(monkeypatch, model_assets):
    calls = []
    model = object()

    def download_model(name, **kwargs):
        calls.append(("download", name, kwargs))
        return str(model_assets)

    def whisper_model(name, **kwargs):
        calls.append(("load", name, kwargs))
        return model

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(
        download_model=download_model, WhisperModel=whisper_model))
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(
        disable_telemetry_events=lambda: None))
    return model, calls


def test_faster_whisper_defaults_to_offline_cpu_int8(monkeypatch, model_assets):
    expected, calls = fake_faster_whisper(monkeypatch, model_assets)
    model, metadata = phase.load_asr_model({"backend": "faster_whisper", "name": "tiny"})
    assert model is expected
    assert calls[0][2]["local_files_only"] is True
    assert calls[1][2] == {"device": "cpu", "compute_type": "int8", "cpu_threads": 4,
                          "local_files_only": True}
    assert metadata["backend"] == "faster_whisper"
    assert metadata["name"] == "tiny"


def test_faster_whisper_download_requires_explicit_opt_in(monkeypatch, model_assets):
    _, calls = fake_faster_whisper(monkeypatch, model_assets)
    phase.load_asr_model({"backend": "faster_whisper", "name": "tiny", "revision": "test-revision"},
                         allow_downloads=True)
    assert calls[0][2]["local_files_only"] is False
    assert calls[0][2]["revision"] == "test-revision"
    assert calls[1][2]["local_files_only"] is True


def test_local_faster_whisper_does_not_call_download(monkeypatch, model_assets):
    _, calls = fake_faster_whisper(monkeypatch, model_assets)
    phase.load_asr_model({"backend": "faster_whisper", "name": str(model_assets)})
    assert [call[0] for call in calls] == ["load"]


def test_incomplete_tokenizer_cache_cannot_trigger_fallback(monkeypatch, model_assets):
    _, calls = fake_faster_whisper(monkeypatch, model_assets)
    (model_assets / "tokenizer.json").unlink()
    with pytest.raises(FileNotFoundError, match="tokenizer.json"):
        phase.load_asr_model({"backend": "faster_whisper", "name": "tiny"})
    assert [call[0] for call in calls] == ["download"]


def test_uncached_whisper_does_not_import_or_download(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "whisper", None)
    with pytest.raises(FileNotFoundError, match="explicitly opt in"):
        phase.load_asr_model({"name": "base", "download_root": str(tmp_path)})


def test_default_whisper_accepts_local_weights(monkeypatch, tmp_path):
    weights = tmp_path / "base.pt"
    weights.touch()
    calls = []
    monkeypatch.setitem(sys.modules, "whisper", SimpleNamespace(
        load_model=lambda name, **kwargs: calls.append((name, kwargs)) or "model"))
    model, metadata = phase.load_asr_model({"name": "base", "download_root": str(tmp_path)})
    assert model == "model"
    assert calls[0][0] == str(weights)
    assert metadata["backend"] == "whisper"
    model, _ = phase.load_asr_model({"name": str(weights)})
    assert model == "model"


def test_invalid_backend_is_explicit():
    with pytest.raises(ValueError, match="Unknown ASR backend"):
        phase.load_asr_model({"backend": "unknown"})


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    no_speech_prob: float


def test_faster_segments_are_consumed_and_provenance_preserved(tmp_path):
    consumed = []

    def segments():
        consumed.append(True)
        yield Segment(0, 1.25, 2.75, " synthetic unit-test phrase", .03)

    info = SimpleNamespace(language="en", language_probability=.8, duration=4.0, duration_after_vad=2.0)
    model = SimpleNamespace(transcribe=lambda *args, **kwargs: (segments(), info))
    metadata = {"backend": "faster_whisper", "name": "tiny", "device": "cpu", "compute_type": "int8"}
    result = phase.transcribe_audio(model, metadata, tmp_path / "audio.wav", {})
    assert consumed == [True]
    assert result["segments"][0]["start"] == 1.25
    assert result["segments"][0]["end"] == 2.75
    assert result["text"] == "synthetic unit-test phrase"
    assert result["source_type"] == "model_transcript"
    assert result["model"] == metadata
    assert result["accuracy_validated"] is False
    assert result["inference_options"]["vad_filter"] is True
    json.dumps(result)


def test_empty_hypothesis_is_not_no_audio(tmp_path):
    info = SimpleNamespace(language="en", language_probability=0, duration=4, duration_after_vad=0)
    model = SimpleNamespace(transcribe=lambda *args, **kwargs: (iter([]), info))
    result = phase.transcribe_audio(model, {"backend": "faster_whisper"}, tmp_path / "audio.wav", {})
    assert result["text"] == ""
    assert result["segments"] == []
    assert result["source_type"] == "model_transcript"
    assert result["accuracy_validated"] is False


def test_no_audio_does_not_load_any_model(monkeypatch, tmp_path):
    folder = tmp_path / "no_audio"
    folder.mkdir()
    (folder / "processed_meta.json").write_text('{"has_audio": false}')
    monkeypatch.setattr(phase, "selected_video_dirs", lambda *args: [folder])
    monkeypatch.setattr(phase, "load_asr_model", lambda *args: pytest.fail("model must not load"))
    phase.main("no_audio")
    result = json.loads((folder / "transcript.json").read_text())
    assert result == {"text": "", "segments": [], "source_type": "no_audio_stream"}


def test_asr_opt_in_environment(monkeypatch):
    from slm_pipeline.config import load_config
    monkeypatch.setenv("SLM_ALLOW_DOWNLOADS", "1")
    assert load_config()["pipeline"]["allow_model_downloads"] is True
    monkeypatch.delenv("SLM_ALLOW_DOWNLOADS")
    monkeypatch.delenv("SLM_CONFIG", raising=False)
    assert load_config()["pipeline"]["allow_model_downloads"] is False

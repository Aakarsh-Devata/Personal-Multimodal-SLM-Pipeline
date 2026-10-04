"""Portable config/entrypoint regressions, without models or personal files."""
from pathlib import Path
import subprocess
import sys

import pytest
from slm_pipeline.config import load_config, PROJECT_ROOT
from slm_pipeline.runtime import selected_raw_videos, selected_video_dirs


def test_relative_paths_are_cwd_independent(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('SLM_DATA_ROOT', raising=False)
    assert load_config()['paths']['raw_dir'] == str(PROJECT_ROOT / 'data/raw')


def test_environment_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv('SLM_DATA_ROOT', str(tmp_path))
    assert load_config()['paths']['memory_dir'] == str(tmp_path / 'memory')
    monkeypatch.setenv('SLM_RAW_DIR', str(tmp_path / 'custom'))
    assert load_config()['paths']['raw_dir'] == str(tmp_path / 'custom')


def test_input_selector_is_exact(tmp_path):
    for name in ('alpha.mp4', 'alphabet.mov'):
        (tmp_path / name).touch()
    cfg = {'paths': {'raw_dir': str(tmp_path), 'processed_dir': str(tmp_path)}}
    assert [p.name for p in selected_raw_videos(cfg, 'alpha')] == ['alpha.mp4']
    with pytest.raises(FileNotFoundError):
        selected_raw_videos(cfg, 'missing')
    with pytest.raises(ValueError):
        selected_video_dirs(cfg, '../elsewhere')


def test_missing_input_has_nonzero_exit(tmp_path, monkeypatch):
    monkeypatch.setenv('SLM_DATA_ROOT', str(tmp_path))
    result = subprocess.run([sys.executable, '-m', 'slm_pipeline.orchestrator', '--video', 'missing'], capture_output=True, text=True)
    assert result.returncode == 1
    assert 'Pipeline failed' in result.stderr


def test_orchestrator_preserves_requested_scope(monkeypatch):
    import slm_pipeline.orchestrator as module
    calls = []
    class Phase:
        def main(self, video_id=None):
            calls.append(video_id)
    monkeypatch.setattr(module.importlib, 'import_module', lambda _: Phase())
    module.PipelineOrchestrator().process_single_video('only', start_phase=4, end_phase=5)
    assert calls == ['only', 'only']


def test_orchestrator_propagates_phase_failure(monkeypatch):
    import slm_pipeline.orchestrator as module
    class Phase:
        def main(self, video_id=None):
            raise RuntimeError('synthetic stage failure')
    monkeypatch.setattr(module.importlib, 'import_module', lambda _: Phase())
    assert module.main(['--video', 'only', '--start-phase', '4']) == 1


def test_invalid_fractional_phase_cannot_succeed_without_work():
    from slm_pipeline.orchestrator import PipelineOrchestrator
    with pytest.raises(ValueError, match='Select ordered phases'):
        PipelineOrchestrator().run_full_pipeline(3.6, 3.9, 'synthetic')

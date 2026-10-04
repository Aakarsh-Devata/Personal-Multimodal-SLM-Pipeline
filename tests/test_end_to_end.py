"""Synthetic tests of real media/timestamp plumbing; no inference-quality claim."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from slm_pipeline.config import config
from slm_pipeline.ingestion.phase_1r_preprocess import process_video, extract_frames
from slm_pipeline.ingestion.phase_3r_quality import process_video_folder
from slm_pipeline.pipelines.phase_5_align import MultimodalAligner


@pytest.fixture
def isolated_data(tmp_path, monkeypatch):
    for key in config['paths']:
        path = tmp_path / key
        path.mkdir()
        monkeypatch.setitem(config['paths'], key, str(path))
    return tmp_path


@pytest.fixture
def synthetic_video(isolated_data):
    if not shutil.which('ffmpeg'):
        pytest.skip('FFmpeg is required for genuine synthetic-media integration tests')
    path = Path(config['paths']['raw_dir']) / 'synthetic.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                    'testsrc2=size=320x240:rate=10:duration=3', '-c:v', 'mpeg4', str(path)], check=True)
    return path


def test_real_video_preprocess_retains_source_timestamps(synthetic_video):
    meta = process_video(synthetic_video)
    folder = Path(config['paths']['processed_dir']) / 'synthetic'
    timing = json.loads((folder / 'frame_timestamps.json').read_text())['frames']
    assert meta['has_audio'] is False
    assert [v['timestamp_sec'] for k, v in timing.items() if k.startswith('frames_fixed/')] == [0, 1, 2]
    process_video_folder(folder)
    manifest = json.loads((folder / 'frames_manifest.json').read_text())
    assert manifest['usable_frames']
    for frame in manifest['usable_frames']:
        assert frame['timestamp_sec'] == timing[frame['source_path']]['timestamp_sec']
        assert (folder / frame['path']).is_file()


def test_scene_indices_never_substitute_timestamps(tmp_path):
    if not shutil.which('ffmpeg'):
        pytest.skip('FFmpeg is required')
    video = tmp_path / 'scene.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                    "color=black:size=64x64:rate=10:duration=1.7", '-f', 'lavfi', '-i',
                    "color=white:size=64x64:rate=10:duration=1.3", '-filter_complex',
                    '[0:v][1:v]concat=n=2:v=1:a=0', '-c:v', 'mpeg4', str(video)], check=True)
    frames = extract_frames(video, tmp_path / 'frames_scene', 'scene', 'gt(scene,0.2)')
    assert list(frames.values())[0]['timestamp_sec'] == pytest.approx(1.7)


def test_alignment_retains_visual_evidence_and_silent_video():
    aligner = MultimodalAligner()
    frames = [{'frame_id': 'scene_1.jpg', 'timestamp_sec': 8.25, 'source_path': 'frames_scene/scene_1.jpg',
               'caption': 'synthetic object', 'success': True}]
    blocks = aligner.create_context_blocks({'segments': []}, frames)
    assert blocks[0]['time_range'] == [8.25, 8.25]
    assert blocks[0]['source_type'] == 'visual_only'
    assert blocks[0]['visual']['frame_timestamps_sec'] == [8.25]
    assert blocks[0]['visual']['frames'][0]['source_path'].startswith('frames_scene/')


def test_alignment_preserves_raw_model_hypothesis_provenance_without_acceptance():
    from slm_pipeline.pipelines.evidence_policy import raw_visual_hypothesis
    frame = {
        'frame_id': 'scene_2.jpg', 'timestamp_sec': 8.25, 'source_timestamp_sec': 8.125,
        'source_path': 'frames_scene/scene_2.jpg', 'caption': 'A synthetic object is visible.',
        'success': True, 'source_type': 'model', 'model': 'fixture-vlm', 'backend': 'fixture',
        'device': 'cpu', 'model_revision': 'abc123', 'run_id': 'fixture-run',
        'prompt_sha256': 'prompt-hash', 'frame_hash': 'pixels-hash',
    }
    observation = MultimodalAligner().create_context_blocks({'segments': []}, [frame])[0]['visual']['model_observations'][0]
    normalized = raw_visual_hypothesis(observation)
    assert normalized['source_pts_sec'] == [8.125]
    assert normalized['frame_hashes'] == ['pixels-hash']
    assert normalized['model_provenance']['model_revision'] == 'abc123'
    assert normalized['model_provenance']['run_id'] == 'fixture-run'
    assert observation['accepted_as_fact'] is False


def test_alignment_does_not_invent_timestamp():
    with pytest.raises(KeyError):
        MultimodalAligner().create_context_blocks({'segments': []}, [{'caption': 'untimed'}])


def test_vision_uses_manifest_not_filename_number(tmp_path):
    from slm_pipeline.pipelines.phase_4_vision import VisionProcessor
    frame = tmp_path / 'ready_scene_000001.jpg'
    frame.touch()
    (tmp_path / 'frames_manifest.json').write_text(json.dumps({'usable_frames': [
        {'path': frame.name, 'source_path': 'frames_scene/scene_000001.jpg', 'timestamp_sec': 9.75}]}))
    vision = VisionProcessor.__new__(VisionProcessor)
    vision.generate_caption = lambda _: {'caption': 'synthetic fixture label', 'success': True}
    vision.process_video_folder(tmp_path)
    result = json.loads((tmp_path / 'vision_captions.json').read_text())
    assert result['captions'][0]['timestamp_sec'] == 9.75
    assert result['captions'][0]['source_type'] == 'model'


def test_caption_only_retrieval_preserves_model_timestamp_and_excludes_asr(tmp_path):
    import numpy as np
    from slm_pipeline.pipelines.phase_7_embed import VectorStore
    from slm_pipeline.pipelines.vision_retrieval import CaptionOnlyRetriever

    class FakeEmbedder:
        dimension = 2

        def encode(self, texts):
            return np.array([[1, 0]] * len(texts), dtype=np.float32)

        def encode_single(self, text):
            return np.array([1, 0], dtype=np.float32)

    retriever = CaptionOnlyRetriever(
        embedder=FakeEmbedder(), vector_store=VectorStore(2, memory_dir=tmp_path / 'memory')
    )
    payload = {'video_id': 'public_fixture', 'captions': [
        {'frame_id': 'frame.jpg', 'source_path': 'frames/frame.jpg', 'timestamp_sec': 3.5,
         'source_timestamp_sec': 3.5, 'caption': 'A person holds a glass.', 'source_type': 'model',
         'model': 'test-model', 'backend': 'test', 'device': 'cpu'},
    ]}
    assert retriever.index_caption_data(payload) == 1
    result = retriever.search('glass', video_id='public_fixture')[0]
    assert result['frame_timestamp_sec'] == 3.5
    assert result['source_type'] == 'model'
    assert result['asr_status'] == 'not_run'


def test_caption_only_retrieval_rejects_non_model_evidence():
    from slm_pipeline.pipelines.vision_retrieval import CaptionOnlyRetriever
    with pytest.raises(ValueError, match='source_type=model'):
        CaptionOnlyRetriever.observation_documents({
            'video_id': 'fixture', 'captions': [{'timestamp_sec': 0, 'caption': 'label', 'source_type': 'human'}]
        })


def test_vision_device_selection_has_safe_mps_cpu_fallback():
    from types import SimpleNamespace
    from slm_pipeline.pipelines.phase_4_vision import VisionProcessor

    unavailable = SimpleNamespace(backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)))
    available = SimpleNamespace(backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)))
    assert VisionProcessor._select_device(unavailable, 'auto') == 'cpu'
    assert VisionProcessor._select_device(unavailable, 'mps') == 'cpu'
    assert VisionProcessor._select_device(available, 'auto') == 'mps'
    with pytest.raises(ValueError, match='auto, mps, or cpu'):
        VisionProcessor._select_device(unavailable, 'cuda')


def test_uniform_temporal_windows_preserve_manifest_source_times():
    from slm_pipeline.pipelines.temporal_experiment import uniform_windows
    frames = [
        {'path': f'f{index}.jpg', 'source_timestamp_sec': timestamp}
        for index, timestamp in enumerate((0.0, 2.002, 4.004, 6.006, 8.008))
    ]
    windows = list(uniform_windows(frames, window_size=3))
    assert [[frame['source_timestamp_sec'] for frame in window] for window in windows] == [
        [0.0, 2.002, 4.004], [6.006, 8.008]
    ]


def test_mlx_video_runner_is_local_only_and_traces_multiframe_input(tmp_path):
    from types import SimpleNamespace
    from slm_pipeline.pipelines.mlx_video import (
        MLXVideoRunner, decode_video_trace, map_clip_times_to_source, video_input_trace,
    )

    model = tmp_path / 'local-model'
    model.mkdir()
    window = tmp_path / 'window.mp4'
    window.touch()
    runner = MLXVideoRunner(model, '/isolated/mlx/bin/python', max_new_tokens=24, fps=1.0)
    command = runner.command(window)
    assert command[:3] == ['/isolated/mlx/bin/python', '-m', 'mlx_vlm.generate']
    assert '--trust-remote-code' not in command
    assert command[command.index('--model') + 1] == str(model)
    assert command[command.index('--video-num-frames') + 1] == '4'
    metadata = SimpleNamespace(
        timestamps=[0.0, 1.1678, 2.3357, 3.5035],
        frames_indices=[0, 70, 140, 210], fps=59.94, sampled_fps=1.1363,
    )
    decoded = decode_video_trace(
        window, 4.004, fps=1.0, video_num_frames=4,
        load_video_fn=lambda _path: (object(), metadata),
    )
    assert decoded['decoded_frame_indices'] == [0, 70, 140, 210]
    assert decoded['source_timestamps_sec'] == [4.004, 5.1718, 6.3397, 7.5075]
    trace = video_input_trace(metadata, [2, 78, 138], 4.004)
    assert trace['source_timestamps_sec'] == [4.004, 5.1718, 6.3397, 7.5075]
    assert trace['decoded_frame_count'] == 4
    assert trace['visual_patch_rows'] == 21528
    assert trace['merged_visual_tokens'] == 5382
    with pytest.raises(ValueError, match='increasing'):
        map_clip_times_to_source([2.0, 2.0], 0)
    with pytest.raises(ValueError, match='requested number'):
        decode_video_trace(
            window, 0, video_num_frames=4,
            load_video_fn=lambda _path: (object(), SimpleNamespace(
                timestamps=[0.0, 1.0], frames_indices=[0, 1], fps=1.0, sampled_fps=1.0,
            )),
        )


def test_persistent_mlx_runner_uses_one_load_and_native_video_tensors(tmp_path):
    from types import SimpleNamespace
    from slm_pipeline.pipelines.mlx_video import PersistentMLXVideoRunner

    model_dir = tmp_path / 'local-model'
    model_dir.mkdir()
    window = tmp_path / 'public-window.mp4'
    window.touch()
    calls = {'load': 0, 'generate': []}
    metadata = SimpleNamespace(
        timestamps=[0.0, 0.5, 1.0, 1.5], frames_indices=[0, 30, 60, 90],
        fps=60.0, sampled_fps=2.0,
    )

    class Tensor:
        def __init__(self, shape, values=None):
            self.shape = shape
            self.values = values
        def tolist(self):
            return self.values

    model = SimpleNamespace(config=SimpleNamespace(image_token_index=151655))
    processor = SimpleNamespace(video_processor=SimpleNamespace(
        max_pixels=None, patch_size=16, temporal_patch_size=2, merge_size=2,
        do_convert_rgb=True, do_rescale=True, rescale_factor=1 / 255,
        do_normalize=True, image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5],
    ))
    implementations = {
        'load': lambda path: (calls.__setitem__('load', calls['load'] + 1) or (model, processor)),
        'VideoSampling': lambda **kwargs: kwargs,
        'load_video': lambda path, sampling: ([object()] * 4, metadata),
        'apply_chat_template': lambda processor, config, question, **kwargs: 'video-chat-prompt',
        'prepare_inputs': lambda *args, **kwargs: {
            'input_ids': 'ids', 'attention_mask': 'mask',
            'pixel_values_videos': Tensor((5382, 1176)),
            'video_grid_thw': Tensor((1, 3), [[2, 78, 138]]),
        },
        'generate': lambda *args, **kwargs: (
            calls['generate'].append(kwargs) or SimpleNamespace(text='uncertain', peak_memory=2.5)
        ),
    }
    runner = PersistentMLXVideoRunner(
        model_dir, fps=2.0, video_num_frames=4, max_pixels=1000,
        max_merged_visual_tokens=6000, implementations=implementations,
    )
    first = runner.observe_window(window, 4.0)
    second = runner.observe_window(window, 8.0)
    assert calls['load'] == 1
    assert first['input_type'] == 'native_video_tensors'
    assert first['input_trace']['decoded_frame_indices'] == [0, 30, 60, 90]
    assert first['input_trace']['pixel_values_videos_shape'] == [5382, 1176]
    assert first['input_trace']['video_grid_thw'] == [2, 78, 138]
    assert first['input_trace']['source_timestamps_sec'] == [4.0, 4.5, 5.0, 5.5]
    assert first['input_trace']['configured_video_max_pixels'] == 1000
    assert first['input_trace']['video_preprocessor_contract']['patch_size'] == 16
    assert first['input_trace']['video_preprocessor_contract']['max_pixels'] == 1000
    assert first['input_trace']['preprocessing_parity_status'].startswith('recorded_local_contract')
    assert processor.video_processor.max_pixels == 1000
    assert second['input_trace']['source_timestamps_sec'] == [8.0, 8.5, 9.0, 9.5]
    overridden = runner.observe_window(window, 8.0, source_frame_pts=[8.008, 8.125, 8.258, 8.375])
    assert overridden['input_trace']['source_timestamps_sec'] == [8.008, 8.125, 8.258, 8.375]
    assert overridden['input_trace']['source_timestamp_mapping'].startswith('caller_provided')
    with pytest.raises(ValueError, match='strictly increasing'):
        runner.observe_window(window, 8.0, source_frame_pts=[8.0, 8.0, 8.2, 8.3])
    assert calls['generate'][0]['input_ids'] == 'ids'
    assert calls['generate'][0]['pixel_values_videos'].shape == (5382, 1176)
    assert calls['generate'][0]['temperature'] == 0.0


def test_native_frame_hashes_can_prove_a_repeated_still_control_is_static():
    from slm_pipeline.pipelines.mlx_video import (
        FROZEN_DEVELOPMENT_QUESTIONS,
        decoded_frame_hashes,
    )

    class Frame:
        mode, size = 'RGB', (2, 1)
        def __init__(self, pixels):
            self.pixels = pixels
        def tobytes(self):
            return self.pixels

    same = decoded_frame_hashes([Frame(b'ab'), Frame(b'ab'), Frame(b'ab')])
    changed = decoded_frame_hashes([Frame(b'ab'), Frame(b'ac')])
    assert len(set(same)) == 1
    assert len(set(changed)) == 2
    assert "say exactly 'no action observed'" in FROZEN_DEVELOPMENT_QUESTIONS[
        'wearer_hands_temporal_v2_no_action'
    ]
    assert 'at most one directly visible hand-object transition' in FROZEN_DEVELOPMENT_QUESTIONS[
        'structured_temporal_action_evidence_v1'
    ]


def test_persistent_mlx_runner_rejects_excess_visual_tokens_before_generation(tmp_path):
    from types import SimpleNamespace
    from slm_pipeline.pipelines.mlx_video import PersistentMLXVideoRunner

    model_dir = tmp_path / 'local-model'
    model_dir.mkdir()
    window = tmp_path / 'public-window.mp4'
    window.touch()
    metadata = SimpleNamespace(
        timestamps=[0.0, 1.0, 2.0, 3.0], frames_indices=[0, 1, 2, 3],
        fps=1.0, sampled_fps=1.0,
    )

    class Tensor:
        def __init__(self, shape, values=None):
            self.shape, self.values = shape, values
        def tolist(self):
            return self.values

    calls = {'generate': 0}
    processor = SimpleNamespace(video_processor=SimpleNamespace(max_pixels=None))
    implementations = {
        'load': lambda path: (SimpleNamespace(config=SimpleNamespace(image_token_index=1)), processor),
        'VideoSampling': lambda **kwargs: kwargs,
        'load_video': lambda path, sampling: ([object()] * 4, metadata),
        'apply_chat_template': lambda *args, **kwargs: 'prompt',
        'prepare_inputs': lambda *args, **kwargs: {
            'input_ids': 'ids', 'attention_mask': 'mask',
            'pixel_values_videos': Tensor((21528, 1536)),
            'video_grid_thw': Tensor((1, 3), [[2, 78, 138]]),
        },
        'generate': lambda *args, **kwargs: calls.__setitem__('generate', calls['generate'] + 1),
    }
    runner = PersistentMLXVideoRunner(
        model_dir, video_num_frames=4, max_merged_visual_tokens=2304,
        implementations=implementations,
    )
    with pytest.raises(RuntimeError, match='visual token cap exceeded'):
        runner.observe_window(window, 0.0)
    assert calls['generate'] == 0


def test_preprocess_corrupt_file_fails(isolated_data):
    video = Path(config['paths']['raw_dir']) / 'bad.mp4'
    video.write_bytes(b'not a video')
    with pytest.raises(subprocess.CalledProcessError):
        process_video(video)


def test_malformed_semantic_output_fails():
    from slm_pipeline.pipelines.phase_6_semantic import SemanticExtractor
    extractor = SemanticExtractor.__new__(SemanticExtractor)
    with pytest.raises(ValueError):
        extractor.parse_llm_response('not JSON')


def test_unequal_audio_video_origins_are_rejected(isolated_data, monkeypatch):
    from slm_pipeline.ingestion import phase_1r_preprocess as module
    video = Path(config['paths']['raw_dir']) / 'offset.mp4'
    video.touch()
    monkeypatch.setattr(module, 'ffprobe_info', lambda _: {'streams': [
        {'codec_type': 'video', 'start_time': '0', 'duration': '3'},
        {'codec_type': 'audio', 'start_time': '0.936'}], 'format': {'duration': '3'}})
    with pytest.raises(ValueError, match='Nonzero audio/video'):
        module.process_video(video)


def test_one_video_to_evidence_query_with_explicit_fake_models(synthetic_video, monkeypatch):
    """Actual FFmpeg/quality/alignment/FAISS; fake neural stages test contracts only."""
    import numpy as np
    from slm_pipeline.orchestrator import PipelineOrchestrator
    from slm_pipeline.pipelines.phase_4_vision import VisionProcessor
    from slm_pipeline.pipelines.phase_6_semantic import SemanticExtractor
    from slm_pipeline.pipelines.phase_7_embed import EmbeddingGenerator, VectorStore
    from slm_pipeline.agents.memory_agent import MemoryAgent
    # An unselected corrupt video must never be opened by any phase.
    (synthetic_video.parent / 'unselected.mp4').write_bytes(b'do not process')
    monkeypatch.setattr(VisionProcessor, '__init__', lambda self: None)
    monkeypatch.setattr(VisionProcessor, 'generate_caption', lambda self, path:
                        {'caption': 'synthetic test pattern', 'success': True})
    monkeypatch.setattr(SemanticExtractor, '__init__', lambda self: None)
    monkeypatch.setattr(SemanticExtractor, 'extract_semantic_info', lambda self, block:
                        {'summary': 'synthetic fixture summary', 'topics': [], 'tasks': [],
                         'decisions': [], 'entities': [], 'intent': 'informational', 'confidence': 0})
    def fake_init(self):
        self.dimension = 2
    monkeypatch.setattr(EmbeddingGenerator, '__init__', fake_init)
    monkeypatch.setattr(EmbeddingGenerator, 'encode', lambda self, texts: np.array([[1, 0]] * len(texts), dtype=np.float32))
    pipeline = PipelineOrchestrator()
    for _ in range(2):
        pipeline.process_single_video('synthetic')
    store = VectorStore(dimension=2)
    agent = MemoryAgent(embedder=EmbeddingGenerator(), vector_store=store)
    results = agent.search_memory('synthetic query', video_id='synthetic')
    assert results and results[0]['frame_timestamps_sec']
    assert results[0]['source_type'] == 'visual_only'
    assert len({row['block_id'] for row in store.metadata}) == store.index.ntotal
    assert {row['video_id'] for row in store.metadata} == {'synthetic'}
    assert not (Path(config['paths']['processed_dir']) / 'unselected').exists()


@pytest.mark.parametrize('response', ['{}', '[]', '{"summary":"text"}'])
def test_incomplete_semantic_schema_is_rejected(response):
    from slm_pipeline.pipelines.phase_6_semantic import SemanticExtractor
    with pytest.raises(ValueError):
        SemanticExtractor.__new__(SemanticExtractor).parse_llm_response(response)


def test_model_cannot_override_evidence_times():
    from slm_pipeline.pipelines.phase_6_semantic import SemanticExtractor
    valid = {'topics': [], 'decisions': [], 'tasks': [], 'people': [], 'entities': [],
             'intent': 'informational', 'summary': 'synthetic', 'confidence': 0.5,
             'context_block_id': 'injected', 'time_range': [999, 1000]}
    extractor = SemanticExtractor.__new__(SemanticExtractor)
    result = extractor.parse_llm_response(json.dumps(valid))
    assert 'context_block_id' not in result and 'time_range' not in result
    valid['confidence'] = True
    with pytest.raises(ValueError):
        extractor.parse_llm_response(json.dumps(valid))


def test_frame_command_explicitly_maps_verified_first_video(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from slm_pipeline.ingestion import phase_1r_preprocess as module
    commands = []
    def fake_run(command):
        commands.append(command)
        return SimpleNamespace(stderr='')
    monkeypatch.setattr(module, 'run', fake_run)
    module.extract_frames(tmp_path / 'multi.mp4', tmp_path / 'frames', 'scene', 'gt(scene,0.2)')
    command = commands[0]
    assert command[command.index('-map') + 1] == '0:v:0'

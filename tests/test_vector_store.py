"""Offline vector-store regressions using synthetic embeddings and temp folders."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from slm_pipeline.agents.memory_agent import MemoryAgent
from slm_pipeline.pipelines import phase_7_embed
from slm_pipeline.pipelines.phase_7_embed import MemoryIndexer, VectorStore


@pytest.fixture
def store(tmp_path):
    return VectorStore(dimension=2, memory_dir=tmp_path / 'memory')


def rows(video_id, count):
    return [
        {'video_id': video_id, 'block_id': f'ctx_{idx:04d}', 'time_range': [idx, idx + 1]}
        for idx in range(count)
    ]


class SyntheticEmbedder:
    dimension = 2

    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return np.asarray([[len(text), 1] for text in texts], dtype=np.float32)

    def encode_single(self, text):
        return np.zeros(2, dtype=np.float32)


def write_video(root, summaries):
    video_dir = root / 'video-a'
    video_dir.mkdir(exist_ok=True)
    semantic = []
    context = []
    for idx, summary in enumerate(summaries):
        block_id = f'ctx_{idx:04d}'
        semantic.append({'context_block_id': block_id, 'summary': summary})
        context.append({
            'block_id': block_id,
            'time_range': [idx * 2, idx * 2 + 1],
            'speech': {'text': f'Original speech {idx}'},
            'visual': {
                'caption': f'Visual caption {idx}',
                'frame_ids': [f'frame_{idx}.jpg'],
                'frame_timestamps_sec': [idx * 2 + 0.5],
                'frames': [{
                    'frame_id': f'frame_{idx}.jpg',
                    'timestamp_sec': idx * 2 + 0.5,
                    'source_path': f'frames_clean/frame_{idx}.jpg',
                }],
            },
            'raw_transcript_segment_id': idx,
        })
    (video_dir / 'semantic_structure.json').write_text(
        json.dumps({'semantic_blocks': semantic}), encoding='utf-8'
    )
    (video_dir / 'context_blocks.json').write_text(
        json.dumps({'context_blocks': context}), encoding='utf-8'
    )
    return video_dir


def test_module_import_does_not_load_sentence_transformers():
    source = '''
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] == 'sentence_transformers':
        raise AssertionError('Heavy model package imported during module import')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import slm_pipeline.pipelines.phase_7_embed
import slm_pipeline.agents.memory_agent
'''
    result = subprocess.run(
        [sys.executable, '-c', source],
        cwd=Path(__file__).resolve().parents[1],
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('dimension', [0, -1, 1.5, True])
def test_dimension_must_be_a_positive_integer(tmp_path, dimension):
    with pytest.raises(ValueError, match='positive integer'):
        VectorStore(dimension=dimension, memory_dir=tmp_path)


@pytest.mark.parametrize('vectors,metadata,message', [
    (np.zeros((1, 3)), rows('a', 1), 'shape'),
    (np.zeros(2), rows('a', 1), 'shape'),
    (np.zeros((2, 2)), rows('a', 1), 'match metadata'),
    (np.array([[np.nan, 1]]), rows('a', 1), 'finite'),
    (np.array([[np.inf, 1]]), rows('a', 1), 'finite'),
    (np.zeros((1, 2)), ['bad'], 'list of objects'),
])
def test_invalid_batches_leave_store_unchanged(store, vectors, metadata, message):
    with pytest.raises(ValueError, match=message):
        store.replace_video('a', vectors, metadata)
    assert store.index.ntotal == 0
    assert store.metadata == []


def test_replacement_rejects_wrong_video_metadata(store):
    store.replace_video('a', np.array([[1, 2]]), rows('a', 1))
    with pytest.raises(ValueError, match='belong to video_id'):
        store.replace_video('a', np.zeros((1, 2)), rows('b', 1))
    assert store.index.ntotal == 1
    assert store.metadata[0]['video_id'] == 'a'


def test_replacement_is_idempotent_and_preserves_other_videos(store):
    other = [{'video_id': 'b', 'block_id': 'other', 'summary': 'Keep me'}]
    store.replace_video('b', np.array([[20, 20]]), other)
    vectors = np.array([[1, 0], [2, 0], [3, 0]])
    for _ in range(3):
        store.replace_video('a', vectors, rows('a', 3))
        assert store.index.ntotal == 4
        assert len(store.metadata) == 4
    changed = [{'video_id': 'a', 'block_id': 'new', 'summary': 'Changed'}]
    store.replace_video('a', np.array([[9, 9]]), changed)
    assert store.index.ntotal == 2
    assert store.metadata == other + changed
    assert store.search(np.array([9, 9]), video_id='a')[0]['distance'] == 0
    assert store.search(np.array([20, 20]), video_id='b')[0]['distance'] == 0

    store.replace_video('a', np.empty((0, 2)), [])
    assert store.index.ntotal == 1
    assert store.metadata == other
    assert store.search(np.array([9, 9]), video_id='a') == []


def test_save_reload_retains_vectors_and_replacement(store):
    store.replace_video('a', np.array([[1, 2], [3, 4]]), rows('a', 2))
    store.save()
    reloaded = VectorStore(dimension=2, memory_dir=store.index_path.parent.parent)
    assert reloaded.index.ntotal == 2
    assert reloaded.search(np.array([3, 4]), k=1)[0]['block_id'] == 'ctx_0001'
    reloaded.replace_video('a', np.array([[8, 9]]), rows('a', 1))
    reloaded.save()
    reloaded_again = VectorStore(dimension=2, memory_dir=store.index_path.parent.parent)
    assert reloaded_again.index.ntotal == 1
    assert reloaded_again.search(np.array([8, 9]))[0]['distance'] == 0


def test_reloading_rejects_incompatible_dimension(store):
    store.save()
    with pytest.raises(ValueError, match='dimension'):
        VectorStore(dimension=3, memory_dir=store.index_path.parent.parent)


def test_reloading_rejects_mismatched_cardinality(store):
    store.replace_video('a', np.ones((1, 2)), rows('a', 1))
    store.save()
    store.metadata_path.write_text('[]', encoding='utf-8')
    with pytest.raises(ValueError, match='cardinality'):
        VectorStore(dimension=2, memory_dir=store.index_path.parent.parent)


@pytest.mark.parametrize('missing', ['index_path', 'metadata_path'])
def test_reloading_rejects_partial_store(store, missing):
    store.save()
    getattr(store, missing).unlink()
    with pytest.raises(FileNotFoundError, match='requires both'):
        VectorStore(dimension=2, memory_dir=store.index_path.parent.parent)


def test_empty_and_oversized_search_never_returns_phantom_rows(store):
    assert store.search(np.zeros(2), k=50) == []
    store.replace_video('a', np.ones((1, 2)), rows('a', 1))
    assert len(store.search(np.zeros(2), k=50)) == 1
    assert store.search(np.zeros(2), k=0) == []
    assert store.search(np.zeros(2), video_id='absent') == []
    with pytest.raises(ValueError, match='nonnegative integer'):
        store.search(np.zeros(2), k=-1)
    with pytest.raises(ValueError, match='dimension'):
        store.search(np.zeros(3))
    with pytest.raises(ValueError, match='finite'):
        store.search(np.array([np.nan, 0]))


def test_faiss_negative_neighbor_is_ignored(store):
    store.metadata = rows('a', 1)
    store.index = SimpleNamespace(
        d=2, ntotal=1,
        search=lambda query, k: (np.array([[0.0]]), np.array([[-1]])),
    )
    assert store.search(np.zeros(2)) == []


def test_video_filter_returns_full_k_despite_many_closer_other_videos(store):
    store.replace_video('other', np.zeros((100, 2)), rows('other', 100))
    store.replace_video('target', np.array([[10, 0], [12, 0], [11, 0]]), rows('target', 3))
    result = store.search(np.zeros(2), k=2, video_id='target')
    assert [row['block_id'] for row in result] == ['ctx_0000', 'ctx_0002']
    assert all(row['video_id'] == 'target' for row in result)
    assert len(store.search(np.zeros(2), k=20, video_id='target')) == 3


def test_time_filter_uses_inclusive_overlap_before_ranking(store):
    store.replace_video('a', np.array([[0, 0], [4, 0], [2, 0]]), [
        {'video_id': 'a', 'block_id': 'early', 'time_range': [0, 1]},
        {'video_id': 'a', 'block_id': 'boundary', 'time_range': [4, 5]},
        {'video_id': 'a', 'block_id': 'overlap', 'time_range': [3, 6]},
    ])
    result = store.search(np.zeros(2), k=2, video_id='a', time_range=(5, 10))
    assert [row['block_id'] for row in result] == ['overlap', 'boundary']
    assert store.search(np.zeros(2), time_range=(20, 21)) == []
    with pytest.raises(ValueError, match='time_range'):
        store.search(np.zeros(2), time_range=(4, 1))


def test_unvalidated_semantic_tasks_cannot_be_promoted_to_memory_facts(tmp_path):
    root = tmp_path / 'processed'
    video_dir = root / 'video-a'
    video_dir.mkdir(parents=True)
    (video_dir / 'semantic_structure.json').write_text(json.dumps({'semantic_blocks': [{
        'context_block_id': 'ctx_0000', 'summary': 'A model claims an action.',
        'topics': ['claimed-topic'], 'decisions': ['claimed decision'], 'tasks': ['claimed task'],
        'entities': [], 'confidence': 0.99,
        'verification_status': 'unvalidated_derived_model_hypothesis', 'accepted_as_fact': False,
    }]}), encoding='utf-8')
    (video_dir / 'context_blocks.json').write_text(json.dumps({'context_blocks': [{
        'block_id': 'ctx_0000', 'time_range': [0, 1], 'speech': {'text': ''},
        'visual': {'caption': 'raw caption', 'frame_ids': ['f0'], 'frame_timestamps_sec': [0], 'frames': []},
    }]}), encoding='utf-8')
    embedder = SyntheticEmbedder()
    vector = VectorStore(2, memory_dir=tmp_path / 'memory')
    indexer = phase_7_embed.MemoryIndexer(embedder=embedder, vector_store=vector)
    assert indexer.process_video_folder(video_dir) == 1
    row = vector.metadata[0]
    assert row['accepted_as_fact'] is False
    assert row['tasks'] == row['decisions'] == row['topics'] == []
    assert row['unvalidated_tasks'] == ['claimed task']
    agent = MemoryAgent(embedder=embedder, vector_store=vector)
    assert agent.get_all_tasks() == []
    assert agent.get_all_decisions() == []


def test_memory_agent_delegates_filters_to_store(store):
    store.replace_video('other', np.zeros((100, 2)), rows('other', 100))
    store.replace_video('target', np.array([[10, 0], [11, 0]]), rows('target', 2))
    agent = MemoryAgent(embedder=SyntheticEmbedder(), vector_store=store)
    assert len(agent.search_memory('query', k=2, video_id='target')) == 2
    assert agent.search_memory('query', k=0) == []
    assert agent.search_memory('query', video_id='missing') == []
    assert agent.search_memory('query', time_range=(10, 20), video_id='target') == []


def test_indexer_replaces_reduced_changed_and_empty_blocks_with_evidence(tmp_path, store):
    embedder = SyntheticEmbedder()
    indexer = MemoryIndexer(embedder=embedder, vector_store=store)
    video_dir = write_video(tmp_path, ['First summary', 'Second summary'])
    store.replace_video('video-b', np.array([[9, 9]]), rows('video-b', 1))
    for _ in range(2):
        assert indexer.process_video_folder(video_dir) == 2
        indexer.save()
        assert store.index.ntotal == 3
    result = store.search(np.array([13, 1]), video_id='video-a')[0]
    assert result['speech_text'] == 'Original speech 0'
    assert result['time_range'] == [0, 1]
    assert result['frame_ids'] == ['frame_0.jpg']
    assert result['frame_timestamps_sec'] == [0.5]
    assert result['frames'][0] == {
        'frame_id': 'frame_0.jpg', 'timestamp_sec': 0.5,
        'source_path': 'frames_clean/frame_0.jpg',
    }
    assert result['raw_transcript_segment_id'] == 0

    write_video(tmp_path, ['Replacement summary'])
    assert indexer.process_video_folder(video_dir) == 1
    assert store.index.ntotal == 2
    assert store.search(np.zeros(2), video_id='video-a')[0]['summary'] == 'Replacement summary'

    write_video(tmp_path, [''])
    call_count = len(embedder.calls)
    assert indexer.process_video_folder(video_dir) == 0
    assert len(embedder.calls) == call_count  # Never ask the model to encode an empty batch.
    assert store.index.ntotal == 1
    write_video(tmp_path, [])
    assert indexer.process_video_folder(video_dir) == 0
    indexer.save()
    assert store.index.ntotal == 1
    assert store.metadata[0]['video_id'] == 'video-b'


@pytest.mark.parametrize('missing', ['semantic_structure.json', 'context_blocks.json'])
def test_indexer_missing_inputs_raise_without_removing_existing_memory(tmp_path, store, missing):
    video_dir = write_video(tmp_path, ['Summary'])
    (video_dir / missing).unlink()
    store.replace_video('video-a', np.ones((1, 2)), rows('video-a', 1))
    indexer = MemoryIndexer(embedder=SyntheticEmbedder(), vector_store=store)
    with pytest.raises(FileNotFoundError):
        indexer.process_video_folder(video_dir)
    assert store.index.ntotal == 1


def test_indexer_embedding_failure_leaves_prior_video_unchanged(tmp_path, store):
    video_dir = write_video(tmp_path, ['New summary'])
    store.replace_video('video-a', np.ones((1, 2)), rows('video-a', 1))

    class BrokenEmbedder(SyntheticEmbedder):
        def encode(self, texts):
            raise RuntimeError('Model failed')

    indexer = MemoryIndexer(embedder=BrokenEmbedder(), vector_store=store)
    with pytest.raises(RuntimeError, match='Model failed'):
        indexer.process_video_folder(video_dir)
    assert store.index.ntotal == 1
    assert store.search(np.ones(2))[0]['distance'] == 0


def test_main_processes_only_selected_video_and_propagates_failure(tmp_path, monkeypatch):
    selected = tmp_path / 'video-a'
    selected.mkdir()
    (tmp_path / 'video-b').mkdir()
    monkeypatch.setitem(phase_7_embed.config['paths'], 'processed_dir', str(tmp_path))
    calls = []

    class StubIndexer:
        def process_video_folder(self, folder):
            calls.append(folder)

        def save(self):
            calls.append('saved')

    monkeypatch.setattr(phase_7_embed, 'MemoryIndexer', StubIndexer)
    phase_7_embed.main(video_id='video-a')
    assert calls == [selected, 'saved']

    class BrokenIndexer(StubIndexer):
        def process_video_folder(self, folder):
            raise RuntimeError('Indexing failed')

    monkeypatch.setattr(phase_7_embed, 'MemoryIndexer', BrokenIndexer)
    with pytest.raises(RuntimeError, match='Indexing failed'):
        phase_7_embed.main(video_id='video-a')

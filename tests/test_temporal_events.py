import json

import pytest
from PIL import Image

from slm_pipeline.pipelines.temporal_events import (
    UNVALIDATED,
    anchored_change_candidate_windows,
    compare_reference_coverage,
    frame_change_records,
    ingest_structured_event_hypotheses,
    select_change_candidate_windows,
    summarize_hypothesis_outcomes,
    temporal_order_controls,
    uniform_candidate_windows,
)


def source_window():
    return ['f0.jpg', 'f1.jpg', 'f2.jpg', 'f3.jpg'], [0.0, 1.0, 2.0, 3.0]


def valid_event_json():
    return json.dumps({"events": [{
        "event_id": "event-1", "verb": "open", "object": "door",
        "actor": "wearer", "direction": "open", "uncertain": False,
        "evidence": [
            {"frame_id": "f1.jpg", "source_pts_sec": 1.0},
            {"frame_id": "f2.jpg", "source_pts_sec": 2.0},
        ],
    }]})


def test_structured_events_keep_raw_text_and_never_self_verify():
    ids, pts = source_window()
    record = ingest_structured_event_hypotheses(
        valid_event_json(), ids, pts, {"model": "fixture", "backend": "local"},
    )
    assert record['raw_model_text'] == valid_event_json()
    assert record['verification_status'] == UNVALIDATED
    assert record['accepted_as_fact'] is False
    assert 'independently accepted evidence' in record['acceptance_boundary']
    assert record['events'][0]['source_pts_range_sec'] == [1.0, 2.0]
    assert record['events'][0]['evidence'][0]['frame_id'] == 'f1.jpg'
    assert not record['invalid_events']

    indexed = json.dumps({"events": [{
        "event_id": "indexed", "verb": "move", "object": "bowl", "actor": "wearer",
        "direction": "toward", "uncertain": True, "evidence_frame_indices": [1, 3],
    }]})
    materialized = ingest_structured_event_hypotheses(indexed, ids, pts, {"model": "fixture", "backend": "local"})
    assert materialized['events'][0]['evidence'] == [
        {'frame_id': 'f1.jpg', 'source_pts_sec': 1.0, 'frame_index': 1},
        {'frame_id': 'f3.jpg', 'source_pts_sec': 3.0, 'frame_index': 3},
    ]

    digit_strings = json.dumps({"events": [{
        "event_id": "string-indices", "verb": "move", "object": "bowl", "actor": "wearer",
        "direction": "toward", "uncertain": True, "evidence_frame_indices": ["1", "3"],
    }]})
    normalized = ingest_structured_event_hypotheses(digit_strings, ids, pts, {"model": "fixture", "backend": "local"})
    assert normalized['events'][0]['evidence'] == [
        {'frame_id': 'f1.jpg', 'source_pts_sec': 1.0, 'frame_index': 1},
        {'frame_id': 'f3.jpg', 'source_pts_sec': 3.0, 'frame_index': 3},
    ]

    application_ids = json.dumps({"events": [{
        "verb": "move", "object": "bowl", "actor": "wearer", "direction": "toward",
        "uncertain": True, "evidence_frame_indices": [1, 3],
    }]})
    assigned = ingest_structured_event_hypotheses(
        application_ids, ids, pts, {"model": "fixture", "backend": "local"}, assign_event_ids=True,
    )
    assert assigned['events'][0]['event_id'] == 'model_event_01'

    non_integer = json.dumps({"events": [{
        "event_id": "bad-index", "verb": "move", "object": "bowl", "actor": "wearer",
        "direction": "toward", "uncertain": True, "evidence_frame_indices": ["1.0", 99],
    }]})
    rejected = ingest_structured_event_hypotheses(non_integer, ids, pts, {"model": "fixture", "backend": "local"})
    assert rejected['events'] == []
    assert 'integer or digit-string' in rejected['invalid_events'][0]['schema_errors'][0]

    reverse = ingest_structured_event_hypotheses(indexed, list(reversed(ids)), list(reversed(pts)),
                                                  {"model": "fixture", "backend": "local"}, input_order='source_reverse')
    assert reverse['input_order'] == 'source_reverse'
    assert reverse['events'][0]['evidence'][0]['frame_id'] == 'f2.jpg'
    assert reverse['events'][0]['source_pts_range_sec'] == [0.0, 2.0]


def test_invalid_schema_or_frame_reference_is_retained_not_dropped():
    ids, pts = source_window()
    bad_reference = json.dumps({"events": [{
        "event_id": "bad", "verb": "open", "object": "door", "actor": "wearer",
        "direction": "open", "uncertain": False,
        "evidence": [
            {"frame_id": "f1.jpg", "source_pts_sec": 1.0},
            {"frame_id": "made-up.jpg", "source_pts_sec": 2.0},
        ],
    }]})
    record = ingest_structured_event_hypotheses(bad_reference, ids, pts, {"model": "fixture", "backend": "local"})
    assert record['events'] == []
    assert record['invalid_events'][0]['raw_event']['event_id'] == 'bad'
    assert 'exactly match' in record['invalid_events'][0]['schema_errors'][0]

    malformed = ingest_structured_event_hypotheses('{not-json', ids, pts, {"model": "fixture", "backend": "local"})
    assert malformed['invalid_events'][0]['raw_event'] == '{not-json'
    assert malformed['accepted_as_fact'] is False


def test_luma_selector_preserves_all_intervals_and_bounded_selection(tmp_path):
    frames = []
    for index, value in enumerate((0, 0, 255, 255, 8, 8)):
        path = tmp_path / f'f{index}.png'
        Image.new('L', (8, 8), value).save(path)
        frames.append({"frame_id": path.name, "path": path.name, "source_timestamp_sec": float(index)})
    measurements = frame_change_records(tmp_path, frames)
    selection = select_change_candidate_windows(measurements, window_size=3, max_windows=1, max_selected_frames=3)
    assert len(selection['all_candidate_intervals']) == 4
    assert len(selection['selected_candidate_ids']) == 1
    assert len(selection['skipped_candidate_ids']) == 3
    selected = next(item for item in selection['all_candidate_intervals'] if item['selected'])
    assert selected['source_pts_range_sec'] == [0.0, 2.0]
    assert selected['selection_metric'] == 'mean_abs_luma_delta_from_previous'
    assert selection['selected_unique_frame_count'] == 3
    assert all('selection_reason' in item for item in selection['all_candidate_intervals'])
    assert all('confidence' not in item for item in selection['all_candidate_intervals'])


def test_selector_requires_novel_frames_instead_of_repeating_one_high_change_region():
    records = [
        {'frame_id': f'f{index}', 'source_pts_sec': float(index), 'frame_hash': str(index),
         'mean_abs_luma_delta_from_previous': None if index == 0 else float(100 - index)}
        for index in range(7)
    ]
    selection = select_change_candidate_windows(
        records, window_size=4, max_windows=2, max_selected_frames=8,
        minimum_new_frames_per_selected_window=2,
    )
    by_id = {item['candidate_id']: item for item in selection['all_candidate_intervals']}
    assert by_id['change_0000_0003']['selected'] is True
    assert by_id['change_0001_0004']['selected'] is False
    assert 'insufficient novel frames' in by_id['change_0001_0004']['selection_reason']
    assert by_id['change_0002_0005']['selected'] is True


def test_anchored_selector_reserves_uniform_coverage_before_motion_fill():
    records = [
        {'frame_id': f'f{index}', 'source_pts_sec': float(index), 'frame_hash': str(index),
         'mean_abs_luma_delta_from_previous': None if index == 0 else float(100 - index)}
        for index in range(16)
    ]
    selection = anchored_change_candidate_windows(
        records, window_size=4, max_windows=3, max_selected_frames=12,
        uniform_anchor_count=2, minimum_new_frames_per_selected_window=2,
    )
    by_id = {item['candidate_id']: item for item in selection['all_candidate_intervals']}
    assert by_id['change_0000_0003']['selected'] is True
    assert by_id['change_0000_0003']['selection_kind'] == 'required_uniform_anchor'
    assert by_id['change_0012_0015']['selected'] is True
    assert by_id['change_0012_0015']['selection_kind'] == 'required_uniform_anchor'
    assert sum(item['selection_kind'] == 'motion_fill' for item in selection['all_candidate_intervals']) == 1
    selected_ids = {frame_id for item in selection['all_candidate_intervals'] if item['selected'] for frame_id in item['frame_ids']}
    assert selection['selected_unique_frame_count'] == len(selected_ids) == 10
    assert selection['selected_unique_frame_count'] <= selection['max_selected_unique_frames'] == 12
    assert set(selection['selected_candidate_ids']) >= {'change_0000_0003', 'change_0012_0015'}


def test_reference_coverage_compares_candidate_and_uniform_without_reading_labels():
    records = [
        {'frame_id': f'f{index}', 'source_pts_sec': float(index), 'frame_hash': str(index),
         'mean_abs_luma_delta_from_previous': None if index == 0 else float(10 - index)}
        for index in range(8)
    ]
    candidate = select_change_candidate_windows(records, window_size=3, max_windows=1, max_selected_frames=3)
    uniform = uniform_candidate_windows(records, window_size=3, max_windows=1, max_selected_frames=3)
    comparison = compare_reference_coverage(
        [{'reference_id': 'late', 'source_interval_sec': [5.1, 5.9]}],
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], candidate, uniform,
    )
    assert comparison['covered_reference_interval_count'] == 1
    assert comparison['candidate_dropped_reference_interval_count'] == 1
    assert comparison['uniform_dropped_reference_interval_count'] == 1
    assert comparison['candidate_fully_contained_reference_interval_count'] == 0
    assert comparison['reference_intervals'][0]['reference_id'] == 'late'
    assert comparison['reference_intervals'][0]['sampled_frame_observability'].startswith('not assessed')
    assert comparison['reference_intervals'][0]['model_recognition'].startswith('not assessed')


def test_direction_controls_preserve_source_order_and_show_derived_reverse():
    ids, pts = source_window()
    controls = temporal_order_controls(ids, pts)
    assert controls['forward']['frame_ids'] == ids
    assert controls['reverse']['frame_ids'] == list(reversed(ids))
    assert controls['reverse']['source_pts_chronological_sec'] == pts
    assert controls['identical_still']['frame_ids'] == ['f0.jpg'] * 4
    assert controls['identical_still']['source_still_pts_sec'] == 0.0
    assert 'reverse order may legitimately show a reversed action' in controls['pass_condition']
    with pytest.raises(ValueError, match='strictly increasing'):
        temporal_order_controls(['one', 'two'], [1.0, 0.0])


def test_cpu_candidate_cli_writes_selection_artifact_without_model_load(tmp_path):
    from slm_pipeline.pipelines.temporal_candidates import main

    usable = []
    for index, value in enumerate((0, 64, 255, 64)):
        name = f'frame_{index}.png'
        Image.new('L', (8, 8), value).save(tmp_path / name)
        usable.append({'frame_id': name, 'path': name, 'source_timestamp_sec': float(index)})
    (tmp_path / 'frames_manifest.json').write_text(json.dumps({
        'video_id': 'fixture', 'usable_frames': usable,
        'excluded_frames': {'duplicates': [{'path': 'dropped.png', 'source_timestamp_sec': 9.0}]},
    }))
    output = tmp_path / 'candidate.json'
    result = main(['--video-dir', str(tmp_path), '--window-size', '3', '--max-windows', '1',
                   '--max-selected-frames', '3', '--output', str(output)])
    assert output.is_file()
    assert result['source_type'] == 'instrumented_measurement'
    assert result['accepted_as_fact'] is False
    assert result['manifest_excluded_frames']['duplicates'][0]['path'] == 'dropped.png'
    assert len(json.loads(output.read_text())['candidate_selection']['all_candidate_intervals']) == 2
    uniform_output = tmp_path / 'uniform.json'
    uniform = main(['--video-dir', str(tmp_path), '--window-size', '3', '--max-windows', '1',
                    '--max-selected-frames', '3', '--mode', 'uniform', '--output', str(uniform_output)])
    assert uniform['candidate_selection']['selection_method'].startswith('uniform')


def test_fixed_selector_uses_exact_fixed_budget_not_dense_scene_manifest(tmp_path):
    from slm_pipeline.pipelines.temporal_candidates import main

    times = {}
    for index in range(4):
        source = f'frames_fixed/frame_{index:06d}.png'
        path = tmp_path / source
        path.parent.mkdir(exist_ok=True)
        Image.new('L', (8, 8), index * 50).save(path)
        times[source] = {'timestamp_sec': float(index)}
    for index in range(5):
        source = f'frames_scene/scene_{index:06d}.png'
        path = tmp_path / source
        path.parent.mkdir(exist_ok=True)
        Image.new('L', (8, 8), 255).save(path)
        times[source] = {'timestamp_sec': index + 0.1}
    (tmp_path / 'frame_timestamps.json').write_text(json.dumps({'frames': times}))
    output = tmp_path / 'fixed.json'
    fixed = main(['--video-dir', str(tmp_path), '--frame-source', 'fixed', '--expected-frame-count', '4',
                  '--window-size', '3', '--max-windows', '1', '--max-selected-frames', '3', '--output', str(output)])
    assert fixed['candidate_selection']['source_frame_count'] == 4
    assert all('frames_fixed/' in item['source_path'] for item in fixed['frame_change_records'])
    assert 'dense scene frames' in fixed['selection_scope']
    anchored = main(['--video-dir', str(tmp_path), '--frame-source', 'fixed', '--expected-frame-count', '4',
                     '--window-size', '2', '--max-windows', '2', '--max-selected-frames', '4',
                     '--mode', 'anchored_change', '--uniform-anchor-count', '1'])
    assert anchored['candidate_selection']['uniform_anchor_count'] == 1
    assert any(item['selection_kind'] == 'required_uniform_anchor'
               for item in anchored['candidate_selection']['all_candidate_intervals'])


def test_outcome_summary_separates_schema_failures_visual_reviews_and_abstention():
    ids, pts = source_window()
    record = ingest_structured_event_hypotheses(valid_event_json(), ids, pts, {"model": "fixture", "backend": "local"})
    summary = summarize_hypothesis_outcomes(record, [{"status": "unsupported"}])
    assert summary['schema_valid_event_count'] == 1
    assert summary['schema_format_failure_count'] == 0
    assert summary['abstention'] is False
    assert summary['visual_unsupported_count'] == 1
    assert summary['accepted_as_fact'] is False
    with pytest.raises(ValueError, match='cannot exceed'):
        summarize_hypothesis_outcomes(record, [{"status": "supported"}, {"status": "unsupported"}])

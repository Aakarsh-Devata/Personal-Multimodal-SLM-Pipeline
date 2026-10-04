import pytest

from slm_pipeline.pipelines.evidence_policy import (
    ACCEPTED_INDEPENDENT_EVIDENCE,
    answer_temporal_question,
    experimental_calibration_policy,
    raw_visual_hypothesis,
    temporal_sequence_assessment,
)


def hypothesis(text='A hand puts down a glass.', hashes=None, pts=None):
    points = [0.0, 1.0, 2.0] if pts is None else pts
    return {
        'source_type': 'model', 'raw_model_text': text,
        'source_pts_sec': points,
        'frame_ids': [f'f{index}' for index in range(len(points))],
        'frame_hashes': hashes or [],
        'model_provenance': {'model': 'fixture-vlm', 'backend': 'fixture'},
    }


def test_repeated_static_frames_cannot_leak_hallucinated_motion_into_answer():
    raw = hypothesis(hashes=['same', 'same', 'same'])
    assessment = temporal_sequence_assessment(raw)
    assert assessment['sequence_state'] == 'repeated_static_frames'
    answer = answer_temporal_question('What changed?', [0, 2], [raw])
    assert answer['status'] == 'abstain'
    assert 'cannot establish' in answer['answer']
    assert answer['raw_hypotheses'][0]['accepted_as_fact'] is False


def test_nonidentical_frames_are_not_action_proof_and_unseen_range_abstains():
    raw = hypothesis(hashes=['a', 'b', 'c'])
    answer = answer_temporal_question('What action occurred?', [10, 11], [raw])
    assert answer['status'] == 'abstain'
    assert answer['temporal_assessments'][0]['sequence_state'] == 'nonidentical_frames_unverified'


def test_absent_visual_modality_and_contradictory_captions_both_abstain():
    absent = answer_temporal_question('What happened?', [0, 1], [])
    contradictory = answer_temporal_question(
        'What happened?', [0, 1],
        [hypothesis('A red cup appears.'), hypothesis('A blue bowl appears.')],
    )
    assert absent['status'] == contradictory['status'] == 'abstain'
    assert len(contradictory['raw_hypotheses']) == 2


def test_only_independently_accepted_evidence_can_support_answer():
    raw = hypothesis()
    accepted = {
        'evidence_id': 'verified-1', 'verification_status': ACCEPTED_INDEPENDENT_EVIDENCE,
        'source_type': 'human_review', 'accepted_as_fact': True,
        'source_time_range_sec': [0.5, 0.8], 'claim': 'An independently reviewed object movement is visible.',
        'acceptance_provenance': {
            'method': 'independent_human_review', 'reviewer_or_system': 'fixture-reviewer',
            'source_evidence_ids': ['frame-1'],
        },
    }
    answer = answer_temporal_question('What changed?', [0, 1], [raw], [accepted])
    assert answer['status'] == 'supported'
    assert answer['accepted_evidence'][0]['evidence_id'] == 'verified-1'


def test_provenance_and_calibration_are_fail_closed():
    with pytest.raises(ValueError, match='model and backend'):
        raw_visual_hypothesis({'source_type': 'model', 'raw_model_text': 'x', 'source_pts_sec': [0], 'frame_ids': ['f'], 'model_provenance': {}})
    policy = experimental_calibration_policy()
    assert policy['status'] == 'experimental_not_calibrated'
    assert 'retrieval distance' in policy['forbidden_acceptance_signals']
    with pytest.raises(ValueError, match='explicitly accepted'):
        answer_temporal_question('What changed?', [0, 1], [], [{
            'evidence_id': 'spoof', 'verification_status': ACCEPTED_INDEPENDENT_EVIDENCE,
            'source_time_range_sec': [0, 1], 'claim': 'A self-reported model claim.',
        }])
    with pytest.raises(ValueError, match='explicitly accepted'):
        answer_temporal_question('What changed?', [0, 1], [], [{
            'evidence_id': 'model-masquerade', 'verification_status': ACCEPTED_INDEPENDENT_EVIDENCE,
            'source_type': 'model', 'source_time_range_sec': [0, 1], 'claim': 'Self-certified.',
            'acceptance_provenance': {
                'method': 'independent_human_review', 'reviewer_or_system': 'the-model',
                'source_evidence_ids': ['model-output'],
            },
        }])

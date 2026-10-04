"""Fail-closed policy for visual-model hypotheses and temporal answers.

Visual model text is useful for explicitly labelled retrieval, but it is not a
verified event log.  This module keeps raw model output separate from accepted
evidence and makes temporal/action answers abstain by default.
"""

from __future__ import annotations

import math


UNVALIDATED_MODEL_HYPOTHESIS = "unvalidated_model_hypothesis"
ACCEPTED_INDEPENDENT_EVIDENCE = "accepted_independent_evidence"


def _finite_time(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _overlaps(left, right):
    return left[0] <= right[1] and right[0] <= left[1]


def raw_visual_hypothesis(observation):
    """Normalize one raw model observation without accepting its claims."""
    if not isinstance(observation, dict) or observation.get("source_type") != "model":
        raise ValueError("raw visual evidence must be a source_type=model observation")
    source_pts = observation.get("source_pts_sec", observation.get("source_timestamps_sec"))
    if not isinstance(source_pts, list) or not source_pts or not all(_finite_time(item) for item in source_pts):
        raise ValueError("raw visual evidence requires finite source PTS values")
    if any(later < earlier for earlier, later in zip(source_pts, source_pts[1:])):
        raise ValueError("source PTS must be chronological")
    frame_ids = observation.get("frame_ids")
    if not isinstance(frame_ids, list) or len(frame_ids) != len(source_pts) or not all(isinstance(item, str) and item for item in frame_ids):
        raise ValueError("raw visual evidence requires one nonempty frame ID per source PTS")
    provenance = observation.get("model_provenance")
    if not isinstance(provenance, dict) or not all(isinstance(provenance.get(key), str) and provenance[key] for key in ("model", "backend")):
        raise ValueError("raw visual evidence requires model and backend provenance")
    text = observation.get("raw_model_text", observation.get("observation", observation.get("caption")))
    if not isinstance(text, str) or not text.strip():
        raise ValueError("raw visual evidence requires nonempty raw model text")
    hashes = observation.get("frame_hashes", [])
    if hashes and (not isinstance(hashes, list) or len(hashes) != len(source_pts) or not all(isinstance(item, str) and item for item in hashes)):
        raise ValueError("frame hashes, when supplied, must match source PTS")
    return {
        "source_type": "model",
        "raw_model_text": text.strip(),
        "source_pts_sec": [float(item) for item in source_pts],
        "frame_ids": list(frame_ids),
        "frame_hashes": list(hashes),
        "model_provenance": dict(provenance),
        "verification_status": UNVALIDATED_MODEL_HYPOTHESIS,
        "accepted_as_fact": False,
    }


def temporal_sequence_assessment(hypothesis):
    """Describe what the input sequence does *not* establish on its own."""
    record = raw_visual_hypothesis(hypothesis)
    hashes = record["frame_hashes"]
    if len(hashes) > 1 and len(set(hashes)) == 1:
        state = "repeated_static_frames"
        reason = "All supplied frame hashes are identical; a temporal change claim is impossible from this sequence."
    elif len(record["frame_ids"]) < 2:
        state = "single_frame"
        reason = "One frame cannot establish a temporal change."
    else:
        state = "nonidentical_frames_unverified"
        reason = "Different frames establish neither action, causality, nor a temporal claim without independent accepted evidence."
    return {
        "sequence_state": state,
        "can_support_temporal_change": False,
        "reason": reason,
    }


def accepted_evidence_record(evidence):
    """Validate independently accepted evidence for a time-bounded answer."""
    if not isinstance(evidence, dict) or evidence.get("verification_status") != ACCEPTED_INDEPENDENT_EVIDENCE:
        raise ValueError("accepted evidence requires independent acceptance")
    if evidence.get("accepted_as_fact") is not True:
        raise ValueError("accepted evidence must be explicitly accepted as fact")
    interval = evidence.get("source_time_range_sec")
    if not isinstance(interval, list) or len(interval) != 2 or not all(_finite_time(item) for item in interval) or interval[1] < interval[0]:
        raise ValueError("accepted evidence requires a finite source time range")
    claim = evidence.get("claim")
    if not isinstance(claim, str) or not claim.strip():
        raise ValueError("accepted evidence requires a claim")
    if not isinstance(evidence.get("evidence_id"), str) or not evidence["evidence_id"]:
        raise ValueError("accepted evidence requires an evidence ID")
    acceptance = evidence.get("acceptance_provenance")
    if not isinstance(acceptance, dict):
        raise ValueError("accepted evidence requires acceptance provenance")
    if acceptance.get("method") not in {"independent_human_review", "independent_instrumented_measurement"}:
        raise ValueError("accepted evidence requires an independent acceptance method")
    expected_source_type = {
        "independent_human_review": "human_review",
        "independent_instrumented_measurement": "instrumented_measurement",
    }[acceptance["method"]]
    if evidence.get("source_type") != expected_source_type:
        raise ValueError("accepted evidence source type must match its independent acceptance method")
    if not isinstance(acceptance.get("reviewer_or_system"), str) or not acceptance["reviewer_or_system"]:
        raise ValueError("accepted evidence requires an accepting reviewer or system")
    source_ids = acceptance.get("source_evidence_ids")
    if not isinstance(source_ids, list) or not source_ids or not all(isinstance(item, str) and item for item in source_ids):
        raise ValueError("accepted evidence requires source evidence IDs")
    return {
        "evidence_id": evidence["evidence_id"], "claim": claim.strip(),
        "source_time_range_sec": [float(item) for item in interval],
        "source_type": evidence["source_type"],
        "acceptance_provenance": {
            "method": acceptance["method"], "reviewer_or_system": acceptance["reviewer_or_system"],
            "source_evidence_ids": list(source_ids),
        },
    }


def answer_temporal_question(question, requested_source_range_sec, raw_hypotheses, accepted_evidence=()):
    """Return support only from independently accepted evidence, else abstain.

    Retrieval distance, model confidence, and nonidentical frames are never
    acceptance criteria.  Raw hypotheses are returned only as transparent,
    unvalidated context and can never make an answer confident.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be nonempty")
    if (not isinstance(requested_source_range_sec, list) or len(requested_source_range_sec) != 2
            or not all(_finite_time(item) for item in requested_source_range_sec)
            or requested_source_range_sec[1] < requested_source_range_sec[0]):
        raise ValueError("requested source range must be finite and ordered")
    hypotheses = [raw_visual_hypothesis(item) for item in raw_hypotheses]
    assessments = [temporal_sequence_assessment(item) for item in hypotheses]
    accepted = [accepted_evidence_record(item) for item in accepted_evidence]
    requested = [float(item) for item in requested_source_range_sec]
    relevant = [item for item in accepted if _overlaps(item["source_time_range_sec"], requested)]
    if not relevant:
        return {
            "status": "abstain",
            "answer": "I cannot establish that temporal claim from independently accepted evidence in the requested source-time range.",
            "source_time_range_sec": requested,
            "accepted_evidence": [],
            "raw_hypotheses": hypotheses,
            "temporal_assessments": assessments,
            "confidence": "not_calibrated",
        }
    return {
        "status": "supported",
        "answer": " ".join(item["claim"] for item in relevant),
        "source_time_range_sec": requested,
        "accepted_evidence": relevant,
        "raw_hypotheses": hypotheses,
        "temporal_assessments": assessments,
        "confidence": "not_calibrated",
    }


def experimental_calibration_policy():
    """Make explicit that no confidence/distance threshold is calibrated."""
    return {
        "status": "experimental_not_calibrated",
        "forbidden_acceptance_signals": ["model confidence", "retrieval distance", "nonidentical frames"],
    }

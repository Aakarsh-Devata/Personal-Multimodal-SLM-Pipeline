"""A small, fail-closed question layer over timestamped visual retrieval.

This is deliberately retrieval rather than a generative answerer.  It can show
the observations an indexed local model associated with a question, but it
does not turn those observations or their FAISS distance into a factual claim.
"""

from __future__ import annotations

import math
from numbers import Integral
from typing import Optional


MAX_QUERY_LENGTH = 1000
MAX_RESULTS = 10
ACCEPTED_STATUS = "accepted_independent_evidence"


def _finite_seconds(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number of seconds")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return float(value)


def _normalise_range(start_sec=None, end_sec=None):
    if start_sec is None and end_sec is None:
        return None
    if start_sec is None or end_sec is None:
        raise ValueError("start_sec and end_sec must be supplied together")
    interval = [_finite_seconds(start_sec, "start_sec"), _finite_seconds(end_sec, "end_sec")]
    if interval[1] < interval[0]:
        raise ValueError("end_sec must not be earlier than start_sec")
    return interval


class EvidenceQueryService:
    """Query a CaptionOnlyRetriever while keeping evidence status explicit."""

    def __init__(self, retriever):
        self.retriever = retriever

    @staticmethod
    def _validate(question, video_id, limit, start_sec, end_sec):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be nonempty text")
        question = question.strip()
        if len(question) > MAX_QUERY_LENGTH:
            raise ValueError(f"question must be at most {MAX_QUERY_LENGTH} characters")
        if video_id is not None:
            if not isinstance(video_id, str) or not video_id.strip():
                raise ValueError("video_id must be nonempty text when supplied")
            video_id = video_id.strip()
        if isinstance(limit, bool) or not isinstance(limit, Integral) or not 1 <= limit <= MAX_RESULTS:
            raise ValueError(f"limit must be an integer from 1 to {MAX_RESULTS}")
        return question, video_id, int(limit), _normalise_range(start_sec, end_sec)

    @staticmethod
    def _card(result):
        time_range = result.get("time_range")
        if not isinstance(time_range, (list, tuple)) or len(time_range) != 2:
            timestamp = result.get("frame_timestamp_sec")
            if isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool):
                time_range = [float(timestamp), float(timestamp)]
            else:
                time_range = None
        elif all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in time_range):
            time_range = [float(time_range[0]), float(time_range[1])]
        else:
            time_range = None
        status = result.get("verification_status")
        if not isinstance(status, str) or not status:
            status = (
                "unvalidated_model_hypothesis"
                if result.get("source_type") == "model"
                else "unvalidated_or_unknown"
            )
        accepted = status == ACCEPTED_STATUS and result.get("accepted_as_fact") is True
        return {
            "video_id": result.get("video_id"),
            "source_time_range_sec": time_range,
            "observation": result.get("observation", result.get("visual_caption", "")),
            "source_type": result.get("source_type", "unknown"),
            "verification_status": status,
            "accepted_as_fact": accepted,
            "retrieval_distance_ranking_only": result.get("distance"),
            "frame_id": result.get("frame_id"),
            "source_path": result.get("source_path"),
            "model": result.get("model"),
            "backend": result.get("backend"),
        }

    @staticmethod
    def _overlaps(card, interval):
        if interval is None:
            return True
        source = card["source_time_range_sec"]
        return source is not None and source[0] <= interval[1] and interval[0] <= source[1]

    def ask(self, question, *, video_id: Optional[str] = None, limit=5, start_sec=None, end_sec=None):
        """Return cards plus a factual-answer status; never infer from ranking."""
        question, video_id, limit, requested_range = self._validate(
            question, video_id, limit, start_sec, end_sec
        )
        try:
            scope = {} if requested_range is None else {"time_range": requested_range}
            results = self.retriever.search(question, video_id=video_id, limit=limit, **scope)
        except FileNotFoundError:
            return {
                "status": "index_unavailable", "answer_status": "insufficient_evidence",
                "answer": "No local visual-retrieval index is available yet. Index timestamped observations first.",
                "question": question, "video_id": video_id,
                "requested_source_time_range_sec": requested_range, "evidence_cards": [],
            }
        cards = [self._card(result) for result in results]
        cards = [card for card in cards if self._overlaps(card, requested_range)]
        accepted = [card for card in cards if card["accepted_as_fact"]]
        if accepted:
            answer_status = "supported_by_accepted_evidence"
            answer = "Independently accepted evidence was retrieved; inspect the cited cards before relying on the claim."
        elif cards:
            answer_status = "insufficient_evidence"
            answer = "I found potentially relevant timestamped observations, but none is independently accepted evidence for a factual answer."
        else:
            answer_status = "insufficient_evidence"
            scope = " for the supplied source video" if video_id else ""
            answer = f"No indexed observations matched this question{scope} in the requested time range."
        return {
            "status": "ok" if cards else "no_matches",
            "answer_status": answer_status,
            "answer": answer,
            "question": question,
            "video_id": video_id,
            "requested_source_time_range_sec": requested_range,
            "evidence_cards": cards,
        }

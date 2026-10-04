import json

import pytest
from fastapi.testclient import TestClient

from slm_pipeline.api.main import create_app
from slm_pipeline.pipelines.evidence_query import EvidenceQueryService


class FakeRetriever:
    def __init__(self, results=None, error=None):
        self.results = list(results or [])
        self.error = error
        self.calls = []

    def search(self, query, *, video_id=None, limit=5, time_range=None):
        self.calls.append((query, video_id, limit))
        if self.error:
            raise self.error
        results = self.results
        if time_range is not None:
            results = [row for row in results if row['time_range'][0] <= time_range[1]
                       and time_range[0] <= row['time_range'][1]]
        return results[:limit]


def raw_card(timestamp=12.5, text="<b>model observation</b>"):
    return {
        "video_id": "P02_05", "time_range": [timestamp, timestamp],
        "frame_timestamp_sec": timestamp, "frame_id": "frame_001.jpg",
        "source_path": "frames/frame_001.jpg", "source_type": "model",
        "observation": text, "model": "local-test", "backend": "test",
        "verification_status": "unvalidated_model_hypothesis",
        "accepted_as_fact": False, "distance": 0.123,
    }


def test_raw_retrieval_is_useful_but_abstains_on_factual_support():
    service = EvidenceQueryService(FakeRetriever([raw_card()]))
    result = service.ask("What action happens?", video_id="P02_05", limit=1)
    assert result["status"] == "ok"
    assert result["answer_status"] == "insufficient_evidence"
    assert len(result["evidence_cards"]) == 1
    card = result["evidence_cards"][0]
    assert card["source_time_range_sec"] == [12.5, 12.5]
    assert card["retrieval_distance_ranking_only"] == 0.123
    assert "confidence" not in card


def test_accepted_cards_are_marked_supported_without_generating_a_claim():
    accepted = raw_card()
    accepted.update({
        "verification_status": "accepted_independent_evidence",
        "accepted_as_fact": True,
    })
    result = EvidenceQueryService(FakeRetriever([accepted])).ask("What is known?")
    assert result["answer_status"] == "supported_by_accepted_evidence"
    assert "inspect the cited cards" in result["answer"]


@pytest.mark.parametrize("question", ["", "   ", None])
def test_empty_question_is_rejected(question):
    with pytest.raises(ValueError, match="nonempty"):
        EvidenceQueryService(FakeRetriever()).ask(question)


def test_no_index_no_hits_missing_video_and_time_range_states():
    unavailable = EvidenceQueryService(FakeRetriever(error=FileNotFoundError()))
    assert unavailable.ask("find anything")["status"] == "index_unavailable"

    no_hits = EvidenceQueryService(FakeRetriever())
    missing_video = no_hits.ask("find anything", video_id="does-not-exist")
    assert missing_video["status"] == "no_matches"
    assert "supplied source video" in missing_video["answer"]

    out_of_range = EvidenceQueryService(FakeRetriever([raw_card(4.0)])).ask(
        "find anything", start_sec=5, end_sec=6
    )
    assert out_of_range["status"] == "no_matches"
    with pytest.raises(ValueError, match="earlier"):
        no_hits.ask("find anything", start_sec=6, end_sec=5)
    with pytest.raises(ValueError, match="together"):
        no_hits.ask("find anything", start_sec=1)


def test_repeated_questions_are_deterministic_and_bounded():
    retriever = FakeRetriever([raw_card(), raw_card(13.0)])
    service = EvidenceQueryService(retriever)
    first = service.ask("repeat", limit=1)
    second = service.ask("repeat", limit=1)
    assert first == second
    assert retriever.calls == [("repeat", None, 1), ("repeat", None, 1)]
    with pytest.raises(ValueError, match="1 to 10"):
        service.ask("repeat", limit=11)


def test_time_filter_precedes_top_k_with_real_vector_store(tmp_path):
    import numpy as np
    from slm_pipeline.pipelines.phase_7_embed import VectorStore
    from slm_pipeline.pipelines.vision_retrieval import CaptionOnlyRetriever

    class SyntheticEmbedder:
        dimension = 2

        def encode_single(self, question):
            return np.zeros(2, dtype=np.float32)

    store = VectorStore(dimension=2, memory_dir=tmp_path)
    store.replace_video('P02_05', np.asarray([[0, 0], [1, 0]], dtype=np.float32),
                        [raw_card(4.0), raw_card(12.5)])
    retriever = CaptionOnlyRetriever(embedder=SyntheticEmbedder(), vector_store=store)
    result = EvidenceQueryService(retriever).ask(
        'question', video_id='P02_05', limit=1, start_sec=10, end_sec=15,
    )
    assert result['status'] == 'ok'
    assert len(result['evidence_cards']) == 1
    assert result['evidence_cards'][0]['source_time_range_sec'] == [12.5, 12.5]
    assert result['answer_status'] == 'insufficient_evidence'


def test_api_and_ui_states_are_local_and_escape_by_dom_text_content():
    client = TestClient(create_app(EvidenceQueryService(FakeRetriever([raw_card()]))))
    response = client.post("/ask", json={"question": "where?", "limit": 1})
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer_status"] == "insufficient_evidence"
    assert payload["evidence_cards"][0]["observation"] == "<b>model observation</b>"
    assert client.post("/ask", json={"question": "   "}).status_code == 422
    assert client.post("/ask", json={"question": "x", "start_sec": 3}).status_code == 422
    html = client.get("/ask").text
    assert "textContent" in html
    assert "innerHTML" not in html
    assert "calibrated confidence" in html
    root = client.get("/").json()
    assert root["status"] == "local_only"
    assert json.dumps(root)

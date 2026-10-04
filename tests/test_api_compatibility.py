"""Model-free coverage of the legacy API alongside the local evidence UI."""

import subprocess
import sys
import runpy
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from slm_pipeline.api import main


class FakeMemoryAgent:
    def __init__(self):
        self.calls = []
        self.results = [{
            "video_id": "fixture-video", "block_id": "block-1",
            "observation": "Unreviewed model observation",
            "accepted_as_fact": False,
            "verification_status": "unvalidated_model_hypothesis",
        }]

    def search_memory(self, *, query, k):
        self.calls.append((query, k))
        return self.results[:k]

    def get_all_tasks(self):
        return [{"task": "Reviewed task 1"}, {"task": "Reviewed task 2"}]

    def get_all_decisions(self):
        return [{"decision": "Reviewed decision 1"}, {"decision": "Reviewed decision 2"}]

    def get_topics(self, *, k):
        return ["Reviewed topic 1", "Reviewed topic 2"][:k]


class FakeSummaryAgent:
    def __init__(self, result=None):
        self.calls = []
        self.result = result

    def summarize_meeting(self, video_id):
        self.calls.append(video_id)
        if self.result is not None:
            return self.result
        return {
            "video_id": video_id, "summary": "Synthetic summary fixture.",
            "topics": ["Reviewed topic"], "decisions": [], "tasks": [],
            "duration_blocks": 2, "unvalidated_semantic_blocks_excluded": 2,
        }


def test_api_import_and_startup_do_not_import_model_or_agent_backends():
    # A fresh process proves this independently of the rest of the test suite.
    script = """
import sys
from fastapi.testclient import TestClient
from slm_pipeline.api.main import app
with TestClient(app) as client:
    assert client.get('/').status_code == 200
    assert client.get('/ask').status_code == 200
    assert client.get('/openapi.json').status_code == 200
for name in (
    'slm_pipeline.agents.memory_agent', 'slm_pipeline.agents.summary_agent',
    'slm_pipeline.pipelines.vision_retrieval', 'sentence_transformers', 'faiss',
):
    assert name not in sys.modules, name
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_legacy_agents_are_lazy_independent_and_reused(monkeypatch, tmp_path):
    counts = {"memory": 0, "summary": 0}
    memory = FakeMemoryAgent()
    summary = FakeSummaryAgent()

    def memory_factory():
        counts["memory"] += 1
        return memory

    def summary_factory():
        counts["summary"] += 1
        return summary

    monkeypatch.setitem(main.config["paths"], "processed_dir", str(tmp_path))
    app = main.create_app(memory_agent_factory=memory_factory, summary_agent_factory=summary_factory)
    assert counts == {"memory": 0, "summary": 0}
    with TestClient(app) as client:
        for path in ("/", "/ask", "/videos", "/openapi.json"):
            assert client.get(path).status_code == 200
        assert counts == {"memory": 0, "summary": 0}
        for path in ("/tasks", "/decisions", "/topics"):
            assert client.get(path).status_code == 200
        assert client.post("/query", json={"question": "find evidence"}).status_code == 200
        assert counts == {"memory": 1, "summary": 0}
        assert client.get("/summary/fixture-video").status_code == 200
        assert client.get("/summary/fixture-video").status_code == 200
        assert counts == {"memory": 1, "summary": 1}


def test_concurrent_first_requests_share_one_lazy_resource():
    calls = []
    resource = object()
    gate = Barrier(4)

    def factory():
        calls.append("initialized")
        return resource

    resolve = main._lazy_resource(factory)

    def use_resource():
        gate.wait(timeout=5)
        return resolve()

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(use_resource) for _ in range(4)]
        assert all(future.result(timeout=5) is resource for future in futures)
    assert calls == ["initialized"]


def test_query_preserves_legacy_shape_default_limit_and_evidence_status():
    memory = FakeMemoryAgent()
    client = TestClient(main.create_app(memory_agent_factory=lambda: memory))
    response = client.post("/query", json={"question": "find evidence"})
    assert response.status_code == 200
    assert response.json() == {"results": memory.results, "total": 1}
    assert response.json()["results"][0]["accepted_as_fact"] is False
    assert memory.calls == [("find evidence", 5)]
    response = client.post("/query", json={"question": "find again", "limit": 1})
    assert response.status_code == 200
    assert memory.calls[-1] == ("find again", 1)


@pytest.mark.parametrize("resource", ["tasks", "decisions", "topics"])
def test_list_routes_keep_legacy_limit_and_total_semantics(resource):
    client = TestClient(main.create_app(memory_agent_factory=FakeMemoryAgent))
    response = client.get(f"/{resource}?limit=1")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload[resource]) == 1
    # Tasks/decisions count all eligible records; topics counts returned topics.
    assert payload["total"] == (1 if resource == "topics" else 2)


def test_summary_preserves_legacy_response_fields():
    summary = FakeSummaryAgent()
    client = TestClient(main.create_app(summary_agent_factory=lambda: summary))
    response = client.get("/summary/fixture-video")
    assert response.status_code == 200
    assert response.json() == {
        "video_id": "fixture-video", "summary": "Synthetic summary fixture.",
        "topics": ["Reviewed topic"], "decisions": [], "tasks": [],
    }
    assert summary.calls == ["fixture-video"]


@pytest.mark.parametrize("error", ["Video not found", "No semantic data"])
def test_summary_retains_known_not_found_errors(error):
    client = TestClient(main.create_app(
        summary_agent_factory=lambda: FakeSummaryAgent({"error": error}),
    ))
    response = client.get("/summary/missing-video")
    assert response.status_code == 404
    assert response.json() == {"detail": error}


def test_unknown_summary_error_does_not_expose_filesystem_details():
    client = TestClient(main.create_app(summary_agent_factory=lambda: FakeSummaryAgent({
        "error": "Cannot read /private/fixture/semantic_structure.json",
    })))
    response = client.get("/summary/fixture-video")
    assert response.status_code == 404
    assert response.json() == {"detail": "Summary data is unavailable."}


@pytest.mark.parametrize("path", ["/query", "/summary/fixture-video", "/tasks", "/decisions", "/topics"])
def test_backend_failures_are_sanitized(path):
    def broken_factory():
        raise RuntimeError("Cannot load /private/fixture/backend-file")

    client = TestClient(main.create_app(
        memory_agent_factory=broken_factory, summary_agent_factory=broken_factory,
    ))
    response = (
        client.post(path, json={"question": "find evidence"})
        if path == "/query" else client.get(path)
    )
    assert response.status_code == 500
    assert "/private/fixture" not in response.text
    assert "unavailable" in response.json()["detail"]


@pytest.mark.parametrize("video_id", ["%2E", "%2E%2E", "..%5Csecret"])
def test_summary_rejects_unsafe_video_ids_before_resolving_agent(video_id):
    def unexpected_factory():
        pytest.fail("Unsafe video ID must not initialize the summary agent")

    client = TestClient(main.create_app(summary_agent_factory=unexpected_factory))
    assert client.get(f"/summary/{video_id}").status_code == 422


def test_legacy_metadata_routes_use_existing_fact_acceptance_guards():
    from slm_pipeline.agents.memory_agent import MemoryAgent

    metadata = [
        {"accepted_as_fact": False, "topics": ["Unreviewed topic"],
         "decisions": ["Unreviewed decision"], "tasks": ["Unreviewed task"]},
        {"accepted_as_fact": True, "topics": ["Reviewed topic"],
         "decisions": ["Reviewed decision"], "tasks": ["Reviewed task"]},
    ]
    # Neither fake dependency performs inference or initializes a vector index.
    memory = MemoryAgent(
        embedder=SimpleNamespace(dimension=1),
        vector_store=SimpleNamespace(dimension=1, metadata=metadata, index=SimpleNamespace(ntotal=2)),
    )
    client = TestClient(main.create_app(memory_agent_factory=lambda: memory))
    assert client.get("/topics").json() == {"topics": ["Reviewed topic"], "total": 1}
    for path, key in (("tasks", "task"), ("decisions", "decision")):
        payload = client.get(f"/{path}").json()
        assert payload["total"] == 1
        assert payload[path][0][key] == f"Reviewed {key}"


def test_no_permissive_cors_was_reintroduced():
    client = TestClient(main.create_app(memory_agent_factory=FakeMemoryAgent))
    response = client.get("/tasks", headers={"Origin": "https://untrusted.example"})
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers
    preflight = client.options("/query", headers={
        "Origin": "https://untrusted.example", "Access-Control-Request-Method": "POST",
    })
    assert "access-control-allow-origin" not in preflight.headers


def test_executable_entry_point_binds_only_to_loopback(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(
        run=lambda app, **options: calls.append(options),
    ))
    runpy.run_path(main.__file__, run_name="__main__")
    assert calls == [{"host": "127.0.0.1", "port": 8000}]

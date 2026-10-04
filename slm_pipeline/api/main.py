"""Local-only HTTP interface for timestamped retrieval evidence.

The module intentionally does not load an embedding model or legacy agents at
import time. The server binds to loopback in its executable entry point.
"""

from pathlib import Path
from threading import Lock
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from slm_pipeline.config import config
from slm_pipeline.pipelines.evidence_query import EvidenceQueryService


class AskRequest(BaseModel):
    question: str
    video_id: Optional[str] = None
    limit: int = Field(default=5, ge=1, le=10)
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None


class QueryRequest(BaseModel):
    question: str
    limit: int = 5


class QueryResponse(BaseModel):
    results: List[dict]
    total: int


class SummaryResponse(BaseModel):
    video_id: str
    summary: str
    topics: List[str]
    decisions: List[str]
    tasks: List[str]


def _default_query_service():
    # Importing here prevents a request-free service start from loading MiniLM.
    from slm_pipeline.pipelines.vision_retrieval import CaptionOnlyRetriever
    return EvidenceQueryService(CaptionOnlyRetriever())


def _default_memory_agent():
    from slm_pipeline.agents.memory_agent import MemoryAgent
    return MemoryAgent()


def _default_summary_agent():
    from slm_pipeline.agents.summary_agent import SummaryAgent
    return SummaryAgent()


def _lazy_resource(factory, initial=None):
    """Resolve once on first use, including concurrent first requests."""
    resource = initial
    lock = Lock()

    def resolve():
        nonlocal resource
        with lock:
            if resource is None:
                resource = factory()
            return resource

    return resolve


def create_app(query_service=None, *, memory_agent_factory=None, summary_agent_factory=None):
    """Create a same-origin local app; injection keeps API tests model-free."""
    app = FastAPI(
        title="SLM local evidence query",
        description="Local retrieval cards with explicit evidence status; no generative QA.",
        version="1.1.0",
    )

    service = _lazy_resource(_default_query_service, query_service)
    memory_agent = _lazy_resource(memory_agent_factory or _default_memory_agent)
    summary_agent = _lazy_resource(summary_agent_factory or _default_summary_agent)

    @app.get("/")
    def root():
        return {
            "service": "SLM local evidence query",
            "status": "local_only",
            "query_ui": "/ask",
            "answer_policy": "Factual claims require independently accepted evidence.",
        }

    @app.get("/ask", response_class=HTMLResponse)
    def ask_ui():
        return (Path(__file__).with_name("query_ui.html")).read_text(encoding="utf-8")

    @app.post("/ask")
    def ask(request: AskRequest):
        try:
            return service().ask(
                request.question, video_id=request.video_id, limit=request.limit,
                start_sec=request.start_sec, end_sec=request.end_sec,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            # Do not reveal local paths or backend details through the local UI.
            raise HTTPException(status_code=503, detail="Local retrieval is unavailable.") from exc

    @app.post("/query", response_model=QueryResponse)
    def query_memory(request: QueryRequest):
        """Legacy retrieval response; evidence flags are retained unchanged."""
        try:
            results = memory_agent().search_memory(query=request.question, k=request.limit)
            return QueryResponse(results=results, total=len(results))
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Local memory search is unavailable.") from exc

    @app.get("/summary/{video_id}", response_model=SummaryResponse)
    def get_summary(video_id: str):
        # The legacy agent reads a directory under processed_dir for this ID.
        if video_id in {".", ".."} or "/" in video_id or "\\" in video_id:
            raise HTTPException(status_code=422, detail="Invalid video ID.")
        try:
            result = summary_agent().summarize_meeting(video_id)
            if "error" not in result:
                return SummaryResponse(**result)
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Local summary is unavailable.") from exc
        error = result["error"]
        detail = (
            error if error in ("Video not found", "No semantic data")
            else "Summary data is unavailable."
        )
        raise HTTPException(status_code=404, detail=detail)

    @app.get("/tasks")
    def get_tasks(limit: int = 20):
        try:
            tasks = memory_agent().get_all_tasks()
            return {"tasks": tasks[:limit], "total": len(tasks)}
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Local tasks are unavailable.") from exc

    @app.get("/decisions")
    def get_decisions(limit: int = 20):
        try:
            decisions = memory_agent().get_all_decisions()
            return {"decisions": decisions[:limit], "total": len(decisions)}
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Local decisions are unavailable.") from exc

    @app.get("/topics")
    def get_topics(limit: int = 20):
        try:
            topics = memory_agent().get_topics(k=limit)
            return {"topics": topics, "total": len(topics)}
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Local topics are unavailable.") from exc

    @app.get("/videos")
    def list_videos():
        processed_dir = Path(config["paths"]["processed_dir"])
        if not processed_dir.exists():
            return {"videos": [], "total": 0}
        videos = sorted(directory.name for directory in processed_dir.iterdir() if directory.is_dir())
        return {"videos": videos, "total": len(videos)}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

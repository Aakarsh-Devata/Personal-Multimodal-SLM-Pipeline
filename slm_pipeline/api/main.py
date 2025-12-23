"""FastAPI backend for SLM Pipeline."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import config
from agents.memory_agent import MemoryAgent
from agents.summary_agent import SummaryAgent

app = FastAPI(
    title="SLM Pipeline API",
    description="Personal Multimodal SLM Memory System API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agents
memory_agent = MemoryAgent()
summary_agent = SummaryAgent()


# Request/Response models
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


# Routes
@app.get("/")
def root():
    """Root endpoint."""
    return {
        "service": "SLM Pipeline API",
        "version": "1.0.0",
        "status": "running",
        "total_memories": memory_agent.vector_store.index.ntotal
    }


@app.post("/query", response_model=QueryResponse)
def query_memory(request: QueryRequest):
    """Query the memory system with natural language."""
    try:
        results = memory_agent.search_memory(
            query=request.question,
            k=request.limit
        )
        
        return QueryResponse(
            results=results,
            total=len(results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/summary/{video_id}", response_model=SummaryResponse)
def get_summary(video_id: str):
    """Get summary of a specific video."""
    try:
        result = summary_agent.summarize_meeting(video_id)
        
        if 'error' in result:
            raise HTTPException(status_code=404, detail=result['error'])
        
        return SummaryResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks")
def get_tasks(limit: int = 20):
    """Get all tasks."""
    try:
        tasks = memory_agent.get_all_tasks()
        return {
            "tasks": tasks[:limit],
            "total": len(tasks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/decisions")
def get_decisions(limit: int = 20):
    """Get all decisions."""
    try:
        decisions = memory_agent.get_all_decisions()
        return {
            "decisions": decisions[:limit],
            "total": len(decisions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/topics")
def get_topics(limit: int = 20):
    """Get common topics."""
    try:
        topics = memory_agent.get_topics(k=limit)
        return {
            "topics": topics,
            "total": len(topics)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/videos")
def list_videos():
    """List all processed videos."""
    try:
        processed_dir = Path(config['paths']['processed_dir'])
        videos = [d.name for d in processed_dir.iterdir() if d.is_dir()]
        
        return {
            "videos": videos,
            "total": len(videos)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

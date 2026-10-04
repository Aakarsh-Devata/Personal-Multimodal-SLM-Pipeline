"""Memory Agent - Search and recall from vector memory."""

import sys
from pathlib import Path
from typing import List, Dict, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from slm_pipeline.agents.base_agent import BaseAgent
from slm_pipeline.pipelines.phase_7_embed import EmbeddingGenerator, VectorStore


class MemoryAgent(BaseAgent):
    """Agent for searching and retrieving memories."""
    
    def __init__(self, embedder=None, vector_store=None):
        super().__init__("MemoryAgent")
        
        # Load embedding model and vector store
        self.embedder = embedder if embedder is not None else EmbeddingGenerator()
        self.vector_store = vector_store if vector_store is not None else VectorStore(
            dimension=self.embedder.dimension
        )
        if self.embedder.dimension != self.vector_store.dimension:
            raise ValueError("Embedder and vector store dimensions must match")
        
        self.log(f"Loaded vector store with {self.vector_store.index.ntotal} memories")
    
    def search_memory(
        self, 
        query: str, 
        k: int = 5,
        time_range: Optional[tuple] = None,
        video_id: Optional[str] = None
    ) -> List[Dict]:
        """Search memory with a natural language query.
        
        Args:
            query: Natural language search query
            k: Number of results to return
            time_range: Optional inclusive overlap (start_seconds, end_seconds)
            video_id: Optional video_id to filter results
        
        Returns:
            List of memory results with metadata
        """
        self.log(f"Searching for: '{query}' (video_filter={video_id})")
        
        # Generate query embedding
        query_embedding = self.embedder.encode_single(query)
        
        # Filter before ranking so every requested eligible neighbor is considered.
        results = self.vector_store.search(
            query_embedding, k=k, video_id=video_id, time_range=time_range
        )
        self.log(f"Found {len(results)} results")
        return results

    def get_event(self, video_id: str, block_id: str) -> Optional[Dict]:
        """Retrieve a specific event by video and block ID."""
        self.log(f"Retrieving event: {video_id}/{block_id}")
        
        # Search through metadata
        for meta in self.vector_store.metadata:
            if meta.get('video_id') == video_id and meta.get('block_id') == block_id:
                return meta
        
        self.log("Event not found", level="warning")
        return None
    
    def list_events(
        self, 
        video_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """List events, optionally filtered by video ID."""
        self.log(f"Listing events (video_id={video_id}, limit={limit})")
        
        events = []
        for meta in self.vector_store.metadata:
            if video_id is None or meta.get('video_id') == video_id:
                events.append(meta)
                if len(events) >= limit:
                    break
        
        self.log(f"Found {len(events)} events")
        return events
    
    def get_topics(self, k: int = 10) -> List[str]:
        """Get most common topics across all memories."""
        self.log("Extracting common topics")
        
        topic_counts = {}
        for meta in self.vector_store.metadata:
            if meta.get('accepted_as_fact') is not True:
                continue
            for topic in meta.get('topics', []):
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        # Sort by frequency
        sorted_topics = sorted(
            topic_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return [topic for topic, _ in sorted_topics[:k]]
    
    def get_all_decisions(self) -> List[Dict]:
        """Get all decisions from memories."""
        self.log("Extracting all decisions")
        
        decisions = []
        for meta in self.vector_store.metadata:
            if meta.get('accepted_as_fact') is not True:
                continue
            for decision in meta.get('decisions', []):
                decisions.append({
                    'decision': decision,
                    'video_id': meta.get('video_id'),
                    'block_id': meta.get('block_id'),
                    'time_range': meta.get('time_range'),
                    'context': meta.get('speech_text', '')
                })
        
        self.log(f"Found {len(decisions)} decisions")
        return decisions
    
    def get_all_tasks(self) -> List[Dict]:
        """Get all tasks from memories."""
        self.log("Extracting all tasks")
        
        tasks = []
        for meta in self.vector_store.metadata:
            if meta.get('accepted_as_fact') is not True:
                continue
            for task in meta.get('tasks', []):
                tasks.append({
                    'task': task,
                    'video_id': meta.get('video_id'),
                    'block_id': meta.get('block_id'),
                    'time_range': meta.get('time_range'),
                    'context': meta.get('speech_text', '')
                })
        
        self.log(f"Found {len(tasks)} tasks")
        return tasks

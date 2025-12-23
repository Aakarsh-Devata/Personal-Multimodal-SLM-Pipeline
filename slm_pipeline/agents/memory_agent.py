"""Memory Agent - Search and recall from vector memory."""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import config
from agents.base_agent import BaseAgent

# Import vector store components
sys.path.insert(0, str(Path(__file__).parent.parent / "pipelines"))
from phase_7_embed import EmbeddingGenerator, VectorStore


class MemoryAgent(BaseAgent):
    """Agent for searching and retrieving memories."""
    
    def __init__(self):
        super().__init__("MemoryAgent")
        
        # Load embedding model and vector store
        self.embedder = EmbeddingGenerator()
        self.vector_store = VectorStore(dimension=self.embedder.dimension)
        
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
            time_range: Optional (start_time, end_time) tuple to filter results
            video_id: Optional video_id to filter results
        
        Returns:
            List of memory results with metadata
        """
        self.log(f"Searching for: '{query}' (video_filter={video_id})")
        
        # Generate query embedding
        query_embedding = self.embedder.encode_single(query)
        
        # Search vector store - fetch more if filtering
        search_k = k * 5 if (time_range or video_id) else k
        results = self.vector_store.search(query_embedding, k=search_k)
        
        # Apply filters
        filtered_results = []
        for result in results:
            # Filter by video_id
            if video_id and result.get('video_id') != video_id:
                continue
                
            # Filter by time range
            if time_range:
                # Placeholder for time check logic
                pass
                
            filtered_results.append(result)
            if len(filtered_results) >= k:
                break
                
        self.log(f"Found {len(filtered_results)} results after filtering")
        return filtered_results
        
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

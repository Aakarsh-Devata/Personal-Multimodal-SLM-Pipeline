"""Summary Agent - Summarize meetings, days, and time periods."""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import config
from agents.base_agent import BaseAgent

# Import Ollama client
sys.path.insert(0, str(Path(__file__).parent.parent / "pipelines"))
from phase_6_semantic import OllamaClient


class SummaryAgent(BaseAgent):
    """Agent for creating summaries of meetings and time periods."""
    
    def __init__(self):
        super().__init__("SummaryAgent")
        self.llm = OllamaClient(model_name=config['models']['slm']['name'])
        self.processed_dir = Path(config['paths']['processed_dir'])
    
    def summarize_meeting(self, video_id: str) -> Dict:
        """Summarize a specific meeting/video."""
        self.log(f"Summarizing meeting: {video_id}")
        
        video_dir = self.processed_dir / video_id
        if not video_dir.exists():
            self.log(f"Video not found: {video_id}", level="error")
            return {"error": "Video not found"}
        
        # Load semantic structure
        semantic_file = video_dir / "semantic_structure.json"
        if not semantic_file.exists():
            self.log("No semantic data found", level="error")
            return {"error": "No semantic data"}
        
        with open(semantic_file, 'r', encoding='utf-8') as f:
            semantic_data = json.load(f)
        
        # Load context blocks
        context_file = video_dir / "context_blocks.json"
        context_data = {}
        if context_file.exists():
            with open(context_file, 'r', encoding='utf-8') as f:
                context_data = json.load(f)
        
        # Compile all information
        all_topics = set()
        all_decisions = []
        all_tasks = []
        all_text = []
        
        for block in semantic_data.get('semantic_blocks', []):
            all_topics.update(block.get('topics', []))
            all_decisions.extend(block.get('decisions', []))
            all_tasks.extend(block.get('tasks', []))
        
        # Get speech text from context blocks
        for block in context_data.get('context_blocks', []):
            text = block.get('speech', {}).get('text', '')
            if text:
                all_text.append(text)
        
        # Create summary prompt
        full_transcript = ' '.join(all_text)
        
        prompt = f"""Summarize the following meeting transcript:

Transcript: {full_transcript[:2000]}

Topics discussed: {', '.join(list(all_topics)[:10])}
Decisions made: {', '.join(all_decisions[:5]) if all_decisions else 'None'}
Action items: {', '.join(all_tasks[:5]) if all_tasks else 'None'}

Provide a concise 3-4 sentence summary of this meeting, highlighting key points, decisions, and outcomes."""
        
        summary_text = self.llm.generate(prompt)
        
        return {
            "video_id": video_id,
            "summary": summary_text,
            "topics": list(all_topics),
            "decisions": all_decisions,
            "tasks": all_tasks,
            "duration_blocks": len(semantic_data.get('semantic_blocks', []))
        }
    
    def summarize_all_videos(self) -> Dict:
        """Summarize all processed videos."""
        self.log("Summarizing all videos")
        
        summaries = []
        video_folders = [d for d in self.processed_dir.iterdir() if d.is_dir()]
        
        for video_dir in video_folders:
            summary = self.summarize_meeting(video_dir.name)
            if 'error' not in summary:
                summaries.append(summary)
        
        return {
            "total_videos": len(summaries),
            "summaries": summaries
        }

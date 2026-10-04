#!/usr/bin/env python3
"""
Phase 6: Semantic Structuring
Extracts structured semantic meaning from context blocks using Ollama (local LLM).
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import logging
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from slm_pipeline.config import config
from slm_pipeline.runtime import selected_video_dirs, phase_cli

logging.basicConfig(
    level=config['logging']['level'],
    format=config['logging']['format']
)
logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama local LLM."""
    
    def __init__(self, model_name: str = "mistral"):
        self.model_name = model_name
        self.base_url = "http://localhost:11434"
        self.temperature = config['models']['semantic_extraction']['temperature']
        self.max_tokens = config['models']['semantic_extraction']['max_tokens']
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test if Ollama is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Connected to Ollama")
            else:
                response.raise_for_status()
        except Exception as e:
            logger.error(f"❌ Cannot connect to Ollama: {e}")
            raise ConnectionError("Ollama is unavailable; start the configured local server") from e
    
    def generate(self, prompt: str) -> str:
        """Generate text using Ollama."""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                return response.json().get('response', '')
            else:
                logger.error(f"Ollama error: {response.status_code}")
                response.raise_for_status()
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise RuntimeError("Ollama generation failed") from e


class SemanticExtractor:
    """Extract structured semantic information from context blocks."""
    
    def __init__(self):
        self.llm = OllamaClient(
            model_name=config['models']['semantic_extraction']['model']
        )
    
    def create_extraction_prompt(self, context_block: Dict) -> str:
        """Create a prompt for semantic extraction."""
        speech_text = context_block.get('speech', {}).get('text', '')
        visual_caption = context_block.get('visual', {}).get('caption', '')
        
        prompt = f"""Analyze the following multimodal context and extract structured information.

Visual text in this context is an unvalidated model hypothesis, not a fact. Do not
promote a visual action, object, or temporal change to a decision/task/fact unless
the speech explicitly establishes it. Preserve uncertainty rather than completing
missing visual events.

Speech: "{speech_text}"
Visual Context: "{visual_caption}"

Extract and return ONLY a JSON object with the following fields:
- topics: List of main topics discussed (max 3)
- decisions: List of decisions made (if any)
- tasks: List of action items or tasks mentioned (if any)
- people: List of people mentioned (if any)
- entities: List of important entities (objects, concepts, locations)
- intent: The primary intent (one of: informational, planning, discussion, action, personal)
- summary: A one-sentence summary
- confidence: Your confidence in this analysis (0.0 to 1.0)

Return ONLY valid JSON, no additional text.

JSON:"""
        
        return prompt
    
    def parse_llm_response(self, response: str) -> Dict:
        """Parse LLM response and extract JSON."""
        try:
            # Try to find JSON in response
            response = response.strip()
            
            # Remove markdown code blocks if present
            if response.startswith('```'):
                lines = response.split('\n')
                response = '\n'.join(lines[1:-1]) if len(lines) > 2 else response
                if response.startswith('json'):
                    response = response[4:].strip()
            
            # Parse JSON
            data = json.loads(response)
            
            # Validate required fields
            required_fields = ['topics', 'decisions', 'tasks', 'people', 'entities', 'intent', 'summary', 'confidence']
            if not isinstance(data, dict) or any(field not in data for field in required_fields):
                raise ValueError("Structured model output is missing required fields")
            # Generated content may not supply authoritative evidence IDs/times.
            data = {field: data[field] for field in required_fields}
            for field in ('topics', 'decisions', 'tasks', 'people', 'entities'):
                if not isinstance(data[field], list) or not all(isinstance(value, str) for value in data[field]):
                    raise ValueError(f"Invalid semantic field: {field}")
            if not isinstance(data['summary'], str) or not data['summary'].strip() or isinstance(data['confidence'], bool) or not isinstance(data['confidence'], (int, float)) or not 0 <= data['confidence'] <= 1:
                raise ValueError("Invalid semantic summary/confidence")
            if data['intent'] not in {'informational', 'planning', 'discussion', 'action', 'personal'}:
                raise ValueError("Invalid semantic intent")
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.debug(f"Response was: {response[:200]}")
            raise ValueError("Invalid structured model output") from e
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            raise ValueError("Invalid structured model output") from e
    
    def _get_fallback_structure(self) -> Dict:
        """Return fallback structure if parsing fails."""
        return {
            "topics": [],
            "decisions": [],
            "tasks": [],
            "people": [],
            "entities": [],
            "intent": "informational",
            "summary": "",
            "confidence": 0.0
        }
    
    def extract_semantic_info(self, context_block: Dict) -> Dict:
        """Extract semantic information from a context block."""
        prompt = self.create_extraction_prompt(context_block)
        response = self.llm.generate(prompt)
        
        if not response:
            raise ValueError("Empty structured model output")
        
        return self.parse_llm_response(response)
    
    def process_video_folder(self, video_dir: Path):
        """Process all context blocks for a video."""
        logger.info(f"Processing video: {video_dir.name}")
        
        # Load context blocks
        context_file = video_dir / "context_blocks.json"
        if not context_file.exists():
            raise FileNotFoundError(f"No context blocks found for {video_dir.name}")
        
        # Output file
        output_file = video_dir / "semantic_structure.json"
        if output_file.exists() and config['pipeline']['skip_existing']:
            logger.info(f"⏩ Skipping {video_dir.name}, semantic_structure.json already exists")
            return
        
        # Load context blocks
        with open(context_file, 'r', encoding='utf-8') as f:
            context_data = json.load(f)
        
        context_blocks = context_data.get('context_blocks', [])
        logger.info(f"Extracting semantics from {len(context_blocks)} context blocks")
        
        # Process each block
        semantic_data = []
        for block in context_blocks:
            logger.debug(f"Processing {block.get('block_id')}")
            
            semantic_info = self.extract_semantic_info(block)
            
            result = {
                **semantic_info,
                "context_block_id": block['block_id'],
                "time_range": block['time_range'],
                "verification_status": "unvalidated_derived_model_hypothesis",
                "accepted_as_fact": False,
                "visual_evidence_status": block.get('visual', {}).get('verification_status', 'no_visual_observation'),
            }
            
            semantic_data.append(result)
        
        # Save results
        output_data = {
            "video_id": video_dir.name,
            "total_blocks": len(semantic_data),
            "semantic_blocks": semantic_data
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Extracted semantics for {len(semantic_data)} blocks")
        logger.info(f"Saved to {output_file}")


def main(video_id=None):
    folders = selected_video_dirs(config, video_id)
    extractor = SemanticExtractor()
    for video_dir in folders:
        extractor.process_video_folder(video_dir)


if __name__ == "__main__":
    phase_cli(main)

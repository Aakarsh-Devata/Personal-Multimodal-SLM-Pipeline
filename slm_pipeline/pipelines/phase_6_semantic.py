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

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import config

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
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                logger.info("✅ Connected to Ollama")
            else:
                logger.warning("⚠️ Ollama connection issue")
        except Exception as e:
            logger.error(f"❌ Cannot connect to Ollama: {e}")
            logger.error("Please ensure Ollama is running: ollama serve")
    
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
                return ""
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return ""


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
            for field in required_fields:
                if field not in data:
                    data[field] = [] if field != 'intent' and field != 'summary' and field != 'confidence' else (
                        'informational' if field == 'intent' else (
                            '' if field == 'summary' else 0.5
                        )
                    )
            
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.debug(f"Response was: {response[:200]}")
            return self._get_fallback_structure()
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return self._get_fallback_structure()
    
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
            return self._get_fallback_structure()
        
        return self.parse_llm_response(response)
    
    def process_video_folder(self, video_dir: Path):
        """Process all context blocks for a video."""
        logger.info(f"Processing video: {video_dir.name}")
        
        # Load context blocks
        context_file = video_dir / "context_blocks.json"
        if not context_file.exists():
            logger.warning(f"No context blocks found for {video_dir.name}")
            return
        
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
                "context_block_id": block.get('block_id'),
                "time_range": block.get('time_range'),
                **semantic_info
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


def main():
    """Main entry point."""
    processed_dir = Path(config['paths']['processed_dir'])
    
    if not processed_dir.exists():
        logger.error(f"Processed directory not found: {processed_dir}")
        return
    
    # Initialize extractor
    extractor = SemanticExtractor()
    
    # Process all video folders
    video_folders = [d for d in processed_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(video_folders)} video folders to process")
    
    for video_dir in video_folders:
        try:
            extractor.process_video_folder(video_dir)
        except Exception as e:
            logger.error(f"Failed to process {video_dir.name}: {e}")
            continue
    
    logger.info("🎉 Phase 6 (Semantic Structuring) completed!")


if __name__ == "__main__":
    main()

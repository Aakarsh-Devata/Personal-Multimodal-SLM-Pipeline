#!/usr/bin/env python3
"""
Phase 5: Multimodal Alignment
Aligns speech transcripts with vision captions using time-window fusion.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import config

logging.basicConfig(
    level=config['logging']['level'],
    format=config['logging']['format']
)
logger = logging.getLogger(__name__)


class MultimodalAligner:
    """Align speech and vision data using time windows."""
    
    def __init__(self):
        self.time_window = config['pipeline']['time_window_sec']
    
    def find_frames_in_window(
        self, 
        start_time: float, 
        end_time: float, 
        vision_data: List[Dict]
    ) -> List[Dict]:
        """Find all frames within a time window."""
        matching_frames = []
        
        # Expand window by tolerance
        window_start = start_time - self.time_window
        window_end = end_time + self.time_window
        
        for frame in vision_data:
            frame_time = frame.get('timestamp_sec', 0)
            if window_start <= frame_time <= window_end:
                matching_frames.append(frame)
        
        return matching_frames
    
    def merge_captions(self, frames: List[Dict]) -> str:
        """Merge multiple frame captions into a single description."""
        if not frames:
            return ""
        
        captions = [f.get('caption', '') for f in frames if f.get('caption')]
        
        # If all captions are similar, use one
        if len(captions) == 1:
            return captions[0]
        
        # Otherwise, combine with context
        unique_captions = []
        for cap in captions:
            if cap and cap not in unique_captions:
                unique_captions.append(cap)
        
        if len(unique_captions) == 1:
            return unique_captions[0]
        
        # Return combined view
        return " | ".join(unique_captions[:3])  # Max 3 captions
    
    def create_context_blocks(
        self, 
        transcript_data: Dict, 
        vision_data: List[Dict]
    ) -> List[Dict]:
        """Create context blocks by aligning speech and vision."""
        context_blocks = []
        
        segments = transcript_data.get('segments', [])
        
        for idx, segment in enumerate(segments):
            start_time = segment.get('start', 0)
            end_time = segment.get('end', start_time)
            text = segment.get('text', '').strip()
            
            # Find matching frames
            matching_frames = self.find_frames_in_window(
                start_time, end_time, vision_data
            )
            
            # Merge visual information
            visual_caption = self.merge_captions(matching_frames)
            
            # Create context block
            context_block = {
                "block_id": f"ctx_{idx:04d}",
                "time_range": [start_time, end_time],
                "speech": {
                    "text": text,
                    "speaker": "unknown",
                    "confidence": 1.0 - segment.get('no_speech_prob', 0)
                },
                "visual": {
                    "caption": visual_caption,
                    "frame_count": len(matching_frames),
                    "frame_ids": [f.get('frame_id') for f in matching_frames]
                },
                "raw_transcript_segment_id": segment.get('id'),
                "duration_sec": end_time - start_time
            }
            
            context_blocks.append(context_block)
        
        return context_blocks
    
    def process_video_folder(self, video_dir: Path):
        """Process a single video folder."""
        logger.info(f"Processing video: {video_dir.name}")
        
        # Load transcript
        transcript_file = video_dir / "transcript.json"
        if not transcript_file.exists():
            logger.warning(f"No transcript found for {video_dir.name}")
            return
        
        # Load vision captions
        vision_file = video_dir / "vision_captions.json"
        if not vision_file.exists():
            logger.warning(f"No vision captions found for {video_dir.name}")
            return
        
        # Output file
        output_file = video_dir / "context_blocks.json"
        if output_file.exists() and config['pipeline']['skip_existing']:
            logger.info(f"⏩ Skipping {video_dir.name}, context_blocks.json already exists")
            return
        
        # Load data
        with open(transcript_file, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
        
        with open(vision_file, 'r', encoding='utf-8') as f:
            vision_data = json.load(f)
        
        vision_captions = vision_data.get('captions', [])
        
        # Create context blocks
        context_blocks = self.create_context_blocks(
            transcript_data, 
            vision_captions
        )
        
        # Save results
        output_data = {
            "video_id": video_dir.name,
            "total_blocks": len(context_blocks),
            "time_window_sec": self.time_window,
            "context_blocks": context_blocks
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Created {len(context_blocks)} context blocks")
        logger.info(f"Saved to {output_file}")


def main():
    """Main entry point."""
    processed_dir = Path(config['paths']['processed_dir'])
    
    if not processed_dir.exists():
        logger.error(f"Processed directory not found: {processed_dir}")
        return
    
    # Initialize aligner
    aligner = MultimodalAligner()
    
    # Process all video folders
    video_folders = [d for d in processed_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(video_folders)} video folders to process")
    
    for video_dir in video_folders:
        try:
            aligner.process_video_folder(video_dir)
        except Exception as e:
            logger.error(f"Failed to process {video_dir.name}: {e}")
            continue
    
    logger.info("🎉 Phase 5 (Multimodal Alignment) completed!")


if __name__ == "__main__":
    main()

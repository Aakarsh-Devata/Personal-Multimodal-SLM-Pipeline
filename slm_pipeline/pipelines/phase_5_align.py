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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from slm_pipeline.config import config
from slm_pipeline.runtime import selected_video_dirs, phase_cli

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
            if frame.get('success') is False:
                continue
            frame_time = frame['timestamp_sec']
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
        
        segments = list(transcript_data.get('segments', []))
        # Preserve visual evidence for silent inputs and frames outside speech.
        for frame in vision_data:
            if frame.get('success') is False:
                continue
            timestamp = float(frame['timestamp_sec'])
            if not any(seg['start'] - self.time_window <= timestamp <= seg['end'] + self.time_window
                       for seg in segments):
                segments.append({'start': timestamp, 'end': timestamp, 'text': '',
                                 'source_type': 'visual_only'})
        segments.sort(key=lambda seg: (seg['start'], seg['end']))
        
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
                    "frame_ids": [f.get('frame_id') for f in matching_frames],
                    "frame_timestamps_sec": [f['timestamp_sec'] for f in matching_frames],
                    "frames": [{key: f[key] for key in ('frame_id', 'timestamp_sec', 'source_timestamp_sec', 'source_path', 'source_type', 'model', 'backend', 'device') if key in f}
                               for f in matching_frames],
                    "model_observations": [{
                        "source_type": "model",
                        "raw_model_text": f.get("caption", ""),
                        "source_pts_sec": [f.get("source_timestamp_sec", f["timestamp_sec"])],
                        "frame_ids": [f.get("frame_id")],
                        "frame_hashes": [f["frame_hash"]] if f.get("frame_hash") else [],
                        # Retain known run/model identifiers when upstream starts
                        # recording them; provenance is descriptive only and never
                        # makes a model claim accepted evidence.
                        "model_provenance": {key: f.get(key) for key in (
                            "model", "backend", "device", "model_revision",
                            "model_commit", "model_id", "run_id", "prompt_sha256",
                            "input_trace_id",
                        ) if f.get(key) is not None},
                        "verification_status": "unvalidated_model_hypothesis",
                        "accepted_as_fact": False,
                    } for f in matching_frames if f.get("caption") and f.get("frame_id") and f.get("model") and f.get("backend")],
                    "verification_status": "unvalidated_model_hypothesis" if matching_frames else "no_visual_observation",
                    "accepted_as_fact": False,
                },
                "source_type": segment.get('source_type', transcript_data.get('source_type', 'model_transcript')),
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
            raise FileNotFoundError(f"No transcript found for {video_dir.name}")
        
        # Load vision captions
        vision_file = video_dir / "vision_captions.json"
        if not vision_file.exists():
            raise FileNotFoundError(f"No vision captions found for {video_dir.name}")
        
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


def main(video_id=None):
    aligner = MultimodalAligner()
    for video_dir in selected_video_dirs(config, video_id):
        aligner.process_video_folder(video_dir)


if __name__ == "__main__":
    phase_cli(main)

#!/usr/bin/env python3
"""
Phase 4: Vision Understanding
Generates captions for extracted frames using BLIP-2 (local, open-source).
"""

import json
import torch
from pathlib import Path
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from tqdm import tqdm
import logging
import sys

# Add parent to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import config

logging.basicConfig(
    level=config['logging']['level'],
    format=config['logging']['format']
)
logger = logging.getLogger(__name__)


class VisionProcessor:
    """Process frames and generate captions using BLIP-2."""
    
    def __init__(self):
        logger.info(f"Loading BLIP-2 model: {config['models']['vision']['name']}")
        self.device = config['models']['vision']['device']
        self.batch_size = config['models']['vision']['batch_size']
        
        # Load BLIP-2 model and processor
        self.processor = Blip2Processor.from_pretrained(
            config['models']['vision']['name']
        )
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            config['models']['vision']['name'],
            torch_dtype=torch.float32
        ).to(self.device)
        
        logger.info("✅ BLIP-2 model loaded successfully")
    
    def generate_caption(self, image_path: Path) -> dict:
        """Generate caption for a single image."""
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Process image
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            # Generate caption
            generated_ids = self.model.generate(**inputs, max_length=50)
            caption = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0].strip()
            
            return {
                "caption": caption,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error processing {image_path.name}: {e}")
            return {
                "caption": "",
                "success": False,
                "error": str(e)
            }
    
    def process_video_folder(self, video_dir: Path):
        """Process all frames for a video and generate captions."""
        logger.info(f"Processing video: {video_dir.name}")
        
        # Look for frames_ready/ first, then frames_fixed/
        frames_dir = video_dir / "frames_ready"
        if not frames_dir.exists() or not list(frames_dir.glob("*.jpg")):
            frames_dir = video_dir / "frames_fixed"
        
        if not frames_dir.exists():
            logger.warning(f"No frames directory found in {video_dir}")
            return
        
        # Get all frame files
        frame_files = sorted(frames_dir.glob("*.jpg"))
        if not frame_files:
            logger.warning(f"No frames found in {frames_dir}")
            return
        
        logger.info(f"Found {len(frame_files)} frames to process")
        
        # Output file
        output_file = video_dir / "vision_captions.json"
        if output_file.exists() and config['pipeline']['skip_existing']:
            logger.info(f"⏩ Skipping {video_dir.name}, vision_captions.json already exists")
            return
        
        # Process frames
        captions_data = []
        
        for frame_path in tqdm(frame_files, desc="Generating captions"):
            # Extract frame number and calculate timestamp
            frame_name = frame_path.stem
            try:
                # Extract frame number from various formats
                if "frame_" in frame_name:
                    frame_num = int(frame_name.split("_")[-1])
                elif "_" in frame_name:
                    frame_num = int(frame_name.split("_")[-1])
                else:
                    frame_num = int(''.join(filter(str.isdigit, frame_name)))
                
                timestamp_sec = frame_num / config['pipeline']['frame_sampling_fps']
            except:
                timestamp_sec = 0.0
                frame_num = 0
            
            # Generate caption
            result = self.generate_caption(frame_path)
            
            frame_data = {
                "frame_id": frame_path.name,
                "frame_number": frame_num,
                "timestamp_sec": timestamp_sec,
                "caption": result.get("caption", ""),
                "success": result.get("success", False)
            }
            
            if not result.get("success"):
                frame_data["error"] = result.get("error", "Unknown error")
            
            captions_data.append(frame_data)
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "video_id": video_dir.name,
                "total_frames": len(frame_files),
                "captions": captions_data
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Saved vision captions to {output_file}")
        logger.info(f"Processed {len(captions_data)} frames")


def main():
    """Main entry point."""
    processed_dir = Path(config['paths']['processed_dir'])
    
    if not processed_dir.exists():
        logger.error(f"Processed directory not found: {processed_dir}")
        return
    
    # Initialize vision processor
    vision = VisionProcessor()
    
    # Process all video folders
    video_folders = [d for d in processed_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(video_folders)} video folders to process")
    
    for video_dir in video_folders:
        try:
            vision.process_video_folder(video_dir)
        except Exception as e:
            logger.error(f"Failed to process {video_dir.name}: {e}")
            continue
    
    logger.info("🎉 Phase 4 (Vision Understanding) completed!")


if __name__ == "__main__":
    main()

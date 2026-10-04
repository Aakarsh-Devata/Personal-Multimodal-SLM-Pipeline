#!/usr/bin/env python3
"""Generate timestamp-preserving local visual observations for selected frames."""

import json
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import logging
import platform
import resource
import sys
import time

# Add parent to path for config import
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from slm_pipeline.config import config
from slm_pipeline.runtime import selected_video_dirs, phase_cli

logging.basicConfig(
    level=config['logging']['level'],
    format=config['logging']['format']
)
logger = logging.getLogger(__name__)


class VisionProcessor:
    """Generate local frame observations without deriving or altering timestamps.

    ``smolvlm2`` is deliberately frame-based: preprocessing owns source-time
    decoding while this stage only describes pixels from the quality manifest.
    The legacy BLIP-2 backend remains selectable for existing cached setups.
    """
    
    def __init__(self):
        import torch
        vision = config['models']['vision']
        self.backend = vision.get('backend', 'blip2').lower()
        self.model_name = vision['name']
        self.device = self._select_device(torch, vision.get('device', 'auto'))
        self.batch_size = config['models']['vision']['batch_size']
        self.max_new_tokens = int(vision.get('max_new_tokens', 64))
        local_only = not config['pipeline']['allow_model_downloads']
        if local_only:
            # Some processor helpers otherwise make a metadata HEAD request even
            # when every required artifact is cached.  Offline mode keeps the
            # opt-in download contract strict for local reruns.
            os.environ.setdefault('HF_HUB_OFFLINE', '1')
            os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
        if self.backend == 'smolvlm2':
            from transformers import AutoModelForImageTextToText, AutoProcessor
            dtype = torch.float16 if self.device == 'mps' else torch.float32
            logger.info("Loading local SmolVLM2 model: %s on %s", self.model_name, self.device)
            self.processor = AutoProcessor.from_pretrained(self.model_name, local_files_only=local_only)
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name, local_files_only=local_only, torch_dtype=dtype
            ).to(self.device)
        elif self.backend == 'blip2':
            from transformers import Blip2ForConditionalGeneration, Blip2Processor
            logger.info("Loading local BLIP-2 model: %s on %s", self.model_name, self.device)
            self.processor = Blip2Processor.from_pretrained(self.model_name, local_files_only=local_only)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_name, local_files_only=local_only, torch_dtype=torch.float32
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported vision backend: {self.backend}")
        
        self.model.eval()
        logger.info("Local vision model loaded successfully")

    @staticmethod
    def _select_device(torch, requested):
        requested = str(requested).lower()
        mps_available = bool(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())
        if requested == 'auto':
            return 'mps' if mps_available else 'cpu'
        if requested == 'mps' and not mps_available:
            logger.warning("MPS was requested but is unavailable; using CPU")
            return 'cpu'
        if requested not in {'cpu', 'mps'}:
            raise ValueError("Vision device must be auto, mps, or cpu")
        return requested

    def _model_metadata(self):
        return {"source_type": "model", "backend": getattr(self, 'backend', None),
                "model": getattr(self, 'model_name', None), "device": getattr(self, 'device', None)}
    
    def generate_caption(self, image_path: Path) -> dict:
        """Generate caption for a single image."""
        try:
            image = Image.open(image_path).convert('RGB')
            import torch
            started = time.perf_counter()
            with torch.inference_mode():
                if self.backend == 'smolvlm2':
                    messages = [{"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe only visible objects, actions, and scene changes. Do not guess events outside this frame."},
                    ]}]
                    prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
                    inputs = self.processor(text=prompt, images=[image], return_tensors="pt").to(self.device)
                    generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
                    generated_ids = generated_ids[:, inputs.input_ids.shape[-1]:]
                else:
                    inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                    generated_ids = self.model.generate(**inputs, max_length=50, do_sample=False)
            caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            return {"caption": caption, "success": True, "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    **self._model_metadata()}
        except Exception as e:
            logger.error(f"Error processing {image_path.name}: {e}")
            return {
                "caption": "",
                "success": False,
                "error": str(e), **self._model_metadata()
            }

    def generate_video_observation(self, image_paths, source_timestamps_sec):
        """Describe one ordered, pre-sampled video window with retained source times."""
        if self.backend != 'smolvlm2':
            raise ValueError("Temporal observations require the smolvlm2 backend")
        if not image_paths or len(image_paths) != len(source_timestamps_sec):
            raise ValueError("Temporal observation requires equally sized frames and timestamps")
        if any(not isinstance(value, (int, float)) for value in source_timestamps_sec):
            raise ValueError("Temporal observation timestamps must be numeric")
        if any(b <= a for a, b in zip(source_timestamps_sec, source_timestamps_sec[1:])):
            raise ValueError("Temporal observation timestamps must be strictly increasing")
        from transformers.video_utils import VideoMetadata
        import torch
        frames = [Image.open(path).convert('RGB') for path in image_paths]
        intervals = [b - a for a, b in zip(source_timestamps_sec, source_timestamps_sec[1:])]
        fps = 1 / (sum(intervals) / len(intervals)) if intervals else 1.0
        metadata = VideoMetadata(total_num_frames=len(frames), fps=fps,
                                 duration=source_timestamps_sec[-1] - source_timestamps_sec[0],
                                 frames_indices=list(range(len(frames))))
        messages = [{"role": "user", "content": [
            {"type": "video"},
            {"type": "text", "text": "Describe only changes directly visible across this ordered sequence. Do not infer actions that are not visually established."},
        ]}]
        started = time.perf_counter()
        try:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            with torch.inference_mode():
                inputs = self.processor(text=prompt, videos=[[frames]], video_metadata=[metadata], return_tensors='pt').to(self.device)
                generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            generated_ids = generated_ids[:, inputs.input_ids.shape[-1]:]
            return {"observation": self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip(),
                    "success": True, "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "input_type": "video", **self._model_metadata()}
        except Exception as error:
            return {"observation": "", "success": False, "error": str(error), "input_type": "video",
                    **self._model_metadata()}
    
    def process_video_folder(self, video_dir: Path):
        """Process all frames for a video and generate captions."""
        logger.info(f"Processing video: {video_dir.name}")
        
        manifest_file = video_dir / "frames_manifest.json"
        if not manifest_file.is_file():
            raise FileNotFoundError("Missing frames_manifest.json; rerun quality phase")
        manifest = json.loads(manifest_file.read_text())
        frames = manifest["usable_frames"]
        if not frames:
            raise ValueError(f"No usable quality-filtered frames for {video_dir.name}")
        frame_files = [video_dir / frame["path"] for frame in frames]
        if any(not path.is_file() for path in frame_files):
            raise FileNotFoundError("Quality manifest references a missing frame")

        # Output file
        output_file = video_dir / "vision_captions.json"
        if output_file.exists() and config['pipeline']['skip_existing']:
            logger.info(f"⏩ Skipping {video_dir.name}, vision_captions.json already exists")
            return
        
        # Process frames
        started = time.perf_counter()
        captions_data = []
        
        for frame_path, frame in zip(frame_files, frames):
            # Never infer capture time from a scene index or renamed filename.
            timestamp_sec = float(frame["timestamp_sec"])

            # Generate caption
            result = self.generate_caption(frame_path)
            
            frame_data = {
                "frame_id": frame_path.name,
                "source_path": frame["source_path"],
                "source_timestamp_sec": frame.get("source_timestamp_sec", timestamp_sec),
                "timestamp_sec": timestamp_sec,
                "caption": result.get("caption", ""),
                "success": result.get("success", False),
                "source_type": result.get("source_type", "model"),
                "model": result.get("model", getattr(self, 'model_name', None)),
                "backend": result.get("backend", getattr(self, 'backend', None)),
                "device": result.get("device", getattr(self, 'device', None)),
                "verification_status": "unvalidated_model_hypothesis",
                "accepted_as_fact": False,
                "latency_ms": result.get("latency_ms"),
            }
            
            if not result.get("success"):
                raise RuntimeError(result.get("error", "Caption generation failed"))
            
            captions_data.append(frame_data)
        
        # Save results
        elapsed_sec = time.perf_counter() - started
        max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "video_id": video_dir.name,
                "total_frames": len(frame_files),
                "captions": captions_data,
                "model": self._model_metadata(),
                "run_metrics": {"elapsed_sec": round(elapsed_sec, 3), "max_rss_raw": max_rss,
                                "max_rss_unit": "bytes" if platform.system() == "Darwin" else "KiB"}
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Saved vision captions to {output_file}")
        logger.info(f"Processed {len(captions_data)} frames")


def main(video_id=None):
    folders = selected_video_dirs(config, video_id)
    vision = VisionProcessor()
    for video_dir in folders:
        vision.process_video_folder(video_dir)


if __name__ == "__main__":
    phase_cli(main)

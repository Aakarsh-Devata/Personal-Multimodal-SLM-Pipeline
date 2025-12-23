#!/usr/bin/env python3
"""
Pipeline Orchestrator
Runs the complete SLM pipeline end-to-end.
"""

import sys
from pathlib import Path
import logging
import argparse

sys.path.insert(0, str(Path(__file__).parent))
from config import config

logging.basicConfig(
    level=config['logging']['level'],
    format=config['logging']['format']
)
logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrate the complete SLM pipeline."""
    
    def __init__(self):
        self.processed_dir = Path(config['paths']['processed_dir'])
        self.raw_dir = Path(config['paths']['raw_dir'])
        self.project_root = Path(__file__).parent.parent
    
    def run_phase_legacy(self, phase_num: int, phase_name: str, script_path: str):
        """Run a legacy phase (1-3) using existing scripts."""
        logger.info(f"\n{'='*80}")
        logger.info(f"Phase {phase_num}: {phase_name}")
        logger.info(f"{'='*80}\n")
        
        try:
            import subprocess
            result = subprocess.run(
                ['python3', str(script_path)],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(result.stdout)
                logger.info(f"✅ Phase {phase_num} completed successfully\n")
                return True
            else:
                logger.error(f"Phase {phase_num} stderr: {result.stderr}")
                logger.error(f"❌ Phase {phase_num} failed\n")
                return False
        except Exception as e:
            logger.error(f"❌ Phase {phase_num} failed: {e}\n")
            return False
    
    def run_phase(self, phase_num: int, phase_name: str, module_name: str):
        """Run a specific phase of the pipeline."""
        logger.info(f"\n{'='*80}")
        logger.info(f"Phase {phase_num}: {phase_name}")
        logger.info(f"{'='*80}\n")
        
        try:
            # Import and run the phase module
            sys.path.insert(0, str(Path(__file__).parent / "pipelines"))
            module = __import__(module_name)
            module.main()
            logger.info(f"✅ Phase {phase_num} completed successfully\n")
            return True
        except Exception as e:
            logger.error(f"❌ Phase {phase_num} failed: {e}\n")
            return False
    
    def run_full_pipeline(self, start_phase: int = 1, end_phase: int = 7):
        """Run the full pipeline from start_phase to end_phase."""
        logger.info("\n🚀 Starting SLM Pipeline Orchestrator\n")
        
        phases = [
        phases = [
            (1, "Capture & Validation", None, None),  # Manual step
            (2, "Preprocessing (Audio + Frames)", self.project_root / "slm_pipeline/ingestion/phase_1r_preprocess.py", "legacy"),
            (3, "Speech Transcription (Whisper)", self.project_root / "slm_pipeline/ingestion/phase_2r_transcribe.py", "legacy"),
            (3.5, "Frame Quality Filtering", self.project_root / "slm_pipeline/ingestion/phase_3r_quality.py", "legacy"),
            (4, "Vision Understanding", "phase_4_vision", "new"),
            (5, "Multimodal Alignment", "phase_5_align", "new"),
            (6, "Semantic Structuring", "phase_6_semantic", "new"),
            (7, "Embedding & Memory Storage", "phase_7_embed", "new"),
        ]
        
        # Filter phases based on start and end
        phases_to_run = [(n, name, mod, typ) for n, name, mod, typ in phases if start_phase <= n <= end_phase]
        
        failed_phases = []
        
        for phase_num, phase_name, module_or_script, phase_type in phases_to_run:
            if phase_num == 1:
                # Phase 1 is manual - just validate
                logger.info(f"\n{'='*80}")
                logger.info(f"Phase 1: Capture & Validation")
                logger.info(f"{'='*80}\n")
                
                if not self.raw_dir.exists():
                    logger.error(f"❌ Raw directory not found: {self.raw_dir}")
                    failed_phases.append(1)
                    continue
                
                videos = list(self.raw_dir.glob("*.mp4")) + list(self.raw_dir.glob("*.mov"))
                if not videos:
                    logger.warning("⚠️  No videos found in raw directory")
                else:
                    logger.info(f"✅ Found {len(videos)} video(s) in raw directory")
                
                logger.info(f"✅ Phase 1 validation complete\n")
                continue
            
            if phase_type == "legacy":
                success = self.run_phase_legacy(phase_num, phase_name, module_or_script)
            else:
                success = self.run_phase(phase_num, phase_name, module_or_script)
            
            if not success:
                failed_phases.append(phase_num)
                response = input(f"\nContinue despite failure? (y/n): ")
                if not response.lower().startswith('y'):
                    break
        
        logger.info(f"\n{'='*80}")
        if failed_phases:
            logger.warning(f"⚠️  Pipeline completed with failures in phases: {failed_phases}")
        else:
            logger.info("🎉 Pipeline completed successfully!")
        logger.info(f"{'='*80}\n")
    
    def process_single_video(self, video_id: str, start_phase: int = 1):
        """Process a single video through the pipeline."""
        video_dir = self.processed_dir / video_id
        
        logger.info(f"Processing single video: {video_id}")
        
        # Check what's already done
        has_transcript = (video_dir / "transcript.json").exists() if video_dir.exists() else False
        
        if start_phase <= 3 and not has_transcript:
            logger.info("Video not yet preprocessed. Will run phases 1-3.")
            start_phase = 1
        elif has_transcript and start_phase <= 3:
            logger.info("Video already preprocessed. Starting from phase 4.")
            start_phase = 4
        
        # Run pipeline
        self.run_full_pipeline(start_phase=start_phase, end_phase=7)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SLM Pipeline Orchestrator - Run the complete pipeline"
    )
    
    parser.add_argument(
        '--video',
        type=str,
        help='Process a specific video by ID'
    )
    
    parser.add_argument(
        '--start-phase',
        type=int,
        default=4,
        help='Start from this phase (default: 4)'
    )
    
    parser.add_argument(
        '--end-phase',
        type=int,
        default=7,
        help='End at this phase (default: 7)'
    )
    
    args = parser.parse_args()
    
    orchestrator = PipelineOrchestrator()
    
    if args.video:
        orchestrator.process_single_video(args.video)
    else:
        orchestrator.run_full_pipeline(
            start_phase=args.start_phase,
            end_phase=args.end_phase
        )


if __name__ == "__main__":
    main()

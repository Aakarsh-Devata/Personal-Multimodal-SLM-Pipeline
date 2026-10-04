#!/usr/bin/env python3
"""Run the existing seven-stage pipeline with fail-fast, exact video scope."""
import argparse
import importlib
import logging
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slm_pipeline.config import config
from slm_pipeline.runtime import selected_raw_videos, validate_video_id

logger = logging.getLogger(__name__)
PHASES = (
    (2, "Preprocessing", "slm_pipeline.ingestion.phase_1r_preprocess"),
    (3, "Transcription", "slm_pipeline.ingestion.phase_2r_transcribe"),
    (3.5, "Frame quality", "slm_pipeline.ingestion.phase_3r_quality"),
    (4, "Vision", "slm_pipeline.pipelines.phase_4_vision"),
    (5, "Alignment", "slm_pipeline.pipelines.phase_5_align"),
    (6, "Semantics", "slm_pipeline.pipelines.phase_6_semantic"),
    (7, "Indexing", "slm_pipeline.pipelines.phase_7_embed"),
)


class PipelineOrchestrator:
    def run_full_pipeline(self, start_phase=1, end_phase=7, video_id=None):
        supported = {1, *(number for number, _, _ in PHASES)}
        if start_phase not in supported or end_phase not in supported or not 1 <= start_phase <= end_phase <= 7:
            raise ValueError("Select ordered phases from 1, 2, 3, 3.5, 4, 5, 6, 7")
        if video_id is not None:
            validate_video_id(video_id)
        if start_phase <= 1:
            selected_raw_videos(config, video_id)
        for number, name, module_name in PHASES:
            if start_phase <= number <= end_phase:
                logger.info("Phase %s: %s", number, name)
                importlib.import_module(module_name).main(video_id=video_id)
        return True

    def process_single_video(self, video_id, start_phase=1, end_phase=7):
        return self.run_full_pipeline(start_phase, end_phase, video_id)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", help="Process only this exact input filename stem")
    parser.add_argument("--start-phase", type=float, default=1)
    parser.add_argument("--end-phase", type=float, default=7)
    parser.add_argument("--resume", action="store_true", help="Trust existing stage artifacts; use only with unchanged inputs/config")
    args = parser.parse_args(argv)
    logging.basicConfig(level=config["logging"]["level"], format=config["logging"]["format"])
    config["pipeline"]["skip_existing"] = args.resume
    try:
        PipelineOrchestrator().run_full_pipeline(args.start_phase, args.end_phase, args.video)
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc)
        return 1
    logger.info("Selected phases completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())

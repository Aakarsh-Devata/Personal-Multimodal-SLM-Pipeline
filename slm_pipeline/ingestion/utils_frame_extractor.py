#!/usr/bin/env python3
"""Compatibility entrypoint; use the shared timestamp-preserving preprocessor."""
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from slm_pipeline.ingestion.phase_1r_preprocess import main
from slm_pipeline.runtime import phase_cli

if __name__ == '__main__':
    phase_cli(main)

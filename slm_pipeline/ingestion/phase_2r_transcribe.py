#!/usr/bin/env python3
"""
phase_c_transcribe.py
Run from ProjectRoot. Converts audio_clean.wav → transcript.json
Local version, cloud-ready structure.
"""

import json, time
from pathlib import Path
import whisper
import sys

# Import config
sys.path.insert(0, str(Path.cwd()))
from slm_pipeline.config import config

PROC_ROOT = Path(config['paths']['processed_dir'])

# choose model: "base" is fast on CPU, "small" or "medium" for better accuracy
MODEL_NAME = "base"

print(f"🔄 Loading Whisper model: {MODEL_NAME}")
model = whisper.load_model(MODEL_NAME)

for vid_dir in PROC_ROOT.iterdir():
    audio_path = vid_dir / "audio_clean.wav"
    if not audio_path.exists():
        continue

    out_path = vid_dir / "transcript.json"
    if out_path.exists():
        print(f"⏩ Skipping {vid_dir.name}, transcript already exists.")
        continue

    print(f"\n🎙️ Transcribing {vid_dir.name}...")
    start = time.time()
    result = model.transcribe(str(audio_path))
    dur = time.time() - start

    # save result (segments contain text + timestamps)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved transcript.json for {vid_dir.name}  ({dur:.1f}s)")

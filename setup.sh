#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
VENV="${VENV:-.venv}"
profile="requirements-test.txt"
if [[ "${1:-}" == "--inference" ]]; then
  profile="requirements-inference.txt"
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--inference]" >&2
  exit 2
fi
"$PYTHON" -m venv "$VENV"
"$VENV/bin/python" -m pip install -r "$profile"
mkdir -p data/raw data/processed data/memory
if ! command -v ffmpeg >/dev/null || ! command -v ffprobe >/dev/null; then
  echo "Install FFmpeg/ffprobe via your system package manager before processing videos." >&2
fi
echo "Setup complete. Activate: source $VENV/bin/activate"
echo "Model weights and Ollama models were not downloaded. See README.md."

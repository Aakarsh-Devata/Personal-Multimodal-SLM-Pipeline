"""Create optional blank metadata sidecars for explicitly selected input videos."""
import json
from pathlib import Path
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from slm_pipeline.config import config
from slm_pipeline.runtime import selected_raw_videos, phase_cli


def main(video_id=None):
    annotations = Path(config['paths']['raw_dir']).parent / 'annotations'
    annotations.mkdir(parents=True, exist_ok=True)
    for video in selected_raw_videos(config, video_id):
        sidecar = annotations / f'{video.stem}.json'
        if not sidecar.exists():
            sidecar.write_text(json.dumps({'video_filename': video.name, 'notes': ''}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    phase_cli(main)

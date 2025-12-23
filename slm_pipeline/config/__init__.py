"""Configuration management for SLM pipeline."""

import yaml
from pathlib import Path
from typing import Dict, Any

CONFIG_PATH = Path(__file__).parent / "settings.yaml"

def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

# Global config instance
config = load_config()

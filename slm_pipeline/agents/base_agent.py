"""Base agent class for SLM agents."""

import sys
from pathlib import Path
from typing import List, Dict, Any
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))
from slm_pipeline.config import config

logging.basicConfig(
    level=config['logging']['level'],
    format=config['logging']['format']
)
logger = logging.getLogger(__name__)


class BaseAgent:
    """Base class for all agents."""
    
    def __init__(self, name: str):
        self.name = name
        logger.info(f"Initialized {name}")
    
    def log(self, message: str, level: str = "info"):
        """Log a message."""
        log_func = getattr(logger, level, logger.info)
        log_func(f"[{self.name}] {message}")

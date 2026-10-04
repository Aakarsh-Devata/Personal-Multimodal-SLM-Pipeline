"""Offline public-data provenance and evaluation-reference adapters.

Human annotations never enter transcript, caption, alignment, or model outputs.
"""
from .manifest import load_manifest, validate_manifest, verify_source

__all__ = ["load_manifest", "validate_manifest", "verify_source"]

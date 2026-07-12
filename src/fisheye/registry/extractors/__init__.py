"""Zarr-to-registry row extractors.

Extractor modules are intentionally kept independent of ``registry.db`` so
stage-specific Zarr schema readers can evolve without expanding the SQLite
connection layer.
"""
from .chaser_metadata import (
    ChaserMetadataExtraction,
    ChaserMetadataIssue,
    extract_recording_chaser_metadata,
)
from .stimulus_metadata import StimulusMetadataExtraction, extract_stimulus_metadata

__all__ = [
    "ChaserMetadataExtraction",
    "ChaserMetadataIssue",
    "extract_recording_chaser_metadata",
    "StimulusMetadataExtraction",
    "extract_stimulus_metadata",
]

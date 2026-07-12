"""Registry selection and composition of curated visualization montages."""

from .models import MontageArtifactSpec, MontageLayout, RegistryRecording
from .profiles import PLOT_PROFILES, PlotProfile
from .registry import query_registry_recordings
from .render import compose_visualization_montage, load_recording_tiles
from .workflow import SCHEMA_ID, build_registry_visualization_montages

__all__ = [
    "PLOT_PROFILES",
    "SCHEMA_ID",
    "MontageArtifactSpec",
    "MontageLayout",
    "PlotProfile",
    "RegistryRecording",
    "build_registry_visualization_montages",
    "compose_visualization_montage",
    "load_recording_tiles",
    "query_registry_recordings",
]

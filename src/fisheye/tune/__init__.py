# src/fisheye/tune/__init__.py
"""
Interactive parameter tuning tools for the Palette pipeline.

Visual tools for optimizing detection, tracking, and analysis parameters.
"""

__version__ = "1.0.0"

from .base import BaseTuner
from .dispatcher import run_tuner, list_tuners

__all__ = [
    'BaseTuner',
    'run_tuner',
    'list_tuners',
]
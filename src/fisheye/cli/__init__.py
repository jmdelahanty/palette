# src/fisheye/cli/__init__.py
"""
Command-line interface tools for FishEye pipeline.

Includes both traditional CLI and interactive TUI launchers.
"""

from importlib import import_module
from typing import Any

__all__ = [
    'run_interactive_launcher',
    'pick_zarr_path',
    'pick_video_path',
    'pick_config_path',
]

_LAZY_EXPORTS = {
    "run_interactive_launcher": ("fisheye.cli.interactive_launcher", "run_interactive_launcher"),
    "pick_zarr_path": ("fisheye.cli.file_browsers", "pick_zarr_path"),
    "pick_video_path": ("fisheye.cli.file_browsers", "pick_video_path"),
    "pick_config_path": ("fisheye.cli.file_browsers", "pick_config_path"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

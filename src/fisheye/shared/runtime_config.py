"""Resolve version-controlled runtime configuration in source and wheel installs."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import sys


def runtime_config_dirs(relative_path: str) -> tuple[Path, ...]:
    """Return canonical config roots for a safe path below ``configs/fisheye``.

    A source checkout owns the files at repository-level ``configs/fisheye``.
    Wheels install those same files below ``sys.prefix/share/palette`` through
    setuptools ``data-files``.  Returning both locations keeps loaders usable
    without consulting the current working directory.
    """

    raw_relative = str(relative_path).strip()
    relative = PurePosixPath(raw_relative)
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("runtime config path must be a safe non-root relative path")

    module_path = Path(__file__).resolve()
    checkout_dir = module_path.parents[3] / "configs" / "fisheye" / Path(*relative.parts)
    installed_dir = (
        Path(sys.prefix)
        / "share"
        / "palette"
        / "configs"
        / "fisheye"
        / Path(*relative.parts)
    )
    candidates = (checkout_dir, installed_dir)
    return tuple(dict.fromkeys(candidates))


__all__ = ["runtime_config_dirs"]

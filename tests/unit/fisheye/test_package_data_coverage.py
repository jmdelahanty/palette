"""Every runtime asset under src/fisheye is declared as package data.

A file type missing from `[tool.setuptools.package-data]` works in an editable
checkout but is silently absent from an installed wheel (as the labeling
templates, `*.html.j2`, were until 2026-09-27).
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "src" / "fisheye"
# Documentation shipped in the source tree only; nothing reads it at runtime.
NOT_RUNTIME_SUFFIXES = {".py", ".pyc", ".md"}


def _patterns() -> list[str]:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    return list(config["tool"]["setuptools"]["package-data"]["fisheye"])


def _declared(relative: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        # setuptools "**/" also matches files at the package root.
        candidates = {pattern, pattern.removeprefix("**/")}
        if any(fnmatch.fnmatch(relative, c) for c in candidates):
            return True
    return False


def test_every_runtime_asset_is_package_data():
    patterns = _patterns()
    missing = sorted(
        path.relative_to(PACKAGE).as_posix()
        for path in PACKAGE.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in NOT_RUNTIME_SUFFIXES
        and not _declared(path.relative_to(PACKAGE).as_posix(), patterns)
    )
    assert missing == [], f"add package-data patterns for: {missing}"


def test_labeling_templates_are_covered():
    patterns = _patterns()
    templates = sorted((PACKAGE / "labeling" / "templates").rglob("*.html.j2"))
    assert len(templates) >= 11
    assert all(_declared(t.relative_to(PACKAGE).as_posix(), patterns) for t in templates)

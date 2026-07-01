"""Small template and static-asset helpers for the labeling web UI."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Mapping

_PACKAGE_ROOT = Path(__file__).resolve().parent


def _safe_asset_path(relative_path: str) -> Path:
    normalized = str(relative_path).replace("\\", "/").lstrip("/")
    path = Path(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe labeling asset path: {relative_path!r}")
    resolved = (_PACKAGE_ROOT / path).resolve()
    try:
        resolved.relative_to(_PACKAGE_ROOT)
    except ValueError as exc:
        raise ValueError(f"unsafe labeling asset path: {relative_path!r}") from exc
    return resolved


@lru_cache(maxsize=None)
def read_labeling_asset(relative_path: str) -> str:
    """Read a labeling package asset as UTF-8 text."""
    return _safe_asset_path(relative_path).read_text(encoding="utf-8")


def render_labeling_template(relative_path: str, context: Mapping[str, object]) -> str:
    """Render a tiny @@name@@ placeholder template from labeling/templates/."""
    template_path = str(relative_path)
    if not template_path.startswith("templates/"):
        template_path = f"templates/{template_path}"
    rendered = read_labeling_asset(template_path)
    for key, value in context.items():
        replacement = "" if value is None else str(value)
        rendered = rendered.replace(f"@@{key}@@", replacement)
    return rendered

"""Read-only personal HTML renderers for labeling web surfaces."""

from __future__ import annotations

from fisheye.labeling.template_assets import read_labeling_asset

__all__ = ["_dashboard_html", "_datasets_html"]


def _dashboard_html() -> bytes:
    return read_labeling_asset("templates/personal/my_work.html").encode("utf-8")


def _datasets_html() -> bytes:
    return read_labeling_asset("templates/personal/my_datasets.html").encode("utf-8")

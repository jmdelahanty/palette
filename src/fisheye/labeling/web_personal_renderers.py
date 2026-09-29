"""Read-only personal HTML renderers for labeling web surfaces."""

from __future__ import annotations

from fisheye.labeling.template_assets import read_labeling_asset, render_labeling_template
from fisheye.labeling.web_static import static_url

__all__ = ["_dashboard_html", "_datasets_html", "_queue_html"]


def _dashboard_html() -> bytes:
    return read_labeling_asset("templates/personal/my_work.html").encode("utf-8")


def _datasets_html() -> bytes:
    return read_labeling_asset("templates/personal/my_datasets.html").encode("utf-8")


def _queue_html() -> bytes:
    return render_labeling_template(
        "personal/queue.html",
        {
            "palette_css": static_url("css/palette.css"),
            "queue_js": static_url("js/queue_page.js"),
        },
    ).encode("utf-8")

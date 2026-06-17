"""Datasette plugin: color status cells in the registry's wide/overview views.

Mirrors the TUI's status palette (registry_tui.py _STATUS_STYLES) and the
registry's own "blocking" classification (status_page query._blocking_expr,
which keys off MISS/STALE/UNVER/ERR/FAIL prefixes).

The render_cell hook fires for every cell on a table/view page, so we gate on:
  1. the object name (only the status-bearing views), and
  2. the cell text matching a known status token.
Identity columns (Recording / Camera names, paths) never match a token, so
they are left untouched without needing an explicit column allow-list.

Load with:  datasette ... --plugins-dir docs/registry_browser/plugins
"""

from __future__ import annotations

from datasette import hookimpl
from markupsafe import Markup, escape

# Views whose cells carry pipeline status tokens.
STATUS_VIEWS = frozenset(
    {
        "recording_step_status_wide",
        "recording_step_status_latest",
        "recording_step_overview",
        "recording_overview",
    }
)


def classify(text: str) -> str | None:
    """Return a CSS suffix for a status cell, or None to leave it uncolored."""
    u = text.upper()
    # Worst-first so decorated cells like "0 (MISS)" resolve to the bad state.
    if "FAIL" in u or "ERROR" in u or u.startswith("ERR"):
        return "err"
    if "MISS" in u:
        return "miss"
    if "STALE" in u:
        return "stale"
    if "UNVER" in u or "PENDING" in u or "NEEDS" in u or "WARN" in u:
        return "warn"
    if u in {"N/A", "NA", "ABSENT", "—", "-", "–"}:
        return "muted"
    # Good states: explicit OK, review approvals/completions, and coverage %.
    if "OK" in u or "APPROV" in u or "COMPLET" in u or u.endswith("%"):
        return "ok"
    return None


@hookimpl
def render_cell(row, value, column, table, database, datasette):
    if table not in STATUS_VIEWS or value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    suffix = classify(text)
    if suffix is None:
        return None
    # escape() neutralizes any HTML in the value before we wrap it.
    return Markup('<span class="s s-{}">{}</span>').format(suffix, escape(text))

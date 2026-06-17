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

import json

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


# Legend: (swatch CSS class, label). Swatch colors come from registry.css
# variables, so the legend always matches the cell tints. Labels mirror the
# classify() rules above.
LEGEND_ITEMS = [
    ("sw-ok", "OK · approved · coverage%"),
    ("sw-miss", "missing"),
    ("sw-stale", "stale"),
    ("sw-warn", "unverified · pending · warn"),
    ("sw-err", "error · failed"),
    ("sw-muted", "N/A · absent"),
]


def _legend_script() -> str:
    """Client-side legend, injected above the table. Done in JS rather than a
    template override so it survives Datasette version bumps."""
    items_js = json.dumps(LEGEND_ITEMS)
    return (
        "(function(){"
        "if(document.querySelector('.status-legend'))return;"
        "var items=" + items_js + ";"
        "var L=document.createElement('div');L.className='status-legend';"
        "var t=document.createElement('span');t.className='status-legend-title';"
        "t.textContent='Status';L.appendChild(t);"
        "items.forEach(function(it){"
        "var i=document.createElement('span');i.className='status-legend-item';"
        "var s=document.createElement('span');s.className='status-legend-swatch '+it[0];"
        "i.appendChild(s);i.appendChild(document.createTextNode(it[1]));"
        "L.appendChild(i);});"
        "var tbl=document.querySelector('table.rows-and-columns');"
        "if(tbl&&tbl.parentNode){tbl.parentNode.insertBefore(L,tbl);}"
        "else{document.body.insertBefore(L,document.body.firstChild);}"
        "})();"
    )


@hookimpl
def extra_body_script(table, view_name):
    # Only on the colored status views (table is None on query/db pages).
    if table not in STATUS_VIEWS:
        return None
    return _legend_script()

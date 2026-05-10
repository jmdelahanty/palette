#!/usr/bin/env python3
"""Textual TUI scaffold for browsing and managing the Palette registry."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from rich.text import Text as RichText

try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.widgets import DataTable, Footer, Header, Input, Label, ListItem, ListView, Static
except ImportError:
    raise SystemExit("Textual is required. Install with: scripts/py -m pip install textual rich")

try:
    from fisheye.registry.step_cascade import STEP_DEPENDENTS, get_transitive_dependents
    _HAS_STEP_CASCADE = True
except Exception:
    _HAS_STEP_CASCADE = False
    STEP_DEPENDENTS: Dict[str, frozenset] = {}  # type: ignore[no-redef]

    def get_transitive_dependents(step_name: str) -> frozenset:  # type: ignore[misc]
        return frozenset()


CURATED_VIEWS: List[Tuple[str, str]] = [
    ("datasets", "SELECT * FROM datasets ORDER BY created_utc DESC"),
    ("dataset_lineage_current", "SELECT * FROM dataset_lineage_current ORDER BY child_dataset_id, relationship_type, parent_dataset_id"),
    ("step_status_wide", "SELECT * FROM recording_step_status_wide"),
    ("step_overview", "SELECT * FROM recording_step_overview"),
    ("recording_overview", "SELECT * FROM recording_overview"),
    ("provenance", "SELECT * FROM provenance ORDER BY dataset_id"),
    ("training_sets", "SELECT * FROM training_sets ORDER BY created_utc DESC"),
    ("training_runs", "SELECT * FROM training_runs ORDER BY created_utc DESC"),
    ("training_models", "SELECT * FROM training_models ORDER BY created_utc DESC"),
    ("onnx_models", "SELECT * FROM onnx_models ORDER BY created_utc DESC"),
    ("tensorrt_models", "SELECT * FROM tensorrt_models ORDER BY created_utc DESC"),
    ("keypoint_quality_current", "SELECT * FROM keypoint_quality_current ORDER BY quality_updated_utc DESC"),
    (
        "refined_detect_review_current",
        "SELECT * FROM detect_quality_current ORDER BY quality_updated_utc DESC",
    ),
    ("detect_perf", "SELECT * FROM detect_performance_latest"),
    ("keypoint_perf", "SELECT * FROM keypoint_performance_latest"),
    ("eye_mask_perf", "SELECT * FROM eye_mask_performance_latest"),
    ("pose_skeleton_specs", "SELECT * FROM pose_skeleton_specs ORDER BY created_utc DESC"),
]

# ---------------------------------------------------------------------------
# Status color-coding
# ---------------------------------------------------------------------------

_STATUS_STYLES: Dict[str, str] = {
    "ok": "green",
    "OK": "green",
    "missing": "bold red",
    "MISS": "bold red",
    "absent": "dim",
    "na": "dim",
    "N/A": "dim",
    "error": "bold yellow",
    "failed": "bold yellow",
}

_STATUS_COLORIZED_VIEWS: frozenset = frozenset({
    "step_status_wide",
    "step_overview",
    "recording_overview",
    "detect_perf",
    "keypoint_perf",
    "eye_mask_perf",
    "keypoint_quality_current",
    "detect_quality_current",
    "refined_detect_review_current",
})


def _view_exists(conn: sqlite3.Connection, view_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'view' AND name = ? LIMIT 1;",
        (view_name,),
    ).fetchone()
    return row is not None


def _preferred_detect_review_view_name(conn: sqlite3.Connection) -> str:
    if _view_exists(conn, "refined_detect_review_current"):
        return "refined_detect_review_current"
    return "detect_quality_current"


def _style_cell_value(value: Any, view_name: str) -> Any:
    """Return a Rich Text object with color styling for known status values."""
    if view_name not in _STATUS_COLORIZED_VIEWS:
        return "" if value is None else str(value)
    s = "" if value is None else str(value)
    lookup = s.strip()
    style = _STATUS_STYLES.get(lookup)
    if style:
        return RichText(s, style=style)
    upper = lookup.upper()
    if upper.startswith("OK"):
        return RichText(s, style="green")
    if upper.startswith("MISS"):
        return RichText(s, style="bold red")
    return s


# ---------------------------------------------------------------------------
# Column-aware filter parsing
# ---------------------------------------------------------------------------


def _parse_filter_expr(raw: str) -> Tuple[List[Tuple[str, str]], Optional[str]]:
    """Parse filter text into column-specific filters and optional global substring.

    Syntax examples:
        "detect:miss"           -> column filter on cols matching "detect"
        "status:ok crop:miss"   -> AND of two column filters
        "cedar"                 -> global substring across all columns
        "detect:miss cedar"     -> column filter AND global substring
    """
    tokens = raw.strip().split()
    col_filters: List[Tuple[str, str]] = []
    global_parts: List[str] = []
    for token in tokens:
        if ":" in token:
            parts = token.split(":", 1)
            col_name, col_val = parts[0].strip(), parts[1].strip()
            if col_name and col_val:
                col_filters.append((col_name.lower(), col_val.lower()))
                continue
        global_parts.append(token)
    global_sub = " ".join(global_parts).lower() if global_parts else None
    return col_filters, global_sub


# ---------------------------------------------------------------------------
# Step dependency visualization
# ---------------------------------------------------------------------------

# Maps the wide view's human-readable column names to step_cascade step names.
_WIDE_COL_TO_STEP: Dict[str, str] = {
    "import": "raw",
    "bg full": "background",
    "bg ds": "background",
    "detect": "detect",
    "detect quality": "detect_quality",
    "refine detect": "refined_detect",
    "detect group": "refined_detect",
    "crop": "crop",
    "keypoints": "keypoints",
    "refined keypoints (analysis/train)": "refined_keypoints",
    "eye masks": "eye_masks",
    "refined eye masks": "refined_eye_masks",
    "subject masks": "subject_masks",
    "refined subject masks": "refined_subject_masks",
    "arena assignment": "arena_assignment",
    "track": "tracks",
    "stimulus": "stimulus",
    "calib": "calibration",
}

_STEP_DEPENDENCY_VIEWS: frozenset = frozenset({
    "step_status_wide",
    "step_overview",
})


def _render_step_dependencies(step_name: str) -> List[str]:
    """Render concise step dependency info for the details panel."""
    if not _HAS_STEP_CASCADE or step_name not in STEP_DEPENDENTS:
        return []
    lines: List[str] = []
    # Upstream
    upstream = sorted(p for p, children in STEP_DEPENDENTS.items() if step_name in children)
    if upstream:
        lines.append(f"  {step_name} <- {', '.join(upstream)}")
    # Direct dependents
    direct = STEP_DEPENDENTS.get(step_name, frozenset())
    if direct:
        lines.append(f"  {step_name} -> {', '.join(sorted(direct))}")
    # Cascade impact
    transitive = get_transitive_dependents(step_name)
    if transitive:
        lines.append(f"  cascade: re-running {step_name} invalidates {len(transitive)} step(s)")
    return lines


# ---------------------------------------------------------------------------
# Registry client
# ---------------------------------------------------------------------------


class RegistryClient:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()

    def list_tables_and_views(self) -> List[str]:
        rows = self.conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%'
            ORDER BY name;
            """
        ).fetchall()
        return [str(r["name"]) for r in rows]

    def fetch_rows(self, sql: str, limit: int = 500) -> List[Dict[str, Any]]:
        wrapped = f"SELECT * FROM ({sql}) LIMIT {int(limit)}"
        rows = self.conn.execute(wrapped).fetchall()
        return [dict(r) for r in rows]

    def fetch_rows_for_name(self, name: str, limit: int = 500) -> List[Dict[str, Any]]:
        safe = "".join(ch for ch in name if ch.isalnum() or ch == "_")
        if safe != name:
            raise ValueError(f"Unsafe table/view name: {name}")
        sql = f"SELECT * FROM {safe}"
        rows = self.conn.execute(f"{sql} LIMIT {int(limit)}").fetchall()
        return [dict(r) for r in rows]

    def fetch_relationships(self, view_name: str, row: Dict[str, Any]) -> List[str]:
        lines: List[str] = []
        dataset_id = str(row.get("dataset_id") or "").strip()
        set_id = str(row.get("set_id") or "").strip()
        run_id = str(row.get("run_id") or "").strip()

        if dataset_id:
            lines.append(f"dataset_id={dataset_id}")
            parents = self.conn.execute(
                """
                SELECT parent_dataset_id, relationship_type, source_set_id
                FROM dataset_lineage_current
                WHERE child_dataset_id = ?
                ORDER BY relationship_type, parent_dataset_id;
                """,
                (dataset_id,),
            ).fetchall()
            children = self.conn.execute(
                """
                SELECT child_dataset_id, relationship_type, source_set_id
                FROM dataset_lineage_current
                WHERE parent_dataset_id = ?
                ORDER BY relationship_type, child_dataset_id;
                """,
                (dataset_id,),
            ).fetchall()
            if parents:
                lines.append("dataset_lineage_current (parents):")
                for r in parents[:12]:
                    lines.append(
                        "  - "
                        f"{r['parent_dataset_id']} rel={r['relationship_type']} set={r['source_set_id'] or '—'}"
                    )
            if children:
                lines.append("dataset_lineage_current (children):")
                for r in children[:12]:
                    lines.append(
                        "  - "
                        f"{r['child_dataset_id']} rel={r['relationship_type']} set={r['source_set_id'] or '—'}"
                    )
            kq = self.conn.execute(
                "SELECT refined_run, keypoint_method, review_state, usable_keypoints_rate FROM keypoint_quality_current WHERE dataset_id=?",
                (dataset_id,),
            ).fetchall()
            detect_review_view = _preferred_detect_review_view_name(self.conn)
            dq = self.conn.execute(
                f"SELECT refined_run, detect_method, review_state, interpolated_detections_rate FROM {detect_review_view} WHERE dataset_id=?",
                (dataset_id,),
            ).fetchall()
            if kq:
                lines.append("keypoint_quality_current:")
                for r in kq[:5]:
                    lines.append(
                        f"  - run={r['refined_run']} method={r['keypoint_method']} state={r['review_state']} usable_rate={r['usable_keypoints_rate']}"
                    )
            if dq:
                lines.append(f"{detect_review_view}:")
                for r in dq[:5]:
                    lines.append(
                        f"  - run={r['refined_run']} method={r['detect_method']} state={r['review_state']} interp_rate={r['interpolated_detections_rate']}"
                    )
            ts = self.conn.execute(
                "SELECT set_id, created_utc FROM training_sets WHERE dataset_ids_json LIKE ? ORDER BY created_utc DESC",
                (f"%{dataset_id}%",),
            ).fetchall()
            if ts:
                lines.append("training_sets containing dataset:")
                for r in ts[:8]:
                    lines.append(f"  - {r['set_id']} ({r['created_utc']})")

        if set_id:
            lines.append(f"set_id={set_id}")
            runs = self.conn.execute(
                "SELECT run_id, status, created_utc FROM training_runs WHERE set_id=? ORDER BY created_utc DESC",
                (set_id,),
            ).fetchall()
            if runs:
                lines.append("training_runs:")
                for r in runs[:10]:
                    lines.append(f"  - {r['run_id']} status={r['status']} ({r['created_utc']})")

        if run_id:
            lines.append(f"run_id={run_id}")
            onnx = self.conn.execute(
                "SELECT path, opset, created_utc FROM onnx_models WHERE run_id=?",
                (run_id,),
            ).fetchall()
            trt = self.conn.execute(
                "SELECT path, precision, created_utc FROM tensorrt_models WHERE run_id=?",
                (run_id,),
            ).fetchall()
            if onnx:
                lines.append("onnx_models:")
                for r in onnx:
                    lines.append(f"  - opset={r['opset']} created={r['created_utc']} path={r['path']}")
            if trt:
                lines.append("tensorrt_models:")
                for r in trt:
                    lines.append(f"  - precision={r['precision']} created={r['created_utc']} path={r['path']}")

        if not lines:
            lines.append("No relationship hints available for this row.")
        return lines


# ---------------------------------------------------------------------------
# TUI application
# ---------------------------------------------------------------------------


class RegistryTUI(App[None]):
    CSS = """
    Screen {
      layout: vertical;
    }
    #main {
      layout: horizontal;
      height: 1fr;
    }
    #left {
      width: 28;
      border: solid $primary;
    }
    #center {
      width: 1fr;
      border: solid $accent;
    }
    #right {
      width: 48;
      border: solid $secondary;
      padding: 0 1;
    }
    #status {
      height: 3;
      border: solid $surface;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("/", "focus_filter", "Filter"),
        Binding("n", "next_view", "Next View"),
        Binding("p", "prev_view", "Prev View"),
        Binding("c", "clear_filter", "Clear Filter"),
        Binding("s", "cycle_sort", "Sort Column"),
    ]

    def __init__(self, *, registry_path: Path, start_view: Optional[str], row_limit: int, readonly: bool) -> None:
        super().__init__()
        self.registry_path = registry_path
        self.start_view = start_view
        self.row_limit = int(row_limit)
        self.readonly = bool(readonly)
        self.client = RegistryClient(registry_path)
        self.view_entries: List[Tuple[str, str]] = []
        self.active_view_index = 0
        self.active_rows: List[Dict[str, Any]] = []
        self.display_rows: List[Dict[str, Any]] = []
        self.filter_text = ""
        self.sort_column: Optional[str] = None
        self.sort_reverse: bool = False
        self._column_keys: List[str] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main"):
            with Vertical(id="left"):
                yield Label("Views")
                yield ListView(id="view_list")
            with Vertical(id="center"):
                yield Label("Rows")
                yield DataTable(id="row_table", zebra_stripes=True, cursor_type="row")
            with Vertical(id="right"):
                yield Label("Relationships")
                yield Static("Select a row to inspect links.", id="detail")
        with Vertical(id="status"):
            yield Input(placeholder="Filter: text or column:value (e.g. detect:miss). Enter to apply.", id="filter_input")
            yield Static("", id="status_text")
        yield Footer()

    def on_mount(self) -> None:
        curated = list(CURATED_VIEWS)
        existing = set(self.client.list_tables_and_views())
        raw = sorted(name for name in existing if name not in {v[0] for v in curated})
        for name in raw:
            curated.append((f"table:{name}", name))
        self.view_entries = curated

        view_list = self.query_one("#view_list", ListView)
        for label, _ in self.view_entries:
            view_list.append(ListItem(Label(label)))

        if self.start_view:
            for idx, (label, _) in enumerate(self.view_entries):
                if label == self.start_view or label == f"table:{self.start_view}":
                    self.active_view_index = idx
                    break

        view_list.index = self.active_view_index
        self._load_active_view()
        self._set_status(f"Connected: {self.registry_path} | readonly={self.readonly}")
        self.query_one("#filter_input", Input).blur()

    def on_unmount(self) -> None:
        self.client.close()

    # --- helpers ---

    def _set_status(self, text: str) -> None:
        self.query_one("#status_text", Static).update(text)

    def _active_view_name_and_sql(self) -> Tuple[str, str]:
        label, payload = self.view_entries[self.active_view_index]
        if label.startswith("table:"):
            return label, f"TABLE::{payload}"
        return label, payload

    def _build_status_text(self, cursor_row: Optional[int] = None) -> str:
        """Build status bar text with view name, row counts, sort, filter."""
        label, _ = self._active_view_name_and_sql()
        parts = [f"View={label}"]
        if cursor_row is not None:
            parts.append(f"Row {cursor_row + 1}/{len(self.display_rows)}")
        else:
            parts.append(f"rows={len(self.display_rows)}")
        parts.append(f"loaded={len(self.active_rows)}")
        if self.sort_column:
            arrow = "\u25b2" if not self.sort_reverse else "\u25bc"
            parts.append(f"sort: {self.sort_column} {arrow}")
        if self.filter_text:
            parts.append(f"filter: '{self.filter_text}'")
        return " | ".join(parts)

    # --- filtering ---

    def _apply_filter(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        needle = self.filter_text.strip()
        if not needle:
            return list(rows)
        col_filters, global_sub = _parse_filter_expr(needle)
        out: List[Dict[str, Any]] = []
        for row in rows:
            # Column-specific filters (all must match)
            col_match = True
            for col_name, col_val in col_filters:
                matched_any = False
                for k, v in row.items():
                    if col_name in k.lower():
                        cell_text = "" if v is None else str(v).lower()
                        if col_val in cell_text:
                            matched_any = True
                            break
                if not matched_any:
                    col_match = False
                    break
            if not col_match:
                continue
            # Global substring (across all columns)
            if global_sub:
                text = " | ".join("" if v is None else str(v) for v in row.values()).lower()
                if global_sub not in text:
                    continue
            out.append(row)
        return out

    # --- sorting ---

    def _apply_sort(self) -> None:
        """Sort display_rows by current sort_column and re-render."""
        if not self.sort_column or not self.display_rows:
            return
        col = self.sort_column
        if col not in self.display_rows[0]:
            return

        def sort_key(row: Dict[str, Any]) -> Tuple[int, Any]:
            val = row.get(col)
            if val is None:
                return (1, "")
            s = str(val).strip()
            try:
                return (0, float(s.rstrip("%")))
            except (ValueError, TypeError):
                return (0, s.lower())

        self.display_rows.sort(key=sort_key, reverse=self.sort_reverse)
        self._render_table()

    def _update_sort_indicator(self) -> None:
        """Show sort direction arrow on the sorted column header."""
        table = self.query_one("#row_table", DataTable)
        for col_key_obj in list(table.columns):
            col_meta = table.columns[col_key_obj]
            key_str = str(col_key_obj)
            # Original label is the column key itself
            original = key_str
            if key_str == self.sort_column:
                arrow = " \u25b2" if not self.sort_reverse else " \u25bc"
                col_meta.label = RichText(original + arrow)
            else:
                col_meta.label = RichText(original)
        table.refresh()

    # --- view loading and rendering ---

    def _load_active_view(self) -> None:
        label, sql_or_table = self._active_view_name_and_sql()
        try:
            if sql_or_table.startswith("TABLE::"):
                table_name = sql_or_table.replace("TABLE::", "", 1)
                rows = self.client.fetch_rows_for_name(table_name, limit=self.row_limit)
            else:
                rows = self.client.fetch_rows(sql_or_table, limit=self.row_limit)
        except sqlite3.OperationalError as exc:
            rows = []
            self._set_status(f"Error loading {label}: {exc}")
            self.active_rows = rows
            self.display_rows = rows
            self._render_table()
            return
        self.active_rows = rows
        self.sort_column = None
        self.sort_reverse = False
        self.display_rows = self._apply_filter(rows)
        self._render_table()
        self._set_status(self._build_status_text())

    def _render_table(self) -> None:
        table = self.query_one("#row_table", DataTable)
        table.clear(columns=True)
        if not self.display_rows:
            table.add_column("No rows")
            table.add_row(" ")
            self.query_one("#detail", Static).update("No rows matched.")
            return
        columns = list(self.display_rows[0].keys())
        self._column_keys = columns
        for col in columns:
            table.add_column(str(col), key=str(col))
        view_name = self._active_view_name_and_sql()[0]
        for row in self.display_rows:
            table.add_row(*[_style_cell_value(row.get(c), view_name) for c in columns])
        if self.sort_column:
            self._update_sort_indicator()
        table.move_cursor(row=0, column=0)
        self._update_details_for_row(0)

    def _update_details_for_row(self, row_index: int) -> None:
        if row_index < 0 or row_index >= len(self.display_rows):
            return
        row = self.display_rows[row_index]
        view_name, _ = self._active_view_name_and_sql()
        lines = [f"view={view_name}", ""]
        for k, v in row.items():
            lines.append(f"{k}: {v}")
        lines.append("")
        lines.append("links:")
        lines.extend(self.client.fetch_relationships(view_name, row))

        # Step dependency visualization
        if _HAS_STEP_CASCADE and view_name in _STEP_DEPENDENCY_VIEWS:
            blocking_steps = self._find_blocking_steps(view_name, row)
            if blocking_steps:
                lines.append("")
                lines.append("step dependencies (blocking steps):")
                for step in list(blocking_steps)[:3]:
                    lines.extend(_render_step_dependencies(step))
            else:
                lines.append("")
                lines.append("step dependencies: all tracked steps OK")

        self.query_one("#detail", Static).update("\n".join(lines))

    def _find_blocking_steps(self, view_name: str, row: Dict[str, Any]) -> List[str]:
        """Identify pipeline steps that are MISS/error in the current row."""
        blocking: List[str] = []
        seen: Set[str] = set()
        if view_name == "step_status_wide":
            for col_name, val in row.items():
                val_str = str(val or "").strip().upper()
                if val_str in ("MISS", "MISSING", "ERROR", "FAILED"):
                    step = _WIDE_COL_TO_STEP.get(col_name.lower())
                    if step and step not in seen:
                        blocking.append(step)
                        seen.add(step)
        elif view_name == "step_overview":
            csv_val = str(row.get("blocking_steps_csv") or "")
            if csv_val:
                for s in csv_val.split(","):
                    s = s.strip()
                    if s and s not in seen:
                        blocking.append(s)
                        seen.add(s)
        return blocking

    # --- event handlers ---

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        idx = int(event.list_view.index or 0)
        if idx < 0 or idx >= len(self.view_entries):
            return
        self.active_view_index = idx
        self._load_active_view()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.cursor_row is None:
            return
        row_idx = int(event.cursor_row)
        self._update_details_for_row(row_idx)
        self._set_status(self._build_status_text(cursor_row=row_idx))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.cursor_row is None:
            return
        self._update_details_for_row(int(event.cursor_row))

    def on_data_table_header_selected(self, event: DataTable.HeaderSelected) -> None:
        """Sort by the clicked column. Toggle direction on repeated clicks."""
        col_key = str(event.column_key)
        if col_key == self.sort_column:
            self.sort_reverse = not self.sort_reverse
        else:
            self.sort_column = col_key
            self.sort_reverse = False
        self._apply_sort()
        self._set_status(self._build_status_text())

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "filter_input":
            return
        self.filter_text = str(event.value or "")
        self.display_rows = self._apply_filter(self.active_rows)
        if self.sort_column:
            self._apply_sort()
        else:
            self._render_table()
        self._set_status(self._build_status_text())

    # --- actions ---

    def action_refresh(self) -> None:
        self._load_active_view()

    def action_focus_filter(self) -> None:
        self.query_one("#filter_input", Input).focus()

    def action_clear_filter(self) -> None:
        self.filter_text = ""
        inp = self.query_one("#filter_input", Input)
        inp.value = ""
        self.display_rows = list(self.active_rows)
        if self.sort_column:
            self._apply_sort()
        else:
            self._render_table()
        self._set_status(self._build_status_text())

    def action_next_view(self) -> None:
        if not self.view_entries:
            return
        self.active_view_index = (self.active_view_index + 1) % len(self.view_entries)
        self.query_one("#view_list", ListView).index = self.active_view_index
        self._load_active_view()

    def action_prev_view(self) -> None:
        if not self.view_entries:
            return
        self.active_view_index = (self.active_view_index - 1) % len(self.view_entries)
        self.query_one("#view_list", ListView).index = self.active_view_index
        self._load_active_view()

    def action_cycle_sort(self) -> None:
        """Cycle sort through columns for keyboard users."""
        if not self.display_rows or not self._column_keys:
            return
        columns = self._column_keys
        if self.sort_column is None or self.sort_column not in columns:
            self.sort_column = columns[0]
            self.sort_reverse = False
        elif not self.sort_reverse:
            self.sort_reverse = True
        else:
            idx = columns.index(self.sort_column)
            self.sort_column = columns[(idx + 1) % len(columns)]
            self.sort_reverse = False
        self._apply_sort()
        self._set_status(self._build_status_text())


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("/nvme1/palette_registry.sqlite"),
        help="Registry SQLite path.",
    )
    parser.add_argument("--view", type=str, help="Start view name (e.g., training_runs or table:datasets).")
    parser.add_argument("--limit", type=int, default=500, help="Max rows to load per view.")
    parser.add_argument("--readonly", action="store_true", default=True, help="Read-only mode (default: true).")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    db_path = Path(args.registry).expanduser().resolve()
    if not db_path.exists():
        raise SystemExit(f"Registry not found: {db_path}")
    app = RegistryTUI(
        registry_path=db_path,
        start_view=args.view,
        row_limit=int(args.limit),
        readonly=bool(args.readonly),
    )
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

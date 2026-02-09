#!/usr/bin/env python3
"""Textual TUI scaffold for browsing and managing the Palette registry."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.widgets import DataTable, Footer, Header, Input, Label, ListItem, ListView, Static
except ImportError:
    raise SystemExit("Textual is required. Install with: scripts/py -m pip install textual rich")


CURATED_VIEWS: List[Tuple[str, str]] = [
    ("datasets", "SELECT * FROM datasets ORDER BY created_utc DESC"),
    ("dataset_lineage_current", "SELECT * FROM dataset_lineage_current ORDER BY child_dataset_id, relationship_type, parent_dataset_id"),
    ("training_sets", "SELECT * FROM training_sets ORDER BY created_utc DESC"),
    ("training_runs", "SELECT * FROM training_runs ORDER BY created_utc DESC"),
    ("training_models", "SELECT * FROM training_models ORDER BY created_utc DESC"),
    ("onnx_models", "SELECT * FROM onnx_models ORDER BY created_utc DESC"),
    ("tensorrt_models", "SELECT * FROM tensorrt_models ORDER BY created_utc DESC"),
    ("keypoint_quality_current", "SELECT * FROM keypoint_quality_current ORDER BY quality_updated_utc DESC"),
    ("detect_quality_current", "SELECT * FROM detect_quality_current ORDER BY quality_updated_utc DESC"),
    ("pose_skeleton_specs", "SELECT * FROM pose_skeleton_specs ORDER BY created_utc DESC"),
]


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
            dq = self.conn.execute(
                "SELECT refined_run, detect_method, review_state, interpolated_detections_rate FROM detect_quality_current WHERE dataset_id=?",
                (dataset_id,),
            ).fetchall()
            if kq:
                lines.append("keypoint_quality_current:")
                for r in kq[:5]:
                    lines.append(
                        f"  - run={r['refined_run']} method={r['keypoint_method']} state={r['review_state']} usable_rate={r['usable_keypoints_rate']}"
                    )
            if dq:
                lines.append("detect_quality_current:")
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
            yield Input(placeholder="Filter (substring across visible row values). Enter to apply.", id="filter_input")
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

    def _set_status(self, text: str) -> None:
        self.query_one("#status_text", Static).update(text)

    def _active_view_name_and_sql(self) -> Tuple[str, str]:
        label, payload = self.view_entries[self.active_view_index]
        if label.startswith("table:"):
            return label, f"TABLE::{payload}"
        return label, payload

    def _apply_filter(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        needle = self.filter_text.strip().lower()
        if not needle:
            return rows
        out: List[Dict[str, Any]] = []
        for row in rows:
            text = " | ".join("" if v is None else str(v) for v in row.values()).lower()
            if needle in text:
                out.append(row)
        return out

    def _load_active_view(self) -> None:
        label, sql_or_table = self._active_view_name_and_sql()
        if sql_or_table.startswith("TABLE::"):
            table_name = sql_or_table.replace("TABLE::", "", 1)
            rows = self.client.fetch_rows_for_name(table_name, limit=self.row_limit)
        else:
            rows = self.client.fetch_rows(sql_or_table, limit=self.row_limit)
        self.active_rows = rows
        self.display_rows = self._apply_filter(rows)
        self._render_table()
        self._set_status(
            f"View={label} rows={len(self.display_rows)} (loaded={len(self.active_rows)}, limit={self.row_limit})"
        )

    def _render_table(self) -> None:
        table = self.query_one("#row_table", DataTable)
        table.clear(columns=True)
        if not self.display_rows:
            table.add_column("No rows")
            table.add_row(" ")
            self.query_one("#detail", Static).update("No rows matched.")
            return
        columns = list(self.display_rows[0].keys())
        for col in columns:
            table.add_column(str(col), key=str(col))
        for row in self.display_rows:
            table.add_row(*[("" if row.get(c) is None else str(row.get(c))) for c in columns])
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
        self.query_one("#detail", Static).update("\n".join(lines))

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        idx = int(event.list_view.index or 0)
        if idx < 0 or idx >= len(self.view_entries):
            return
        self.active_view_index = idx
        self._load_active_view()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.cursor_row is None:
            return
        self._update_details_for_row(int(event.cursor_row))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.cursor_row is None:
            return
        self._update_details_for_row(int(event.cursor_row))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "filter_input":
            return
        self.filter_text = str(event.value or "")
        self.display_rows = self._apply_filter(self.active_rows)
        self._render_table()
        self._set_status(
            f"Filter applied: '{self.filter_text}' | rows={len(self.display_rows)} loaded={len(self.active_rows)}"
        )

    def action_refresh(self) -> None:
        self._load_active_view()

    def action_focus_filter(self) -> None:
        self.query_one("#filter_input", Input).focus()

    def action_clear_filter(self) -> None:
        self.filter_text = ""
        inp = self.query_one("#filter_input", Input)
        inp.value = ""
        self.display_rows = list(self.active_rows)
        self._render_table()
        self._set_status("Filter cleared.")

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

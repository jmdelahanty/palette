#!/usr/bin/env python3
"""Format multiple detection compute-smoke JSON reports side by side."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import check_detect_compute_smoke


def _get_nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _as_int(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _fmt_float(value: Any, digits: int = 2) -> str:
    number = _as_float(value)
    return f"{number:.{digits}f}" if number is not None else "-"


def _fmt_int(value: Any) -> str:
    number = _as_int(value)
    return str(number) if number is not None else "-"


def _short_path(path: Path) -> str:
    name = path.name
    if len(name) <= 44:
        return name
    return f"{name[:20]}...{name[-20:]}"


def _row_from_payload(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    failures = check_detect_compute_smoke._validate_payload(payload)  # noqa: SLF001
    return {
        "path": str(path),
        "file": _short_path(path),
        "ok": not failures,
        "failures": failures,
        "status": payload.get("status"),
        "backend": _get_nested(payload, "inputs", "decode_backend"),
        "backend_requested": _get_nested(payload, "inputs", "decode_backend_requested"),
        "pipeline": _get_nested(payload, "inputs", "pipeline_mode") or "sequential",
        "timing_policy": _get_nested(payload, "inputs", "timing_policy"),
        "device": _get_nested(payload, "inputs", "device"),
        "host": _get_nested(payload, "cluster", "HOSTNAME"),
        "job": _get_nested(payload, "cluster", "LSB_JOBID"),
        "frames": _get_nested(payload, "summary", "frames_processed"),
        "batches": _get_nested(payload, "summary", "batches_processed"),
        "detections": _get_nested(payload, "summary", "detections_total"),
        "total_s": _get_nested(payload, "summary", "total_seconds"),
        "e2e_fps": _get_nested(payload, "summary", "end_to_end_fps"),
        "inference_fps": _get_nested(payload, "summary", "inference_fps"),
        "steady_inference_fps": _get_nested(
            payload,
            "summary",
            "steady_state_excluding_first_batch",
            "inference_fps",
        ),
        "video_open_s": _get_nested(payload, "stages", "video_open_seconds"),
        "decode_s": _get_nested(payload, "summary", "decode_seconds_total"),
        "preprocess_s": _get_nested(payload, "summary", "preprocess_seconds_total"),
        "predict_return_s": _get_nested(
            payload, "summary", "predict_return_seconds_total"
        ),
        "first_predict_s": _get_nested(
            payload, "summary", "first_batch", "predict_return_seconds"
        ),
    }


def _load_rows(paths: Iterable[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in paths:
        expanded = path.expanduser()
        try:
            payload = json.loads(expanded.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"{expanded}: failed reading JSON: {exc}")
            continue
        if not isinstance(payload, dict):
            errors.append(f"{expanded}: JSON root is not an object")
            continue
        rows.append(_row_from_payload(expanded, payload))
    return rows, errors


def _sort_rows(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    if key == "input":
        return list(rows)
    reverse = key in {"e2e_fps", "inference_fps", "steady_inference_fps"}

    def _sort_value(row: dict[str, Any]) -> tuple[int, Any]:
        value = row.get(key)
        if key in {"backend", "pipeline", "host", "job", "file"}:
            return (0, str(value or ""))
        number = _as_float(value)
        if number is None:
            return (1, 0.0)
        return (0, number)

    return sorted(rows, key=_sort_value, reverse=reverse)


def _render_markdown(rows: list[dict[str, Any]]) -> str:
    headers = [
        "ok",
        "backend",
        "pipeline",
        "frames",
        "total_s",
        "e2e_fps",
        "steady_inf_fps",
        "open_s",
        "decode_s",
        "pre_s",
        "predict_s",
        "first_predict_s",
        "host",
        "job",
        "file",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = [
            "yes" if row["ok"] else "no",
            str(row.get("backend") or "-"),
            str(row.get("pipeline") or "-"),
            _fmt_int(row.get("frames")),
            _fmt_float(row.get("total_s")),
            _fmt_float(row.get("e2e_fps")),
            _fmt_float(row.get("steady_inference_fps")),
            _fmt_float(row.get("video_open_s")),
            _fmt_float(row.get("decode_s")),
            _fmt_float(row.get("preprocess_s")),
            _fmt_float(row.get("predict_return_s")),
            _fmt_float(row.get("first_predict_s")),
            str(row.get("host") or "-"),
            str(row.get("job") or "-"),
            str(row.get("file") or "-"),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a side-by-side report from detect compute-smoke JSON files."
    )
    parser.add_argument("json_paths", nargs="+", type=Path, help="Compute-smoke JSON reports.")
    parser.add_argument(
        "--sort",
        choices=(
            "input",
            "backend",
            "e2e_fps",
            "inference_fps",
            "steady_inference_fps",
            "total_s",
            "video_open_s",
        ),
        default="e2e_fps",
        help="Sort output rows (default: e2e_fps descending).",
    )
    parser.add_argument("--json", action="store_true", help="Emit rows as JSON.")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    rows, errors = _load_rows(args.json_paths)
    rows = _sort_rows(rows, args.sort)
    if args.json:
        print(json.dumps({"rows": rows, "errors": errors}, indent=2, sort_keys=True))
    else:
        if rows:
            print(_render_markdown(rows))
        if errors:
            if rows:
                print()
            print("Errors:", file=sys.stderr)
            for error in errors:
                print(f"- {error}", file=sys.stderr)
    failed_rows = [row for row in rows if not row["ok"]]
    return 1 if errors or failed_rows else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Report and manifest output helpers for Palette web labeling."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Mapping, Sequence


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))

def _csv_export_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)

def _write_optional_json_report(payload: Mapping[str, object], output: str | None, *, overwrite: bool, description: str) -> None:
    if not output:
        return
    output_path = Path(output)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing {description}: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

def _write_row_export(
    *,
    payload: dict[str, object],
    rows: list[dict[str, object]],
    output: str | None,
    output_format: str,
    overwrite: bool,
) -> dict[str, object]:
    if output_format == "json":
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    elif output_format == "jsonl":
        text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    elif output_format == "csv":
        import csv
        import io

        fieldnames: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(str(key))
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_export_value(row.get(key)) for key in fieldnames})
        text = buffer.getvalue()
    else:
        raise ValueError(f"Unsupported export format: {output_format}")

    summary = {
        "ok": True,
        "count": len(rows),
        "format": output_format,
        "output": output,
        "filters": payload.get("filters", {}),
    }
    if output:
        output_path = Path(output)
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing export: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        return summary
    print(text, end="")
    return summary

from __future__ import annotations

import csv
import json
from pathlib import Path

from fisheye.utils.audit_zarr_group_counts import (
    audit_archive,
    build_report,
    discover_analysis_zarrs,
    main,
)


def _write_zarr_json(path: Path, *, node_type: str = "group") -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": node_type, "attributes": {}}),
        encoding="utf-8",
    )


def _make_analysis_zarr(path: Path) -> Path:
    _write_zarr_json(path)
    _write_zarr_json(path / "analysis")
    _write_zarr_json(path / "analysis" / "swim_bout_runs")
    _write_zarr_json(path / "analysis" / "swim_bout_runs" / "bouts_a")
    _write_zarr_json(path / "analysis" / "swim_bout_runs" / "bouts_a" / "speed_exponential")
    _write_zarr_json(
        path / "analysis" / "swim_bout_runs" / "bouts_a" / "speed_exponential" / "start_frame",
        node_type="array",
    )
    _write_zarr_json(
        path / "analysis" / "swim_bout_runs" / "bouts_a" / "speed_exponential" / "end_frame",
        node_type="array",
    )
    _write_zarr_json(path / "analysis" / "bout_kinematics_runs")
    _write_zarr_json(path / "analysis" / "bout_kinematics_runs" / "bk_a")
    _write_zarr_json(
        path / "analysis" / "bout_kinematics_runs" / "bk_a" / "net_delta_heading_deg",
        node_type="array",
    )
    (path / "analysis" / "bad_metadata").mkdir(parents=True)
    (path / "analysis" / "bad_metadata" / "zarr.json").write_text("{bad", encoding="utf-8")
    return path


def test_audit_archive_counts_groups_arrays_and_prefixes(tmp_path: Path) -> None:
    zarr_path = _make_analysis_zarr(tmp_path / "rec_a_analysis.zarr")

    archive, families, components = audit_archive(zarr_path)

    assert archive.zarr_json_count == 11
    assert archive.group_count == 7
    assert archive.array_count == 3
    assert archive.invalid_json_count == 1
    family_counts = {row.prefix: row.zarr_json_count for row in families}
    assert family_counts["analysis/swim_bout_runs"] == 5
    assert family_counts["analysis/bout_kinematics_runs"] == 3
    assert family_counts["analysis/bad_metadata"] == 1
    component_counts = {row.prefix: row.zarr_json_count for row in components}
    assert component_counts["analysis/swim_bout_runs/bouts_a"] == 4
    assert component_counts["analysis/bout_kinematics_runs/bk_a"] == 2


def test_discover_build_report_and_write_outputs(tmp_path: Path, capsys) -> None:
    root = tmp_path / "recordings"
    zarr_path = _make_analysis_zarr(
        root / "rec_a" / "zarr" / "rec_a_analysis.zarr"
    )
    _write_zarr_json(root / "rec_a" / "zarr" / "rec_a_training.zarr")
    output_dir = tmp_path / "audit"

    discovered = discover_analysis_zarrs([root])
    assert discovered == [zarr_path.resolve()]

    report = build_report(discovered)
    assert report["totals"]["archive_count"] == 1
    assert report["totals"]["zarr_json_count"] == 11
    assert report["global_families"][0]["prefix"] == "analysis/swim_bout_runs"

    assert main(
        [
            "--recordings-root",
            str(root),
            "--output-dir",
            str(output_dir),
            "--format",
            "json",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["totals"]["archive_count"] == 1
    assert (output_dir / "audit_summary.json").is_file()
    assert (output_dir / "audit_summary.md").is_file()
    assert (output_dir / "archive_summary.csv").is_file()
    assert (output_dir / "family_summary.csv").is_file()
    assert (output_dir / "component_summary.csv").is_file()

    with (output_dir / "archive_summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["zarr_json_count"] == "11"
    assert rows[0]["top_family_prefix"] == "analysis/swim_bout_runs"

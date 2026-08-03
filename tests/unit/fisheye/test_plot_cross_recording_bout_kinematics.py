from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from fisheye.analytics_exports.contracts import (
    EXPORT_SCHEMA_ID,
    EXPORT_SCHEMA_VERSION,
    TABLE_CONTRACTS,
    contract_snapshot,
)
from fisheye.analytics_exports.publication import sha256_file
from fisheye.registry.db import Registry
from fisheye.utils.plot_cross_recording_bout_kinematics import (
    BOUT_KINEMATICS_TABLE,
    plot_export,
)


def _write_parquet_export(root: Path, export_run_id: str) -> None:
    generation_path = (
        Path("v1")
        / ".generations"
        / f"export_run_id={export_run_id}"
        / "generation=test"
    )
    table_dir = root / generation_path / "tables" / BOUT_KINEMATICS_TABLE
    table_dir.mkdir(parents=True)
    rows = [
        {
            "recording_id": "rec_a",
            "measurement_level": "heading_smoothed",
            "stimulus_mode": "MOVING_GRATING",
            "net_delta_heading_deg": 10.0,
            "abs_net_delta_heading_deg": 10.0,
            "within_heading_path_deg": 15.0,
            "within_heading_peak_to_peak_deg": 12.0,
            "within_angular_speed_mean_deg_s": 90.0,
            "within_angular_speed_max_deg_s": 250.0,
        },
        {
            "recording_id": "rec_a",
            "measurement_level": "heading_smoothed",
            "stimulus_mode": "MOVING_GRATING",
            "net_delta_heading_deg": -20.0,
            "abs_net_delta_heading_deg": 20.0,
            "within_heading_path_deg": 30.0,
            "within_heading_peak_to_peak_deg": 25.0,
            "within_angular_speed_mean_deg_s": 120.0,
            "within_angular_speed_max_deg_s": 300.0,
        },
        {
            "recording_id": "rec_a",
            "measurement_level": "heading_raw",
            "stimulus_mode": "MOVING_GRATING",
            "net_delta_heading_deg": 99.0,
            "abs_net_delta_heading_deg": 99.0,
            "within_heading_path_deg": 99.0,
            "within_heading_peak_to_peak_deg": 99.0,
            "within_angular_speed_mean_deg_s": 99.0,
            "within_angular_speed_max_deg_s": 99.0,
        },
    ]
    table = pa.Table.from_pylist(rows).replace_schema_metadata(
        {
            b"palette.export_schema_id": EXPORT_SCHEMA_ID.encode(),
            b"palette.export_schema_version": str(EXPORT_SCHEMA_VERSION).encode(),
            b"palette.table_contract": json.dumps(
                TABLE_CONTRACTS[BOUT_KINEMATICS_TABLE].to_dict(),
                sort_keys=True,
                separators=(",", ":"),
            ).encode(),
        }
    )
    part = table_dir / "part-00000.parquet"
    pq.write_table(table, part)
    manifest_dir = root / "v1" / "manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / f"export_run_id={export_run_id}.json").write_text(
        json.dumps(
            {
                "export_run_id": export_run_id,
                "schema_id": EXPORT_SCHEMA_ID,
                "schema_version": EXPORT_SCHEMA_VERSION,
                "created_at_utc": "2026-05-08T00:00:00Z",
                "tables_requested": [BOUT_KINEMATICS_TABLE],
                "table_contracts": contract_snapshot([BOUT_KINEMATICS_TABLE]),
                "row_counts_by_table": {BOUT_KINEMATICS_TABLE: len(rows)},
                "part_files_by_table": {
                    BOUT_KINEMATICS_TABLE: [
                        (
                            generation_path
                            / "tables"
                            / BOUT_KINEMATICS_TABLE
                            / part.name
                        ).as_posix()
                    ]
                },
                "capabilities": [],
                "publication": {
                    "schema_id": "palette.analytics_export.publication",
                    "schema_version": 1,
                    "state": "complete",
                    "generation_id": "test",
                    "generation_path": generation_path.as_posix(),
                    "parts_by_table": {
                        BOUT_KINEMATICS_TABLE: [
                            {
                                "path": (
                                    generation_path
                                    / "tables"
                                    / BOUT_KINEMATICS_TABLE
                                    / part.name
                                ).as_posix(),
                                "sha256": sha256_file(part),
                                "size_bytes": part.stat().st_size,
                                "row_count": len(rows),
                            }
                        ]
                    },
                },
            }
        )
        + "\n"
    )


def _index_parquet_export(registry_path: Path, root: Path, export_run_id: str) -> None:
    registry = Registry(registry_path)
    try:
        registry.upsert_analytics_collection(
            collection_id="collection_test",
            manifest_sha256="abc123",
            manifest_path=root / "collection.manifest.json",
            collection_name="Collection Test",
            record_count=1,
            included_record_count=1,
        )
        registry.upsert_analytics_export(
            export_run_id=export_run_id,
            collection_id="collection_test",
            collection_manifest_sha256="abc123",
            export_manifest_path=root / "v1" / "manifests" / f"export_run_id={export_run_id}.json",
            output_root=root,
            source_recording_count=1,
            row_counts_by_table={"bout_kinematics_metrics": 3},
            part_files_by_table={
                "bout_kinematics_metrics": [
                    str(
                        root
                        / "v1"
                        / ".generations"
                        / f"export_run_id={export_run_id}"
                        / "generation=test"
                        / "tables"
                        / BOUT_KINEMATICS_TABLE
                        / "part-00000.parquet"
                    )
                ]
            },
            created_at_utc="2026-05-08T00:00:00Z",
        )
    finally:
        registry.close()


def test_plot_cross_recording_bout_kinematics_writes_artifacts(tmp_path: Path) -> None:
    export_root = tmp_path / "exports"
    output_dir = tmp_path / "plots"
    _write_parquet_export(export_root, "run_test")

    summary = plot_export(
        export_root=export_root,
        export_run_id="run_test",
        parquet_root=(
            export_root
            / "v1"
            / ".generations"
            / "export_run_id=run_test"
            / "generation=test"
            / "tables"
            / BOUT_KINEMATICS_TABLE
        ),
        output_dir=output_dir,
        measurement_level="heading_smoothed",
        dpi=60,
    )

    assert summary["export_run_id"] == "run_test"
    assert summary["measurement_level"] == "heading_smoothed"
    assert summary["row_count"] == 2
    assert summary["metrics"]["net_delta_heading_deg"]["n"] == 2
    assert summary["metrics"]["net_delta_heading_deg"]["median"] == -5.0
    assert summary["net_delta_by_mode"]["MOVING_GRATING"]["n"] == 2

    expected = {
        "plot_summary.json",
        "swim_bout_heading_summary.tsv",
        "swim_bout_heading_histograms_overall.png",
        "swim_bout_net_heading_by_stimulus.png",
        "swim_bout_angular_speed_histograms.png",
    }
    assert expected == {path.name for path in output_dir.iterdir()}
    for path in output_dir.iterdir():
        assert path.stat().st_size > 0

    payload = json.loads((output_dir / "plot_summary.json").read_text())
    assert payload["source"] == "parquet:bout_kinematics_metrics"


def test_plot_cross_recording_bout_kinematics_resolves_registry_export(tmp_path: Path) -> None:
    export_root = tmp_path / "exports"
    output_dir = tmp_path / "plots"
    registry_path = tmp_path / "registry.sqlite"
    _write_parquet_export(export_root, "run_test")
    _index_parquet_export(registry_path, export_root, "run_test")

    summary = plot_export(
        export_root=tmp_path / "unused_exports",
        export_run_id="latest",
        output_dir=output_dir,
        registry_path=registry_path,
        collection_id="collection_test",
        measurement_level="heading_smoothed",
        dpi=60,
    )

    assert summary["export_run_id"] == "run_test"
    assert summary["collection_id"] == "collection_test"
    assert summary["registry_path"] == str(registry_path.resolve())
    assert summary["row_count"] == 2
    assert summary["table_path"] == str(
        (
            export_root
            / "v1"
            / ".generations"
            / "export_run_id=run_test"
            / "generation=test"
            / "tables"
            / BOUT_KINEMATICS_TABLE
        ).resolve()
    )


def test_plot_cross_recording_bout_kinematics_uses_strict_publication_v1(
    tmp_path: Path,
) -> None:
    export_root = tmp_path / "exports"
    output_dir = tmp_path / "strict-plots"
    _write_parquet_export(export_root, "run_strict")

    summary = plot_export(
        export_root=export_root,
        export_run_id="latest",
        output_dir=output_dir,
        measurement_level="heading_smoothed",
        dpi=60,
    )

    assert summary["export_run_id"] == "run_strict"
    assert "/.generations/export_run_id=run_strict/" in summary["table_path"]
    assert summary["row_count"] == 2

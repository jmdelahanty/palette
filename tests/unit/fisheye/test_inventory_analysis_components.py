from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from fisheye.utils.inventory_analysis_components import (
    build_cohort_inventory,
    inventory_archive,
    load_registry_targets,
    render_markdown,
)


def _write_node(
    path: Path,
    *,
    node_type: str = "group",
    attrs: dict[str, object] | None = None,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "zarr_format": 3,
        "node_type": node_type,
        "attributes": attrs or {},
    }
    if node_type == "array":
        payload.update(
            {
                "shape": [1],
                "data_type": "uint8",
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [1]},
                },
                "chunk_key_encoding": {
                    "name": "default",
                    "configuration": {"separator": "/"},
                },
                "fill_value": 0,
                "codecs": [],
            }
        )
    (path / "zarr.json").write_text(json.dumps(payload), encoding="utf-8")


def _make_archive(tmp_path: Path, name: str, *, include_cra: bool) -> Path:
    zarr_path = tmp_path / f"{name}_analysis.zarr"
    _write_node(
        zarr_path,
        attrs={"recording_id": name, "dataset_id": f"dataset-{name}"},
    )
    _write_node(zarr_path / "analysis")
    family = zarr_path / "analysis" / "chaser_distance_runs"
    _write_node(family, attrs={"latest": f"chaser_{name}"})
    run = family / f"chaser_{name}"
    _write_node(
        run,
        attrs={
            "schema_id": "palette.chaser_distance.v1",
            "schema_version": 1,
            "method": "offline_detection_to_chaser_distance",
            "palette_run_completion_contract": "palette.zarr_run_completion.v1",
            "palette_run_completion_status": "complete",
            "source_refs": {"refined_detect": "refined_detect_runs/refined_1"},
        },
    )
    _write_node(run / "distances")
    _write_node(run / "distances" / "distance_mm", node_type="array")
    # Traversal must stop at arrays even if a malformed child directory exists.
    _write_node(run / "distances" / "distance_mm" / "c" / "fake_group")

    bout_parent = run / "chaser_bout_response"
    _write_node(bout_parent, attrs={"latest": "bout_v1"})
    _write_node(
        bout_parent / "bout_v1",
        attrs={
            "schema_id": "palette.chaser_bout_response.v1",
            "method": "chaser_relative_bout_kinematics_with_virtual_controls",
            "palette_run_completion_status": "complete",
        },
    )
    if include_cra:
        cra_parent = run / "cra_primary_endpoint"
        _write_node(cra_parent, attrs={"latest": "goodcopbadcop_cra_v1"})
        _write_node(
            cra_parent / "goodcopbadcop_cra_v1",
            attrs={
                "schema_id": "palette.goodcopbadcop.cra_primary_endpoint.v1",
                "method": "goodcopbadcop_object_relative_pre_post_endpoint",
                "palette_run_completion_status": "complete",
            },
        )
    return zarr_path


def _group_by_path(inventory: dict[str, object], path: str) -> dict[str, object]:
    for group in inventory["groups"]:  # type: ignore[index]
        if group["node_path"] == path:
            return group
    raise AssertionError(f"missing group {path}")


def test_inventory_archive_is_schema_first_and_stops_at_arrays(tmp_path: Path) -> None:
    zarr_path = _make_archive(tmp_path, "GoodCopBadCop_recording", include_cra=True)

    inventory = inventory_archive(
        zarr_path,
        branding_tokens=["GoodCopBadCop"],
    )

    run_path = "analysis/chaser_distance_runs/chaser_GoodCopBadCop_recording"
    run = _group_by_path(inventory, run_path)
    assert run["role"] == "run"
    assert run["run_family_path"] == "analysis/chaser_distance_runs"
    assert run["completion_status"] == "complete"
    assert run["selected_by_parent_pointers"] == ["latest"]
    assert run["source"]["source_refs"] == {
        "refined_detect": "refined_detect_runs/refined_1"
    }
    assert run["protocol_branding_fields"] == ["path"]

    cra_path = f"{run_path}/cra_primary_endpoint/goodcopbadcop_cra_v1"
    cra = _group_by_path(inventory, cra_path)
    assert cra["role"] == "component_group"
    assert cra["component_family_path"] == (
        "analysis/chaser_distance_runs/*/cra_primary_endpoint"
    )
    assert cra["selected_by_parent_pointers"] == ["latest"]
    assert cra["protocol_branding_fields"] == ["schema_id", "method", "path"]
    assert all("fake_group" not in group["node_path"] for group in inventory["groups"])


def test_build_cohort_inventory_aggregates_physical_and_schema_coverage(
    tmp_path: Path,
) -> None:
    first = _make_archive(tmp_path, "GoodCopBadCop_a", include_cra=True)
    second = _make_archive(tmp_path, "GoodCopBadCop_b", include_cra=False)

    inventory = build_cohort_inventory(
        [
            {"recording_id": "GoodCopBadCop_a", "zarr_path": str(first)},
            {"recording_id": "GoodCopBadCop_b", "zarr_path": str(second)},
        ],
        branding_tokens=["GoodCopBadCop"],
    )

    assert inventory["summary"]["archive_count"] == 2
    assert inventory["summary"]["archive_error_count"] == 0
    family = next(
        row
        for row in inventory["run_families"]
        if row["run_family_path"] == "analysis/chaser_distance_runs"
    )
    assert family["recording_count"] == 2
    assert "palette.chaser_distance.v1" in family["schema_ids"]

    cra = next(
        row
        for row in inventory["component_families"]
        if row["component_family_path"]
        == "analysis/chaser_distance_runs/*/cra_primary_endpoint"
    )
    assert cra["recording_count"] == 1
    assert cra["schema_protocol_branded"] is True

    generic_schema = next(
        row
        for row in inventory["schemas"]
        if row["schema_id"] == "palette.chaser_bout_response.v1"
    )
    assert generic_schema["recording_count"] == 2
    assert generic_schema["schema_protocol_branded"] is False
    assert "## Declared schemas" in render_markdown(inventory)


def test_load_registry_targets_is_read_only_and_filters_active_analysis(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "registry.sqlite"
    connection = sqlite3.connect(registry)
    connection.execute(
        """
        CREATE TABLE dataset_context_current (
            dataset_id TEXT,
            recording_id TEXT,
            zarr_path TEXT,
            dataset_status TEXT,
            zarr_use TEXT,
            protocol_name TEXT
        )
        """
    )
    connection.executemany(
        "INSERT INTO dataset_context_current VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("a", "rec-a", str(tmp_path / "a.zarr"), "active", "analysis", "ChaserA"),
            ("b", "rec-b", str(tmp_path / "b.zarr"), "missing", "analysis", "ChaserA"),
            ("c", "rec-c", str(tmp_path / "c.zarr"), "active", "training", "ChaserA"),
            ("d", "rec-d", str(tmp_path / "d.zarr"), "active", "analysis", "Other"),
        ],
    )
    connection.commit()
    before = registry.read_bytes()
    connection.close()

    targets = load_registry_targets(registry, protocol_name="ChaserA")

    assert targets == [
        {
            "dataset_id": "a",
            "recording_id": "rec-a",
            "zarr_path": str((tmp_path / "a.zarr").resolve()),
        }
    ]
    assert registry.read_bytes() == before

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fisheye.shared.run_resolution import LEGACY_DETECT_REVIEW_AUTHORITY_ATTR
from fisheye.shared.zarr_run_completion import (
    AUTHORITATIVE_RUN_ATTR,
    COMPLETION_EPOCH_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
)
from fisheye.utils import backfill_detect_review_authoritative_run as mod


def _write_group(path: Path, attrs: dict[str, object] | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": attrs or {},
    }
    (path / "zarr.json").write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _read_attrs(path: Path) -> dict[str, object]:
    payload = json.loads((path / "zarr.json").read_text(encoding="utf-8"))
    attrs = payload.get("attributes")
    assert isinstance(attrs, dict)
    return attrs


def _write_store(root: Path, name: str) -> Path:
    zarr_path = root / f"{name}.zarr"
    _write_group(zarr_path)
    return zarr_path


def _write_backfillable_parent(zarr_path: Path, run_name: str = "reviewed") -> Path:
    parent = zarr_path / "refined_detect_runs"
    _write_group(
        parent,
        {
            LEGACY_DETECT_REVIEW_AUTHORITY_ATTR: run_name,
            "latest": run_name,
            COMPLETION_EPOCH_ATTR: 1,
        },
    )
    _write_group(parent / run_name, {RUN_COMPLETION_STATUS_ATTR: RUN_STATUS_COMPLETE})
    return parent


def test_dry_run_is_default_and_does_not_mutate_fixture_store(tmp_path: Path) -> None:
    zarr_path = _write_store(tmp_path, "backfillable")
    parent = _write_backfillable_parent(zarr_path)

    report = mod.backfill_stores([zarr_path])

    assert report.dry_run is True
    assert report.planned_mutation_count == 1
    assert report.applied_mutation_count == 0
    attrs = _read_attrs(parent)
    assert AUTHORITATIVE_RUN_ATTR not in attrs
    assert attrs[LEGACY_DETECT_REVIEW_AUTHORITY_ATTR] == "reviewed"


def test_execute_requires_explicit_store_list() -> None:
    with pytest.raises(SystemExit) as excinfo:
        mod.main(["--execute"])

    assert excinfo.value.code == 2


def test_execute_backfills_only_backfillable_store_and_keeps_legacy_pointer(tmp_path: Path) -> None:
    zarr_path = _write_store(tmp_path, "backfillable")
    parent = _write_backfillable_parent(zarr_path)

    report = mod.backfill_stores([zarr_path], execute=True)

    assert report.dry_run is False
    assert report.applied_mutation_count == 1
    attrs = _read_attrs(parent)
    assert attrs[AUTHORITATIVE_RUN_ATTR] == "reviewed"
    assert attrs[LEGACY_DETECT_REVIEW_AUTHORITY_ATTR] == "reviewed"


def test_execute_skips_safe_store_without_mutation(tmp_path: Path) -> None:
    zarr_path = _write_store(tmp_path, "safe")
    parent = _write_backfillable_parent(zarr_path)
    attrs = _read_attrs(parent)
    attrs[AUTHORITATIVE_RUN_ATTR] = "reviewed"
    payload = json.loads((parent / "zarr.json").read_text(encoding="utf-8"))
    payload["attributes"] = attrs
    (parent / "zarr.json").write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    report = mod.backfill_stores([zarr_path], execute=True)

    assert report.applied_mutation_count == 0
    assert report.stores[0].plan.bucket == "SAFE"
    assert _read_attrs(parent)[AUTHORITATIVE_RUN_ATTR] == "reviewed"


def test_execute_skips_ambiguous_store_without_mutation(tmp_path: Path) -> None:
    zarr_path = _write_store(tmp_path, "ambiguous")
    parent = zarr_path / "refined_detect_runs"
    _write_group(
        parent,
        {
            LEGACY_DETECT_REVIEW_AUTHORITY_ATTR: "reviewed",
            "latest": "newer",
            COMPLETION_EPOCH_ATTR: 1,
        },
    )
    _write_group(parent / "reviewed", {RUN_COMPLETION_STATUS_ATTR: RUN_STATUS_COMPLETE})
    _write_group(parent / "newer", {RUN_COMPLETION_STATUS_ATTR: RUN_STATUS_COMPLETE})

    report = mod.backfill_stores([zarr_path], execute=True)

    assert report.applied_mutation_count == 0
    assert report.stores[0].plan.bucket == "AMBIGUOUS"
    assert AUTHORITATIVE_RUN_ATTR not in _read_attrs(parent)


def test_no_reader_retirement_behavior_legacy_attr_and_reader_modules_unchanged(tmp_path: Path) -> None:
    zarr_path = _write_store(tmp_path, "backfillable")
    parent = _write_backfillable_parent(zarr_path)

    mod.backfill_stores([zarr_path], execute=True)

    attrs = _read_attrs(parent)
    assert attrs[LEGACY_DETECT_REVIEW_AUTHORITY_ATTR] == "reviewed"
    assert mod.LEGACY_DETECT_REVIEW_AUTHORITY_ATTR == "detect_review_status_latest"

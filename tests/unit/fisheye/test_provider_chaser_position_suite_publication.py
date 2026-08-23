from __future__ import annotations

import json
from pathlib import Path

import pytest
import zarr

from fisheye.analysis_workflows import (
    provider_chaser_position_suite_publication as publication_module,
)
from fisheye.analysis_workflows.provider_chaser_position_suite_publication import (
    MAX_MANIFEST_BYTES,
    build_provider_chaser_position_suite_publication_plan,
    deep_audit_provider_chaser_position_suite_run,
    load_provider_chaser_position_suite_source_handle,
    publish_provider_chaser_position_suite_run,
)
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.zarr_io import open_zarr_root
from tests.unit.fisheye.test_provider_chaser_position_suite import _result


def _source_bindings() -> dict[str, object]:
    return {
        "provider_chaser_distance": {
            "run_name": "provider-distance-v1",
            "run_path": "analysis/provider_chaser_distance_runs/provider-distance-v1",
            "manifest_sha256": "a" * 64,
            "source_receipt_sha256": "b" * 64,
            "verification_mode": "bounded_publication",
            "source_position_provider": {
                "provider_id": "detection_bbox_centroid.v1",
                "provider_digest": "c" * 64,
            },
        },
        "relative_frame": {"record_sha256": "d" * 64},
        "epoch_candidate": {"run_path": "analysis/stimulus_epoch_runs/epochs-v1"},
        "epoch_selection": {"selection_sha256": "e" * 64},
        "arena_geometry_and_scale": {"selection_sha256": "f" * 64},
        "provider_physical_frame": {"record_sha256": "1" * 64},
        "recording_physical_frame": {"record_sha256": "2" * 64},
        "physical_frame_equivalence": {"policy_id": "equivalent-v1"},
        "source_camera_to_arena_mm_transform": {"record_sha256": "3" * 64},
    }


def _report(archive: Path) -> dict[str, object]:
    return {
        "schema_id": "palette.provider_chaser_position_suite_canary",
        "schema_version": 1,
        "disposition": "selector_ineligible_operational_canary",
        "status": "computed_read_only",
        "recording_id": "recording-fixture",
        "analysis_zarr": str(archive.resolve()),
        "selector_eligible": False,
        "selection": "none",
        "production_authority": False,
        "registry_update": False,
        "source_bindings": _source_bindings(),
        "temporal_alignment": {
            "temporal_alignment_class": "controller_input_provenance_proxy",
            "physical_presentation_verified": False,
            "timestamp_matching_performed": False,
        },
        "temporal_caveat": "Controller-input provenance proxy; presentation is unverified.",
        "suite": json_attr_safe(_result()),
    }


def _plan(tmp_path: Path, *, run_name: str = "position-suite-v1"):
    archive = tmp_path / "analysis.zarr"
    zarr.open_group(str(archive), mode="w", zarr_format=3, use_consolidated=False)
    report = _report(archive)
    plan = build_provider_chaser_position_suite_publication_plan(
        archive,
        report=report,
        run_name=run_name,
        expected_recording_id="recording-fixture",
    )
    return archive, report, plan


def test_dry_run_reveals_typed_tables_without_creating_target(tmp_path: Path) -> None:
    archive, _report_value, plan = _plan(tmp_path)

    result = plan.to_json()

    assert result["status"] == "dry_run_plan"
    assert result["table_row_counts"]["per_epoch_chaser_metrics"] == 4
    assert result["array_count"] > 7
    assert result["selector_eligible"] is False
    assert result["selection"] == "none"
    assert result["registry_update"] is False
    assert result["target_exists"] is False
    assert result["manifest_bytes"] < MAX_MANIFEST_BYTES
    assert not (archive / plan.run_path).exists()


def test_manifest_keeps_row_evidence_out_of_attributes(tmp_path: Path) -> None:
    _archive, _report_value, plan = _plan(tmp_path)

    manifest = dict(plan.manifest)

    assert set(manifest["table_contracts"]) == {
        "epoch_roles",
        "per_epoch_chaser_metrics",
        "distance_cdf",
        "radial_occupancy",
        "quadrant_joint_occupancy",
        "role_contrasts",
        "role_radial_contrasts",
    }
    assert not set(manifest["suite_metadata"]).intersection(manifest["table_contracts"])
    assert manifest["row_evidence_storage"] == "typed_zarr_arrays_not_attributes_v1"
    assert len(json.dumps(manifest)) < MAX_MANIFEST_BYTES


def test_float_nulls_use_explicit_validity_and_strings_use_registries(
    tmp_path: Path,
) -> None:
    _archive, _report_value, plan = _plan(tmp_path)

    median = plan.prepared.arrays[
        "per_epoch_chaser_metrics__near_zone_complete_visit_median_dwell_s"
    ]
    median_valid = plan.prepared.arrays[
        "per_epoch_chaser_metrics__near_zone_complete_visit_median_dwell_s_valid"
    ]
    role_codes = plan.prepared.arrays["epoch_roles__analysis_role"]

    assert median.dtype.str == "<f8"
    assert median_valid.dtype == bool
    assert (~median_valid).any()
    assert role_codes.dtype.str == "<i4"
    assert plan.manifest["value_registries"]["epoch_roles.analysis_role"]["values"] == [
        "pre",
        "training",
    ]


def test_publication_is_selector_ineligible_and_reader_decodes_exact_rows(
    tmp_path: Path,
) -> None:
    archive, report, plan = _plan(tmp_path)

    result = publish_provider_chaser_position_suite_run(
        plan, scratch_root=tmp_path / "scratch"
    )
    handle = load_provider_chaser_position_suite_source_handle(
        archive,
        run_name=plan.run_name,
        expected_recording_id="recording-fixture",
    )

    assert result["status"] == "published_selector_ineligible"
    assert result["selector_eligible"] is False
    assert result["selection"] == "none"
    assert result["production_authority"] is False
    assert result["registry_update"] is False
    assert list(handle.table_rows("epoch_roles")) == report["suite"]["epoch_roles"]
    assert (
        list(handle.table_rows("role_contrasts")) == report["suite"]["role_contrasts"]
    )
    assert all(not value.flags.writeable for value in handle.arrays.values())
    root = open_zarr_root(archive, mode="r", use_consolidated=False)
    parent = root["analysis/provider_chaser_position_suite_runs"]
    assert not set(parent.attrs).intersection({"latest", "selected", "current"})


def test_atomic_receipt_contains_only_bounded_array_evidence(tmp_path: Path) -> None:
    archive, _report_value, plan = _plan(tmp_path)
    publish_provider_chaser_position_suite_run(plan, scratch_root=tmp_path / "scratch")

    root = open_zarr_root(archive, mode="r", use_consolidated=False)
    run = root[plan.run_path]
    publication = dict(run.attrs["cluster_output_staging"])
    for field in (
        "local_validation",
        "temporary_validation",
        "pre_pointer_validation",
        "final_validation",
    ):
        validation = publication[field]
        assert "arrays" not in validation
        assert validation["array_count"] == len(plan.prepared.arrays)
        assert validation["array_path_count"] == len(plan.prepared.arrays)
        assert len(validation["array_paths_sha256"]) == 64
        assert validation["readable_array_declarations"].endswith(".array_declarations")
        assert validation["row_evidence_storage"] == (
            "typed_zarr_arrays_not_publication_metadata_v1"
        )
    assert len(json.dumps(publication)) < 30_000


def test_bounded_reader_does_not_hash_and_deep_audit_is_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive, _report_value, plan = _plan(tmp_path)
    publish_provider_chaser_position_suite_run(plan, scratch_root=tmp_path / "scratch")

    def unexpected_hash(_value: object) -> str:
        raise AssertionError("bounded persistent reader must not hash arrays")

    monkeypatch.setattr(publication_module, "array_values_sha256", unexpected_hash)
    bounded = load_provider_chaser_position_suite_source_handle(
        archive,
        run_name=plan.run_name,
        expected_recording_id="recording-fixture",
    )
    assert bounded.verification_mode == "bounded_publication"

    with pytest.raises(AssertionError, match="must not hash"):
        deep_audit_provider_chaser_position_suite_run(
            archive,
            run_name=plan.run_name,
            expected_recording_id="recording-fixture",
        )


def test_plan_fails_closed_for_missing_authority_or_selector_name(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "analysis.zarr"
    zarr.open_group(str(archive), mode="w", zarr_format=3, use_consolidated=False)
    report = _report(archive)
    del report["source_bindings"]["epoch_selection"]

    with pytest.raises(ValueError, match="source bindings are incomplete"):
        build_provider_chaser_position_suite_publication_plan(
            archive, report=report, run_name="position-suite-v1"
        )

    with pytest.raises(ValueError, match="concrete bare run name"):
        build_provider_chaser_position_suite_publication_plan(
            archive, report=_report(archive), run_name="latest"
        )

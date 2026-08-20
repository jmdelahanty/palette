from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.analysis.provider_occupancy_v2 import (
    OccupancyGrid,
    OccupancyTimingPolicy,
    ProviderOccupancySamples,
    calculate_provider_occupancy_v2,
)
from fisheye.analysis_workflows.materializers.provider_occupancy_v2 import (
    PROVIDER_OCCUPANCY_PARENT_PATH,
    ProviderOccupancyV2MaterializationError,
    materialize_provider_occupancy_v2,
    plan_provider_occupancy_v2_run,
    publish_provider_occupancy_v2_run,
    validate_provider_occupancy_v2_run,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr_io import open_zarr_root


def _bindings(*, config_digest: str | None = None) -> dict[str, dict[str, object]]:
    values: dict[str, dict[str, object]] = {}
    for name in (
        "trajectory",
        "compiled_selection",
        "provider",
        "timing",
        "geometry",
        "transform",
        "fixed_grid_policy",
    ):
        record: dict[str, object] = {
            "schema_id": f"test.{name}",
            "schema_version": 1,
            "immutable_id": f"{name}-run-v1",
        }
        if name == "timing":
            record.update({"fps_hz": 10.0, "timing_policy_id": "valid_in_grid_sample_count_divided_by_fps_v1"})
        if name == "fixed_grid_policy" and config_digest is not None:
            record["config_digest"] = config_digest
        values[name] = {
            "record": record,
            "sha256": canonical_json_sha256(record),
        }
    return values


def _result(*, empty: bool = False):
    if empty:
        samples = ProviderOccupancySamples(
            x_mm=np.asarray([], dtype=np.float64),
            y_mm=np.asarray([], dtype=np.float64),
            selected=np.asarray([], dtype=bool),
            provider_present=np.asarray([], dtype=bool),
            provider_valid=np.asarray([], dtype=bool),
            transform_valid=np.asarray([], dtype=bool),
            occurrence_ids=[],
            expected_occurrence_ids=[],
        )
    else:
        samples = ProviderOccupancySamples(
            x_mm=np.asarray([0.0, 2.0, 1.0], dtype=np.float64),
            y_mm=np.asarray([0.0, 2.0, 1.0], dtype=np.float64),
            selected=np.asarray([True, True, True], dtype=bool),
            provider_present=np.asarray([True, True, True], dtype=bool),
            provider_valid=np.asarray([True, True, True], dtype=bool),
            transform_valid=np.asarray([True, True, True], dtype=bool),
            occurrence_ids=[("first", "overlap"), ("first",), ("overlap",)],
            # Four expected frames: one selected frame is absent from the
            # source rowset and must remain in the expected denominator.
            expected_occurrence_ids=[
                ("first", "overlap"),
                ("first",),
                ("overlap",),
                ("first",),
            ],
        )
    result = calculate_provider_occupancy_v2(
        samples,
        OccupancyGrid([0.0, 1.0, 2.0], [0.0, 1.0, 2.0]),
        OccupancyTimingPolicy(10.0),
    )
    return result


def _archive(tmp_path: Path) -> Path:
    path = tmp_path / "analysis.zarr"
    root = zarr.open_group(str(path), mode="w-", zarr_format=3, use_consolidated=False)
    parent = root.require_group(PROVIDER_OCCUPANCY_PARENT_PATH)
    parent.attrs["latest"] = "preexisting-run"
    parent.attrs["authoritative_run"] = "preexisting-run"
    return path


def _published(tmp_path: Path, *, empty: bool = False):
    archive = _archive(tmp_path)
    result = _result(empty=empty)
    bindings = _bindings(config_digest=result.config_digest)
    output = materialize_provider_occupancy_v2(
        archive,
        result,
        bindings,
        scratch_root=tmp_path / "scratch",
        run_name="occupancy_canary_v1",
        apply=True,
        keep_scratch=True,
    )
    return archive, result, bindings, output


def test_materializer_preserves_expected_denominator_and_overlap(tmp_path: Path) -> None:
    archive, _result_value, _bindings_value, output = _published(tmp_path)
    run_path = f"{PROVIDER_OCCUPANCY_PARENT_PATH}/occupancy_canary_v1"
    root = open_zarr_root(archive, mode="r", use_consolidated=True)
    run = root[run_path]

    assert run["pooled/expected_selected_frames"][:].tolist() == [4]
    assert run["pooled/provider_present_count"][:].tolist() == [3]
    assert run["pooled/counts"][:].sum() == 3
    assert run["per_occurrence/expected_selected_frames"][:].tolist() == [3, 2]
    assert run["per_occurrence/counts"][:].sum(axis=(1, 2)).tolist() == [2, 2]
    conservation = run.attrs["provider_occupancy_v2_manifest"]["payload"][
        "conservation"
    ]
    assert conservation["per_occurrence_count"] == 2
    assert conservation["per_occurrence_count_sum_array"] == (
        "per_occurrence/counts"
    )
    assert "per_occurrence_count_sums" not in conservation
    assert "per_occurrence_valid_in_grid_sample_counts" not in conservation
    assert output["acceptance"]["consolidated_validation"]["valid"] is True
    assert output["acceptance"]["direct_consolidated"]["declarations_sha256"]

    parent = root[PROVIDER_OCCUPANCY_PARENT_PATH]
    assert parent.attrs["palette_completion_epoch"] == 2
    assert parent.attrs["latest"] == "preexisting-run"
    assert parent.attrs["authoritative_run"] == "preexisting-run"

    for node_path in (
        "occurrence_id_offsets",
        "occurrence_id_utf8",
        "per_occurrence/counts",
        "pooled/counts",
    ):
        assert run[node_path].dtype.kind != "O"


def test_complete_run_retains_exact_manifest_and_rejects_selector_attribute(
    tmp_path: Path,
) -> None:
    archive, result, bindings, _output = _published(tmp_path)
    run_path = f"{PROVIDER_OCCUPANCY_PARENT_PATH}/occupancy_canary_v1"
    root = open_zarr_root(archive, mode="a", use_consolidated=False)
    run = root[run_path]
    assert run.attrs["palette_run_completion_status"] == "complete"
    assert run.attrs["provider_occupancy_v2_manifest"]["payload"]["status"] == "complete"
    run.attrs["latest_materialized"] = "occupancy_canary_v1"
    with pytest.raises(ProviderOccupancyV2MaterializationError, match="Invalid provider occupancy run"):
        validate_provider_occupancy_v2_run(
            archive,
            run_path,
            result=result,
            source_bindings=bindings,
            use_consolidated=False,
        )


def test_final_grid_edges_are_inclusive(tmp_path: Path) -> None:
    result = _result()
    assert result.pooled.counts[-1, -1] == 2
    assert result.pooled.valid_in_grid_sample_count == 3


def test_empty_selection_persists_nan_fraction_policy(tmp_path: Path) -> None:
    archive, _result_value, _bindings_value, _output = _published(tmp_path, empty=True)
    run_path = f"{PROVIDER_OCCUPANCY_PARENT_PATH}/occupancy_canary_v1"
    root = open_zarr_root(archive, mode="r", use_consolidated=True)
    run = root[run_path]
    assert run["per_occurrence/counts"].shape[0] == 0
    assert np.isnan(run["pooled/occupancy_fraction"][:]).all()
    assert run["pooled/valid_in_grid_sample_count"][:].tolist() == [0]


def test_retry_rejects_existing_target(tmp_path: Path) -> None:
    archive, result, bindings, _output = _published(tmp_path)
    with pytest.raises(FileExistsError, match="existing occupancy run"):
        plan_provider_occupancy_v2_run(
            archive,
            result,
            bindings,
            scratch_root=tmp_path / "scratch-second",
            run_name="occupancy_canary_v1",
        )


def test_mutated_result_is_rejected_before_publication(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    result = _result()
    bindings = _bindings(config_digest=result.config_digest)
    plan = plan_provider_occupancy_v2_run(
        archive,
        result,
        bindings,
        scratch_root=tmp_path / "scratch",
        run_name="occupancy_canary_v1",
    )
    mutated = result.pooled.counts.copy()
    mutated[0, 0] += 1
    object.__setattr__(result.pooled, "counts", mutated)
    with pytest.raises(ProviderOccupancyV2MaterializationError, match="mutated"):
        publish_provider_occupancy_v2_run(plan)


def test_plan_rejects_binding_tamper(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    result = _result()
    bindings = _bindings(config_digest=result.config_digest)
    bindings["provider"]["record"]["immutable_id"] = "tampered"
    with pytest.raises(ProviderOccupancyV2MaterializationError, match="digest"):
        plan_provider_occupancy_v2_run(
            archive,
            result,
            bindings,
            scratch_root=tmp_path / "scratch",
            run_name="occupancy_canary_v1",
        )

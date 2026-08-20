from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from fisheye.analysis.provider_occupancy_contrast import compute_occupancy_contrast
from fisheye.analysis.provider_occupancy_v2 import (
    OccupancyGrid,
    OccupancyTimingPolicy,
    ProviderOccupancySamples,
    calculate_provider_occupancy_v2,
)
from fisheye.analysis_workflows.materializers.provider_occupancy_contrast import (
    MANIFEST_ATTR,
    PARENT_PATH,
    SOURCE_SCOPE_POOLED,
    ProviderOccupancyContrastMaterializationError,
    build_pooled_occupancy_contrast_summary,
    build_provider_occupancy_contrast_materialization_plan,
    publish_provider_occupancy_contrast_run,
)
from fisheye.analysis_workflows.materializers.provider_occupancy_v2 import (
    PROVIDER_OCCUPANCY_MANIFEST_ATTR,
    PROVIDER_OCCUPANCY_MANIFEST_DIGEST_ATTR,
    PROVIDER_OCCUPANCY_PARENT_PATH,
    plan_provider_occupancy_v2_run,
    publish_provider_occupancy_v2_run,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_helpers import consolidate_metadata_capture_expected_warnings


def _bindings(
    selection_id: str,
    *,
    config_digest: str,
    track_policy_id: str = "provider_track_policy_v1",
    trajectory_run_id: str | None = None,
) -> dict[str, dict[str, object]]:
    values: dict[str, dict[str, object]] = {}
    trajectory_run_id = trajectory_run_id or f"trajectory-{selection_id}"
    source_rows_sha256 = "1" * 64
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
            "schema_id": f"test.provider_occupancy.{name}",
            "schema_version": 1,
            "immutable_id": (
                trajectory_run_id
                if name == "trajectory"
                else f"{name}-authority-v1"
            ),
            "recording_id": "recording-occupancy-v1",
            "subject_id": "subject-occupancy-v1",
        }
        if name == "trajectory":
            record.update(
                {
                    "trajectory_run_id": trajectory_run_id,
                    "source_rows_sha256": source_rows_sha256,
                    "position_track_policy": {
                        "policy_id": track_policy_id,
                        "row_axis": "provider_track_samples",
                        "source_rows_sha256": source_rows_sha256,
                        "provider_id": "detection",
                        "recording_id": "recording-occupancy-v1",
                    },
                    "sample_unit": {
                        "sample_unit_id": "one_provider_track_sample_v1",
                        "row_axis": "provider_track_samples",
                    },
                }
            )
        if name == "compiled_selection":
            record["selection_id"] = selection_id
        if name == "provider":
            record["provider_id"] = "detection"
            record["estimator_id"] = "detection-centroid-v1"
        if name == "timing":
            record.update(
                {
                    "fps_hz": 10.0,
                    "timing_policy_id": "valid_in_grid_sample_count_divided_by_fps_v1",
                }
            )
        if name == "fixed_grid_policy":
            record.update(
                {
                    "config_digest": config_digest,
                    "grid_id": "arena-mm-grid-v1",
                    "x_edges": [0.0, 1.0, 2.0],
                    "y_edges": [0.0, 1.0, 2.0],
                }
            )
        if name == "transform":
            record["coordinate_frame"] = {
                "frame_id": "arena_mm_v1",
                "space_id": "arena_relative_mm",
            }
        values[name] = {
            "record": record,
            "sha256": canonical_json_sha256(record),
        }
    return values


def _result(*, extra_sample: bool = False) -> object:
    x_mm = [0.0, 2.0, 1.0]
    y_mm = [0.0, 2.0, 1.0]
    occurrence_ids = [("first", "overlap"), ("first",), ("overlap",)]
    if extra_sample:
        x_mm.append(0.0)
        y_mm.append(0.0)
        occurrence_ids.append(("first",))
    samples = ProviderOccupancySamples(
        x_mm=np.asarray(x_mm, dtype=np.float64),
        y_mm=np.asarray(y_mm, dtype=np.float64),
        selected=np.ones(len(x_mm), dtype=bool),
        provider_present=np.ones(len(x_mm), dtype=bool),
        provider_valid=np.ones(len(x_mm), dtype=bool),
        transform_valid=np.ones(len(x_mm), dtype=bool),
        occurrence_ids=occurrence_ids,
        expected_occurrence_ids=[
            ("first", "overlap"),
            ("first",),
            ("overlap",),
            ("first",),
        ],
    )
    return calculate_provider_occupancy_v2(
        samples,
        OccupancyGrid([0.0, 1.0, 2.0], [0.0, 1.0, 2.0]),
        OccupancyTimingPolicy(10.0),
    )


def _archive(tmp_path: Path) -> Path:
    import zarr

    archive = tmp_path / "analysis.zarr"
    root = zarr.open_group(str(archive), mode="w-", zarr_format=3, use_consolidated=False)
    parent = root.require_group(PROVIDER_OCCUPANCY_PARENT_PATH)
    parent.attrs["latest"] = "preexisting-occupancy"
    parent.attrs["authoritative_run"] = "preexisting-occupancy"
    contrast_parent = root.require_group(PARENT_PATH)
    contrast_parent.attrs["latest"] = "preexisting-contrast"
    contrast_parent.attrs["latest_complete"] = "preexisting-contrast"
    return archive


def _publish_occupancy_source(
    archive: Path,
    tmp_path: Path,
    *,
    run_name: str,
    selection_id: str,
) -> tuple[str, str]:
    result = _result(extra_sample=run_name == "occupancy_treatment")
    bindings = _bindings(selection_id, config_digest=result.config_digest)
    plan = plan_provider_occupancy_v2_run(
        archive,
        result,
        bindings,
        scratch_root=tmp_path / f"scratch-{run_name}",
        run_name=run_name,
    )
    publish_provider_occupancy_v2_run(plan, keep_scratch=False)
    run_path = f"{PROVIDER_OCCUPANCY_PARENT_PATH}/{run_name}"
    root = open_zarr_root(archive, mode="r", use_consolidated=True)
    digest = root[run_path].attrs[PROVIDER_OCCUPANCY_MANIFEST_DIGEST_ATTR]
    return run_path, digest


def _fixture(tmp_path: Path) -> tuple[Path, str, str, str, str]:
    archive = _archive(tmp_path)
    baseline_path, baseline_digest = _publish_occupancy_source(
        archive,
        tmp_path,
        run_name="occupancy_baseline",
        selection_id="black-before-v1",
    )
    treatment_path, treatment_digest = _publish_occupancy_source(
        archive,
        tmp_path,
        run_name="occupancy_treatment",
        selection_id="chaser-v1",
    )
    return archive, baseline_path, baseline_digest, treatment_path, treatment_digest


def _contrast_fixture(tmp_path: Path):
    archive, baseline_path, baseline_digest, treatment_path, treatment_digest = _fixture(tmp_path)
    baseline = build_pooled_occupancy_contrast_summary(
        archive,
        run_path=baseline_path,
        manifest_sha256=baseline_digest,
        arm_role="baseline",
        source_scope=SOURCE_SCOPE_POOLED,
    )
    treatment = build_pooled_occupancy_contrast_summary(
        archive,
        run_path=treatment_path,
        manifest_sha256=treatment_digest,
        arm_role="treatment",
        source_scope=SOURCE_SCOPE_POOLED,
    )
    contrast = compute_occupancy_contrast(
        baseline,
        treatment,
        config={"selection_policy": "same-provider-v2-pooled-v1"},
    )
    return archive, baseline_path, baseline_digest, treatment_path, treatment_digest, contrast


def _plan(tmp_path: Path):
    archive, baseline_path, baseline_digest, treatment_path, treatment_digest, contrast = _contrast_fixture(tmp_path)
    plan = build_provider_occupancy_contrast_materialization_plan(
        archive,
        baseline_run_path=baseline_path,
        treatment_run_path=treatment_path,
        baseline_manifest_digest=baseline_digest,
        treatment_manifest_digest=treatment_digest,
        contrast_result=contrast,
        run_name="occupancy-contrast-v1",
        scratch_root=tmp_path / "contrast-scratch",
        source_scope=SOURCE_SCOPE_POOLED,
        software_record={"git_sha": "test", "materializer": "test"},
    )
    return archive, plan, contrast


def test_true_occupancy_v2_integration_round_trip(tmp_path: Path) -> None:
    archive, plan, contrast = _plan(tmp_path)
    published = publish_provider_occupancy_contrast_run(plan)
    run_path = "analysis/provider_occupancy_contrast_runs/occupancy-contrast-v1"
    root = open_zarr_root(archive, mode="r", use_consolidated=True)
    run = root[run_path]
    assert published["status"] == "complete"
    assert run.attrs["stage_selector_eligible"] is False
    assert root[PARENT_PATH].attrs["palette_completion_epoch"] == 2
    assert isinstance(run.attrs["run_provenance"], dict)
    assert run.attrs["source_scope"] == SOURCE_SCOPE_POOLED
    assert "occupancy_fraction" not in run
    np.testing.assert_array_equal(
        run["occupancy_fraction_difference"][:],
        contrast["occupancy_fraction_difference"],
    )
    assert run["occupancy_fraction_difference"].dtype == np.dtype("<f8")
    assert run["x_edges"].dtype == np.dtype("<f8")
    assert run["y_edges"].dtype == np.dtype("<f8")
    assert run.attrs[MANIFEST_ATTR]["payload"]["source_scope"] == SOURCE_SCOPE_POOLED
    manifest_payload = run.attrs[MANIFEST_ATTR]["payload"]
    assert "x_edges" not in manifest_payload["grid"]
    assert "y_edges" not in manifest_payload["grid"]
    assert manifest_payload["grid"]["x_edges_array"]["array_path"] == "x_edges"
    assert manifest_payload["grid"]["y_edges_array"]["array_path"] == "y_edges"
    for arm in ("baseline", "treatment"):
        occurrence_evidence = manifest_payload["source_arm_records"][arm][
            "occurrence_evidence"
        ]
        assert occurrence_evidence["count"] > 0
        assert occurrence_evidence["offsets_array"]["array_path"] == (
            "occurrence_id_offsets"
        )
        assert "source_occurrences" not in manifest_payload["source_arm_records"][arm]
    assert run.attrs["direct_consolidated_metadata_equality"]["array_count"] == 3
    assert validate_direct_consolidated_subtree(archive, subtree_path=run_path).array_count == 3
    for arm in ("baseline", "treatment"):
        source_identity = contrast["source_arms"][arm]["source_manifest"]
        assert "payload" not in source_identity
        assert source_identity["sha256"] == source_identity["manifest_sha256"]
        assert source_identity["manifest_attr"] == PROVIDER_OCCUPANCY_MANIFEST_ATTR
        assert source_identity["occupancy_payload_digest"] in {
            plan.baseline_manifest_digest,
            plan.treatment_manifest_digest,
        }


def test_compact_contrast_reader_accepts_explicit_v1_occupancy_source(
    tmp_path: Path,
) -> None:
    (
        archive,
        baseline_path,
        baseline_digest,
        treatment_path,
        treatment_digest,
    ) = _fixture(tmp_path)
    root = open_zarr_root(archive, mode="a", use_consolidated=False)
    baseline_run = root[baseline_path]
    baseline_manifest = deepcopy(
        baseline_run.attrs[PROVIDER_OCCUPANCY_MANIFEST_ATTR]
    )
    baseline_run.attrs["schema_version"] = 1
    baseline_manifest["schema_version"] = 1
    baseline_run.attrs[PROVIDER_OCCUPANCY_MANIFEST_ATTR] = baseline_manifest
    consolidate_metadata_capture_expected_warnings(archive)

    baseline = build_pooled_occupancy_contrast_summary(
        archive,
        run_path=baseline_path,
        manifest_sha256=baseline_digest,
        arm_role="baseline",
    )
    treatment = build_pooled_occupancy_contrast_summary(
        archive,
        run_path=treatment_path,
        manifest_sha256=treatment_digest,
        arm_role="treatment",
    )
    contrast = compute_occupancy_contrast(
        baseline,
        treatment,
        config={"selection_policy": "mixed-publication-schema-compatibility-v1"},
    )
    plan = build_provider_occupancy_contrast_materialization_plan(
        archive,
        baseline_run_path=baseline_path,
        treatment_run_path=treatment_path,
        baseline_manifest_digest=baseline_digest,
        treatment_manifest_digest=treatment_digest,
        contrast_result=contrast,
        run_name="occupancy-contrast-mixed-source-schema-v1",
        scratch_root=tmp_path / "contrast-scratch",
        source_scope=SOURCE_SCOPE_POOLED,
    )

    assert plan.manifest["payload"]["source_runs"]["baseline"][
        "run_schema_version"
    ] == 1
    assert plan.manifest["payload"]["source_runs"]["treatment"][
        "run_schema_version"
    ] == 2
    assert "source_manifest_bindings" not in plan.manifest["payload"]


def test_selector_like_child_attribute_is_rejected(tmp_path: Path) -> None:
    archive, plan, _contrast = _plan(tmp_path)
    publish_provider_occupancy_contrast_run(plan)
    run = open_zarr_root(plan.source_zarr, mode="a", use_consolidated=False)[plan.run_path]
    run.attrs["latest_materialized"] = plan.run_name
    from fisheye.analysis_workflows.materializers import provider_occupancy_contrast as materializer

    with pytest.raises(ProviderOccupancyContrastMaterializationError, match="selector attributes"):
        materializer._validate_run(
            plan.source_zarr.joinpath(*plan.run_path.split("/")),
            expected_manifest=plan.manifest,
        )


def test_compact_source_references_are_persisted_without_recursive_manifests(
    tmp_path: Path,
) -> None:
    archive, plan, _contrast = _plan(tmp_path)
    source_root_before = open_zarr_root(archive, mode="r", use_consolidated=True)
    source_manifests_before = {
        arm: deepcopy(source_root_before[path].attrs[PROVIDER_OCCUPANCY_MANIFEST_ATTR])
        for arm, path in (
            ("baseline", plan.baseline_run_path),
            ("treatment", plan.treatment_run_path),
        )
    }
    publish_provider_occupancy_contrast_run(plan)
    root = open_zarr_root(archive, mode="r", use_consolidated=True)
    payload = root["analysis/provider_occupancy_contrast_runs/occupancy-contrast-v1"].attrs[MANIFEST_ATTR]["payload"]
    assert "source_manifest_bindings" not in payload
    for arm, path in (("baseline", plan.baseline_run_path), ("treatment", plan.treatment_run_path)):
        source_manifest = root[path].attrs[PROVIDER_OCCUPANCY_MANIFEST_ATTR]
        source_run = payload["source_runs"][arm]
        assert source_run["run_path"] == path
        assert source_run["manifest_sha256"] == source_manifest["payload_digest"]
        assert source_run["source_bindings_sha256"] == source_manifest["payload"][
            "source_bindings_sha256"
        ]
        assert payload["source_arm_records"][arm]["source_manifest"][
            "manifest_attr"
        ] == PROVIDER_OCCUPANCY_MANIFEST_ATTR
        assert "payload" not in payload["source_arm_records"][arm][
            "source_manifest"
        ]
        assert source_manifest == source_manifests_before[arm]
    assert root[PARENT_PATH].attrs["latest"] == "preexisting-contrast"


def test_selection_and_trajectory_runs_differ_but_stable_policies_match(
    tmp_path: Path,
) -> None:
    archive, plan, _contrast = _plan(tmp_path)
    root = open_zarr_root(archive, mode="r", use_consolidated=True)
    baseline_bindings = root[plan.baseline_run_path].attrs[
        PROVIDER_OCCUPANCY_MANIFEST_ATTR
    ]["payload"]["source_bindings"]
    treatment_bindings = root[plan.treatment_run_path].attrs[
        PROVIDER_OCCUPANCY_MANIFEST_ATTR
    ]["payload"]["source_bindings"]
    assert (
        baseline_bindings["compiled_selection"]
        != treatment_bindings["compiled_selection"]
    )
    assert baseline_bindings["trajectory"] != treatment_bindings["trajectory"]
    assert (
        baseline_bindings["trajectory"]["record"]["position_track_policy"]
        == treatment_bindings["trajectory"]["record"]["position_track_policy"]
    )
    assert (
        baseline_bindings["transform"]["record"]["coordinate_frame"]
        == treatment_bindings["transform"]["record"]["coordinate_frame"]
    )
    assert (
        plan.source_evidence["baseline"]["payload"]["conservation"]
        != plan.source_evidence["treatment"]["payload"]["conservation"]
    )
    assert (
        plan.source_evidence["baseline"]["summary"]["denominator"]
        == plan.source_evidence["treatment"]["summary"]["denominator"]
    )
    assert (
        plan.source_evidence["baseline"]["summary"]["normalization"]
        == plan.source_evidence["treatment"]["summary"]["normalization"]
    )
    published = publish_provider_occupancy_contrast_run(plan)
    assert published["status"] == "complete"


def test_mismatched_underlying_position_policy_fails_closed(tmp_path: Path) -> None:
    archive, baseline_path, baseline_digest, treatment_path, treatment_digest, contrast = _contrast_fixture(tmp_path)
    root = open_zarr_root(archive, mode="a", use_consolidated=False)
    run = root[baseline_path]
    manifest = deepcopy(run.attrs[PROVIDER_OCCUPANCY_MANIFEST_ATTR])
    payload = manifest["payload"]
    bindings = deepcopy(payload["source_bindings"])
    trajectory_record = deepcopy(bindings["trajectory"]["record"])
    trajectory_record["position_track_policy"]["policy_id"] = "different_track_policy_v1"
    bindings["trajectory"] = {
        "record": trajectory_record,
        "sha256": canonical_json_sha256(trajectory_record),
    }
    payload["source_bindings"] = bindings
    payload["source_bindings_sha256"] = canonical_json_sha256(bindings)
    manifest["payload_digest"] = canonical_json_sha256(payload)
    run.attrs[PROVIDER_OCCUPANCY_MANIFEST_ATTR] = manifest
    run.attrs[PROVIDER_OCCUPANCY_MANIFEST_DIGEST_ATTR] = manifest["payload_digest"]
    consolidate_metadata_capture_expected_warnings(archive)

    fresh_baseline = build_pooled_occupancy_contrast_summary(
        archive,
        run_path=baseline_path,
        manifest_sha256=manifest["payload_digest"],
        arm_role="baseline",
        source_scope=SOURCE_SCOPE_POOLED,
    )
    stale_result = deepcopy(contrast)
    stale_result["source_arms"]["baseline"]["source_manifest"] = fresh_baseline[
        "source_manifest"
    ]
    with pytest.raises(
        ProviderOccupancyContrastMaterializationError,
        match="compatibility.position_track_policy",
    ):
        build_provider_occupancy_contrast_materialization_plan(
            archive,
            baseline_run_path=baseline_path,
            treatment_run_path=treatment_path,
            baseline_manifest_digest=manifest["payload_digest"],
            treatment_manifest_digest=treatment_digest,
            contrast_result=stale_result,
            run_name="mismatched-position-policy",
            scratch_root=tmp_path / "scratch-mismatched-policy",
            source_scope=SOURCE_SCOPE_POOLED,
        )


def test_stale_policy_and_explicit_nonpooled_scope_fail_closed(tmp_path: Path) -> None:
    archive, baseline_path, baseline_digest, treatment_path, treatment_digest, contrast = _contrast_fixture(tmp_path)
    stale = deepcopy(contrast)
    stale["policy_digest"] = "0" * 64
    with pytest.raises(ProviderOccupancyContrastMaterializationError, match="policy digest"):
        build_provider_occupancy_contrast_materialization_plan(
            archive,
            baseline_run_path=baseline_path,
            treatment_run_path=treatment_path,
            baseline_manifest_digest=baseline_digest,
            treatment_manifest_digest=treatment_digest,
            contrast_result=stale,
            run_name="stale-policy",
            scratch_root=tmp_path / "scratch",
            source_scope=SOURCE_SCOPE_POOLED,
        )
    with pytest.raises(ProviderOccupancyContrastMaterializationError, match="only explicit pooled"):
        build_provider_occupancy_contrast_materialization_plan(
            archive,
            baseline_run_path=baseline_path,
            treatment_run_path=treatment_path,
            baseline_manifest_digest=baseline_digest,
            treatment_manifest_digest=treatment_digest,
            contrast_result=contrast,
            run_name="per-occurrence",
            scratch_root=tmp_path / "scratch",
            source_scope="per_occurrence",
        )


def test_tampered_published_source_array_is_rejected(tmp_path: Path) -> None:
    archive, baseline_path, baseline_digest, treatment_path, treatment_digest, _contrast = _contrast_fixture(tmp_path)
    root = open_zarr_root(archive, mode="a", use_consolidated=False)
    values = root[baseline_path]["pooled/occupancy_fraction"][:]
    values[0, 0] = 0.75
    root[baseline_path]["pooled/occupancy_fraction"][:] = values
    with pytest.raises(ProviderOccupancyContrastMaterializationError, match="does not match its declared bytes"):
        build_pooled_occupancy_contrast_summary(
            archive,
            run_path=baseline_path,
            manifest_sha256=baseline_digest,
            arm_role="baseline",
            source_scope=SOURCE_SCOPE_POOLED,
        )


def test_stale_source_manifest_digest_and_existing_retry_are_rejected(tmp_path: Path) -> None:
    archive, plan, _contrast = _plan(tmp_path)
    with pytest.raises(ProviderOccupancyContrastMaterializationError, match="manifest digest"):
        build_provider_occupancy_contrast_materialization_plan(
            archive,
            baseline_run_path=plan.baseline_run_path,
            treatment_run_path=plan.treatment_run_path,
            baseline_manifest_digest="0" * 64,
            treatment_manifest_digest=plan.treatment_manifest_digest,
            contrast_result=plan.contrast_result,
            run_name="stale-source",
            scratch_root=tmp_path / "scratch-stale",
            source_scope=SOURCE_SCOPE_POOLED,
        )
    publish_provider_occupancy_contrast_run(plan)
    with pytest.raises(FileExistsError, match="existing immutable contrast run"):
        build_provider_occupancy_contrast_materialization_plan(
            archive,
            baseline_run_path=plan.baseline_run_path,
            treatment_run_path=plan.treatment_run_path,
            baseline_manifest_digest=plan.baseline_manifest_digest,
            treatment_manifest_digest=plan.treatment_manifest_digest,
            contrast_result=plan.contrast_result,
            run_name=plan.run_name,
            scratch_root=tmp_path / "scratch-retry",
            source_scope=SOURCE_SCOPE_POOLED,
        )

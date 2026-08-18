from __future__ import annotations

import numpy as np
import pytest
import zarr

from fisheye.analysis_workflows.materializers.single_subject_tracking import (
    plan_single_subject_tracking_run,
    publish_single_subject_tracking_run,
)
from fisheye.analysis_workflows.materializers.subject_position import (
    plan_subject_position_run,
    publish_subject_position_run,
)
from fisheye.analysis_workflows.subject_position_source_handle import (
    load_subject_position_source_handle,
)
from fisheye.analysis_workflows.tracking_source_handle import (
    load_tracking_source_handle,
)
from fisheye.shared.json_safety import json_attr_safe
from fisheye.tracking.run_manifest import TRACKING_RUN_MANIFEST_ATTR
from tests.unit.fisheye.test_subject_position_materializer import _prepared


def test_subject_position_bound_tracking_is_atomic_exact_and_nonpromoting(
    tmp_path,
) -> None:
    analysis_zarr = tmp_path / "analysis.zarr"
    root = zarr.open_group(
        str(analysis_zarr),
        mode="w-",
        zarr_format=3,
        use_consolidated=False,
    )
    tracking_parent = root.require_group("tracking_runs")
    tracking_parent.attrs.update(
        {
            "latest": "tracking_existing",
            "latest_complete": "tracking_existing",
            "authoritative_run": "tracking_existing",
        }
    )

    position_plan = plan_subject_position_run(
        analysis_zarr,
        _prepared(3),
        run_name="position_exact",
        scratch_root=tmp_path / "position-scratch",
    )
    publish_subject_position_run(position_plan)
    position = load_subject_position_source_handle(
        analysis_zarr,
        position_plan.run_path,
        expected_selector_eligible=False,
        use_consolidated=True,
    )

    tracking_plan = plan_single_subject_tracking_run(
        position,
        arena_id=7,
        run_name="tracking_position_exact",
        scratch_root=tmp_path / "tracking-scratch",
    )
    result = publish_single_subject_tracking_run(tracking_plan)
    handle = load_tracking_source_handle(
        analysis_zarr,
        tracking_plan.run_path,
        expected_selector_eligible=False,
        use_consolidated=True,
    )

    assert handle.instance_key.tolist() == [100, 101, 102]
    assert handle.frame_indices.tolist() == [0, 1, 2]
    assert handle.track_ids.tolist() == [0, 0, 0]
    assert handle.arena_ids.tolist() == [7, 7, 7]
    assert result["acceptance"]["verification_digest"] == (
        handle.verification_digest
    )

    direct = zarr.open_group(
        str(analysis_zarr),
        mode="r",
        zarr_format=3,
        use_consolidated=False,
    )
    parent = direct["tracking_runs"]
    assert parent.attrs["latest"] == "tracking_existing"
    assert parent.attrs["latest_complete"] == "tracking_existing"
    assert parent.attrs["authoritative_run"] == "tracking_existing"
    run = parent[tracking_plan.run_name]
    assert run.attrs["stage_selector_eligible"] is False
    source = run.attrs[TRACKING_RUN_MANIFEST_ATTR]["payload"]["source"]
    assert source["source_authority_kind"] == "subject_position_run"
    assert source["source_subject_position_run"] == position.run_path
    assert source["source_subject_position_manifest_sha256"] == (
        position.manifest_sha256
    )
    assert source["source_subject_position_decoded_content_sha256"] == (
        position.decoded_content_sha256
    )


def test_tracking_plan_rejects_duplicate_observations_in_one_frame(tmp_path) -> None:
    analysis_zarr = tmp_path / "analysis.zarr"
    zarr.open_group(
        str(analysis_zarr),
        mode="w-",
        zarr_format=3,
        use_consolidated=False,
    )
    prepared = _prepared(3)
    arrays = {name: np.array(value, copy=True) for name, value in prepared.arrays.items()}
    arrays["source_acquisition_frame_index"][:] = [0, 0, 1]
    duplicate = type(prepared)(
        arrays=arrays,
        estimator_record=json_attr_safe(prepared.estimator_record),
        estimator_sha256=prepared.estimator_sha256,
        anatomy_record=json_attr_safe(prepared.anatomy_record),
        anatomy_sha256=prepared.anatomy_sha256,
        source_record=json_attr_safe(prepared.source_record),
        source_sha256=prepared.source_sha256,
        policy_record=json_attr_safe(prepared.policy_record),
        policy_sha256=prepared.policy_sha256,
        software_record=json_attr_safe(prepared.software_record),
        software_sha256=prepared.software_sha256,
        coordinate_record=json_attr_safe(prepared.coordinate_record),
        coordinate_sha256=prepared.coordinate_sha256,
    )
    position_plan = plan_subject_position_run(
        analysis_zarr,
        duplicate,
        run_name="position_duplicate_frame",
        scratch_root=tmp_path / "position-scratch",
    )
    publish_subject_position_run(position_plan)
    position = load_subject_position_source_handle(
        analysis_zarr,
        position_plan.run_path,
        expected_selector_eligible=False,
        use_consolidated=True,
    )

    with pytest.raises(ValueError, match="multiple detections"):
        plan_single_subject_tracking_run(
            position,
            arena_id=0,
            run_name="tracking_duplicate_frame",
            scratch_root=tmp_path / "tracking-scratch",
        )

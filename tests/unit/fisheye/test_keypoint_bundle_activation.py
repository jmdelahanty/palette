from __future__ import annotations

from pathlib import Path

import pytest
import zarr

from fisheye.shared.run_provenance import build_run_provenance
from fisheye.shared.zarr.clipped_keypoint_finalization import (
    publish_selector_ineligible_clipped_keypoint_chain,
)
from fisheye.shared.zarr.keypoint_bundle_activation import (
    KEYPOINT_BUNDLE_AUTHORITY_ATTR,
    KEYPOINT_BUNDLE_AUTHORITY_GENERATION_ATTR,
    KEYPOINT_BUNDLE_AUTHORITY_LEASE_ATTR,
    KeypointBundleActivationError,
    activate_keypoint_bundle_from_plan,
    build_keypoint_bundle_activation_plan,
    validate_active_keypoint_bundle,
)
from fisheye.shared.zarr.keypoint_bundle_production_publication import (
    publish_keypoint_v2_production_candidate_chain,
)
from fisheye.shared.zarr.keypoint_publication_mode import (
    KEYPOINT_PUBLICATION_MODE_PRODUCTION_CANDIDATE,
    KeypointChainPublicationDispositions,
    KeypointPublicationDisposition,
)
from fisheye.shared.zarr.refined_keypoint_manifest import (
    initial_refined_keypoint_snapshot_identity,
)
from fisheye.shared.zarr_helpers import consolidate_metadata_capture_expected_warnings
from fisheye.shared.zarr_run_completion import (
    COMPLETION_EPOCH_STRICT,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
)
from tests.unit.fisheye.test_clipped_keypoint_finalization import (
    _clip_results,
    _crop,
    _pose_binding,
    _preprocessing,
)


def _candidate_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    crop = _crop(tmp_path)
    monkeypatch.setattr(
        "fisheye.shared.zarr.clipped_keypoint_finalization."
        "validate_crop_geometry_shadow_publication",
        lambda publication: (),
    )
    identity = initial_refined_keypoint_snapshot_identity(
        recording_identity="keypoint_v2_canary",
        lineage_id="33333333-3333-4333-8333-333333333333",
        snapshot_id="44444444-4444-4444-8444-444444444444",
    )
    provenance = build_run_provenance(
        command="pytest keypoint bundle activation",
        params={"selector_activation": "deferred"},
        input_run_ids={"crop": crop.run_id},
        cwd=Path.cwd(),
    )

    def disposition(owner: str) -> KeypointPublicationDisposition:
        return KeypointPublicationDisposition(
            mode=KEYPOINT_PUBLICATION_MODE_PRODUCTION_CANDIDATE,
            publication_owner_uuid=owner,
            run_provenance=provenance,
        )

    chain = publish_selector_ineligible_clipped_keypoint_chain(
        crop,
        _clip_results(crop),
        pose_model_schema_binding=_pose_binding(),
        preprocessing=_preprocessing(),
        bundle_root=tmp_path / "production",
        raw_run_id="raw_v2",
        quality_run_id="quality_v1",
        refined_run_id="refined_v2",
        body_frame_run_id="body_frame_v1",
        refined_identity=identity,
        created_by="pytest",
        dispositions=KeypointChainPublicationDispositions(
            raw=disposition("a" * 32),
            quality=disposition("b" * 32),
            refined=disposition("c" * 32),
            body_frame=disposition("d" * 32),
        ),
    )
    archive = tmp_path / "analysis.zarr"
    root = zarr.open_group(str(archive), mode="w", zarr_format=3)
    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["palette_completion_epoch"] = COMPLETION_EPOCH_STRICT
    crop_run = crop_parent.create_group(crop.run_id)
    crop_run.attrs.update(
        {
            "status": "complete",
            "stage_selector_eligible": False,
            RUN_COMPLETION_STATUS_ATTR: RUN_STATUS_COMPLETE,
            "run_manifest": crop.manifest,
        }
    )
    consolidate_metadata_capture_expected_warnings(archive)
    publish_keypoint_v2_production_candidate_chain(
        analysis_zarr=archive,
        chain=chain,
    )
    return archive


def _plan(archive: Path) -> dict[str, object]:
    return build_keypoint_bundle_activation_plan(
        archive,
        crop_run_id="crop_v2_source",
        raw_run_id="raw_v2",
        quality_run_id="quality_v1",
        refined_run_id="refined_v2",
        body_frame_run_id="body_frame_v1",
    )


def test_bundle_activation_is_one_root_authority_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _candidate_archive(tmp_path, monkeypatch)
    plan = _plan(archive)

    result = activate_keypoint_bundle_from_plan(plan)

    assert result["status"] == "activated"
    assert result["activation_performed"] is True
    authority = validate_active_keypoint_bundle(archive)
    assert authority == result["authority"]
    direct = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    consolidated = zarr.open_group(str(archive), mode="r", use_consolidated=True)
    assert direct.attrs[KEYPOINT_BUNDLE_AUTHORITY_ATTR] == authority
    assert consolidated.attrs[KEYPOINT_BUNDLE_AUTHORITY_ATTR] == authority
    assert direct.attrs[KEYPOINT_BUNDLE_AUTHORITY_GENERATION_ATTR] == 1
    for path in (
        "keypoints_runs/raw_v2",
        "keypoint_quality_runs/quality_v1",
        "refined_keypoints_runs/refined_v2",
        "analysis/body_frame_runs/body_frame_v1",
    ):
        assert direct[path].attrs["stage_selector_eligible"] is False
    for parent_path in (
        "keypoints_runs",
        "keypoint_quality_runs",
        "refined_keypoints_runs",
        "analysis/body_frame_runs",
    ):
        assert "latest" not in direct[parent_path].attrs
        assert "latest_complete" not in direct[parent_path].attrs

    repeated = activate_keypoint_bundle_from_plan(plan)
    assert repeated["status"] == "already_active"
    assert repeated["activation_performed"] is False
    assert direct.attrs[KEYPOINT_BUNDLE_AUTHORITY_GENERATION_ATTR] == 1


def test_bundle_activation_rejects_a_stale_reviewed_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _candidate_archive(tmp_path, monkeypatch)
    plan = _plan(archive)
    root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
    root.attrs[KEYPOINT_BUNDLE_AUTHORITY_GENERATION_ATTR] = 7
    consolidate_metadata_capture_expected_warnings(archive)

    with pytest.raises(KeypointBundleActivationError, match="stale"):
        activate_keypoint_bundle_from_plan(plan)

    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    assert KEYPOINT_BUNDLE_AUTHORITY_ATTR not in root.attrs
    assert root.attrs[KEYPOINT_BUNDLE_AUTHORITY_GENERATION_ATTR] == 7


def test_bundle_activation_rolls_back_its_owned_lease_before_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _candidate_archive(tmp_path, monkeypatch)
    plan = _plan(archive)

    def fail_before_commit(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("synthetic precommit failure")

    monkeypatch.setattr(
        "fisheye.shared.zarr.keypoint_bundle_activation._authority_from_plan",
        fail_before_commit,
    )
    with pytest.raises(RuntimeError, match="synthetic precommit failure"):
        activate_keypoint_bundle_from_plan(plan)

    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    assert KEYPOINT_BUNDLE_AUTHORITY_ATTR not in root.attrs
    assert KEYPOINT_BUNDLE_AUTHORITY_LEASE_ATTR not in root.attrs
    assert KEYPOINT_BUNDLE_AUTHORITY_GENERATION_ATTR not in root.attrs


def test_bundle_plan_rehashes_and_rejects_changed_logical_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _candidate_archive(tmp_path, monkeypatch)
    root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
    values = root["keypoints_runs/raw_v2/pose_success"]
    values[0] = not bool(values[0])

    with pytest.raises(KeypointBundleActivationError, match="logical array changed"):
        _plan(archive)

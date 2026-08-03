from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from fisheye.analysis import tail_posture_view_runs as mod
from fisheye.analysis import subject_shape_io
from fisheye.analysis.tail_posture_view_schema import (
    TAIL_POSTURE_VIEW_ARRAY_SCHEMA_ATTR,
    TAIL_POSTURE_VIEW_ARRAY_SCHEMA_DIGEST_ATTR,
    TailPostureViewDimensions,
    tail_posture_view_manifest_digest,
    validate_tail_posture_view_arrays,
)
from fisheye.shared.detect_reason_codec import decode_reason_bytes


def _patch_provenance(monkeypatch) -> None:
    monkeypatch.setattr(
        mod,
        "get_git_info",
        lambda repo_path=None: {  # noqa: ARG005
            "commit_hash": "d" * 40,
            "short_hash": "dddddddd",
            "branch": "main",
            "is_dirty": False,
            "remote_url": "git@example.com:palette.git",
        },
    )
    monkeypatch.setattr(
        subject_shape_io,
        "load_persisted_subject_shape_coordinate_publication",
        lambda _root, _run_path: SimpleNamespace(
            manifest=SimpleNamespace(record_sha256="a" * 64)
        ),
    )
    monkeypatch.setattr(
        mod,
        "get_environment_info",
        lambda **kwargs: {  # noqa: ARG005
            "environment": {"python": "3.11"},
            "platform": {
                "hostname": "posture-host",
                "system": "Linux",
                "release": "6.8",
                "python_version": "3.11.0",
                "machine": "x86_64",
            },
        },
    )
    monkeypatch.setattr(
        mod,
        "publish_tail_posture_coordinate_surfaces",
        lambda _root, run_group: run_group.attrs.__setitem__(
            "tail_coordinate_publication_manifest_sha256", "9" * 64
        ),
    )

    def _activate(
        _root,
        parent,
        run_group,
        *,
        run_name,
        expected_publication_owner_uuid,
        additional_selector_attrs=(),
    ):
        assert (
            expected_publication_owner_uuid
            == run_group.attrs[mod.TAIL_PUBLICATION_OWNER_ATTR]
        )
        assert run_group.attrs["palette_run_completion_status"] == "complete"
        assert run_group.attrs["stage_selector_eligible"] is False
        parent.attrs["latest_complete"] = run_name
        parent.attrs["latest"] = run_name
        for name in additional_selector_attrs:
            parent.attrs[name] = run_name
        run_group.attrs["stage_selector_eligible"] = True

    monkeypatch.setattr(mod, "activate_tail_coordinate_publication", _activate)


def _source_arrays() -> dict[str, np.ndarray]:
    source_s = np.linspace(0.0, 1.0, 4, dtype=np.float32)
    tail_xy = np.zeros((2, 4, 2), dtype=np.float32)
    tail_xy[:, :, 0] = -np.linspace(0.0, 10.0, 4, dtype=np.float32)[None, :]
    return {
        "source_tail_sample_s": source_s,
        "tail_sample_xy": tail_xy,
        "head_xy": np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        "tail_sample_valid": np.ones((2,), dtype=bool),
        "bspline_valid": np.ones((2,), dtype=bool),
    }


def test_cumulative_segment_angles_straight_tail_are_zero() -> None:
    sources = _source_arrays()

    batch = mod.compute_tail_posture_view_from_subject_shape_arrays(
        **sources, keypoint_count=11
    )

    assert batch.valid.tolist() == [True, True]
    assert batch.tail_keypoints_xy.shape == (2, 11, 2)
    assert batch.tail_angle_rad.shape == (2, 10)
    np.testing.assert_allclose(batch.tail_angle_rad, 0.0, atol=1e-7)
    np.testing.assert_allclose(batch.head_yaw_rad, 0.0, atol=1e-7)


def test_cumulative_segment_angles_follow_keypoint_turns() -> None:
    head_xy = np.asarray([[1.0, 0.0]], dtype=np.float32)
    tail_keypoints = np.asarray(
        [[[0.0, 0.0], [-1.0, 0.0], [-2.0, 1.0]]],
        dtype=np.float32,
    )

    angle, head_yaw, valid = mod.compute_cumulative_segment_angles_from_keypoints(
        head_xy=head_xy,
        tail_keypoints_xy=tail_keypoints,
    )

    assert valid.tolist() == [True]
    np.testing.assert_allclose(head_yaw, 0.0, atol=1e-7)
    np.testing.assert_allclose(angle[0, 0], 0.0, atol=1e-7)
    np.testing.assert_allclose(angle[0, 1], -math.pi / 4.0, atol=1e-7)


def test_invalid_rows_are_nan_and_preserve_failure_reason() -> None:
    sources = _source_arrays()
    sources["tail_sample_valid"] = np.asarray([True, False], dtype=bool)

    batch = mod.compute_tail_posture_view_from_subject_shape_arrays(
        **sources,
        tail_sample_failure_reason=np.asarray(["ok", "scratch_artifact"], dtype=object),
        keypoint_count=11,
    )

    assert batch.valid.tolist() == [True, False]
    assert str(batch.failure_reason[1]) == "scratch_artifact"
    assert np.all(np.isnan(batch.head_xy[1]))
    assert np.all(np.isnan(batch.tail_keypoints_xy[1]))
    assert np.all(np.isnan(batch.tail_angle_rad[1]))
    assert np.isnan(batch.head_yaw_rad[1])
    decoded = decode_reason_bytes(batch.failure_reason_bytes)
    assert decoded.tolist() == ["ok", "scratch_artifact"]


def _build_shape_root() -> zarr.Group:
    root = zarr.group()
    analysis = root.create_group("analysis")
    parent = analysis.create_group("subject_shape_runs")
    parent.attrs["latest"] = "shape_001"
    shape = parent.create_group("shape_001")
    shape.attrs["source_refined_subject_masks_run"] = "refined_001"

    shape.create_array(
        "source_acquisition_frame_index",
        data=np.asarray([10, 11], dtype=np.int32),
        overwrite=True,
    )
    shape.create_array(
        "source_crop_row_ids",
        data=np.asarray([100, 101], dtype=np.int64),
        overwrite=True,
    )
    shape.create_array(
        "instance_key",
        data=np.asarray([1000, 1001], dtype=np.uint64),
        overwrite=True,
    )

    components = shape.create_group("components")
    body = components.create_group("subject_body")
    sources = _source_arrays()
    body.create_array(
        "tail_sample_s", data=sources["source_tail_sample_s"], overwrite=True
    )
    body.create_array("tail_sample_xy", data=sources["tail_sample_xy"], overwrite=True)
    body.create_array("head_endpoint_xy", data=sources["head_xy"], overwrite=True)
    body.create_array(
        "tail_sample_valid", data=sources["tail_sample_valid"], overwrite=True
    )
    body.create_array("bspline_valid", data=sources["bspline_valid"], overwrite=True)
    body.create_array(
        "tail_sample_failure_reason_bytes",
        data=mod._encode_reasons(["ok", "ok"]),
        overwrite=True,
    )
    body.create_array(
        "bspline_failure_reason_bytes",
        data=mod._encode_reasons(["ok", "ok"]),
        overwrite=True,
    )
    return root


def test_normal_tail_posture_writer_rejects_unpublished_subject_shape_source() -> None:
    root = _build_shape_root()

    with pytest.raises(
        subject_shape_io.SubjectShapeIOError,
        match="not a valid canonical coordinate publication",
    ):
        mod.write_tail_posture_view_run_group(
            root,
            subject_shape_run="shape_001",
            run_name="posture_rejected",
            dry_run=True,
        )


@pytest.mark.parametrize("overwrite", [False, True])
def test_tail_posture_existing_name_is_immutable_and_unchanged(
    overwrite: bool,
) -> None:
    root = zarr.group()
    parent = root.require_group("analysis/tail_posture_view_runs")
    existing = parent.create_group("posture_existing")
    existing.attrs.update(
        {
            mod.TAIL_PUBLICATION_OWNER_ATTR: ("11111111-1111-4111-8111-111111111111"),
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": True,
            "sentinel": "preserve",
        }
    )
    existing.create_array("sentinel_values", data=np.asarray([7, 8], dtype=np.int16))
    parent.attrs.update(
        {"latest": "posture_existing", "latest_complete": "posture_existing"}
    )
    before_attrs = dict(existing.attrs)
    before_parent = dict(parent.attrs)
    before_values = np.asarray(existing["sentinel_values"][:]).copy()

    with pytest.raises(ValueError, match="immutable"):
        mod._prepare_run_group(
            root,
            target_run="posture_existing",
            shape_run_name="shape",
            shape_group=root.require_group("analysis/subject_shape_runs/shape"),
            source_subject_shape_publication_manifest_sha256="a" * 64,
            row_count=2,
            view_family="family",
            head_source="head_endpoint_xy",
            keypoint_count=5,
            source_tail_kinematics_run=None,
            stage_command="fixture",
            publication_owner_uuid="22222222-2222-4222-8222-222222222222",
            overwrite=overwrite,
        )

    assert dict(existing.attrs) == before_attrs
    assert dict(parent.attrs) == before_parent
    np.testing.assert_array_equal(existing["sentinel_values"][:], before_values)


def test_write_tail_posture_view_run_group_writes_schema_and_arrays(
    monkeypatch,
) -> None:
    _patch_provenance(monkeypatch)
    root = _build_shape_root()

    summary = mod.write_tail_posture_view_run_group(
        root,
        subject_shape_run="shape_001",
        run_name="megabouts_view_001",
        source_tail_kinematics_run="tail_001",
    )

    assert summary["status"] == "updated"
    assert summary["valid_row_count"] == 2
    parent = root["analysis"]["tail_posture_view_runs"]
    assert parent.attrs["latest"] == "megabouts_view_001"
    assert parent.attrs["latest_megabouts_compatible"] == "megabouts_view_001"
    run = parent["megabouts_view_001"]
    assert run.attrs["schema_id"] == "analysis.tail_posture_view_runs"
    assert run.attrs["schema_version"] == 3
    assert run.attrs["view_family"] == "megabouts_compatible"
    assert run.attrs["dependency_policy"] == "no_megabouts_dependency_required"
    assert run.attrs["angle_convention"] == "megabouts_cumulative_segment_angle"
    assert run.attrs["keypoint_count"] == 11
    assert run.attrs["angle_count"] == 10
    assert run.attrs["source_subject_shape_run"] == "shape_001"
    assert run.attrs["source_subject_shape_publication_manifest_sha256"] == "a" * 64
    assert run.attrs["source_tail_kinematics_run"] == "tail_001"
    assert run["source_acquisition_frame_index"][:].tolist() == [10, 11]
    assert run["instance_key"][:].tolist() == [1000, 1001]
    assert np.asarray(run["tail_keypoints_xy"][:], dtype=np.float32).shape == (2, 11, 2)
    assert np.asarray(run["tail_angle_rad"][:], dtype=np.float32).shape == (2, 10)
    np.testing.assert_allclose(
        np.asarray(run["tail_angle_rad"][:], dtype=np.float32), 0.0, atol=1e-7
    )
    assert run.attrs["provenance"]["stage"] == "analysis.tail_posture_view_runs"
    assert (
        run.attrs["provenance"]["inputs"][
            "source_subject_shape_publication_manifest_sha256"
        ]
        == "a" * 64
    )


def test_tail_posture_writer_freezes_exact_array_contract(monkeypatch) -> None:
    _patch_provenance(monkeypatch)
    root = _build_shape_root()
    mod.write_tail_posture_view_run_group(
        root,
        subject_shape_run="shape_001",
        run_name="posture_exact",
    )
    run = root["analysis/tail_posture_view_runs/posture_exact"]
    manifest = run.attrs[TAIL_POSTURE_VIEW_ARRAY_SCHEMA_ATTR]

    assert len(manifest["arrays"]) == 10
    assert manifest["byte_planner_adopted"] is False
    assert run.attrs[TAIL_POSTURE_VIEW_ARRAY_SCHEMA_DIGEST_ATTR] == (
        tail_posture_view_manifest_digest(manifest)
    )
    assert run["source_crop_row_ids"].dtype == np.dtype("int64")
    assert run["source_acquisition_frame_index"].dtype == np.dtype("int64")
    assert run["failure_reason_bytes"].shape == (2, 64)


def test_tail_posture_recomputed_digest_does_not_authorize_manifest_tampering(
    monkeypatch,
) -> None:
    _patch_provenance(monkeypatch)
    root = _build_shape_root()
    mod.write_tail_posture_view_run_group(
        root,
        subject_shape_run="shape_001",
        run_name="posture_tampered",
    )
    run = root["analysis/tail_posture_view_runs/posture_tampered"]
    manifest = run.attrs[TAIL_POSTURE_VIEW_ARRAY_SCHEMA_ATTR]
    manifest["arrays"][0]["logical_contract"]["axis_names"] = ["wrong_axis"]
    run.attrs[TAIL_POSTURE_VIEW_ARRAY_SCHEMA_ATTR] = manifest
    run.attrs[TAIL_POSTURE_VIEW_ARRAY_SCHEMA_DIGEST_ATTR] = (
        tail_posture_view_manifest_digest(manifest)
    )

    issues = validate_tail_posture_view_arrays(
        run,
        dimensions=TailPostureViewDimensions(
            n_rows=2,
            n_keypoints=11,
            n_angles=10,
        ),
    )

    assert {issue.code for issue in issues} == {
        "array_schema_manifest_mismatch",
        "array_schema_digest_mismatch",
    }


def test_tail_posture_wrong_dtype_fails_exact_array_validation(monkeypatch) -> None:
    _patch_provenance(monkeypatch)
    root = _build_shape_root()
    mod.write_tail_posture_view_run_group(
        root,
        subject_shape_run="shape_001",
        run_name="posture_wrong_dtype",
    )
    run = root["analysis/tail_posture_view_runs/posture_wrong_dtype"]
    values = np.asarray(run["head_yaw_rad"][:], dtype=np.float64)
    del run["head_yaw_rad"]
    run.create_array("head_yaw_rad", data=values)

    issues = validate_tail_posture_view_arrays(
        run,
        dimensions=TailPostureViewDimensions(
            n_rows=2,
            n_keypoints=11,
            n_angles=10,
        ),
    )

    assert any("dtype mismatch" in issue.message for issue in issues)


@pytest.mark.parametrize("overwrite", [False, True])
def test_tail_posture_retry_never_invalidates_existing_publication(
    monkeypatch: pytest.MonkeyPatch,
    overwrite: bool,
) -> None:
    _patch_provenance(monkeypatch)
    root = _build_shape_root()
    mod.write_tail_posture_view_run_group(
        root,
        subject_shape_run="shape_001",
        run_name="posture_immutable",
    )
    parent = root["analysis/tail_posture_view_runs"]
    existing = parent["posture_immutable"]
    before_attrs = dict(existing.attrs)
    before_parent = dict(parent.attrs)
    before_values = np.asarray(existing["tail_angle_rad"][:]).copy()

    with pytest.raises(ValueError, match="immutable"):
        mod.write_tail_posture_view_run_group(
            root,
            subject_shape_run="shape_001",
            run_name="posture_immutable",
            overwrite=overwrite,
        )

    assert dict(existing.attrs) == before_attrs
    assert dict(parent.attrs) == before_parent
    np.testing.assert_array_equal(existing["tail_angle_rad"][:], before_values)


def test_tail_posture_publication_failure_leaves_owned_tombstone_ineligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_provenance(monkeypatch)
    root = _build_shape_root()
    monkeypatch.setattr(
        mod,
        "publish_tail_posture_coordinate_surfaces",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("injected publication failure")
        ),
    )

    with pytest.raises(RuntimeError, match="injected publication failure"):
        mod.write_tail_posture_view_run_group(
            root,
            subject_shape_run="shape_001",
            run_name="posture_failed",
        )

    parent = root["analysis/tail_posture_view_runs"]
    failed = parent["posture_failed"]
    assert failed.attrs["palette_run_completion_status"] == "failed"
    assert failed.attrs["stage_selector_eligible"] is False
    assert parent.attrs.get("latest") != "posture_failed"
    assert parent.attrs.get("latest_complete") != "posture_failed"


def test_tail_posture_early_lifecycle_failure_retains_owned_tombstone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_provenance(monkeypatch)
    root = _build_shape_root()
    observed_owner: list[str] = []

    def fail_started(run_group, **_kwargs):
        owner = run_group.attrs.get(mod.TAIL_PUBLICATION_OWNER_ATTR)
        assert isinstance(owner, str) and owner
        assert run_group.attrs.get("stage_selector_eligible") is False
        observed_owner.append(owner)
        raise RuntimeError("injected early posture lifecycle failure")

    monkeypatch.setattr(mod, "mark_run_started", fail_started)

    with pytest.raises(RuntimeError, match="injected early posture lifecycle failure"):
        mod.write_tail_posture_view_run_group(
            root,
            subject_shape_run="shape_001",
            run_name="posture_early_failed",
        )

    failed = root["analysis/tail_posture_view_runs/posture_early_failed"]
    assert failed.attrs[mod.TAIL_PUBLICATION_OWNER_ATTR] == observed_owner[0]
    assert failed.attrs["palette_run_completion_status"] == "failed"
    assert failed.attrs["stage_selector_eligible"] is False
    parent = root["analysis/tail_posture_view_runs"]
    assert parent.attrs.get("latest") != "posture_early_failed"
    assert parent.attrs.get("latest_complete") != "posture_early_failed"


def test_tail_posture_persist_then_raise_create_recovers_owned_tombstone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_provenance(monkeypatch)
    root = _build_shape_root()
    run_name = "posture_create_ambiguous"
    parent = root.require_group("analysis/tail_posture_view_runs")
    parent.attrs["latest_pending"] = run_name
    original_create = mod._create_tail_posture_public_candidate

    def persist_then_raise(parent, *, run_name, publication_owner_uuid):
        original_create(
            parent,
            run_name=run_name,
            publication_owner_uuid=publication_owner_uuid,
        )
        raise RuntimeError("injected posture create acknowledgement loss")

    monkeypatch.setattr(
        mod,
        "_create_tail_posture_public_candidate",
        persist_then_raise,
    )

    with pytest.raises(RuntimeError, match="create acknowledgement loss"):
        mod.write_tail_posture_view_run_group(
            root,
            subject_shape_run="shape_001",
            run_name=run_name,
        )

    failed = root[f"analysis/tail_posture_view_runs/{run_name}"]
    owner = failed.attrs[mod.TAIL_PUBLICATION_OWNER_ATTR]
    tombstone = failed.attrs[mod.TAIL_PUBLICATION_TOMBSTONE_ATTR]
    assert failed.attrs["palette_run_completion_status"] == "failed"
    assert failed.attrs["stage_selector_eligible"] is False
    assert tombstone["publication_owner_uuid"] == owner
    assert tombstone["run_family"] == "tail_posture_view"
    assert parent.attrs["latest_pending"] == run_name


def test_tail_posture_failure_cleanup_never_clobbers_recreated_successor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_provenance(monkeypatch)
    root = _build_shape_root()
    parent = root.require_group("analysis/tail_posture_view_runs")
    run_name = "posture_cleanup_takeover"
    original_write = mod._write_tail_posture_failure_attr
    takeover_injected = False

    monkeypatch.setattr(
        mod,
        "publish_tail_posture_coordinate_surfaces",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("injected posture source failure")
        ),
    )

    def hostile_write(attrs, name, value):
        nonlocal takeover_injected
        original_write(attrs, name, value)
        if name == "stage_selector_eligible" and not takeover_injected:
            takeover_injected = True
            del parent[run_name]
            parent.create_group(
                run_name,
                attributes={
                    mod.TAIL_PUBLICATION_OWNER_ATTR: "alien-posture-owner",
                    "palette_run_completion_status": "complete",
                    "stage_selector_eligible": True,
                    "sentinel": "successor-preserved",
                },
            )

    monkeypatch.setattr(mod, "_write_tail_posture_failure_attr", hostile_write)

    with pytest.raises(RuntimeError, match="injected posture source failure"):
        mod.write_tail_posture_view_run_group(
            root,
            subject_shape_run="shape_001",
            run_name=run_name,
        )

    successor = parent[run_name]
    assert takeover_injected is True
    assert successor.attrs[mod.TAIL_PUBLICATION_OWNER_ATTR] == "alien-posture-owner"
    assert successor.attrs["palette_run_completion_status"] == "complete"
    assert successor.attrs["stage_selector_eligible"] is True
    assert successor.attrs["sentinel"] == "successor-preserved"
    assert mod.TAIL_PUBLICATION_TOMBSTONE_ATTR not in successor.attrs


def test_write_tail_posture_view_run_group_copies_instance_key_lineage(
    monkeypatch,
) -> None:
    _patch_provenance(monkeypatch)
    root = _build_shape_root()
    shape = root["analysis"]["subject_shape_runs"]["shape_001"]
    shape.create_array(
        "instance_key", data=np.asarray([11, 22], dtype=np.uint64), overwrite=True
    )
    shape.create_array(
        "source_crop_row_ids", data=np.asarray([5, 6], dtype=np.int64), overwrite=True
    )

    mod.write_tail_posture_view_run_group(
        root,
        subject_shape_run="shape_001",
        run_name="megabouts_view_001",
        source_tail_kinematics_run="tail_001",
    )

    run = root["analysis"]["tail_posture_view_runs"]["megabouts_view_001"]
    assert run["instance_key"][:].tolist() == [11, 22]
    assert run["source_crop_row_ids"][:].tolist() == [5, 6]
    assert "instance_key" in run.attrs["row_lineage_copied"]
    assert "source_crop_row_ids" in run.attrs["row_lineage_copied"]


def test_write_tail_posture_view_run_group_rejects_missing_direct_instance_key(
    monkeypatch,
) -> None:
    _patch_provenance(monkeypatch)
    root = _build_shape_root()

    del root["analysis/subject_shape_runs/shape_001/instance_key"]

    with pytest.raises(
        subject_shape_io.SubjectShapeIOError, match="direct 'instance_key'"
    ):
        mod.write_tail_posture_view_run_group(
            root,
            subject_shape_run="shape_001",
            run_name="megabouts_view_001",
            source_tail_kinematics_run="tail_001",
        )


def test_write_tail_posture_view_rejects_acquisition_frame_alias(monkeypatch) -> None:
    _patch_provenance(monkeypatch)
    root = _build_shape_root()
    shape = root["analysis/subject_shape_runs/shape_001"]
    values = np.asarray(shape["source_acquisition_frame_index"][:])
    del shape["source_acquisition_frame_index"]
    shape.create_array("frame_indices", data=values)

    with pytest.raises(
        subject_shape_io.SubjectShapeIOError,
        match="source_acquisition_frame_index",
    ):
        mod.write_tail_posture_view_run_group(
            root,
            subject_shape_run="shape_001",
            run_name="megabouts_view_001",
            source_tail_kinematics_run="tail_001",
        )

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import zarr

import fisheye.shared.zarr.subject_mask_bundle_publication as bundle_publication
from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.subject_mask_bundle_publication import (
    SUBJECT_MASK_BUNDLE_AUTHORITY_ATTR,
    SUBJECT_MASK_BUNDLE_AUTHORITY_GENERATION_ATTR,
    SUBJECT_MASK_BUNDLE_AUTHORITY_LEASE_ATTR,
    SUBJECT_MASK_BUNDLE_FAMILY,
    SUBJECT_MASK_BUNDLE_SELECTOR_ELIGIBLE_ATTR,
    activate_subject_mask_bundle,
    publish_subject_mask_bundle_candidate,
)
from fisheye.shared.zarr.subject_mask_core_publication import (
    publish_selector_ineligible_subject_mask_core_snapshot,
)
from fisheye.shared.zarr.subject_mask_quality_publication import (
    publish_selector_ineligible_subject_mask_quality_snapshot,
)
from fisheye.shared.zarr.subject_mask_quality_schema import (
    SubjectMaskQualitySourceReference,
)
from fisheye.shared.zarr.subject_mask_schema import (
    SubjectMaskComponentRegistry,
    derive_subject_mask_frame_row_offsets,
    derive_subject_mask_metrics,
)


def _components() -> SubjectMaskComponentRegistry:
    return SubjectMaskComponentRegistry(
        ("subject_body", "eye_left", "eye_right", "swim_bladder")
    )


def _surfaces() -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    masks = np.zeros((4, 4, 8, 8), dtype=np.uint8)
    masks[:, 0, 1:7, 1:7] = 1
    masks[:, 1, 2, 2] = 1
    masks[:, 2, 2, 5] = 1
    masks[:, 3, 5, 3] = 1
    metrics = derive_subject_mask_metrics(masks)
    frames = np.asarray([0, 0, 2, 3], dtype=np.int64)
    common = {
        "source_crop_row_ids": np.arange(4, dtype=np.int64),
        "instance_key": np.asarray([101, 102, 201, 301], dtype=np.uint64),
        "source_acquisition_frame_index": frames,
        "frame_row_offsets": derive_subject_mask_frame_row_offsets(frames, n_frames=4),
        "source_crop_xywh": np.asarray(
            [[0, 0, 8, 8], [1, 0, 8, 8], [0, 1, 8, 8], [1, 1, 8, 8]],
            dtype=np.float32,
        ),
        "available_channels": np.ones((4,), dtype=bool),
        **{f"metrics/{name}": values for name, values in metrics.items()},
    }
    probabilities = masks * np.uint8(255)
    raw = {
        **common,
        "mask_probs_roi": probabilities,
        "metrics/prob_max": (
            np.max(probabilities, axis=(2, 3)).astype(np.float32) / np.float32(255.0)
        ),
    }
    refined = {**common, "masks_roi": masks}
    crop = {
        "instance_key": common["instance_key"],
        "source_acquisition_frame_index": frames,
        "source_crop_xywh": common["source_crop_xywh"],
    }
    return raw, refined, crop


def _publish_members(tmp_path: Path) -> tuple[object, object, object]:
    raw, refined, crop = _surfaces()
    source_manifest = {
        "schema_id": "palette.subject_mask.bundle_test_source",
        "schema_version": 1,
        "run_id": "source_001",
    }
    raw_publication = publish_selector_ineligible_subject_mask_core_snapshot(
        raw,
        source_crop_arrays=crop,
        source_manifest=source_manifest,
        n_frames=4,
        components=_components(),
        destination=tmp_path / "raw.zarr",
        run_id="raw_001",
        kind="raw_probability_uint8",
        source_run_path="subject_mask_shard_runs/source_001",
        created_by="pytest",
    )
    refined_publication = publish_selector_ineligible_subject_mask_core_snapshot(
        refined,
        source_crop_arrays=crop,
        source_manifest=source_manifest,
        n_frames=4,
        components=_components(),
        destination=tmp_path / "refined.zarr",
        run_id="refined_001",
        kind="refined_dense_core",
        source_run_path="refined_subject_mask_shard_runs/source_001",
        created_by="pytest",
    )
    source_paths = (
        "masks_roi",
        "instance_key",
        "source_crop_row_ids",
        "source_acquisition_frame_index",
        "frame_row_offsets",
        "available_channels",
    )
    source = SubjectMaskQualitySourceReference(
        run_name="refined_001",
        manifest_digest=canonical_json_sha256(refined_publication.manifest),
        dense_array_values_sha256=sha256_array(refined["masks_roi"]),
        component_registry_digest=canonical_json_sha256(_components().as_manifest()),
        source_array_values_sha256={
            path: sha256_array(refined[path]) for path in source_paths
        },
    )
    quality_root = tmp_path / "quality"
    quality_publication = publish_selector_ineligible_subject_mask_quality_snapshot(
        refined,
        n_frames=4,
        components=_components(),
        source=source,
        source_manifest=refined_publication.manifest,
        destination=quality_root / "quality.zarr",
        run_id="quality_001",
        shadow_root=quality_root,
        source_compute_block_bytes=512,
        created_by="pytest",
    )
    return raw_publication, refined_publication, quality_publication


def _analysis_archive(tmp_path: Path) -> Path:
    path = tmp_path / "analysis.zarr"
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs["recording_id"] = "recording_001"
    return path


def _publish_bundle(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    raw, refined, quality = _publish_members(tmp_path)
    archive = _analysis_archive(tmp_path)
    receipt = publish_subject_mask_bundle_candidate(
        analysis_zarr=archive,
        recording_identity="recording_001",
        raw_snapshot_root=raw.output_path,
        raw_run_id=raw.run_id,
        refined_snapshot_root=refined.output_path,
        refined_run_id=refined.run_id,
        quality_snapshot_root=quality.output_path,
        quality_run_id=quality.run_id,
        bundle_id="bundle_001",
    )
    return archive, receipt


def test_bundle_candidate_is_complete_but_not_authoritative(tmp_path: Path) -> None:
    archive, receipt = _publish_bundle(tmp_path)

    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    assert receipt["status"] == "complete"
    assert SUBJECT_MASK_BUNDLE_AUTHORITY_ATTR not in root.attrs
    assert SUBJECT_MASK_BUNDLE_AUTHORITY_LEASE_ATTR not in root.attrs
    assert (
        root[f"{SUBJECT_MASK_BUNDLE_FAMILY}/bundle_001"].attrs[
            SUBJECT_MASK_BUNDLE_SELECTOR_ELIGIBLE_ATTR
        ]
        is False
    )
    for family, run_id in (
        ("subject_mask_runs", "raw_001"),
        ("refined_subject_masks_runs", "refined_001"),
        ("subject_mask_quality_runs", "quality_001"),
    ):
        run = root[f"{family}/{run_id}"]
        assert run.attrs["stage_selector_eligible"] is False
        assert run.attrs.get(SUBJECT_MASK_BUNDLE_SELECTOR_ELIGIBLE_ATTR) is None
        for selector in ("latest", "latest_complete", "authoritative_run"):
            assert root[family].attrs.get(selector) is None


def test_bundle_activation_commits_one_root_authority(tmp_path: Path) -> None:
    archive, receipt = _publish_bundle(tmp_path)

    authority = activate_subject_mask_bundle(
        analysis_zarr=archive, bundle_id="bundle_001"
    )

    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    assert authority["bundle_manifest_digest"] == receipt["bundle_manifest_digest"]
    assert root.attrs[SUBJECT_MASK_BUNDLE_AUTHORITY_ATTR] == authority
    assert root.attrs[SUBJECT_MASK_BUNDLE_AUTHORITY_GENERATION_ATTR] == 1
    assert SUBJECT_MASK_BUNDLE_AUTHORITY_LEASE_ATTR not in root.attrs
    for path in (
        f"{SUBJECT_MASK_BUNDLE_FAMILY}/bundle_001",
        "subject_mask_runs/raw_001",
        "refined_subject_masks_runs/refined_001",
        "subject_mask_quality_runs/quality_001",
    ):
        group = root[path]
        assert group.attrs[SUBJECT_MASK_BUNDLE_SELECTOR_ELIGIBLE_ATTR] is True
        assert group.attrs["stage_selector_eligible"] is False


def test_bundle_rejects_cross_run_identity_mismatch(tmp_path: Path) -> None:
    raw, refined, quality = _publish_members(tmp_path)
    archive = _analysis_archive(tmp_path)
    refined_run = zarr.open_group(
        str(refined.output_path / "refined_subject_masks_runs" / refined.run_id),
        mode="a",
        use_consolidated=False,
    )
    manifest = copy.deepcopy(refined_run.attrs["run_manifest"])
    manifest["payload"]["logical_content"]["document"]["arrays"]["instance_key"][
        "sha256"
    ] = ("00" * 32)
    manifest["payload"]["logical_content"]["digest"] = canonical_json_sha256(
        manifest["payload"]["logical_content"]["document"]
    )
    manifest["payload_digest"] = canonical_json_sha256(manifest["payload"])
    refined_run.attrs["run_manifest"] = manifest

    with pytest.raises(ValueError, match="identity differs"):
        publish_subject_mask_bundle_candidate(
            analysis_zarr=archive,
            recording_identity="recording_001",
            raw_snapshot_root=raw.output_path,
            raw_run_id=raw.run_id,
            refined_snapshot_root=refined.output_path,
            refined_run_id=refined.run_id,
            quality_snapshot_root=quality.output_path,
            quality_run_id=quality.run_id,
            bundle_id="bundle_001",
        )

    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    assert SUBJECT_MASK_BUNDLE_AUTHORITY_ATTR not in root.attrs
    assert SUBJECT_MASK_BUNDLE_FAMILY not in root


def test_bundle_preflights_all_immutable_names_before_import(tmp_path: Path) -> None:
    raw, refined, quality = _publish_members(tmp_path)
    archive = _analysis_archive(tmp_path)
    root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
    root.require_group("refined_subject_masks_runs").create_group("refined_001")

    with pytest.raises(FileExistsError, match="refined_subject_masks_runs/refined_001"):
        publish_subject_mask_bundle_candidate(
            analysis_zarr=archive,
            recording_identity="recording_001",
            raw_snapshot_root=raw.output_path,
            raw_run_id=raw.run_id,
            refined_snapshot_root=refined.output_path,
            refined_run_id=refined.run_id,
            quality_snapshot_root=quality.output_path,
            quality_run_id=quality.run_id,
            bundle_id="bundle_001",
        )

    reopened = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    assert "subject_mask_runs" not in reopened
    assert "subject_mask_quality_runs" not in reopened
    assert SUBJECT_MASK_BUNDLE_FAMILY not in reopened


def test_activation_recovers_interrupted_precommit_lease(tmp_path: Path) -> None:
    archive, receipt = _publish_bundle(tmp_path)
    root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
    bundle = root[f"{SUBJECT_MASK_BUNDLE_FAMILY}/bundle_001"]
    manifest = bundle.attrs["run_manifest"]
    paths = (
        f"{SUBJECT_MASK_BUNDLE_FAMILY}/bundle_001",
        *(
            manifest["payload"]["members"][role]["run_path"]
            for role in ("raw", "refined", "quality")
        ),
    )
    for path in paths:
        root[path].attrs[SUBJECT_MASK_BUNDLE_SELECTOR_ELIGIBLE_ATTR] = True
    root.attrs[SUBJECT_MASK_BUNDLE_AUTHORITY_LEASE_ATTR] = {
        "owner_uuid": "dead-worker",
        "bundle_id": "bundle_001",
        "bundle_manifest_digest": receipt["bundle_manifest_digest"],
        "next_generation": 1,
        "policy": bundle_publication.SUBJECT_MASK_BUNDLE_ACTIVATION_POLICY,
    }

    authority = activate_subject_mask_bundle(
        analysis_zarr=archive, bundle_id="bundle_001"
    )

    reopened = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    assert authority["generation"] == 1
    assert SUBJECT_MASK_BUNDLE_AUTHORITY_LEASE_ATTR not in reopened.attrs
    assert reopened.attrs[SUBJECT_MASK_BUNDLE_AUTHORITY_ATTR] == authority
    for path in paths:
        assert reopened[path].attrs[SUBJECT_MASK_BUNDLE_SELECTOR_ELIGIBLE_ATTR] is True


def test_activation_failure_restores_readiness_and_root_attrs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, _receipt = _publish_bundle(tmp_path)
    original = bundle_publication._validate_live_bundle
    calls = 0

    def fail_second_validation(
        archive_path: Path, *, bundle_id: str
    ) -> tuple[dict[str, object], dict[str, object]]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected activation validation failure")
        return original(archive_path, bundle_id=bundle_id)  # type: ignore[return-value]

    monkeypatch.setattr(
        bundle_publication, "_validate_live_bundle", fail_second_validation
    )
    with pytest.raises(RuntimeError, match="injected activation"):
        activate_subject_mask_bundle(analysis_zarr=archive, bundle_id="bundle_001")

    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    assert SUBJECT_MASK_BUNDLE_AUTHORITY_ATTR not in root.attrs
    assert SUBJECT_MASK_BUNDLE_AUTHORITY_LEASE_ATTR not in root.attrs
    assert SUBJECT_MASK_BUNDLE_AUTHORITY_GENERATION_ATTR not in root.attrs
    for path in (
        f"{SUBJECT_MASK_BUNDLE_FAMILY}/bundle_001",
        "subject_mask_runs/raw_001",
        "refined_subject_masks_runs/refined_001",
        "subject_mask_quality_runs/quality_001",
    ):
        assert root[path].attrs.get(SUBJECT_MASK_BUNDLE_SELECTOR_ELIGIBLE_ATTR) in (
            None,
            False,
        )

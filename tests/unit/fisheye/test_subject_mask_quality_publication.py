from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.subject_mask_quality_manifest import (
    validate_subject_mask_quality_run_manifest,
)
from fisheye.shared.zarr.subject_mask_quality_producer import (
    SUBJECT_V1_LR_COMPONENTS,
    prepare_in_memory_observation_local_subject_mask_quality,
)
from fisheye.shared.zarr.subject_mask_quality_publication import (
    publish_selector_ineligible_subject_mask_quality_snapshot,
    require_local_subject_mask_quality_scratch_root,
    validate_subject_mask_quality_shadow_publication,
)
from fisheye.shared.zarr.subject_mask_quality_schema import (
    SUBJECT_MASK_QUALITY_SCHEMA_V1,
    SubjectMaskQualitySourceReference,
)
from fisheye.shared.zarr.subject_mask_schema import SubjectMaskComponentRegistry
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_CONTRACT,
    RUN_COMPLETION_CONTRACT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
)


class _ReadTrackingArray:
    def __init__(self, values: np.ndarray) -> None:
        self._values = values
        self.shape = values.shape
        self.dtype = values.dtype
        self.selections: list[object] = []

    def __getitem__(self, selection: object) -> np.ndarray:
        self.selections.append(selection)
        if selection is Ellipsis:
            raise AssertionError("Full-array dense-mask reads are forbidden.")
        return self._values[selection]


def _components() -> SubjectMaskComponentRegistry:
    return SubjectMaskComponentRegistry(SUBJECT_V1_LR_COMPONENTS)


def _fixture() -> tuple[
    dict[str, np.ndarray],
    dict[str, object],
    SubjectMaskQualitySourceReference,
]:
    masks = np.zeros((4, 4, 8, 8), dtype=np.uint8)
    body, left, right, bladder = range(4)
    masks[:, body, 1:7, 1:7] = 1
    masks[:, left, 2, 2] = 1
    masks[:, right, 2, 5] = 1
    masks[:, bladder, 5, 3] = 1
    masks[1, left, 0, 0] = 1
    masks[2, right, 2, 2] = 1
    masks[3, right] = 0
    arrays = {
        "masks_roi": masks,
        "instance_key": np.asarray([101, 102, 201, 301], dtype=np.uint64),
        "source_crop_row_ids": np.arange(4, dtype=np.int64),
        "source_acquisition_frame_index": np.asarray(
            [0, 0, 2, 3], dtype=np.int64
        ),
        "frame_row_offsets": np.asarray([0, 2, 2, 3, 4], dtype=np.int64),
        "available_channels": np.ones((4,), dtype=bool),
    }
    source_manifest: dict[str, object] = {
        "schema_id": "palette.refined_subject_mask.test_manifest",
        "schema_version": 1,
        "run_id": "refined_subject_masks_001",
        "dense_array_values_sha256": sha256_array(masks),
        "components": _components().as_manifest(),
    }
    source = SubjectMaskQualitySourceReference(
        run_name="refined_subject_masks_001",
        manifest_digest=canonical_json_sha256(source_manifest),
        dense_array_values_sha256=sha256_array(masks),
        component_registry_digest=canonical_json_sha256(
            _components().as_manifest()
        ),
        source_array_values_sha256={
            path: sha256_array(arrays[path])
            for path in (
                "masks_roi",
                "instance_key",
                "source_crop_row_ids",
                "source_acquisition_frame_index",
                "frame_row_offsets",
                "available_channels",
            )
        },
    )
    return arrays, source_manifest, source


def test_bounded_selector_ineligible_publication_round_trip(tmp_path: object) -> None:
    arrays, source_manifest, source = _fixture()
    expected = prepare_in_memory_observation_local_subject_mask_quality(
        arrays,
        n_frames=4,
        components=_components(),
        source=source,
    )
    tracked_masks = _ReadTrackingArray(arrays["masks_roi"])
    lazy_source = {**arrays, "masks_roi": tracked_masks}
    shadow_root = tmp_path / "quality"  # type: ignore[operator]
    scratch_root = tmp_path / "scratch"  # type: ignore[operator]
    publication = publish_selector_ineligible_subject_mask_quality_snapshot(
        lazy_source,
        n_frames=4,
        components=_components(),
        source=source,
        source_manifest=source_manifest,
        destination=shadow_root / "fixture.zarr",
        run_id="quality_v1_001",
        shadow_root=shadow_root,
        scratch_root=scratch_root,
        source_compute_block_bytes=512,
        created_by="pytest",
    )

    assert validate_subject_mask_quality_shadow_publication(publication) == ()
    assert validate_subject_mask_quality_run_manifest(publication.manifest) == ()
    assert tracked_masks.selections == [slice(0, 2), slice(2, 4)]
    assert publication.write_receipt["source_compute_block_rows"] == 2
    assert publication.write_receipt["source_compute_block_count"] == 2
    assert list(scratch_root.iterdir()) == []

    family = zarr.open_group(
        str(publication.output_path / "subject_mask_quality_runs"),
        mode="r",
        use_consolidated=False,
    )
    for selector in (
        "latest",
        "latest_complete",
        "latest_pending",
        "authoritative_run",
    ):
        assert family.attrs.get(selector) is None
    run = family["quality_v1_001"]
    assert run.attrs["status"] == "complete"
    assert run.attrs[RUN_COMPLETION_CONTRACT_ATTR] == RUN_COMPLETION_CONTRACT
    assert run.attrs[RUN_COMPLETION_STATUS_ATTR] == "complete"
    assert run.attrs["stage_selector_eligible"] is False
    assert set(run.array_keys()) == set(SUBJECT_MASK_QUALITY_SCHEMA_V1.binding_paths)
    for path in SUBJECT_MASK_QUALITY_SCHEMA_V1.binding_paths:
        observed = np.asarray(run[path][...])
        wanted = np.asarray(expected.arrays[path])
        if np.issubdtype(observed.dtype, np.floating):
            np.testing.assert_allclose(observed, wanted, equal_nan=True)
        else:
            np.testing.assert_array_equal(observed, wanted)


def test_manifest_rejects_recomputed_nested_tampering(tmp_path: object) -> None:
    arrays, source_manifest, source = _fixture()
    root = tmp_path / "quality"  # type: ignore[operator]
    publication = publish_selector_ineligible_subject_mask_quality_snapshot(
        arrays,
        n_frames=4,
        components=_components(),
        source=source,
        source_manifest=source_manifest,
        destination=root / "fixture.zarr",
        run_id="quality_v1_001",
        shadow_root=root,
        source_compute_block_bytes=512,
        created_by="pytest",
    )
    tampered = copy.deepcopy(publication.manifest)
    tampered["payload"]["policy"]["exclusive_pairs"] = []
    tampered["payload"]["storage_plan"]["arrays"][0][
        "access_unit_semantics"
    ] = "tampered"
    tampered["payload"]["write_receipt"]["output_array_write_units"][
        "instance_key"
    ]["row_count"] = 999
    logical_content = tampered["payload"]["logical_content"]
    logical_content["document"]["arrays"]["instance_key"]["shape"] = [999]
    logical_content["digest"] = canonical_json_sha256(logical_content["document"])
    tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])

    errors = validate_subject_mask_quality_run_manifest(tampered)

    assert any("policy differs" in error for error in errors)
    assert "subject-mask quality storage plan differs from planner output" in errors
    assert (
        "subject-mask quality output write units differ from storage plan" in errors
    )
    assert "subject-mask quality shape mismatch at instance_key" in errors


def test_publication_rejects_dense_digest_mismatch_before_creating_store(
    tmp_path: object,
) -> None:
    arrays, source_manifest, source = _fixture()
    wrong_source = SubjectMaskQualitySourceReference(
        run_name=source.run_name,
        manifest_digest=source.manifest_digest,
        dense_array_values_sha256="00" * 32,
        component_registry_digest=source.component_registry_digest,
        source_array_values_sha256={
            **dict(source.source_array_values_sha256),
            "masks_roi": "00" * 32,
        },
    )
    root = tmp_path / "quality"  # type: ignore[operator]
    destination = root / "fixture.zarr"

    with pytest.raises(ValueError, match="source-array digest mismatch"):
        publish_selector_ineligible_subject_mask_quality_snapshot(
            arrays,
            n_frames=4,
            components=_components(),
            source=wrong_source,
            source_manifest=source_manifest,
            destination=destination,
            run_id="quality_v1_001",
            shadow_root=root,
            source_compute_block_bytes=512,
            created_by="pytest",
        )

    assert not destination.exists()


def test_publication_rejects_caller_substituted_identity_column(
    tmp_path: object,
) -> None:
    arrays, source_manifest, source = _fixture()
    arrays["instance_key"] = arrays["instance_key"].copy()
    arrays["instance_key"][0] = np.uint64(999)
    root = tmp_path / "quality"  # type: ignore[operator]
    destination = root / "fixture.zarr"

    with pytest.raises(
        ValueError,
        match="Exact refined source-array digest mismatch for: instance_key",
    ):
        publish_selector_ineligible_subject_mask_quality_snapshot(
            arrays,
            n_frames=4,
            components=_components(),
            source=source,
            source_manifest=source_manifest,
            destination=destination,
            run_id="quality_v1_001",
            shadow_root=root,
            source_compute_block_bytes=512,
            created_by="pytest",
        )

    assert not destination.exists()


def test_shared_storage_cannot_be_declared_node_local_scratch() -> None:
    with pytest.raises(ValueError, match="must be node-local"):
        require_local_subject_mask_quality_scratch_root(
            Path("/groups/johnson/shared_scratch")
        )

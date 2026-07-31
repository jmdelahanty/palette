from __future__ import annotations

import numpy as np
import pytest
import zarr

from fisheye.shared.row_source_signature import (
    ROW_SOURCE_SIGNATURE_BASIS_CONTENT,
    ROW_SOURCE_SIGNATURE_BASIS_CONTENT_AND_REVISION,
    ROW_SOURCE_SIGNATURE_WIDTH_BYTES,
    RowSourceSignatureError,
    assert_row_source_signature_specs_match,
    build_row_source_signatures,
    copy_selected_row_source_signatures,
    load_group_row_source_signature_spec,
    load_row_source_signature_spec,
    validate_row_source_signature_array,
)


def _build(
    keys: np.ndarray,
    boxes: np.ndarray,
    *,
    revisions: np.ndarray | None = None,
    context: dict[str, object] | None = None,
):
    return build_row_source_signatures(
        stage="crop",
        instance_keys=keys,
        content_components={"bbox_norm_coords": boxes},
        revision_components=(
            {"refined_detect/row_revision": revisions}
            if revisions is not None
            else None
        ),
        revision_authorities=(
            {"refined_detect/row_revision": "refined_detect_authority_v1"}
            if revisions is not None
            else None
        ),
        compatibility_context=context or {"pixel_source_sha256": "abc123"},
    )


def test_content_edit_changes_only_the_affected_keyed_row() -> None:
    keys = np.asarray([11, 22, 33], dtype=np.uint64)
    boxes = np.asarray(
        [[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5], [0.3, 0.4, 0.5, 0.6]],
        dtype=np.float64,
    )
    before = _build(keys, boxes)
    changed_boxes = boxes.copy()
    changed_boxes[1, 0] += 0.01
    after = _build(keys, changed_boxes)

    assert before.signatures.shape == (3, ROW_SOURCE_SIGNATURE_WIDTH_BYTES)
    assert before.signatures.dtype == np.uint8
    assert np.array_equal(before.signatures[0], after.signatures[0])
    assert not np.array_equal(before.signatures[1], after.signatures[1])
    assert np.array_equal(before.signatures[2], after.signatures[2])
    assert before.spec.basis == ROW_SOURCE_SIGNATURE_BASIS_CONTENT


def test_physical_reorder_preserves_key_to_signature_mapping() -> None:
    keys = np.asarray([11, 22, 33], dtype=np.uint64)
    boxes = np.arange(12, dtype=np.float32).reshape(3, 4)
    original = _build(keys, boxes)
    order = np.asarray([2, 0, 1])
    reordered = _build(keys[order], boxes[order])

    original_by_key = {
        int(key): bytes(signature)
        for key, signature in zip(keys, original.signatures, strict=True)
    }
    reordered_by_key = {
        int(key): bytes(signature)
        for key, signature in zip(keys[order], reordered.signatures, strict=True)
    }
    assert reordered_by_key == original_by_key
    assert_row_source_signature_specs_match(original.spec, reordered.spec)


def test_bounded_batches_match_one_full_build() -> None:
    keys = np.arange(100, 108, dtype=np.uint64)
    boxes = np.arange(32, dtype=np.float32).reshape(8, 4)
    full = _build(keys, boxes)
    first = _build(keys[:3], boxes[:3])
    second = _build(keys[3:], boxes[3:])

    assert first.spec.spec_digest == full.spec.spec_digest == second.spec.spec_digest
    assert np.array_equal(
        np.concatenate([first.signatures, second.signatures], axis=0),
        full.signatures,
    )


def test_revision_inputs_are_semantic_uint64_and_change_only_one_row() -> None:
    keys = np.asarray([11, 22, 33], dtype=np.uint64)
    boxes = np.arange(12, dtype=np.float64).reshape(3, 4)
    revision_i32 = np.asarray([0, 2, 0], dtype=np.int32)
    revision_i64 = revision_i32.astype(np.int64)
    first = _build(keys, boxes, revisions=revision_i32)
    same = _build(keys, boxes, revisions=revision_i64)
    assert first.spec.basis == ROW_SOURCE_SIGNATURE_BASIS_CONTENT_AND_REVISION
    assert first.spec.spec_digest == same.spec.spec_digest
    assert np.array_equal(first.signatures, same.signatures)

    changed_revision = revision_i64.copy()
    changed_revision[2] += 1
    changed = _build(keys, boxes, revisions=changed_revision)
    assert np.array_equal(first.signatures[:2], changed.signatures[:2])
    assert not np.array_equal(first.signatures[2], changed.signatures[2])


def test_context_change_invalidates_the_whole_signature_spec() -> None:
    keys = np.asarray([11, 22], dtype=np.uint64)
    boxes = np.arange(8, dtype=np.float32).reshape(2, 4)
    first = _build(keys, boxes, context={"pixel_source_sha256": "first"})
    changed = _build(keys, boxes, context={"pixel_source_sha256": "second"})

    assert first.spec.spec_digest != changed.spec.spec_digest
    assert np.all(np.any(first.signatures != changed.signatures, axis=1))
    with pytest.raises(RowSourceSignatureError, match="specifications differ"):
        assert_row_source_signature_specs_match(first.spec, changed.spec)


def test_float_nan_and_signed_zero_are_canonical() -> None:
    keys = np.asarray([11], dtype=np.uint64)
    positive = np.asarray([[0.0, np.nan]], dtype=np.float64)
    negative = np.asarray([[-0.0, np.nan]], dtype=np.float64)
    first = build_row_source_signatures(
        stage="example",
        instance_keys=keys,
        content_components={"values": positive},
    )
    second = build_row_source_signatures(
        stage="example",
        instance_keys=keys,
        content_components={"values": negative},
    )
    assert first.spec.spec_digest == second.spec.spec_digest
    assert np.array_equal(first.signatures, second.signatures)


def test_attrs_capture_complete_comparison_specification() -> None:
    batch = _build(
        np.asarray([11], dtype=np.uint64),
        np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float64),
    )
    attrs = batch.spec.to_attrs()
    assert attrs["source_row_signature_schema_id"] == "palette.row_source_signature"
    assert attrs["source_row_signature_width_bytes"] == 32
    assert attrs["source_row_signature_spec_digest"] == batch.spec.spec_digest
    assert (
        attrs["source_row_signature_spec"]["components"][0]["name"]
        == "bbox_norm_coords"
    )
    loaded = load_row_source_signature_spec(attrs)
    assert loaded == batch.spec


def test_persisted_spec_tampering_fails_closed() -> None:
    batch = _build(
        np.asarray([11], dtype=np.uint64),
        np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float64),
    )
    attrs = batch.spec.to_attrs()
    attrs["source_row_signature_spec"]["compatibility_context"][
        "pixel_source_sha256"
    ] = "tampered"
    with pytest.raises(RowSourceSignatureError, match="digest does not match"):
        load_row_source_signature_spec(attrs)


def test_signature_array_metadata_validation_does_not_read_payload() -> None:
    signatures = np.empty((7, 32), dtype=np.uint8)
    assert validate_row_source_signature_array(signatures, expected_row_count=7) == 7
    with pytest.raises(RowSourceSignatureError, match=r"shape \(N, 32\)"):
        validate_row_source_signature_array(np.empty((7, 16), dtype=np.uint8))
    with pytest.raises(RowSourceSignatureError, match="uint8"):
        validate_row_source_signature_array(np.empty((7, 32), dtype=np.uint64))
    with pytest.raises(RowSourceSignatureError, match="row count"):
        validate_row_source_signature_array(signatures, expected_row_count=8)


def test_copy_selected_signatures_persists_exact_inference_binding() -> None:
    root = zarr.group(store=zarr.storage.MemoryStore(), zarr_format=3)
    source = root.create_group("crop")
    target = root.create_group("delta")
    batch = _build(
        np.asarray([11, 22, 33], dtype=np.uint64),
        np.arange(12, dtype=np.float32).reshape(3, 4),
    )
    source.create_array(
        "source_row_signature", data=batch.signatures, chunks=(1, 32)
    )
    source.attrs.update(batch.spec.to_attrs())

    copied = copy_selected_row_source_signatures(
        target,
        source,
        np.asarray([2, 0], dtype=np.int64),
        shard_rows=2,
    )

    np.testing.assert_array_equal(copied, batch.signatures[[2, 0]])
    np.testing.assert_array_equal(target["source_row_signature"][:], copied)
    assert (
        target.attrs["source_row_signature_spec_digest"]
        == batch.spec.spec_digest
    )


def test_copy_selected_signatures_accepts_strict_crop_envelope() -> None:
    root = zarr.group(store=zarr.storage.MemoryStore(), zarr_format=3)
    source = root.create_group("crop")
    target = root.create_group("delta")
    batch = _build(
        np.asarray([11, 22, 33], dtype=np.uint64),
        np.arange(12, dtype=np.float32).reshape(3, 4),
    )
    source.create_array(
        "source_row_signature", data=batch.signatures, chunks=(1, 32)
    )
    source.attrs["row_signature"] = {
        "array_path": "source_row_signature",
        **batch.spec.to_attrs(prefix=""),
    }

    copied = copy_selected_row_source_signatures(
        target,
        source,
        np.asarray([2, 0], dtype=np.int64),
        shard_rows=2,
    )

    np.testing.assert_array_equal(copied, batch.signatures[[2, 0]])
    assert load_group_row_source_signature_spec(source.attrs) == batch.spec
    assert target.attrs["source_row_signature_spec_digest"] == batch.spec.spec_digest


def test_strict_signature_envelope_rejects_wrong_array_and_extra_fields() -> None:
    batch = _build(
        np.asarray([11], dtype=np.uint64),
        np.zeros((1, 4), dtype=np.float32),
    )
    nested = {
        "array_path": "wrong",
        **batch.spec.to_attrs(prefix=""),
    }
    with pytest.raises(RowSourceSignatureError, match="wrong array path"):
        load_group_row_source_signature_spec({"row_signature": nested})
    nested["array_path"] = "source_row_signature"
    nested["unexpected"] = True
    with pytest.raises(RowSourceSignatureError, match="fields do not match"):
        load_group_row_source_signature_spec({"row_signature": nested})


def test_copy_selected_signatures_bootstraps_verified_auxiliary_proxy() -> None:
    root = zarr.group(store=zarr.storage.MemoryStore(), zarr_format=3)
    root.attrs.update(
        {
            "recording_id": "recording_fixture",
            "height": 8,
            "width": 8,
        }
    )
    source = root.create_group("crop")
    target = root.create_group("delta")
    source.create_array(
        "instance_key", data=np.asarray([11, 22, 33], dtype=np.uint64)
    )
    source.create_array(
        "frame_indices", data=np.asarray([0, 1, 2], dtype=np.int64)
    )
    source.create_array(
        "roi_coordinates_full",
        data=np.asarray([[0, 0], [1, 1], [2, 2]], dtype=np.int32),
    )
    source.create_array(
        "source_clip_indices", data=np.zeros(3, dtype=np.int64)
    )
    source.create_array(
        "source_clip_local_frame_indices",
        data=np.asarray([0, 1, 2], dtype=np.int64),
    )
    source.attrs.update(
        {
            "palette_run_completion_status": "auxiliary",
            "proxy_crop_complete": True,
            "stage_selector_eligible": False,
            "crop_storage_mode": "geometry_only",
            "stage": "crop_proxy",
            "schema": "palette_clipped_collection_proxy_crop_run_v1",
            "source_collection_id": "collection_fixture",
            "source_collection_path": "experiment_index/finalized_runs/collection_fixture",
            "height": 8,
            "width": 8,
            "roi_shape": [2, 2],
            "crop_policy": "centered_refined_bbox",
        }
    )

    copied = copy_selected_row_source_signatures(
        target,
        source,
        np.asarray([2, 0], dtype=np.int64),
        shard_rows=2,
        root=root,
    )

    assert copied.shape == (2, 32)
    assert target.attrs["source_row_signature_stage"] == "crop"
    assert (
        target.attrs["source_row_signature_spec"]["compatibility_context"][
            "bootstrap_schema_id"
        ]
        == "palette.legacy_proxy_crop_signature_bootstrap"
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "instance_keys": np.asarray([1, 1], dtype=np.uint64),
                "content_components": {"bbox": np.zeros((2, 4), dtype=np.float32)},
            },
            "instance_key values must be unique",
        ),
        (
            {
                "instance_keys": np.asarray([1.5]),
                "content_components": {"bbox": np.zeros((1, 4), dtype=np.float32)},
            },
            "integer dtype",
        ),
        (
            {
                "instance_keys": np.asarray([-1]),
                "content_components": {"bbox": np.zeros((1, 4), dtype=np.float32)},
            },
            "nonnegative",
        ),
        (
            {
                "instance_keys": np.asarray([1, 2], dtype=np.uint64),
                "content_components": {"bbox": np.zeros((1, 4), dtype=np.float32)},
            },
            "leading row count",
        ),
        (
            {
                "instance_keys": np.asarray([1], dtype=np.uint64),
                "revision_components": {"row_revision": np.asarray([-1])},
                "revision_authorities": {"row_revision": "authority_v1"},
            },
            "negative revision",
        ),
        (
            {
                "instance_keys": np.asarray([1], dtype=np.uint64),
                "revision_components": {"row_revision": np.asarray([0])},
            },
            "requires a named revision authority",
        ),
        (
            {
                "instance_keys": np.asarray([1], dtype=np.uint64),
                "content_components": {"label": np.asarray(["fish"], dtype=object)},
            },
            "fixed-width",
        ),
        (
            {
                "instance_keys": np.asarray([1], dtype=np.uint64),
                "content_components": {"same": np.asarray([1], dtype=np.int32)},
                "revision_components": {"same": np.asarray([0], dtype=np.int64)},
            },
            "declared more than once",
        ),
    ],
)
def test_invalid_signature_inputs_fail_closed(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(RowSourceSignatureError, match=message):
        build_row_source_signatures(stage="example", **kwargs)


def test_empty_batch_is_valid_when_component_shapes_are_explicit() -> None:
    batch = _build(
        np.asarray([], dtype=np.uint64),
        np.empty((0, 4), dtype=np.float32),
    )
    assert batch.signatures.shape == (0, 32)

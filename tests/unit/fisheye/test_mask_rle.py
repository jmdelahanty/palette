import numpy as np
import pytest

from fisheye.shared.mask_rle import (
    MASK_RLE_ENCODING,
    MASK_RLE_SCHEMA_ID,
    binary_mask_bbox_xyxy,
    component_stack_from_flat_rle,
    concatenate_encoded_mask_component_stacks,
    concatenate_encoded_mask_stacks,
    decode_binary_mask_rle,
    decode_mask_component_stack_rle,
    decode_mask_stack_rle,
    encode_binary_mask_rle,
    encode_mask_component_stack_rle,
    encode_mask_stack_rle,
)


def test_encode_decode_empty_mask_round_trips() -> None:
    mask = np.zeros((4, 5), dtype=np.uint8)

    counts = encode_binary_mask_rle(mask)
    decoded = decode_binary_mask_rle(counts, mask.shape)

    assert MASK_RLE_SCHEMA_ID == "palette_mask_rle_binary_v1"
    assert MASK_RLE_ENCODING == "coco_rle_fortran_v1"
    np.testing.assert_array_equal(decoded, mask)
    np.testing.assert_array_equal(counts, np.asarray([20], dtype=np.uint32))
    assert binary_mask_bbox_xyxy(mask) == (0, 0, 0, 0)


def test_encode_decode_all_foreground_uses_leading_zero_run() -> None:
    mask = np.ones((3, 2), dtype=np.uint8)

    counts = encode_binary_mask_rle(mask)
    decoded = decode_binary_mask_rle(counts, mask.shape)

    np.testing.assert_array_equal(counts, np.asarray([0, 6], dtype=np.uint32))
    np.testing.assert_array_equal(decoded, mask)
    assert binary_mask_bbox_xyxy(mask) == (0, 0, 2, 3)


def test_encode_decode_holes_and_disconnected_components_are_exact() -> None:
    mask = np.zeros((6, 7), dtype=np.uint8)
    mask[1:5, 1:3] = 1
    mask[2:4, 2] = 0
    mask[4, 5] = 1

    counts = encode_binary_mask_rle(mask)
    decoded = decode_binary_mask_rle(counts, mask.shape)

    np.testing.assert_array_equal(decoded, mask)
    assert int(np.sum(counts, dtype=np.int64)) == int(mask.size)
    assert binary_mask_bbox_xyxy(mask) == (1, 1, 6, 5)


def test_mask_stack_rle_round_trips_and_records_metadata() -> None:
    masks = np.zeros((2, 3, 5, 6), dtype=np.uint8)
    masks[0, 0, 1:3, 2:4] = 1
    masks[0, 2, :, :] = 1
    masks[1, 1, 4, 5] = 1

    encoded = encode_mask_stack_rle(masks)
    decoded = decode_mask_stack_rle(
        encoded.counts,
        encoded.indptr,
        n_rows=2,
        n_channels=3,
        shape_hw=encoded.shape_hw,
    )

    np.testing.assert_array_equal(decoded, masks)
    assert encoded.indptr.shape == (2 * 3 + 1,)
    np.testing.assert_array_equal(
        encoded.present,
        np.asarray([[True, False, True], [False, True, False]], dtype=bool),
    )
    np.testing.assert_array_equal(
        encoded.area_px,
        np.asarray([[4, 0, 30], [0, 1, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(encoded.bbox_xyxy[0, 0], np.asarray([2, 1, 4, 3], dtype=np.int32))
    np.testing.assert_array_equal(encoded.bbox_xyxy[0, 1], np.asarray([0, 0, 0, 0], dtype=np.int32))
    np.testing.assert_array_equal(encoded.bbox_xyxy[1, 1], np.asarray([5, 4, 6, 5], dtype=np.int32))


def test_decode_rejects_counts_with_wrong_shape_sum() -> None:
    with pytest.raises(ValueError, match="RLE counts sum"):
        decode_binary_mask_rle(np.asarray([1, 2, 3], dtype=np.uint32), (4, 4))


def test_decode_stack_rejects_bad_indptr() -> None:
    with pytest.raises(ValueError, match="indptr has"):
        decode_mask_stack_rle(
            np.asarray([4], dtype=np.uint32),
            np.asarray([0, 1], dtype=np.int64),
            n_rows=1,
            n_channels=2,
            shape_hw=(2, 2),
        )


def test_concatenate_encoded_mask_stacks_matches_one_shot_encoding() -> None:
    masks = np.zeros((5, 4, 8, 9), dtype=np.uint8)
    masks[0, 0, 1:4, 2:5] = 1
    masks[1, 2, :, :] = 1
    masks[2, 3, 5, 7] = 1
    masks[3, 1, 2:7, 3:8] = np.tri(5, 5, dtype=np.uint8)
    masks[4, 0, 0, 0] = 1
    masks[4, 3, 7, 8] = 1

    one_shot = encode_mask_stack_rle(masks)
    sharded = concatenate_encoded_mask_stacks(
        [
            encode_mask_stack_rle(masks[:2]),
            encode_mask_stack_rle(masks[2:4]),
            encode_mask_stack_rle(masks[4:]),
        ]
    )

    np.testing.assert_array_equal(sharded.counts, one_shot.counts)
    np.testing.assert_array_equal(sharded.indptr, one_shot.indptr)
    np.testing.assert_array_equal(sharded.present, one_shot.present)
    np.testing.assert_array_equal(sharded.area_px, one_shot.area_px)
    np.testing.assert_array_equal(sharded.bbox_xyxy, one_shot.bbox_xyxy)
    decoded = decode_mask_stack_rle(
        sharded.counts,
        sharded.indptr,
        n_rows=masks.shape[0],
        n_channels=masks.shape[1],
        shape_hw=sharded.shape_hw,
    )
    np.testing.assert_array_equal(decoded, masks)


def test_concatenate_encoded_mask_stacks_rejects_shape_mismatch() -> None:
    first = encode_mask_stack_rle(np.zeros((1, 2, 4, 4), dtype=np.uint8))
    second = encode_mask_stack_rle(np.zeros((1, 2, 4, 5), dtype=np.uint8))

    with pytest.raises(ValueError, match="different shapes"):
        concatenate_encoded_mask_stacks([first, second])


def test_concatenate_encoded_mask_stacks_rejects_channel_mismatch() -> None:
    first = encode_mask_stack_rle(np.zeros((1, 2, 4, 4), dtype=np.uint8))
    second = encode_mask_stack_rle(np.zeros((1, 3, 4, 4), dtype=np.uint8))

    with pytest.raises(ValueError, match="different channel counts"):
        concatenate_encoded_mask_stacks([first, second])


def test_component_stack_rle_round_trips_with_names() -> None:
    masks = np.zeros((3, 4, 8, 9), dtype=np.uint8)
    masks[0, 0, 1:4, 2:5] = 1
    masks[1, 1, :, :] = 1
    masks[2, 2, 3, 4] = 1
    masks[2, 3, 5:7, 6:8] = 1
    names = ("subject_body", "eye_left", "eye_right", "swim_bladder")

    encoded = encode_mask_component_stack_rle(masks, component_names=names)
    decoded = decode_mask_component_stack_rle(encoded)

    assert encoded.n_rows == 3
    assert tuple(component.component_name for component in encoded.components) == names
    np.testing.assert_array_equal(decoded, masks)
    body = encoded.components[0]
    assert body.component_name == "subject_body"
    assert body.component_index == 0
    assert body.indptr.shape == (4,)
    np.testing.assert_array_equal(body.present, np.asarray([True, False, False], dtype=bool))
    np.testing.assert_array_equal(body.area_px, np.asarray([9, 0, 0], dtype=np.int32))
    np.testing.assert_array_equal(body.bbox_xyxy[0], np.asarray([2, 1, 5, 4], dtype=np.int32))


def test_component_stack_shard_concatenation_matches_one_shot() -> None:
    masks = np.zeros((5, 3, 7, 8), dtype=np.uint8)
    masks[0, 0, 1:3, 1:4] = 1
    masks[1, 1, 2:5, 2:6] = 1
    masks[2, 2, :, :] = 1
    masks[3, 0, 6, 7] = 1
    masks[4, 1, 0, 0] = 1
    names = ("body", "eye", "bladder")

    one_shot = encode_mask_component_stack_rle(masks, component_names=names)
    sharded = concatenate_encoded_mask_component_stacks(
        [
            encode_mask_component_stack_rle(masks[:2], component_names=names),
            encode_mask_component_stack_rle(masks[2:4], component_names=names),
            encode_mask_component_stack_rle(masks[4:], component_names=names),
        ]
    )

    assert sharded.n_rows == one_shot.n_rows
    assert tuple(component.component_name for component in sharded.components) == names
    np.testing.assert_array_equal(decode_mask_component_stack_rle(sharded), masks)
    for actual, expected in zip(sharded.components, one_shot.components):
        np.testing.assert_array_equal(actual.counts, expected.counts)
        np.testing.assert_array_equal(actual.indptr, expected.indptr)
        np.testing.assert_array_equal(actual.present, expected.present)
        np.testing.assert_array_equal(actual.area_px, expected.area_px)
        np.testing.assert_array_equal(actual.bbox_xyxy, expected.bbox_xyxy)


def test_component_stack_requires_matching_component_names() -> None:
    masks = np.zeros((1, 2, 4, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match="Expected 2 component names"):
        encode_mask_component_stack_rle(masks, component_names=("only_one",))


def test_component_stack_concatenation_rejects_identity_mismatch() -> None:
    first = encode_mask_component_stack_rle(np.zeros((1, 2, 4, 4), dtype=np.uint8), component_names=("a", "b"))
    second = encode_mask_component_stack_rle(np.zeros((1, 2, 4, 4), dtype=np.uint8), component_names=("a", "c"))

    with pytest.raises(ValueError, match="different component identities"):
        concatenate_encoded_mask_component_stacks([first, second])


def test_component_stack_from_flat_rle_preserves_masks_and_metadata() -> None:
    masks = np.zeros((4, 3, 6, 7), dtype=np.uint8)
    masks[0, 0, 1:4, 2:5] = 1
    masks[1, 1, :, :] = 1
    masks[2, 2, 3, 4] = 1
    masks[3, 0, 5, 6] = 1
    names = ("body", "eye", "bladder")

    flat = encode_mask_stack_rle(masks)
    components = component_stack_from_flat_rle(flat, component_names=names)

    assert tuple(component.component_name for component in components.components) == names
    np.testing.assert_array_equal(decode_mask_component_stack_rle(components), masks)
    np.testing.assert_array_equal(components.components[0].present, flat.present[:, 0])
    np.testing.assert_array_equal(components.components[1].area_px, flat.area_px[:, 1])
    np.testing.assert_array_equal(components.components[2].bbox_xyxy, flat.bbox_xyxy[:, 2])

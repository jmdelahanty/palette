from __future__ import annotations

import numpy as np
import pytest
import zarr

from fisheye.shared.zarr.subject_mask_core_publication import (
    SUBJECT_MASK_CORE_RUN_MANIFEST_ATTRIBUTE,
    publish_selector_ineligible_subject_mask_core_snapshot,
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


def _fixture() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
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
        "metrics/prob_max": np.max(
            probabilities.astype(np.float32) / np.float32(255.0),
            axis=(2, 3),
        ).astype(np.float32),
    }
    refined = {**common, "masks_roi": masks}
    crop = {
        "instance_key": common["instance_key"],
        "source_acquisition_frame_index": frames,
        "source_crop_xywh": common["source_crop_xywh"],
    }
    return raw, {**refined, "_crop": crop}  # type: ignore[dict-item]


@pytest.mark.parametrize(
    ("kind", "family", "payload"),
    (
        ("raw_probability_uint8", "subject_mask_runs", "mask_probs_roi"),
        ("refined_dense_core", "refined_subject_masks_runs", "masks_roi"),
    ),
)
def test_subject_mask_core_publication_round_trip(
    tmp_path: object,
    kind: str,
    family: str,
    payload: str,
) -> None:
    raw, refined_with_crop = _fixture()
    crop = refined_with_crop.pop("_crop")
    arrays = raw if kind == "raw_probability_uint8" else refined_with_crop
    run_id = f"{kind}_001"
    publication = publish_selector_ineligible_subject_mask_core_snapshot(
        arrays,
        source_crop_arrays=crop,  # type: ignore[arg-type]
        source_manifest={
            "schema_id": "palette.subject_mask.source_fixture",
            "schema_version": 1,
            "run_id": "source_001",
        },
        n_frames=4,
        components=_components(),
        destination=tmp_path / f"{kind}.zarr",  # type: ignore[operator]
        run_id=run_id,
        kind=kind,
        source_run_path="subject_mask_shard_runs/source_001",
        source_attributes={"source_crop_run": "crop_001"},
        created_by="pytest",
    )

    root = zarr.open_group(
        str(publication.output_path), mode="r", use_consolidated=True
    )
    run = root[f"{family}/{run_id}"]
    assert run.attrs["stage_selector_eligible"] is False
    assert run.attrs["status"] == "complete"
    assert run.attrs[SUBJECT_MASK_CORE_RUN_MANIFEST_ATTRIBUTE] == publication.manifest
    assert set(run.array_keys()) | {
        f"metrics/{name}" for name in run["metrics"].array_keys()
    } == set(
        publication.plans.entries[index].rule.path
        for index in range(len(publication.plans.entries))
    )
    np.testing.assert_array_equal(run[payload][...], arrays[payload])
    parent = root[family]
    for selector in (
        "latest",
        "latest_complete",
        "latest_pending",
        "authoritative_run",
    ):
        assert parent.attrs.get(selector) is None


def test_subject_mask_core_publication_rejects_crop_identity_mismatch(
    tmp_path: object,
) -> None:
    raw, refined_with_crop = _fixture()
    crop = refined_with_crop.pop("_crop")
    crop["instance_key"] = np.asarray([999, 102, 201, 301], dtype=np.uint64)

    with pytest.raises(ValueError, match="schema validation failed"):
        publish_selector_ineligible_subject_mask_core_snapshot(
            raw,
            source_crop_arrays=crop,  # type: ignore[arg-type]
            source_manifest={"schema_id": "fixture", "schema_version": 1},
            n_frames=4,
            components=_components(),
            destination=tmp_path / "raw.zarr",  # type: ignore[operator]
            run_id="raw_001",
            kind="raw_probability_uint8",
            source_run_path="subject_mask_shard_runs/source_001",
            created_by="pytest",
        )

    assert not (tmp_path / "raw.zarr").exists()  # type: ignore[operator]


def test_raw_publication_canonicalizes_one_ulp_probability_max(
    tmp_path: object,
) -> None:
    raw, refined_with_crop = _fixture()
    crop = refined_with_crop.pop("_crop")
    source_prob_max = raw["metrics/prob_max"].copy()
    source_prob_max[0, 0] = np.nextafter(
        source_prob_max[0, 0],
        np.float32(np.inf),
        dtype=np.float32,
    )
    raw["metrics/prob_max"] = source_prob_max

    publication = publish_selector_ineligible_subject_mask_core_snapshot(
        raw,
        source_crop_arrays=crop,  # type: ignore[arg-type]
        source_manifest={"schema_id": "fixture", "schema_version": 1},
        n_frames=4,
        components=_components(),
        destination=tmp_path / "raw.zarr",  # type: ignore[operator]
        run_id="raw_001",
        kind="raw_probability_uint8",
        source_run_path="subject_mask_shard_runs/source_001",
        created_by="pytest",
    )

    run = zarr.open_group(
        str(publication.output_path), mode="r", use_consolidated=True
    )["subject_mask_runs/raw_001"]
    canonical = np.max(raw["mask_probs_roi"], axis=(2, 3)).astype(
        np.float32
    ) / np.float32(255.0)
    np.testing.assert_array_equal(run["metrics/prob_max"][...], canonical)
    receipt = publication.manifest["payload"]["write_receipt"][
        "derived_metric_canonicalization"
    ]
    assert receipt["source_mismatch_count"] == 1


def test_raw_publication_rejects_material_probability_max_drift(
    tmp_path: object,
) -> None:
    raw, refined_with_crop = _fixture()
    crop = refined_with_crop.pop("_crop")
    raw["metrics/prob_max"] = raw["metrics/prob_max"].copy()
    raw["metrics/prob_max"][0, 0] += np.float32(0.01)

    with pytest.raises(ValueError, match="differs materially"):
        publish_selector_ineligible_subject_mask_core_snapshot(
            raw,
            source_crop_arrays=crop,  # type: ignore[arg-type]
            source_manifest={"schema_id": "fixture", "schema_version": 1},
            n_frames=4,
            components=_components(),
            destination=tmp_path / "raw.zarr",  # type: ignore[operator]
            run_id="raw_001",
            kind="raw_probability_uint8",
            source_run_path="subject_mask_shard_runs/source_001",
            created_by="pytest",
        )

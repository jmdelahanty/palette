from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

import fisheye.shared.refined_subject_masks_io as io_module
from fisheye.shared.mask_rle import encode_mask_component_stack_rle
from fisheye.shared.mask_store import MaskStoreError, open_mask_store
from fisheye.shared.refined_subject_masks_io import (
    RefinedSubjectMasksIOError,
    discover_refined_subject_masks_run_options,
    load_canonical_refined_subject_masks_run_tables,
    load_refined_subject_masks_run_tables,
    resolve_historical_refined_subject_masks_run,
    resolve_refined_subject_masks_run,
)


def _write_array(group: zarr.Group, name: str, values: np.ndarray) -> None:
    group.create_array(name, data=values, chunks=values.shape, overwrite=True)


def _write_component_rle(run: zarr.Group, masks: np.ndarray, labels: tuple[str, ...]) -> None:
    encoded = encode_mask_component_stack_rle(masks, component_names=labels)
    rle = run.create_group("mask_rle")
    rle.attrs.update(
        {
            "schema_id": "palette_mask_rle_binary_v1",
            "mask_encoding": "coco_rle_fortran_v1",
            "mask_value_semantics": "binary_0_1",
            "encoded_shape_hw": [int(encoded.shape_hw[0]), int(encoded.shape_hw[1])],
            "layout": "component_groups",
            "component_names": list(labels),
        }
    )
    components = rle.create_group("components")
    for component in encoded.components:
        group = components.create_group(f"{component.component_index:02d}_{component.component_name}")
        group.attrs.update(
            {
                "component_name": component.component_name,
                "component_index": int(component.component_index),
            }
        )
        _write_array(group, "counts", component.counts)
        _write_array(group, "indptr", component.indptr)
        _write_array(group, "present", component.present)
        _write_array(group, "area_px", component.area_px)
        _write_array(group, "bbox_xyxy", component.bbox_xyxy)


def _make_refined_subject_archive(tmp_path: Path) -> zarr.Group:
    root = zarr.open_group(str(tmp_path / "refined_subject.zarr"), mode="w")
    parent = root.create_group("refined_subject_masks_runs")
    parent.attrs["latest"] = "refined_1"
    run = parent.create_group("refined_1")
    run.attrs.update(
        {
            "method": "smart_finalizer",
            "label_schema_id": "subject_v1_lr",
            "mask_labels": ["subject_body", "eye_left", "eye_right", "swim_bladder"],
            "component_review_statuses": {
                "subject_body": {"state": "approved"},
                "eye_left": {"state": "approved"},
                "eye_right": {"state": "approved"},
                "swim_bladder": {"state": "missing"},
            },
        }
    )

    masks = np.zeros((3, 4, 8, 8), dtype=np.uint8)
    masks[:, 0, 2:6, 2:6] = 1
    masks[:, 1, 2:4, 1:3] = 1
    masks[:, 2, 2:4, 5:7] = 1
    _write_array(run, "masks_roi", masks)
    _write_array(run, "available_channels", np.asarray([True, True, True, False], dtype=bool))
    _write_array(run, "edit_applied", np.asarray([[False, False, False, False]] * 3, dtype=bool))
    _write_array(run, "detection_source", np.asarray([0, 0, 0], dtype=np.int8))
    _write_array(run, "frame_indices", np.asarray([10, 11, 12], dtype=np.int64))
    _write_array(run, "detection_indices", np.asarray([0, 1, 2], dtype=np.int64))

    metrics = run.create_group("metrics")
    _write_array(metrics, "mask_present", np.asarray([[True, True, True, False]] * 3, dtype=bool))
    _write_array(metrics, "area_px", np.asarray([[16.0, 4.0, 4.0, np.nan]] * 3, dtype=np.float32))
    _write_array(metrics, "centroid_xy", np.zeros((3, 4, 2), dtype=np.float32))
    _write_array(metrics, "centroid_valid", np.asarray([[True, True, True, False]] * 3, dtype=bool))
    _write_array(metrics, "bbox_xyxy", np.zeros((3, 4, 4), dtype=np.float32))
    _write_array(metrics, "bbox_valid", np.asarray([[True, True, True, False]] * 3, dtype=bool))

    components = run.create_group("components")
    body = components.create_group("subject_body")
    body.attrs["component_schema_id"] = "subject_body_v1"
    _write_array(body, "row_revision", np.asarray([0, 1, 0], dtype=np.int64))
    _write_array(body, "manual_override", np.asarray([False, True, False], dtype=bool))
    body_metrics = body.create_group("metrics")
    _write_array(body_metrics, "component_count", np.asarray([1, 1, 0], dtype=np.int32))
    _write_array(body_metrics, "largest_component_fraction", np.asarray([1.0, 1.0, np.nan], dtype=np.float32))
    qc = body.create_group("qc")
    _write_array(qc, "severe_qc_failure", np.asarray([False, True, False], dtype=bool))
    _write_array(qc, "requires_review", np.asarray([False, True, False], dtype=bool))

    for component_name in ("eye_left", "eye_right"):
        eye = components.create_group(component_name)
        _write_array(eye, "row_revision", np.asarray([0, 0, 0], dtype=np.int64))
        geometry = eye.create_group("geometry")
        _write_array(geometry, "ellipse_params", np.zeros((3, 5), dtype=np.float32))
        _write_array(geometry, "ellipse_success", np.asarray([True, True, False], dtype=bool))

    relations = run.create_group("relations")
    eye_pair = relations.create_group("eye_pair")
    eye_pair.attrs["relation_schema_id"] = "refined_subject_eye_pair_relation_v1"
    eye_pair_metrics = eye_pair.create_group("metrics")
    _write_array(eye_pair_metrics, "separation_px", np.asarray([20.0, 21.0, np.nan], dtype=np.float32))
    _write_array(eye_pair_metrics, "separation_valid", np.asarray([True, True, False], dtype=bool))
    return root


def _refined_run(root: zarr.Group) -> zarr.Group:
    return root["refined_subject_masks_runs/refined_1"]


def test_discover_refined_subject_masks_options_uses_latest_and_mask_metadata(tmp_path: Path) -> None:
    root = _make_refined_subject_archive(tmp_path)

    options = discover_refined_subject_masks_run_options(root)

    assert len(options) == 1
    option = options[0]
    assert option.run_name == "refined_1"
    assert option.run_path == "refined_subject_masks_runs/refined_1"
    assert option.method == "smart_finalizer"
    assert option.label_schema_id == "subject_v1_lr"
    assert option.mask_labels == ("subject_body", "eye_left", "eye_right", "swim_bladder")
    assert option.available_components == ("subject_body", "eye_left", "eye_right")
    assert option.n_rows == 3
    assert option.channel_count == 4
    assert option.roi_shape == (8, 8)
    assert option.is_latest is True
    assert "latest" in option.label


def test_discover_refined_subject_masks_options_uses_compact_store_metadata(tmp_path: Path) -> None:
    root = _make_refined_subject_archive(tmp_path)
    run = _refined_run(root)
    labels = tuple(str(value) for value in run.attrs["mask_labels"])
    dense = np.asarray(run["masks_roi"][:], dtype=np.uint8)
    _write_component_rle(run, dense, labels)
    for name in ("masks_roi", "frame_indices", "detection_indices", "edit_applied", "detection_source", "metrics"):
        if name in run:
            del run[name]

    options = discover_refined_subject_masks_run_options(root)

    assert len(options) == 1
    option = options[0]
    assert option.run_name == "refined_1"
    assert option.n_rows == 3
    assert option.channel_count == 4
    assert option.roi_shape == (8, 8)
    assert option.available_components == ("subject_body", "eye_left", "eye_right")


def test_load_refined_subject_masks_run_tables_reads_logical_surfaces(tmp_path: Path) -> None:
    root = _make_refined_subject_archive(tmp_path)

    tables = load_refined_subject_masks_run_tables(root, run_name="refined_subject_masks_runs/refined_1")

    assert tables.run_name == "refined_1"
    assert tables.run_path == "refined_subject_masks_runs/refined_1"
    assert tables.mask_labels == ("subject_body", "eye_left", "eye_right", "swim_bladder")
    assert tables.component_index("eye_right") == 2
    assert tables.component_available("swim_bladder") is False
    assert tables.resolve_components(("subject_body", "swim_bladder")) == (("subject_body", 0),)
    assert tables.require_masks_roi().shape == (3, 4, 8, 8)
    assert tables.run_arrays["frame_indices"].tolist() == [10, 11, 12]
    assert tables.metrics["mask_present"].shape == (3, 4)
    assert tables.components["subject_body"].arrays["row_revision"].tolist() == [0, 1, 0]
    assert tables.components["subject_body"].qc["requires_review"].tolist() == [False, True, False]
    assert tables.components["eye_left"].geometry["ellipse_params"].shape == (3, 5)
    np.testing.assert_allclose(
        tables.relations["eye_pair"].metrics["separation_px"],
        [20.0, 21.0, np.nan],
        equal_nan=True,
    )
    assert tables.source_paths["masks_roi"] == "refined_subject_masks_runs/refined_1/masks_roi"
    assert (
        tables.source_paths["components/subject_body/qc/requires_review"]
        == "refined_subject_masks_runs/refined_1/components/subject_body/qc/requires_review"
    )
    assert tables.require_mask_store().encoding == "dense_uint8"
    np.testing.assert_array_equal(
        tables.require_mask_store().read_dense(rows=[1], channels=["subject_body"]),
        np.asarray(root["refined_subject_masks_runs/refined_1/masks_roi"][1:2, 0:1], dtype=np.uint8),
    )


def test_load_refined_subject_masks_run_tables_can_skip_dense_masks(tmp_path: Path) -> None:
    root = _make_refined_subject_archive(tmp_path)

    tables = load_refined_subject_masks_run_tables(root, include_masks_roi=False)

    assert tables.masks_roi is None
    assert tables.n_rows == 3
    with pytest.raises(RefinedSubjectMasksIOError, match="loaded without masks_roi"):
        tables.require_masks_roi()


def test_canonical_tables_fail_closed_on_legacy_compatibility_run(tmp_path: Path) -> None:
    root = _make_refined_subject_archive(tmp_path)

    with pytest.raises(RefinedSubjectMasksIOError, match="not a valid canonical"):
        load_canonical_refined_subject_masks_run_tables(root)


def test_canonical_tables_carry_the_exact_strict_coordinate_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _make_refined_subject_archive(tmp_path)
    run = root["refined_subject_masks_runs/refined_1"]
    _write_array(run, "source_crop_row_ids", np.asarray([0, 1, 2], dtype=np.int64))
    _write_array(run, "instance_key", np.asarray([100, 101, 102], dtype=np.uint64))
    _write_array(
        run,
        "source_acquisition_frame_index",
        np.asarray([10, 11, 12], dtype=np.int64),
    )
    _write_array(
        run,
        "source_crop_xywh",
        np.asarray([[1, 2, 8, 8], [1, 2, 8, 8], [1, 2, 8, 8]], dtype=np.int32),
    )
    sentinel = SimpleNamespace(
        context=SimpleNamespace(
            run_path="refined_subject_masks_runs/refined_1",
            labels=("subject_body", "eye_left", "eye_right", "swim_bladder"),
        )
    )
    monkeypatch.setattr(
        io_module,
        "load_persisted_refined_subject_mask_coordinate_surfaces",
        lambda _root, _path: sentinel,
    )
    revalidated: list[object] = []
    monkeypatch.setattr(
        io_module,
        "require_bound_refined_subject_mask_coordinate_surfaces",
        lambda value: revalidated.append(value) or value,
    )

    loaded = load_canonical_refined_subject_masks_run_tables(root)

    assert loaded.coordinate_surfaces is sentinel
    assert loaded.run_path == "refined_subject_masks_runs/refined_1"
    assert loaded.masks_roi.shape == (3, 4, 8, 8)
    assert set(loaded.tables.run_arrays) == {
        "available_channels",
        "source_crop_row_ids",
        "instance_key",
        "source_acquisition_frame_index",
        "source_crop_xywh",
    }
    assert loaded.tables.components == {}
    assert loaded.tables.relations == {}
    assert "edit_applied" not in loaded.tables.run_arrays
    assert revalidated == [sentinel]


def test_load_refined_subject_masks_run_tables_rejects_missing_component(tmp_path: Path) -> None:
    root = _make_refined_subject_archive(tmp_path)

    with pytest.raises(RefinedSubjectMasksIOError, match="not present"):
        load_refined_subject_masks_run_tables(root, component_names=("missing_component",))


def test_resolve_refined_subject_masks_run_rejects_missing_run(tmp_path: Path) -> None:
    root = _make_refined_subject_archive(tmp_path)

    with pytest.raises(RefinedSubjectMasksIOError, match="not found"):
        resolve_refined_subject_masks_run(root, "missing_refined")


def test_normal_refined_resolver_never_uses_lexicographic_child_fallback(
    tmp_path: Path,
) -> None:
    root = _make_refined_subject_archive(tmp_path)
    parent = root["refined_subject_masks_runs"]
    del parent.attrs["latest"]
    fallback = parent.create_group("zzz_historical")
    fallback.attrs["mask_labels"] = ["subject_body"]

    with pytest.raises(
        RefinedSubjectMasksIOError,
        match="no valid controlled selector",
    ):
        resolve_refined_subject_masks_run(root)

    _group, run_name, run_path = resolve_historical_refined_subject_masks_run(root)
    assert run_name == "zzz_historical"
    assert run_path == "refined_subject_masks_runs/zzz_historical"


def test_normal_refined_resolver_rejects_invalid_controlled_selector(
    tmp_path: Path,
) -> None:
    root = _make_refined_subject_archive(tmp_path)
    parent = root["refined_subject_masks_runs"]
    parent.attrs["authoritative_run"] = "missing_authority"

    with pytest.raises(
        RefinedSubjectMasksIOError,
        match="authoritative_run='missing_authority'",
    ):
        resolve_refined_subject_masks_run(root)


def test_open_mask_store_reads_dense_masks_by_rows_and_components(tmp_path: Path) -> None:
    root = _make_refined_subject_archive(tmp_path)
    run = _refined_run(root)

    store = open_mask_store(run, source_path="refined_subject_masks_runs/refined_1")

    assert store.encoding == "dense_uint8"
    assert store.storage_surface == "masks_roi"
    assert store.storage_path == "refined_subject_masks_runs/refined_1/masks_roi"
    assert store.mask_labels == ("subject_body", "eye_left", "eye_right", "swim_bladder")
    assert store.shape == (3, 4, 8, 8)
    masks = store.read_dense(rows=[0, 2], channels=["eye_left", "eye_right"])
    expected = np.asarray(run["masks_roi"][:], dtype=np.uint8)[[0, 2]][:, [1, 2]]
    np.testing.assert_array_equal(masks, expected)


def test_open_mask_store_reads_component_rle_without_dense_masks(tmp_path: Path) -> None:
    root = _make_refined_subject_archive(tmp_path)
    run = _refined_run(root)
    labels = tuple(str(value) for value in run.attrs["mask_labels"])
    dense = np.asarray(run["masks_roi"][:], dtype=np.uint8)
    _write_component_rle(run, dense, labels)
    del run["masks_roi"]

    store = open_mask_store(run, source_path="refined_subject_masks_runs/refined_1", prefer="rle")

    assert store.encoding == "component_rle_v1"
    assert store.storage_surface == "mask_rle"
    assert store.storage_path == "refined_subject_masks_runs/refined_1/mask_rle"
    assert store.shape == (3, 4, 8, 8)
    masks = store.read_dense(rows=slice(1, 3), channels=("subject_body", "eye_right"))
    np.testing.assert_array_equal(masks, dense[1:3][:, [0, 2]])
    one = store.read_dense(rows=0, channels="eye_left")
    np.testing.assert_array_equal(one, dense[0:1, 1:2])


def test_open_mask_store_prefers_dense_when_both_surfaces_exist(tmp_path: Path) -> None:
    root = _make_refined_subject_archive(tmp_path)
    run = _refined_run(root)
    labels = tuple(str(value) for value in run.attrs["mask_labels"])
    dense = np.asarray(run["masks_roi"][:], dtype=np.uint8)
    _write_component_rle(run, dense, labels)

    dense_store = open_mask_store(run)
    compact_store = open_mask_store(run, prefer="rle")
    assert dense_store.encoding == "dense_uint8"
    assert dense_store.storage_surface == "masks_roi"
    assert compact_store.encoding == "component_rle_v1"
    assert compact_store.storage_surface == "mask_rle"


def test_open_mask_store_rejects_missing_component_payload(tmp_path: Path) -> None:
    root = _make_refined_subject_archive(tmp_path)
    run = _refined_run(root)
    labels = tuple(str(value) for value in run.attrs["mask_labels"])
    dense = np.asarray(run["masks_roi"][:], dtype=np.uint8)
    _write_component_rle(run, dense, labels)
    del run["masks_roi"]
    del run["mask_rle/components/02_eye_right"]

    store = open_mask_store(run, prefer="rle")
    with pytest.raises(MaskStoreError, match="missing component index 2"):
        store.read_dense(rows=[0], channels=["eye_right"])


def test_load_refined_subject_masks_tables_uses_component_rle_store_when_dense_skipped(tmp_path: Path) -> None:
    root = _make_refined_subject_archive(tmp_path)
    run = _refined_run(root)
    labels = tuple(str(value) for value in run.attrs["mask_labels"])
    dense = np.asarray(run["masks_roi"][:], dtype=np.uint8)
    _write_component_rle(run, dense, labels)

    tables = load_refined_subject_masks_run_tables(root, include_masks_roi=False)

    assert tables.masks_roi is None
    assert tables.require_mask_store().encoding == "component_rle_v1"
    np.testing.assert_array_equal(
        tables.require_mask_store().read_dense(rows=[0, 2], channels=["eye_left"]),
        dense[[0, 2]][:, 1:2],
    )
    assert tables.source_paths["mask_store"] == "refined_subject_masks_runs/refined_1/mask_rle"


def test_load_refined_subject_masks_tables_uses_mask_store_for_shape_when_dense_and_lineage_absent(
    tmp_path: Path,
) -> None:
    root = _make_refined_subject_archive(tmp_path)
    run = _refined_run(root)
    labels = tuple(str(value) for value in run.attrs["mask_labels"])
    dense = np.asarray(run["masks_roi"][:], dtype=np.uint8)
    _write_component_rle(run, dense, labels)
    for name in ("masks_roi", "frame_indices", "detection_indices", "edit_applied", "detection_source", "metrics"):
        if name in run:
            del run[name]

    tables = load_refined_subject_masks_run_tables(
        root,
        include_masks_roi=True,
        include_metrics=False,
        include_components=False,
        include_relations=False,
        run_array_names=("available_channels",),
    )

    assert tables.masks_roi is None
    assert tables.n_rows == 3
    assert tables.channel_count == 4
    assert tables.require_mask_store().encoding == "component_rle_v1"
    assert tables.source_paths["mask_store"] == "refined_subject_masks_runs/refined_1/mask_rle"


def test_load_refined_subject_masks_tables_prefers_dense_when_rle_is_stale(tmp_path: Path) -> None:
    root = _make_refined_subject_archive(tmp_path)
    run = _refined_run(root)
    labels = tuple(str(value) for value in run.attrs["mask_labels"])
    dense = np.asarray(run["masks_roi"][:], dtype=np.uint8)
    _write_component_rle(run, dense, labels)
    run.attrs["mask_rle_stale"] = True

    tables = load_refined_subject_masks_run_tables(root, include_masks_roi=True)

    assert tables.require_mask_store().encoding == "dense_uint8"
    np.testing.assert_array_equal(
        tables.require_mask_store().read_dense(rows=[0], channels=["subject_body"]),
        dense[0:1, 0:1],
    )


def test_load_refined_subject_masks_tables_rejects_stale_rle_when_dense_skipped(tmp_path: Path) -> None:
    root = _make_refined_subject_archive(tmp_path)
    run = _refined_run(root)
    labels = tuple(str(value) for value in run.attrs["mask_labels"])
    dense = np.asarray(run["masks_roi"][:], dtype=np.uint8)
    _write_component_rle(run, dense, labels)
    run.attrs["mask_rle_stale"] = True

    with pytest.raises(RefinedSubjectMasksIOError, match="mask_rle is marked stale"):
        load_refined_subject_masks_run_tables(root, include_masks_roi=False)


def test_load_refined_subject_masks_tables_rejects_stale_compact_only_store(tmp_path: Path) -> None:
    root = _make_refined_subject_archive(tmp_path)
    run = _refined_run(root)
    labels = tuple(str(value) for value in run.attrs["mask_labels"])
    dense = np.asarray(run["masks_roi"][:], dtype=np.uint8)
    _write_component_rle(run, dense, labels)
    del run["masks_roi"]
    run.attrs["mask_rle_stale"] = True

    with pytest.raises(RefinedSubjectMasksIOError, match="mask_rle is marked stale"):
        load_refined_subject_masks_run_tables(root)

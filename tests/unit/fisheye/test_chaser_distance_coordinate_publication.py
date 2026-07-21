from __future__ import annotations

import hashlib
from pathlib import Path
import shutil

import numpy as np
import pytest
import zarr

from fisheye.analysis import import_stimulus_to_zarr as stimulus_import
from fisheye.analysis import chaser_distance_coordinate_publication as publication_module
from fisheye.analysis import chaser_distance_runs as runs_module
from fisheye.analysis.chaser_distance_coordinate_publication import (
    CHASER_COLLECTION_AUTHORITY_ATTR,
    CHASER_DISTANCE_PARENT_PUBLICATION_LEASE_ATTR,
    CHASER_DISTANCE_PUBLICATION_GENERATION_ATTR,
    CHASER_DISTANCE_PUBLICATION_POLICY,
    CHASER_DISTANCE_PUBLICATION_POLICY_ATTR,
    COORDINATE_CONTRACT,
    INPUT_AUTHORITY_ATTR,
    MEASUREMENT_AUTHORITY_ATTR,
    MEASUREMENT_DESCRIPTOR_ATTR,
    PUBLICATION_SEAL_ATTR,
    ChaserDistanceCoordinateError,
    activate_chaser_distance_run,
    load_bound_chaser_distance_run,
)
from fisheye.analysis.chaser_distance_runs import (
    build_chaser_distance_result,
    write_chaser_distance_run,
)
from fisheye.detection import detect_yolo as detect_mod
from fisheye.shared.coordinate_descriptor import COORDINATE_DESCRIPTOR_ATTR
from fisheye.shared.coordinate_record import coordinate_record_sha256
from fisheye.shared.instance_keys import instance_key_attrs, mint_detection_instance_keys
from fisheye.shared.observation_coordinate_publication import (
    derive_detection_source_camera_geometry,
    publish_detection_backend_result_projection,
    publish_detection_instance_key_derivation,
    publish_detection_observation_cardinality,
    publish_detection_observation_geometry,
)
from fisheye.shared.pixel_frame_authority import (
    load_persisted_acquisition_camera_authority,
)
from fisheye.shared.immutable_yolo_storage import (
    IMMUTABLE_YOLO_STORAGE_ATTR,
    IMMUTABLE_YOLO_STORAGE_SCHEMA,
)
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_CONTRACT,
    RUN_COMPLETION_CONTRACT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    mark_run_started,
)
from tests.unit.fisheye.test_import_stimulus_to_zarr_paths import (
    _prepare_acquisition_authority,
    _write_stimulus_h5_with_arena_relative_chaser_states,
)


_TEST_MODEL_ARTIFACT = {
    "role": "detect_model",
    "path": "/models/chaser-distance-test.pt",
    "fingerprint_scheme": "content_v1",
    "sha256": "c" * 64,
    "size_bytes": 123,
    "mtime_ns": 456,
    "source": "computed",
}


def _publish_detection(root: zarr.Group) -> str:
    _authority_node, acquisition = load_persisted_acquisition_camera_authority(
        root,
        expected_camera_id="2010093",
    )
    parent = root.require_group("detect_runs")
    run = parent.create_group("canonical")
    mark_run_started(run, run_name="canonical", stage="detect")
    evidence, _checkpoints = detect_mod._publish_detection_frame_evidence(  # noqa: SLF001
        root,
        run,
        acquisition_frame=acquisition,
    )
    frame_indices = np.asarray([0, 1], dtype=np.int32)
    source_frames = frame_indices.astype(np.int64)
    normalized = np.asarray(
        [
            [280.0 / 4512.0, 530.0 / 4512.0, 20.0 / 4512.0, 20.0 / 4512.0],
            [290.0 / 4512.0, 540.0 / 4512.0, 20.0 / 4512.0, 20.0 / 4512.0],
        ],
        dtype=np.float64,
    )
    bbox_img, centers = derive_detection_source_camera_geometry(
        normalized,
        frame_evidence=evidence,
    )
    class_ids = np.zeros(2, dtype=np.int32)
    keys = mint_detection_instance_keys(
        recording_identity=acquisition.record.recording_id,
        frame_indices=source_frames,
        bbox_norm_coords=normalized,
        class_ids=class_ids,
    )
    arrays = {
        "frame_indices": frame_indices,
        "source_acquisition_frame_index": source_frames,
        "bbox_norm_coords": normalized,
        "bbox_img_xyxy": bbox_img,
        "centers_img_xy": centers,
        "scores": np.asarray([0.8, 0.9], dtype=np.float32),
        "class_ids": class_ids,
        "instance_key": keys,
        "frame_counts": np.asarray([1, 1], dtype=np.int32),
        "n_detections": np.asarray([1, 1], dtype=np.int32),
    }
    for name, values in arrays.items():
        run.create_array(name, data=values)
    dense_mapping = np.arange(2, dtype=np.int64)
    run.attrs.update(
        {
            **instance_key_attrs(
                acquisition.record.recording_id,
                frame_domain="recording_parent_frame_index",
                frame_mapping_source=(
                    f"{acquisition.record_ref}#"
                    "full_untrimmed_video_decode_identity_v1"
                ),
                frame_mapping_sha256=hashlib.sha256(
                    np.ascontiguousarray(dense_mapping).view(np.uint8)
                ).hexdigest(),
            ),
            detect_mod.OBSERVATION_ROW_COUNT_ATTR: 2,
            "summary_statistics": {
                "total_detections": 2,
                "frames_with_detections": 2,
                "frames_with_zero_detections": 0,
                "frames_with_multiple_detections": 0,
            },
            "detect_row_shard_rows": None,
            "detect_shard_write": None,
            "model_path": _TEST_MODEL_ARTIFACT["path"],
            "model_name": "chaser-distance-test.pt",
            "inference_height": 4512,
            "inference_width": 4512,
            "validated_backend_result_count": 2,
            "validated_backend_result_orig_shape_hw": [4512, 4512],
            "decode_backend_effective": "opencv",
            "video_reader_type": "opencv",
            "parameters": {
                "decode_backend_effective": "opencv",
                "resize_dims": None,
                "pre_resize_dims": None,
                "effective_input_resize_dims": None,
                "tensor_resize_dims": None,
                "imgsz_applied": None,
            },
            IMMUTABLE_YOLO_STORAGE_ATTR: {
                "schema_id": IMMUTABLE_YOLO_STORAGE_SCHEMA,
                "status": "ok",
                "stage": "detect",
                "row_count": 2,
                "frame_count": 2,
                "instance_key_present": True,
                "instance_key_unique": True,
            },
        }
    )
    mapping = detect_mod._publish_detection_acquisition_mapping(  # noqa: SLF001
        run,
        acquisition_frame=acquisition,
    )
    backend_result_projection = publish_detection_backend_result_projection(
        run,
        run["bbox_norm_coords"],
        frame_evidence=evidence,
        model_artifact=_TEST_MODEL_ARTIFACT,
    )
    instance_derivation = publish_detection_instance_key_derivation(
        run,
        run["instance_key"],
        run["source_acquisition_frame_index"],
        run["bbox_norm_coords"],
        run["class_ids"],
        acquisition_frame=acquisition,
        acquisition_mapping=mapping,
    )
    publish_detection_observation_cardinality(
        run,
        acquisition_frame=acquisition,
    )
    publish_detection_observation_geometry(
        run,
        run["instance_key"],
        run["source_acquisition_frame_index"],
        run["bbox_norm_coords"],
        run["bbox_img_xyxy"],
        run["centers_img_xy"],
        frame_evidence=evidence,
        source_lineage_records=(
            mapping,
            backend_result_projection,
            instance_derivation,
        ),
    )
    run.attrs.update(
        {
            "coordinate_contract": "canonical_v2",
            RUN_COMPLETION_CONTRACT_ATTR: RUN_COMPLETION_CONTRACT,
            RUN_COMPLETION_STATUS_ATTR: RUN_STATUS_COMPLETE,
            "stage_selector_eligible": True,
        }
    )
    parent.attrs.update(
        {
            "latest": "canonical",
            "latest_complete": "canonical",
            "authoritative_run": "canonical",
        }
    )
    return "detect_runs/canonical"


def _canonical_sources(tmp_path: Path) -> tuple[Path, str, str]:
    zarr_path = tmp_path / "analysis.zarr"
    h5_path = tmp_path / "stimulus.h5"
    _prepare_acquisition_authority(zarr_path, total_frames=2)
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    root.attrs["fps"] = 120.0
    detection_path = _publish_detection(root)
    _write_stimulus_h5_with_arena_relative_chaser_states(
        h5_path,
        multi_chaser=True,
    )
    stimulus_run = stimulus_import.import_stimulus_to_zarr(
        stimulus_h5=h5_path,
        zarr_path=zarr_path,
        run_name="canonical_stimulus",
        overwrite=False,
        verbose=False,
        repair_chaser_gaps=False,
    )
    return zarr_path, detection_path, stimulus_run


def _add_epoch_run(zarr_path: Path) -> str:
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    parent = root.require_group("analysis").require_group("stimulus_epoch_runs")
    run = parent.create_group("epoch_1")
    windows = run.create_group("windows")
    label = np.zeros((1, 16), dtype=np.uint8)
    label[0, :5] = np.frombuffer(b"trial", dtype=np.uint8)
    for name, values in {
        "window_id": np.asarray([7], dtype=np.int32),
        "label_bytes": label,
        "start_frame": np.asarray([0], dtype=np.int64),
        "end_frame": np.asarray([1], dtype=np.int64),
        "start_time_s": np.asarray([0.0], dtype=np.float64),
        "end_time_s": np.asarray([1.0 / 120.0], dtype=np.float64),
        "duration_s": np.asarray([2.0 / 120.0], dtype=np.float64),
    }.items():
        windows.create_array(name, data=values)
    parent.attrs["latest"] = "epoch_1"
    return "epoch_1"


def _publish_canonical_from_sources(
    zarr_path: Path,
    detection_path: str,
    stimulus_run: str,
) -> str:
    result = build_chaser_distance_result(
        zarr_path,
        run_name="canonical_distance",
        detection_path=detection_path,
        stimulus_run=stimulus_run,
    )
    path = write_chaser_distance_run(
        zarr_path,
        result,
        write_png=False,
        write_interactive_spec=False,
    )
    return path


def _publish_canonical(
    tmp_path: Path,
) -> tuple[Path, zarr.Group, zarr.Group]:
    """Build a fresh canonical archive for test modules that need the writer."""
    zarr_path, detection_path, stimulus_run = _canonical_sources(tmp_path)
    run_path = _publish_canonical_from_sources(
        zarr_path,
        detection_path,
        stimulus_run,
    )
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    return zarr_path, root, root[run_path]


def _copy_zarr_archive(template: Path, target_parent: Path) -> Path:
    target = target_parent / template.name
    shutil.copytree(template, target)
    return target


@pytest.fixture(scope="module")
def canonical_source_template(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, str, str]:
    template_parent = tmp_path_factory.mktemp("chaser-distance-source-template")
    return _canonical_sources(template_parent)


@pytest.fixture
def canonical_source_archive(
    tmp_path: Path,
    canonical_source_template: tuple[Path, str, str],
) -> tuple[Path, str, str]:
    template, detection_path, stimulus_run = canonical_source_template
    return (
        _copy_zarr_archive(template, tmp_path),
        detection_path,
        stimulus_run,
    )


@pytest.fixture(scope="module")
def published_canonical_template(
    tmp_path_factory: pytest.TempPathFactory,
    canonical_source_template: tuple[Path, str, str],
) -> tuple[Path, str]:
    source_template, detection_path, stimulus_run = canonical_source_template
    template_parent = tmp_path_factory.mktemp("chaser-distance-published-template")
    zarr_path = _copy_zarr_archive(source_template, template_parent)
    run_path = _publish_canonical_from_sources(
        zarr_path,
        detection_path,
        stimulus_run,
    )
    return zarr_path, run_path


@pytest.fixture
def published_canonical_archive(
    tmp_path: Path,
    published_canonical_template: tuple[Path, str],
) -> tuple[Path, zarr.Group, zarr.Group]:
    template, run_path = published_canonical_template
    zarr_path = _copy_zarr_archive(template, tmp_path)
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    return zarr_path, root, root[run_path]


def test_canonical_chaser_distance_publishes_exact_coordinate_contract(
    published_canonical_archive: tuple[Path, zarr.Group, zarr.Group],
) -> None:
    _zarr_path, root, run = published_canonical_archive

    bound = load_bound_chaser_distance_run(root, run.path)

    assert run.attrs["coordinate_contract"] == COORDINATE_CONTRACT
    assert INPUT_AUTHORITY_ATTR in run.attrs
    assert MEASUREMENT_AUTHORITY_ATTR in run.attrs
    assert PUBLICATION_SEAL_ATTR in run.attrs
    assert run.attrs[INPUT_AUTHORITY_ATTR]["numeric_transform_direction"] == (
        "source_camera_image_px_to_selected_canvas_px_then_"
        "inverse_arena_to_selected_canvas_to_arena_relative_canvas_px"
    )
    assert run.attrs[MEASUREMENT_AUTHORITY_ATTR]["pixels_per_mm_projector"] == 5.0
    assert run.attrs["stage_selector_eligible"] is True
    assert root["analysis/chaser_distance_runs"].attrs["latest_complete"] == (
        run.path.rsplit("/", 1)[-1]
    )
    assert bound.row_identity.contract.key_array.components == (
        "camera_frame_index",
    )
    fish_img = run["positions/fish_centroid_img_xy"].attrs[COORDINATE_DESCRIPTOR_ATTR]
    fish_arena = run["positions/fish_centroid_arena_xy"].attrs[
        COORDINATE_DESCRIPTOR_ATTR
    ]
    chaser = run["positions/chaser_arena_xy"].attrs[COORDINATE_DESCRIPTOR_ATTR]
    distance_node = run["distances/distance_px"]
    distance = distance_node.attrs[MEASUREMENT_DESCRIPTOR_ATTR]
    assert fish_img["space_id"] == "source_camera_image_px"
    assert fish_img["source_camera_overlay"]["status"] == "direct"
    assert fish_arena["space_id"] == "arena_relative_canvas_px"
    assert fish_arena["source_camera_overlay"]["status"] == "requires_transform"
    assert chaser["collection_axis"]["role"] == "chaser"
    assert chaser["collection_axis"]["cardinality"] == 2
    assert COORDINATE_DESCRIPTOR_ATTR not in distance_node.attrs
    assert distance["quantity"] == "euclidean_distance"
    assert distance["units"] == "px"
    assert distance["collection_axis"]["role"] == "chaser"
    assert distance["source_camera_overlay_status"] == "not_suitable_scalar_measurement"
    assert len(distance["source_coordinate_descriptors"]) == 2
    distance_mm = run["distances/distance_mm"].attrs[MEASUREMENT_DESCRIPTOR_ATTR]
    assert distance_mm["units"] == "mm"
    assert distance_mm["source_measurement_descriptors"][0]["record_ref"].endswith(
        "/distance_px@measurement_descriptor"
    )
    assert "distances/distance_px" in bound.measurement_surfaces
    assert "coordinate_frame" not in run["positions"].attrs
    assert "distance_px_coordinate_frame" not in run["distances"].attrs
    assert run["chasers"].attrs[CHASER_COLLECTION_AUTHORITY_ATTR]["axis"] == 1


def test_epoch_measurements_bind_and_recheck_exact_source_windows(
    canonical_source_archive: tuple[Path, str, str],
) -> None:
    zarr_path, detection_path, stimulus_run = canonical_source_archive
    epoch_run = _add_epoch_run(zarr_path)
    result = build_chaser_distance_result(
        zarr_path,
        run_name="with_epoch",
        detection_path=detection_path,
        stimulus_run=stimulus_run,
        stimulus_epoch_run=epoch_run,
    )
    path = write_chaser_distance_run(
        zarr_path,
        result,
        write_png=False,
        write_interactive_spec=False,
    )
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    run = root[path]
    bound = load_bound_chaser_distance_run(root, path)
    summary = run["epoch_summary/p50_distance_mm"].attrs[
        MEASUREMENT_DESCRIPTOR_ATTR
    ]
    assert summary["output_epoch_window_identity"]["record_ref"].endswith(
        "/epoch_summary@epoch_window_identity_authority"
    )
    assert "epoch_summary/p50_distance_mm" in bound.measurement_surfaces

    root["analysis/stimulus_epoch_runs/epoch_1/windows/end_frame"][0] = np.int64(0)
    with pytest.raises(
        ChaserDistanceCoordinateError,
        match="differs from its exact source run",
    ):
        load_bound_chaser_distance_run(root, path)


def test_canonical_writer_rechecks_source_before_any_output_mutation(
    canonical_source_archive: tuple[Path, str, str],
) -> None:
    zarr_path, detection_path, stimulus_run = canonical_source_archive
    result = build_chaser_distance_result(
        zarr_path,
        run_name="candidate",
        detection_path=detection_path,
        stimulus_run=stimulus_run,
    )
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    root[f"{detection_path}/scores"][0] = np.float32(0.1)

    with pytest.raises(ChaserDistanceCoordinateError, match="changed after binding"):
        write_chaser_distance_run(
            zarr_path,
            result,
            write_png=False,
            write_interactive_spec=False,
        )

    assert "candidate" not in root["analysis"].get("chaser_distance_runs", {})


def test_activation_rolls_back_exact_selectors_on_base_exception(
    published_canonical_archive: tuple[Path, zarr.Group, zarr.Group],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _zarr_path, root, run = published_canonical_archive
    parent = root["analysis/chaser_distance_runs"]
    parent.attrs["latest"] = "previous_latest"
    parent.attrs["latest_complete"] = "previous_complete"
    run.attrs["stage_selector_eligible"] = False
    original = publication_module._write_activation_attr  # noqa: SLF001

    def _interrupt(attrs, key, value):
        if key == "latest":
            raise KeyboardInterrupt("hostile selector-store interruption")
        original(attrs, key, value)

    monkeypatch.setattr(publication_module, "_write_activation_attr", _interrupt)
    with pytest.raises(KeyboardInterrupt, match="hostile selector-store"):
        activate_chaser_distance_run(
            root,
            parent,
            run,
            run_name="canonical_distance",
        )

    assert parent.attrs["latest"] == "previous_latest"
    assert parent.attrs["latest_complete"] == "previous_complete"
    assert run.attrs["stage_selector_eligible"] is False


def test_activation_rolls_back_owned_epoch_on_final_commit_interrupt(
    published_canonical_archive: tuple[Path, zarr.Group, zarr.Group],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _zarr_path, root, run = published_canonical_archive
    parent = root["analysis/chaser_distance_runs"]
    before = {
        name: (name in parent.attrs, parent.attrs.get(name))
        for name in (
            "latest",
            "latest_complete",
            CHASER_DISTANCE_PARENT_PUBLICATION_LEASE_ATTR,
            CHASER_DISTANCE_PUBLICATION_GENERATION_ATTR,
            CHASER_DISTANCE_PUBLICATION_POLICY_ATTR,
        )
    }
    run.attrs["stage_selector_eligible"] = False
    original = publication_module._write_activation_attr  # noqa: SLF001

    def _interrupt(attrs, key, value):
        if key == "stage_selector_eligible":
            raise KeyboardInterrupt("hostile final-commit interruption")
        original(attrs, key, value)

    monkeypatch.setattr(publication_module, "_write_activation_attr", _interrupt)
    with pytest.raises(KeyboardInterrupt, match="hostile final-commit"):
        activate_chaser_distance_run(
            root,
            parent,
            run,
            run_name="canonical_distance",
        )

    for name, (present, value) in before.items():
        assert (name in parent.attrs) is present
        if present:
            assert parent.attrs[name] == value
    assert run.attrs["stage_selector_eligible"] is False


def test_activation_accepts_proven_commit_when_store_interrupts_after_final_write(
    published_canonical_archive: tuple[Path, zarr.Group, zarr.Group],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path, root, run = published_canonical_archive
    parent = root["analysis/chaser_distance_runs"]
    base_generation = parent.attrs[CHASER_DISTANCE_PUBLICATION_GENERATION_ATTR]
    run.attrs["stage_selector_eligible"] = False
    original = publication_module._write_activation_attr  # noqa: SLF001

    def _persist_then_interrupt(attrs, key, value):
        original(attrs, key, value)
        if key == "stage_selector_eligible":
            raise KeyboardInterrupt("interrupt after persisted commit")

    monkeypatch.setattr(
        publication_module,
        "_write_activation_attr",
        _persist_then_interrupt,
    )
    activate_chaser_distance_run(
        root,
        parent,
        run,
        run_name="canonical_distance",
    )

    persisted_root = zarr.open_group(
        str(zarr_path),
        mode="r",
        use_consolidated=False,
    )
    persisted_parent = persisted_root["analysis/chaser_distance_runs"]
    persisted_run = persisted_parent["canonical_distance"]
    assert persisted_run.attrs["stage_selector_eligible"] is True
    assert persisted_parent.attrs["latest"] == "canonical_distance"
    assert persisted_parent.attrs["latest_complete"] == "canonical_distance"
    assert persisted_parent.attrs[CHASER_DISTANCE_PUBLICATION_GENERATION_ATTR] == (
        base_generation + 1
    )
    assert persisted_parent.attrs[CHASER_DISTANCE_PUBLICATION_POLICY_ATTR] == (
        CHASER_DISTANCE_PUBLICATION_POLICY
    )
    lease = persisted_parent.attrs[CHASER_DISTANCE_PARENT_PUBLICATION_LEASE_ATTR]
    assert lease["next_generation"] == base_generation + 1
    assert lease["run_path"] == persisted_run.path


def test_activation_preserves_concurrent_owner_generation_takeover(
    published_canonical_archive: tuple[Path, zarr.Group, zarr.Group],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path, root, run = published_canonical_archive
    parent = root["analysis/chaser_distance_runs"]
    run.attrs["stage_selector_eligible"] = False
    original = publication_module._write_activation_attr  # noqa: SLF001
    winner_lease: dict[str, object] = {}

    def _take_over(attrs, key, value):
        original(attrs, key, value)
        if key != CHASER_DISTANCE_PUBLICATION_GENERATION_ATTR:
            return
        winner_base = int(value)
        winner_lease.update(
            {
                "schema_id": "palette.chaser_distance_publication_lease",
                "schema_version": 1,
                "policy": CHASER_DISTANCE_PUBLICATION_POLICY,
                "owner_uuid": "concurrent-winner",
                "publication_owner": "concurrent-winner",
                "run_path": "analysis/chaser_distance_runs/concurrent_winner",
                "base_generation": winner_base,
                "next_generation": winner_base + 1,
            }
        )
        attrs[CHASER_DISTANCE_PARENT_PUBLICATION_LEASE_ATTR] = dict(winner_lease)
        attrs[CHASER_DISTANCE_PUBLICATION_GENERATION_ATTR] = winner_base + 1
        attrs[CHASER_DISTANCE_PUBLICATION_POLICY_ATTR] = (
            CHASER_DISTANCE_PUBLICATION_POLICY
        )
        attrs["latest_complete"] = "concurrent_winner"
        attrs["latest"] = "concurrent_winner"
        raise KeyboardInterrupt("concurrent activation takeover")

    monkeypatch.setattr(publication_module, "_write_activation_attr", _take_over)
    with pytest.raises(KeyboardInterrupt, match="concurrent activation takeover"):
        activate_chaser_distance_run(
            root,
            parent,
            run,
            run_name="canonical_distance",
        )

    persisted_root = zarr.open_group(
        str(zarr_path),
        mode="r",
        use_consolidated=False,
    )
    persisted_parent = persisted_root["analysis/chaser_distance_runs"]
    persisted_run = persisted_parent["canonical_distance"]
    assert (
        persisted_parent.attrs[CHASER_DISTANCE_PARENT_PUBLICATION_LEASE_ATTR]
        == winner_lease
    )
    assert persisted_parent.attrs[CHASER_DISTANCE_PUBLICATION_GENERATION_ATTR] == (
        winner_lease["next_generation"]
    )
    assert persisted_parent.attrs["latest_complete"] == "concurrent_winner"
    assert persisted_parent.attrs["latest"] == "concurrent_winner"
    assert persisted_run.attrs["stage_selector_eligible"] is False


def test_writer_marks_candidate_failed_on_base_exception(
    canonical_source_archive: tuple[Path, str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path, detection_path, stimulus_run = canonical_source_archive
    result = build_chaser_distance_result(
        zarr_path,
        run_name="interrupted_candidate",
        detection_path=detection_path,
        stimulus_run=stimulus_run,
    )

    def _interrupt(*_args, **_kwargs):
        raise SystemExit("hostile activation interruption")

    monkeypatch.setattr(runs_module, "activate_chaser_distance_run", _interrupt)
    with pytest.raises(SystemExit, match="hostile activation"):
        write_chaser_distance_run(
            zarr_path,
            result,
            write_png=False,
            write_interactive_spec=False,
        )

    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    parent = root["analysis/chaser_distance_runs"]
    run = parent["interrupted_candidate"]
    assert run.attrs["palette_run_completion_status"] == "failed"
    assert run.attrs["stage_selector_eligible"] is False
    assert parent.attrs.get("latest_pending") != "interrupted_candidate"


def test_canonical_writer_refuses_destructive_overwrite(
    published_canonical_archive: tuple[Path, zarr.Group, zarr.Group],
) -> None:
    zarr_path, root, run = published_canonical_archive
    seal_before = run.attrs[PUBLICATION_SEAL_ATTR]
    result = build_chaser_distance_result(
        zarr_path,
        run_name="canonical_distance",
        detection_path=str(run.attrs["source_detection_path"]),
        stimulus_run=str(run.attrs["source_stimulus_run"]),
    )

    with pytest.raises(ValueError, match="immutable"):
        write_chaser_distance_run(
            zarr_path,
            result,
            overwrite=True,
            write_png=False,
            write_interactive_spec=False,
        )

    assert root[run.path].attrs[PUBLICATION_SEAL_ATTR] == seal_before


def test_strict_reader_rejects_tampered_output_payload(
    published_canonical_archive: tuple[Path, zarr.Group, zarr.Group],
) -> None:
    _zarr_path, root, run = published_canonical_archive
    run["positions/fish_centroid_arena_xy"][0, 0] += np.float32(1.0)

    with pytest.raises(ChaserDistanceCoordinateError, match="exact canonical derivation"):
        load_bound_chaser_distance_run(root, run.path)


def test_strict_reader_rejects_direction_relabel_even_with_fresh_record_digest(
    published_canonical_archive: tuple[Path, zarr.Group, zarr.Group],
) -> None:
    _zarr_path, root, run = published_canonical_archive
    record = dict(run.attrs[INPUT_AUTHORITY_ATTR])
    record["numeric_transform_direction"] = (
        "arena_relative_canvas_px_to_source_camera_image_px"
    )
    run.attrs[INPUT_AUTHORITY_ATTR] = record
    run.attrs[f"{INPUT_AUTHORITY_ATTR}_sha256"] = coordinate_record_sha256(record)

    with pytest.raises(ChaserDistanceCoordinateError, match="input authority is stale"):
        load_bound_chaser_distance_run(root, run.path)


def test_strict_reader_rejects_measurement_relabel_with_fresh_digest(
    published_canonical_archive: tuple[Path, zarr.Group, zarr.Group],
) -> None:
    _zarr_path, root, run = published_canonical_archive
    node = run["distances/distance_px"]
    record = dict(node.attrs[MEASUREMENT_DESCRIPTOR_ATTR])
    record["units"] = "mm"
    node.attrs[MEASUREMENT_DESCRIPTOR_ATTR] = record
    node.attrs[f"{MEASUREMENT_DESCRIPTOR_ATTR}_sha256"] = (
        coordinate_record_sha256(record)
    )

    with pytest.raises(ChaserDistanceCoordinateError, match="stale or incomplete"):
        load_bound_chaser_distance_run(root, run.path)


def test_manual_result_requires_explicit_legacy_writer_flag(tmp_path: Path) -> None:
    from tests.unit.fisheye.test_goodcopbadcop_interactive import (
        _make_archive_with_detection_occupancy,
        _make_chaser_result,
    )

    zarr_path = _make_archive_with_detection_occupancy(tmp_path)
    result = _make_chaser_result(zarr_path)
    with pytest.raises(ValueError, match="legacy_compatibility=True"):
        write_chaser_distance_run(
            zarr_path,
            result,
            write_png=False,
            write_interactive_spec=False,
        )


def test_explicit_legacy_writer_never_advances_normal_selectors(
    tmp_path: Path,
) -> None:
    from tests.unit.fisheye.test_goodcopbadcop_interactive import (
        _make_archive_with_detection_occupancy,
        _make_chaser_result,
    )

    zarr_path = _make_archive_with_detection_occupancy(tmp_path)
    result = _make_chaser_result(zarr_path)
    path = write_chaser_distance_run(
        zarr_path,
        result,
        write_png=False,
        write_interactive_spec=False,
        legacy_compatibility=True,
    )
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    parent = root["analysis/chaser_distance_runs"]
    assert root[path].attrs["stage_selector_eligible"] is False
    assert parent.attrs.get("latest") != result.run_name
    assert parent.attrs.get("latest_complete") != result.run_name


def test_canonical_builder_rejects_unsupported_refined_detection_path(
    canonical_source_archive: tuple[Path, str, str],
) -> None:
    zarr_path, _detection_path, stimulus_run = canonical_source_archive
    with pytest.raises(ChaserDistanceCoordinateError, match="detect_runs/<run>"):
        build_chaser_distance_result(
            zarr_path,
            run_name="candidate",
            detection_path="refined_detect_runs/refined",
            stimulus_run=stimulus_run,
        )

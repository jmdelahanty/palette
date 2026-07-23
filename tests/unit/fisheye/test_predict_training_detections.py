import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from fisheye.registry.db import Registry
from fisheye.shared.detection_producer_lifecycle import (
    ARTIFACT_PAYLOAD_INVENTORY_SEAL_ATTR,
    DETECTION_ARTIFACT_FAMILY_CONTRACT,
    EMPTY_ARTIFACT_OBSERVATION_PROOF_ATTR,
    STRICT_ARTIFACT_INTEGRITY_CONTRACT,
    UNBOUND_ARTIFACT_NUMERIC_SEMANTICS_ATTR,
    validate_artifact_payload_inventory_seal,
    validate_empty_artifact_observation_proof,
)
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    RUN_STATUS_FAILED,
)
from fisheye.utils import predict_training_detections as mod


def _write_training_zarr(path: Path, *, ds_shape: tuple[int, int] = (640, 640)) -> None:
    root = zarr.open_group(str(path), mode="w")
    raw = root.require_group("raw_video")
    raw.create_array(
        "images_full",
        data=np.zeros((3, 900, 1200), dtype=np.uint8),
        chunks=(1, 900, 1200),
    )
    raw.create_array(
        "images_ds",
        data=np.zeros((3, ds_shape[0], ds_shape[1]), dtype=np.uint8),
        chunks=(1, ds_shape[0], ds_shape[1]),
    )
    raw.create_array(
        "original_frame_indices",
        data=np.array([0, 5000, 10000], dtype=np.int64),
        chunks=(3,),
    )


def _write_registry(path: Path, model_path: Path, *, imgsz: int = 640) -> None:
    model_path.write_text("fake model", encoding="utf-8")
    registry = Registry(path)
    try:
        registry.record_training_run(
            run_id="detect_run",
            set_id="detect_set",
            task_type="detect",
            config_path=None,
            manifest_path=None,
            model_path=model_path,
            metrics_path=None,
            status="success",
            final_metrics={"imgsz_h": imgsz, "imgsz_w": imgsz},
        )
    finally:
        registry.close()


def test_select_frame_source_prefers_downsampled_match(tmp_path: Path) -> None:
    zarr_path = tmp_path / "training.zarr"
    registry_path = tmp_path / "registry.sqlite"
    model_path = tmp_path / "best.pt"
    _write_training_zarr(zarr_path, ds_shape=(640, 640))
    _write_registry(registry_path, model_path, imgsz=640)

    spec = mod.resolve_model_input_spec(
        registry_path,
        model_run_id="detect_run",
        model_path=None,
        set_id=None,
        artifact_kind="training",
    )
    root = zarr.open_group(str(zarr_path), mode="r")
    selection = mod.select_frame_source(root, spec)

    assert selection.path == "raw_video/images_ds"
    assert selection.matches_model_shape is True
    assert selection.needs_gray_to_rgb is True
    assert selection.reason == "sampled_array_matches_model_shape"


def test_select_frame_source_falls_back_to_full_when_sampled_mismatches(tmp_path: Path) -> None:
    zarr_path = tmp_path / "training.zarr"
    registry_path = tmp_path / "registry.sqlite"
    model_path = tmp_path / "best.pt"
    _write_training_zarr(zarr_path, ds_shape=(320, 320))
    _write_registry(registry_path, model_path, imgsz=640)

    spec = mod.resolve_model_input_spec(
        registry_path,
        model_run_id="detect_run",
        model_path=None,
        set_id=None,
        artifact_kind="training",
    )
    root = zarr.open_group(str(zarr_path), mode="r")
    selection = mod.select_frame_source(root, spec)

    assert selection.path == "raw_video/images_full"
    assert selection.matches_model_shape is False
    assert selection.reason == "fallback_to_available_frame_array"


class _Tensor:
    def __init__(self, value: np.ndarray) -> None:
        self._value = value

    def detach(self) -> "_Tensor":
        return self

    def cpu(self) -> "_Tensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._value


class _Boxes:
    def __init__(self) -> None:
        self.xyxy = _Tensor(np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32))
        self.conf = _Tensor(np.array([0.9], dtype=np.float32))
        self.cls = _Tensor(np.array([1], dtype=np.float32))

    def __len__(self) -> int:
        return 1


class _FakeYOLO:
    calls: list[dict[str, object]] = []

    def __init__(self, path: str) -> None:
        self.path = path

    def to(self, _device: str) -> None:
        return None

    def predict(self, images, **kwargs):
        self.calls.append(
            {
                "n_images": len(images),
                "shape": np.asarray(images[0]).shape,
                "kwargs": kwargs,
            }
        )
        return [SimpleNamespace(boxes=_Boxes()) for _image in images]


class _EmptyBoxes:
    def __len__(self) -> int:
        return 0


class _EmptyYOLO(_FakeYOLO):
    def predict(self, images, **kwargs):
        self.calls.append(
            {
                "n_images": len(images),
                "shape": np.asarray(images[0]).shape,
                "kwargs": kwargs,
            }
        )
        return [SimpleNamespace(boxes=_EmptyBoxes()) for _image in images]


class _PartialYOLO(_FakeYOLO):
    def predict(self, images, **kwargs):
        del kwargs
        return [SimpleNamespace(boxes=_EmptyBoxes()) for _image in images[:-1]]


def _patch_runtime(monkeypatch, *, yolo_type=_FakeYOLO) -> None:
    yolo_type.calls = []
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=yolo_type))
    monkeypatch.setattr(
        mod,
        "get_git_info",
        lambda: {
            "commit_hash": "abc",
            "short_hash": "abc",
            "branch": "test",
            "is_dirty": False,
            "remote_url": None,
        },
    )
    monkeypatch.setattr(
        mod,
        "get_environment_info",
        lambda **_kwargs: {
            "platform": {
                "hostname": "host",
                "system": "Linux",
                "release": "test",
                "python_version": "3.11",
                "machine": "x86_64",
            },
            "environment": {},
        },
    )


def _resolved_spec(
    tmp_path: Path,
    *,
    ds_shape: tuple[int, int] = (640, 640),
) -> tuple[Path, mod.ModelInputSpec]:
    zarr_path = tmp_path / "training.zarr"
    registry_path = tmp_path / "registry.sqlite"
    model_path = tmp_path / "best.pt"
    _write_training_zarr(zarr_path, ds_shape=ds_shape)
    _write_registry(registry_path, model_path, imgsz=640)
    spec = mod.resolve_model_input_spec(
        registry_path,
        model_run_id="detect_run",
        model_path=None,
        set_id=None,
        artifact_kind="training",
    )
    return zarr_path, spec


def _run_prediction(
    zarr_path: Path,
    spec: mod.ModelInputSpec,
    *,
    run_name: str,
    overwrite: bool = False,
) -> dict[str, object]:
    return mod.run_training_zarr_prediction(
        zarr_path=zarr_path,
        spec=spec,
        run_name=run_name,
        batch_size=2,
        conf=0.4,
        iou=0.45,
        max_det=20,
        cpu=True,
        overwrite=overwrite,
        argv=["predict_training_detections"],
    )


def test_run_training_zarr_prediction_writes_selector_free_artifact(
    monkeypatch,
    tmp_path: Path,
) -> None:
    zarr_path, spec = _resolved_spec(tmp_path)
    _patch_runtime(monkeypatch)

    result = _run_prediction(
        zarr_path,
        spec,
        run_name="  detect_seed_test  ",
    )

    root = zarr.open_group(str(zarr_path), mode="r")
    assert "detect_runs" not in root
    parent = root["detection_artifact_runs"]
    for selector in (
        "latest",
        "latest_complete",
        "latest_pending",
        "authoritative_run",
        "authoritative_run_provenance",
    ):
        assert selector not in parent.attrs
    assert (
        parent.attrs["artifact_family_contract"]
        == DETECTION_ARTIFACT_FAMILY_CONTRACT
    )
    assert parent.attrs["stage_selector_eligible"] is False

    run = parent["detect_seed_test"]
    assert run.attrs[RUN_COMPLETION_STATUS_ATTR] == RUN_STATUS_COMPLETE
    assert (
        run.attrs["artifact_integrity_contract"]
        == STRICT_ARTIFACT_INTEGRITY_CONTRACT
    )
    assert run["artifact_row_id"].dtype == np.dtype("uint64")
    assert run["artifact_row_id"][:].tolist() == [0, 1, 2]
    assert run["frame_indices"][:].tolist() == [0, 1, 2]
    assert run["source_frame_indices"][:].tolist() == [0, 5000, 10000]
    assert run["class_ids"][:].tolist() == [1, 1, 1]
    np.testing.assert_allclose(
        run["bbox_norm_coords"][:],
        np.array(
            [
                [0.03125, 0.046875, 0.03125, 0.03125],
                [0.03125, 0.046875, 0.03125, 0.03125],
                [0.03125, 0.046875, 0.03125, 0.03125],
            ],
            dtype=np.float64,
        ),
    )
    assert run.attrs["detection_source"] == "training_zarr_raw_video"
    assert run.attrs["frame_source_path"] == "raw_video/images_ds"
    assert run.attrs["model_registry_run_id"] == "detect_run"
    assert run.attrs["model_input_shape_status"] == "inferred_from_imgsz"
    assert run.attrs["summary_statistics"]["total_detections"] == 3
    assert run.attrs["artifact_row_identity"] == "dense_run_local_uint64_v1"
    assert "instance_key" not in run
    assert not any(name.startswith("instance_key_") for name in run.attrs)
    for name in run.keys():
        node = run[name]
        attrs = getattr(node, "attrs", {})
        assert "coordinate_descriptor" not in attrs
        assert "coordinate_descriptor_sha256" not in attrs

    lineage = run.attrs[mod.ARTIFACT_FRAME_SOURCE_LINEAGE_ATTR]
    assert lineage["status"] == "unbound_artifact_provenance_only"
    assert lineage["selected_array_path"] == "raw_video/images_ds"
    assert lineage["frame_source_extent"] == {
        "width": 640,
        "height": 640,
        "channels": 1,
        "units": "pixels",
        "extent_basis": "selected_training_frame_array_shape",
    }
    assert lineage["model_request"]["exact_preprocessing_transform_persisted"] is False
    assert lineage["source_camera_overlay_suitability"].startswith("unsupported")
    assert run.attrs[f"{mod.ARTIFACT_FRAME_SOURCE_LINEAGE_ATTR}_sha256"] == (
        mod._canonical_sha256(lineage)
    )

    bbox_attrs = run["bbox_norm_coords"].attrs
    semantics = bbox_attrs[UNBOUND_ARTIFACT_NUMERIC_SEMANTICS_ATTR]
    assert semantics["canonical_binding_status"] == "unbound"
    assert semantics["source_camera_overlay_suitability"] == "unsupported"
    assert semantics["canonical_promotion_suitability"] == "unsupported"
    assert semantics["semantic_profile_id"] == (
        "training.bbox_norm_cxcywh.v1"
    )
    assert semantics["geometry_type"] == "bbox_cxcywh"
    assert semantics["component_units"] == ["normalized"] * 4
    assert semantics["pixel_convention"] == "continuous"
    assert semantics["axis_0_domain"] == "observation_rows"
    assert semantics["reference"]["kind"] == "selected_training_frame_array"
    assert semantics["reference"]["node_path"] == "raw_video/images_ds"
    assert semantics["reference"]["width"] == 640
    assert semantics["reference"]["height"] == 640
    assert semantics["components"] == [
        "center_x",
        "center_y",
        "width",
        "height",
    ]
    assert bbox_attrs[f"{UNBOUND_ARTIFACT_NUMERIC_SEMANTICS_ATTR}_sha256"] == (
        mod._canonical_sha256(semantics)
    )
    row_semantics = run["artifact_row_id"].attrs[
        UNBOUND_ARTIFACT_NUMERIC_SEMANTICS_ATTR
    ]
    assert row_semantics["numeric_space_id"] == "run_local_artifact_row"
    assert row_semantics["semantic_profile_id"] == "training.artifact_row_id.v1"
    assert row_semantics["canonical_binding_status"] == "unbound"
    count_semantics = run["frame_counts"].attrs[
        UNBOUND_ARTIFACT_NUMERIC_SEMANTICS_ATTR
    ]
    assert count_semantics["semantic_profile_id"] == "training.frame_counts.v1"
    assert count_semantics["axis_0_domain"] == "dense_frame_rows"
    seal = validate_artifact_payload_inventory_seal(run)
    assert seal == run.attrs[ARTIFACT_PAYLOAD_INVENTORY_SEAL_ATTR]
    assert seal["row_count"] == 3
    assert set(seal["arrays"]) == set(run.keys())
    assert seal["unbound_numeric_manifest_id"] == (
        "training_detection_with_source_mapping.v1"
    )
    mapping_by_array = {
        name: run[name].attrs[UNBOUND_ARTIFACT_NUMERIC_SEMANTICS_ATTR][
            "temporal_evidence"
        ]["source_mapping_sha256"]
        for name in run.keys()
    }
    assert {
        name for name, mapping in mapping_by_array.items() if mapping is not None
    } == {"source_frame_indices"}
    mapping_sha256 = mapping_by_array["source_frame_indices"]
    assert mapping_sha256 is not None
    assert seal["source_mapping_sha256"] == mapping_sha256
    assert run["source_frame_indices"].attrs[
        UNBOUND_ARTIFACT_NUMERIC_SEMANTICS_ATTR
    ]["source_mapping_sha256_policy"] == "required"
    assert semantics["source_mapping_sha256_policy"] == "forbidden"

    assert result["summary_statistics"]["frames_with_detections"] == 3
    assert result["artifact_run"] == "detect_seed_test"
    assert result["artifact_path"] == (
        "detection_artifact_runs/detect_seed_test"
    )
    assert _FakeYOLO.calls[0]["shape"] == (640, 640, 3)


def test_run_training_zarr_prediction_writes_unmapped_manifest(
    monkeypatch,
    tmp_path: Path,
) -> None:
    zarr_path, spec = _resolved_spec(tmp_path)
    mutable = zarr.open_group(str(zarr_path), mode="a")
    del mutable["raw_video/original_frame_indices"]
    _patch_runtime(monkeypatch)

    _run_prediction(zarr_path, spec, run_name="unmapped_seed")

    root = zarr.open_group(str(zarr_path), mode="r")
    run = root["detection_artifact_runs"]["unmapped_seed"]
    assert run.attrs[RUN_COMPLETION_STATUS_ATTR] == RUN_STATUS_COMPLETE
    assert "source_frame_indices" not in run
    seal = validate_artifact_payload_inventory_seal(run)
    assert seal["unbound_numeric_manifest_id"] == (
        "training_detection_without_source_mapping.v1"
    )
    assert seal["source_mapping_sha256"] is None
    assert all(
        run[name].attrs[UNBOUND_ARTIFACT_NUMERIC_SEMANTICS_ATTR][
            "temporal_evidence"
        ]["source_mapping_sha256"]
        is None
        for name in run.keys()
    )


def test_run_training_zarr_prediction_persists_genuine_zero_proof(
    monkeypatch,
    tmp_path: Path,
) -> None:
    zarr_path, spec = _resolved_spec(tmp_path)
    _patch_runtime(monkeypatch, yolo_type=_EmptyYOLO)

    result = _run_prediction(zarr_path, spec, run_name="empty_seed")

    root = zarr.open_group(str(zarr_path), mode="r")
    assert "detect_runs" not in root
    run = root["detection_artifact_runs"]["empty_seed"]
    assert run["artifact_row_id"].dtype == np.dtype("uint64")
    assert run["artifact_row_id"].shape == (0,)
    assert run["frame_indices"].shape == (0,)
    assert run["source_frame_indices"].shape == (0,)
    assert run["frame_counts"][:].tolist() == [0, 0, 0]
    proof = validate_empty_artifact_observation_proof(run)
    assert proof == run.attrs[EMPTY_ARTIFACT_OBSERVATION_PROOF_ATTR]
    assert "artifact_row_id" in proof["row_arrays"]
    assert "source_frame_indices" in proof["row_arrays"]
    evidence = proof["full_domain_evidence"]
    assert evidence["source_frame_count"] == 3
    assert evidence["processed_frame_count"] == 3
    assert evidence["processed_frame_domain"] == {
        "start": 0,
        "stop_exclusive": 3,
        "step": 1,
    }
    assert evidence["inference_result_cardinality_validated"] is True
    seal = validate_artifact_payload_inventory_seal(run)
    assert seal["row_count"] == 0
    assert set(seal["arrays"]) == set(proof["array_inventory"])
    assert result["summary_statistics"]["total_detections"] == 0


def test_run_training_zarr_prediction_rejects_partial_inference_as_false_empty(
    monkeypatch,
    tmp_path: Path,
) -> None:
    zarr_path, spec = _resolved_spec(tmp_path)
    _patch_runtime(monkeypatch, yolo_type=_PartialYOLO)

    with pytest.raises(RuntimeError, match="result count"):
        _run_prediction(zarr_path, spec, run_name="partial_seed")

    root = zarr.open_group(str(zarr_path), mode="r")
    assert "detect_runs" not in root
    assert "detection_artifact_runs" not in root


def test_run_training_zarr_prediction_rejects_overwrite_before_inference(
    monkeypatch,
    tmp_path: Path,
) -> None:
    zarr_path, spec = _resolved_spec(tmp_path)
    _patch_runtime(monkeypatch)

    with pytest.raises(ValueError, match="immutable.*overwrite"):
        _run_prediction(
            zarr_path,
            spec,
            run_name="overwrite_seed",
            overwrite=True,
        )

    root = zarr.open_group(str(zarr_path), mode="r")
    assert "detect_runs" not in root
    assert "detection_artifact_runs" not in root
    assert _FakeYOLO.calls == []


def test_run_training_zarr_prediction_rejects_existing_artifact(
    monkeypatch,
    tmp_path: Path,
) -> None:
    zarr_path, spec = _resolved_spec(tmp_path)
    _patch_runtime(monkeypatch)
    _run_prediction(zarr_path, spec, run_name="immutable_seed")
    calls_after_first = len(_FakeYOLO.calls)

    with pytest.raises(ValueError, match="already exists"):
        _run_prediction(zarr_path, spec, run_name=" immutable_seed ")

    assert len(_FakeYOLO.calls) == calls_after_first
    root = zarr.open_group(str(zarr_path), mode="r")
    assert tuple(root["detection_artifact_runs"].group_keys()) == (
        "immutable_seed",
    )


@pytest.mark.parametrize(
    "tamper_kind",
    ["payload", "row_identity", "inventory", "semantics", "contract_bypass"],
)
def test_run_training_zarr_prediction_revalidates_strict_seal(
    monkeypatch,
    tmp_path: Path,
    tamper_kind: str,
) -> None:
    zarr_path, spec = _resolved_spec(tmp_path)
    _patch_runtime(monkeypatch)
    original_write = mod.write_stage_provenance

    def write_then_tamper(run, provenance):
        original_write(run, provenance)
        if tamper_kind == "payload":
            run["scores"][0] = np.float32(0.1)
        elif tamper_kind == "row_identity":
            run["artifact_row_id"][0] = np.uint64(9)
        elif tamper_kind == "inventory":
            run.create_array(
                "late_array",
                data=np.arange(3, dtype=np.int32),
                chunks=(3,),
            )
        elif tamper_kind == "semantics":
            attrs = run["bbox_norm_coords"].attrs
            record = dict(attrs[UNBOUND_ARTIFACT_NUMERIC_SEMANTICS_ATTR])
            record["origin"] = "tampered"
            attrs[UNBOUND_ARTIFACT_NUMERIC_SEMANTICS_ATTR] = record
        else:
            del run["artifact_row_id"]
            del run.attrs[ARTIFACT_PAYLOAD_INVENTORY_SEAL_ATTR]
            del run.attrs[f"{ARTIFACT_PAYLOAD_INVENTORY_SEAL_ATTR}_sha256"]

    monkeypatch.setattr(mod, "write_stage_provenance", write_then_tamper)

    with pytest.raises(ValueError):
        _run_prediction(
            zarr_path,
            spec,
            run_name=f"tampered_{tamper_kind}",
        )

    root = zarr.open_group(str(zarr_path), mode="r")
    run = root["detection_artifact_runs"][f"tampered_{tamper_kind}"]
    assert run.attrs[RUN_COMPLETION_STATUS_ATTR] == RUN_STATUS_FAILED


def _rewrite_source_mapping_and_reseal(
    run,
    *,
    source_mapping_sha256: str | None,
    rewrite_seal_mapping: bool,
) -> None:
    semantics_attr = UNBOUND_ARTIFACT_NUMERIC_SEMANTICS_ATTR
    semantics_digest_attr = f"{semantics_attr}_sha256"
    semantics = json.loads(
        json.dumps(run["source_frame_indices"].attrs[semantics_attr])
    )
    semantics["temporal_evidence"]["source_mapping_sha256"] = (
        source_mapping_sha256
    )
    semantics_digest = mod._canonical_sha256(semantics)
    run["source_frame_indices"].attrs[semantics_attr] = semantics
    run["source_frame_indices"].attrs[semantics_digest_attr] = semantics_digest

    seal = json.loads(json.dumps(run.attrs[ARTIFACT_PAYLOAD_INVENTORY_SEAL_ATTR]))
    seal["arrays"]["source_frame_indices"][
        "numeric_semantics_sha256"
    ] = semantics_digest
    if rewrite_seal_mapping:
        seal["source_mapping_sha256"] = source_mapping_sha256
    run.attrs[ARTIFACT_PAYLOAD_INVENTORY_SEAL_ATTR] = seal
    run.attrs[f"{ARTIFACT_PAYLOAD_INVENTORY_SEAL_ATTR}_sha256"] = (
        mod._canonical_sha256(seal)
    )


@pytest.mark.parametrize("tamper_kind", ["missing", "mismatched", "coherent"])
def test_run_training_zarr_prediction_rejects_source_mapping_attack(
    monkeypatch,
    tmp_path: Path,
    tamper_kind: str,
) -> None:
    zarr_path, spec = _resolved_spec(tmp_path)
    _patch_runtime(monkeypatch)
    original_write = mod.write_stage_provenance

    def write_then_tamper(run, provenance):
        original_write(run, provenance)
        _rewrite_source_mapping_and_reseal(
            run,
            source_mapping_sha256=(None if tamper_kind == "missing" else "f" * 64),
            rewrite_seal_mapping=tamper_kind != "mismatched",
        )

    monkeypatch.setattr(mod, "write_stage_provenance", write_then_tamper)

    with pytest.raises(ValueError):
        _run_prediction(
            zarr_path,
            spec,
            run_name=f"mapping_{tamper_kind}",
        )

    root = zarr.open_group(str(zarr_path), mode="r")
    run = root["detection_artifact_runs"][f"mapping_{tamper_kind}"]
    assert run.attrs[RUN_COMPLETION_STATUS_ATTR] == RUN_STATUS_FAILED


def test_run_training_zarr_prediction_rolls_back_base_exception(
    monkeypatch,
    tmp_path: Path,
) -> None:
    zarr_path, spec = _resolved_spec(tmp_path)
    _patch_runtime(monkeypatch)

    def interrupt_provenance(*_args, **_kwargs):
        raise KeyboardInterrupt("injected publication interruption")

    monkeypatch.setattr(mod, "write_stage_provenance", interrupt_provenance)

    with pytest.raises(KeyboardInterrupt, match="publication interruption"):
        _run_prediction(zarr_path, spec, run_name="interrupted_seed")

    root = zarr.open_group(str(zarr_path), mode="r")
    assert "detect_runs" not in root
    parent = root["detection_artifact_runs"]
    for selector in (
        "latest",
        "latest_complete",
        "latest_pending",
        "authoritative_run",
        "authoritative_run_provenance",
    ):
        assert selector not in parent.attrs
    assert (
        parent["interrupted_seed"].attrs[RUN_COMPLETION_STATUS_ATTR]
        == RUN_STATUS_FAILED
    )

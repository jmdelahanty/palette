"""Tests for registry-driven detect preflight wrapper."""

from pathlib import Path
import sys

import numpy as np
import pytest
import zarr

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry
from fisheye.utils import prepare_detect_training_from_registry as wrapper


def _seed_dataset(db: Registry, dataset_id: str, zarr_path: Path) -> None:
    db.upsert_dataset(
        dataset_id,
        session_uuid=dataset_id,
        zarr_path=zarr_path,
    )
    db.upsert_provenance(
        dataset_id,
        provenance={},
        context={"canvas_name": "DefaultScreen", "rig_id": "omnifin0"},
        protocol_name=None,
        protocol_hash=None,
        acquisition={
            "dish_design": "cedar",
            "has_images_ds": True,
            "has_images_ds_rgb": False,
            "downsample_formats_json": '["gray"]',
        },
        zarr_purpose=None,
    )


def _seed_registry(registry_path: Path, zarr_path: Path) -> None:
    db = Registry(registry_path)
    _seed_dataset(db, "dataset_1", zarr_path)
    db.close()


def _upsert_detect_quality(
    db: Registry,
    *,
    dataset_id: str,
    review_state: str = "approved",
    review_intended_use: str = "training",
    interpolated_detections_rate: float | None = 0.1,
    zarr_mtime_ns: int | None = 101,
) -> None:
    db.upsert_detect_quality(
        dataset_id=dataset_id,
        refined_run="refined_run_1",
        refined_created_utc=None,
        source_detect_run="detect_run_1",
        detect_method="detect",
        review_state=review_state,
        review_intended_use=review_intended_use,
        review_reviewer=None,
        review_timestamp_utc=None,
        review_resolved_group="filtered",
        total_detections=10,
        real_detections=9,
        interpolated_detections=1,
        interpolated_detections_rate=interpolated_detections_rate,
        zarr_mtime_ns=zarr_mtime_ns,
    )


def test_resolve_detect_quality_from_zarr_handles_sparse_instances(tmp_path: Path) -> None:
    zarr_path = tmp_path / "dataset_1.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    refined_parent = root.create_group("refined_detect_runs")
    refined_parent.attrs["latest"] = "refined_run_1"
    refined = refined_parent.create_group("refined_run_1")
    refined.attrs["source_detect_run"] = "detect_run_1"
    refined.attrs["detect_review_status"] = {
        "state": "approved",
        "intended_use": "training",
        "resolved_group": "refined",
    }
    instances = refined.create_group("instances")
    instances.create_array("refined_row_ids", data=np.array([0, 1], dtype=np.int64))
    instances.create_array("frame_indices", data=np.array([0, 1], dtype=np.int32))
    instances.create_array("frame_offsets", data=np.array([0, 1, 2], dtype=np.int64))
    instances.create_array(
        "bbox_img_xyxy",
        data=np.array([[1.0, 1.0, 4.0, 4.0], [2.0, 2.0, 5.0, 5.0]], dtype=np.float64),
    )
    instances.create_array(
        "bbox_norm_coords",
        data=np.array([[0.5, 0.5, 0.2, 0.2], [0.4, 0.4, 0.2, 0.2]], dtype=np.float64),
    )
    instances.create_array("source_kind_codes", data=np.array([1, 2], dtype=np.int8))
    instances.create_array("manual_edit_flags", data=np.array([0, 1], dtype=np.int8))
    instances.create_array("source_detect_row_index", data=np.array([0, 1], dtype=np.int32))
    instances.create_array("frame_counts", data=np.array([1, 1], dtype=np.int32))

    resolved = wrapper._resolve_detect_quality_from_zarr(
        zarr_path,
        expected_refined_run="refined_run_1",
    )

    assert resolved["source_detect_run"] == "detect_run_1"
    assert resolved["review_state"] == "approved"
    assert resolved["review_intended_use"] == "training"
    assert resolved["review_resolved_group"] == "refined"
    assert resolved["total_detections"] == 2
    assert resolved["real_detections"] == 1
    assert resolved["interpolated_detections"] == 1
    assert resolved["interpolated_detections_rate"] == pytest.approx(0.5)


def test_auto_set_name_is_generated_when_missing(tmp_path: Path, monkeypatch) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry(registry_path, tmp_path / "dataset_1.zarr")

    calls: list[list[str]] = []

    def fake_prepare(cli: list[str]) -> None:
        calls.append(list(cli))

    monkeypatch.setattr(wrapper.pdt, "main", fake_prepare)

    rc = wrapper.main(["--registry", str(registry_path), "--dry-run"])
    assert rc == 0
    rc = wrapper.main(["--registry", str(registry_path), "--dry-run"])
    assert rc == 0
    assert len(calls) == 2

    first_cli = calls[0]
    second_cli = calls[1]
    assert "--set-name" in first_cli
    first_name = first_cli[first_cli.index("--set-name") + 1]
    second_name = second_cli[second_cli.index("--set-name") + 1]
    assert first_name == second_name
    assert first_name.startswith("cedar_defaultscreen_omnifin0_refined_gray_")
    assert len(first_name.rsplit("_", 1)[-1]) == 8


def _naming_args(**overrides):
    defaults = {
        "dish_design": None,
        "dish_design_like": None,
        "fps_min": None,
        "fps_max": None,
        "exposure_min": None,
        "exposure_max": None,
        "frame_rate_min": None,
        "frame_rate_max": None,
        "gain_min": None,
        "gain_max": None,
        "video_codec": None,
        "video_pix_fmt": None,
        "format_encoder": None,
        "format_title": None,
        "format_comment": None,
        "encoder_name": None,
        "encoder_codec": None,
        "encoder_preset": None,
        "encoder_tuning": None,
        "encoder_rc": None,
        "compression": None,
        "camera_model": None,
        "camera_serial": None,
        "camera_id": None,
        "rig_id": None,
        "arena_id": None,
        "path_contains": None,
        "limit": None,
        "source_type": "refined",
        "input_format": "gray",
    }
    defaults.update(overrides)
    return type("Args", (), defaults)()


def test_default_set_name_includes_rig_token_for_single_rig() -> None:
    rows = [
        {"dish_design": "cedar", "canvas_name": "shadow", "rig_id": "omnifin0"},
        {"dish_design": "cedar", "canvas_name": "shadow", "rig_id": "omnifin0"},
    ]
    name = wrapper._default_set_name(_naming_args(), rows, model_input="gray")
    assert name.startswith("cedar_shadow_omnifin0_refined_gray_")
    assert len(name.rsplit("_", 1)[-1]) == 8


def test_default_set_name_uses_mixed_rigs_token_for_multiple_rigs() -> None:
    rows = [
        {"dish_design": "cedar", "canvas_name": "shadow", "rig_id": "omnifin0"},
        {"dish_design": "cedar", "canvas_name": "shadow", "rig_id": "omnifin1"},
    ]
    name = wrapper._default_set_name(_naming_args(), rows, model_input="gray")
    assert name.startswith("cedar_shadow_mixed_rigs_refined_gray_")
    assert len(name.rsplit("_", 1)[-1]) == 8


def test_auto_out_manifest_is_set_when_out_config_is_given(tmp_path: Path, monkeypatch) -> None:
    registry_path = tmp_path / "registry.sqlite"
    out_config = tmp_path / "prep" / "detect.yaml"
    _seed_registry(registry_path, tmp_path / "dataset_1.zarr")

    calls: list[list[str]] = []

    def fake_prepare(cli: list[str]) -> None:
        calls.append(list(cli))

    monkeypatch.setattr(wrapper.pdt, "main", fake_prepare)

    rc = wrapper.main(
        [
            "--registry",
            str(registry_path),
            "--out-config",
            str(out_config),
            "--dry-run",
        ]
    )
    assert rc == 0
    assert calls
    cli = calls[0]
    assert "--out-manifest" in cli
    out_manifest = Path(cli[cli.index("--out-manifest") + 1])
    assert out_manifest == out_config.with_suffix(".manifest.json")


def test_registry_is_forwarded_to_preflight_without_register_flag(tmp_path: Path, monkeypatch) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry(registry_path, tmp_path / "dataset_1.zarr")

    calls: list[list[str]] = []

    def fake_prepare(cli: list[str]) -> None:
        calls.append(list(cli))

    monkeypatch.setattr(wrapper.pdt, "main", fake_prepare)

    rc = wrapper.main(["--registry", str(registry_path), "--dry-run"])
    assert rc == 0
    assert calls
    cli = calls[0]
    assert "--registry" in cli
    assert str(registry_path) in cli


def test_rejects_legacy_train_flag() -> None:
    try:
        wrapper.main(["--train"])
    except SystemExit as exc:
        msg = str(exc)
        assert "prepare-only" in msg
        assert "run_detect_training_pipeline" in msg
    else:  # pragma: no cover - defensive branch
        raise AssertionError("Expected SystemExit when --train is passed to prepare wrapper.")


def test_rejects_legacy_export_flags() -> None:
    try:
        wrapper.main(["--export-merged", "--merge-split", "0.8/0.2"])
    except SystemExit as exc:
        msg = str(exc)
        assert "prepare-only" in msg
        assert "--export-merged" in msg
    else:  # pragma: no cover - defensive branch
        raise AssertionError("Expected SystemExit when merge flags are passed to prepare wrapper.")


@pytest.mark.parametrize(
    ("observed_patch", "expected_error"),
    [
        ({"source_detect_run": "wrong_detect_run"}, "source_detect_run divergence"),
        ({"zarr_mtime_ns": 202}, "refined detect review row is stale"),
    ],
)
def test_detect_quality_gate_rejects_stale_or_divergent_rows(
    tmp_path: Path,
    monkeypatch,
    observed_patch: dict[str, object],
    expected_error: str,
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    dataset_path = tmp_path / "dataset_1.zarr"
    db = Registry(registry_path)
    _seed_dataset(db, "dataset_1", dataset_path)
    _upsert_detect_quality(db, dataset_id="dataset_1")
    db.close()

    observed = {
        "source_detect_run": "detect_run_1",
        "review_state": "approved",
        "review_intended_use": "training",
        "review_resolved_group": "filtered",
        "total_detections": 10,
        "real_detections": 9,
        "interpolated_detections": 1,
        "interpolated_detections_rate": 0.1,
        "zarr_mtime_ns": 101,
    }
    observed.update(observed_patch)

    def fake_resolve(zarr_path: Path, *, expected_refined_run: str) -> dict[str, object]:
        assert zarr_path == dataset_path
        assert expected_refined_run == "refined_run_1"
        return dict(observed)

    def fail_prepare(_cli: list[str]) -> None:  # pragma: no cover - defensive branch
        raise AssertionError("prepare_detect_training should not run when quality validation fails.")

    monkeypatch.setattr(wrapper, "_resolve_detect_quality_from_zarr", fake_resolve)
    monkeypatch.setattr(wrapper.pdt, "main", fail_prepare)

    with pytest.raises(ValueError, match=expected_error):
        wrapper.main(
            [
                "--registry",
                str(registry_path),
                "--require-review-state",
                "approved",
                "--require-review-intended-use",
                "training",
                "--max-interpolated-detections-rate",
                "0.2",
                "--dry-run",
            ]
        )


def test_detect_quality_exclusion_reasons_are_concrete(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    db = Registry(registry_path)
    dataset_ids = [
        "missing_quality",
        "review_state_mismatch",
        "review_use_mismatch",
        "missing_interp_rate",
        "interp_rate_high",
    ]
    for dataset_id in dataset_ids:
        _seed_dataset(db, dataset_id, tmp_path / f"{dataset_id}.zarr")
    _upsert_detect_quality(
        db,
        dataset_id="review_state_mismatch",
        review_state="pending",
        review_intended_use="training",
        interpolated_detections_rate=0.1,
    )
    _upsert_detect_quality(
        db,
        dataset_id="review_use_mismatch",
        review_state="approved",
        review_intended_use="full_recording",
        interpolated_detections_rate=0.1,
    )
    _upsert_detect_quality(
        db,
        dataset_id="missing_interp_rate",
        review_state="approved",
        review_intended_use="training",
        interpolated_detections_rate=None,
    )
    _upsert_detect_quality(
        db,
        dataset_id="interp_rate_high",
        review_state="approved",
        review_intended_use="training",
        interpolated_detections_rate=0.8,
    )
    db.close()

    with pytest.raises(SystemExit, match="No datasets remain after refined detect review filtering."):
        wrapper.main(
            [
                "--registry",
                str(registry_path),
                "--require-review-state",
                "approved",
                "--require-review-intended-use",
                "training",
                "--max-interpolated-detections-rate",
                "0.3",
            ]
        )

    output = capsys.readouterr().out
    assert "Refined detect review SQL filter excluded 5 dataset(s):" in output
    assert "Refined detect review filter summary: passed=0 excluded=5 reasons=" in output
    assert "missing_quality_row=1" in output
    assert "missing_quality_row" in output
    assert "review_state_mismatch:pending!=approved=1" in output
    assert "review_state_mismatch:pending!=approved" in output
    assert "review_use_mismatch:full_recording!=training=1" in output
    assert "review_use_mismatch:full_recording!=training" in output
    assert "missing_interpolated_detections_rate=1" in output
    assert "missing_interpolated_detections_rate" in output
    assert "interpolated_rate_above_threshold:0.800000>0.300000=1" in output
    assert "interpolated_rate_above_threshold:0.800000>0.300000" in output

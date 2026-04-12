from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from fisheye.utils import crop_batch as mod


def _make_archive(
    root: Path,
    recording: str,
    zarr_name: str,
    *,
    zarr_purpose: str | None,
) -> Path:
    zarr_path = root / recording / "zarr" / zarr_name
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(zarr_path), mode="w")
    if zarr_purpose is not None:
        group.attrs["zarr_purpose"] = zarr_purpose
    detect_parent = group.create_group("detect_runs")
    detect_parent.attrs["latest"] = "detect_001"
    detect = detect_parent.create_group("detect_001")
    detect.create_array("frame_indices", data=np.array([0, 1], dtype=np.int32), overwrite=True)
    detect.create_array(
        "bbox_norm_coords",
        data=np.array([[0.5, 0.5, 0.1, 0.1], [0.6, 0.6, 0.1, 0.1]], dtype=np.float32),
        overwrite=True,
    )
    return zarr_path


def test_build_plans_analysis_filter_skips_training(tmp_path: Path) -> None:
    analysis = _make_archive(
        tmp_path,
        "rec_a",
        "rec_a_analysis.zarr",
        zarr_purpose="analysis",
    )
    training = _make_archive(
        tmp_path,
        "rec_b",
        "rec_b_training.zarr",
        zarr_purpose="training",
    )

    plans = mod._build_plans(
        zarr_paths=[analysis, training],
        config={},
        source_type="detect",
        source_path=None,
        selection_policy=None,
        force_new=False,
        zarr_use_filter="analysis",
        crop_storage_mode=None,
    )
    by_name = {plan.zarr_path.name: plan for plan in plans}

    assert by_name[analysis.name].status == "ok"
    assert by_name[analysis.name].source_type == "detect"
    assert by_name[training.name].status == "skipped"
    assert "wanted=analysis" in (by_name[training.name].reason or "")


def test_build_plans_any_filter_includes_all_uses(tmp_path: Path) -> None:
    analysis = _make_archive(
        tmp_path,
        "rec_a",
        "rec_a_analysis.zarr",
        zarr_purpose="analysis",
    )
    training = _make_archive(
        tmp_path,
        "rec_b",
        "rec_b_training.zarr",
        zarr_purpose="training",
    )

    plans = mod._build_plans(
        zarr_paths=[analysis, training],
        config={},
        source_type="detect",
        source_path=None,
        selection_policy=None,
        force_new=False,
        zarr_use_filter="any",
        crop_storage_mode=None,
    )
    by_name = {plan.zarr_path.name: plan for plan in plans}
    assert by_name[analysis.name].status == "ok"
    assert by_name[training.name].status == "ok"


def test_main_strict_stops_after_first_failure(monkeypatch, tmp_path: Path) -> None:
    plans = [
        mod.CropPlan(zarr_path=Path("/tmp/a.zarr"), status="ok", source_type="detect", source_path="detect_runs/x"),
        mod.CropPlan(zarr_path=Path("/tmp/b.zarr"), status="ok", source_type="detect", source_path="detect_runs/y"),
    ]

    monkeypatch.setattr(mod, "_resolve_root", lambda _paths: [tmp_path])
    monkeypatch.setattr(mod, "_resolve_targets", lambda _roots, _recursive: [Path("/tmp/a.zarr"), Path("/tmp/b.zarr")])
    monkeypatch.setattr(mod, "_load_config", lambda _path: {})
    monkeypatch.setattr(mod, "_build_plans", lambda **_kwargs: plans)

    calls: list[str] = []

    def _fake_crop(*args, **kwargs):  # noqa: ANN002, ANN003
        calls.append(str(kwargs.get("zarr_path")))
        raise RuntimeError("boom")

    monkeypatch.setattr(mod, "crop_detections", _fake_crop)

    rc = mod.main(["--apply", "--strict", "--log-dir", str(tmp_path)])
    assert rc == 1
    assert len(calls) == 1


def test_main_non_strict_continues_after_failure(monkeypatch, tmp_path: Path) -> None:
    plans = [
        mod.CropPlan(zarr_path=Path("/tmp/a.zarr"), status="ok", source_type="detect", source_path="detect_runs/x"),
        mod.CropPlan(zarr_path=Path("/tmp/b.zarr"), status="ok", source_type="detect", source_path="detect_runs/y"),
    ]

    monkeypatch.setattr(mod, "_resolve_root", lambda _paths: [tmp_path])
    monkeypatch.setattr(mod, "_resolve_targets", lambda _roots, _recursive: [Path("/tmp/a.zarr"), Path("/tmp/b.zarr")])
    monkeypatch.setattr(mod, "_load_config", lambda _path: {})
    monkeypatch.setattr(mod, "_build_plans", lambda **_kwargs: plans)

    calls: list[str] = []

    def _fake_crop(*args, **kwargs):  # noqa: ANN002, ANN003
        calls.append(str(kwargs.get("zarr_path")))
        if len(calls) == 1:
            raise RuntimeError("boom")
        return {"detection_source_type": "detect", "detection_source_path": "detect_runs/y"}

    monkeypatch.setattr(mod, "crop_detections", _fake_crop)

    rc = mod.main(["--apply", "--log-dir", str(tmp_path)])
    assert rc == 1
    assert len(calls) == 2


def test_main_forwards_external_write_options(monkeypatch, tmp_path: Path) -> None:
    plans = [
        mod.CropPlan(
            zarr_path=Path("/tmp/a.zarr"),
            status="ok",
            source_type="detect",
            source_path="detect_runs/x",
        ),
    ]

    monkeypatch.setattr(mod, "_resolve_root", lambda _paths: [tmp_path])
    monkeypatch.setattr(mod, "_resolve_targets", lambda _roots, _recursive: [Path("/tmp/a.zarr")])
    monkeypatch.setattr(mod, "_load_config", lambda _path: {})
    monkeypatch.setattr(mod, "_build_plans", lambda **_kwargs: plans)

    seen_kwargs: dict = {}

    def _fake_crop(*args, **kwargs):  # noqa: ANN002, ANN003
        seen_kwargs.update(kwargs)
        return {"detection_source_type": "detect", "detection_source_path": "detect_runs/x"}

    monkeypatch.setattr(mod, "crop_detections", _fake_crop)

    rc = mod.main(
        [
            "--apply",
            "--log-dir",
            str(tmp_path),
            "--external-write-backend",
            "kvikio",
            "--external-roi-storage",
            "uncompressed",
            "--external-use-sharding",
            "--external-roi-chunk-size",
            "256",
            "--external-roi-shard-size",
            "2048",
            "--external-gpu-chunk-frames",
            "8",
            "--require-kvikio",
        ]
    )
    assert rc == 0
    assert seen_kwargs["external_write_backend"] == "kvikio"
    assert seen_kwargs["external_roi_storage"] == "uncompressed"
    assert seen_kwargs["external_use_sharding"] is True
    assert seen_kwargs["external_roi_chunk_size"] == 256
    assert seen_kwargs["external_roi_shard_size"] == 2048
    assert seen_kwargs["external_gpu_chunk_frames"] == 8
    assert seen_kwargs["external_require_kvikio"] is True


def test_build_plan_geometry_only_compares_against_latest_any(tmp_path: Path) -> None:
    zarr_path = _make_archive(
        tmp_path,
        "rec_a",
        "rec_a_analysis.zarr",
        zarr_purpose="analysis",
    )
    root = zarr.open_group(str(zarr_path), mode="a")
    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_materialized"
    crop_parent.attrs["latest_materialized"] = "crop_materialized"
    crop_parent.attrs["latest_any"] = "crop_geometry"

    crop_materialized = crop_parent.create_group("crop_materialized")
    crop_materialized.attrs.update(
        {
            "detection_source_path": "detect_runs/detect_001",
            "detection_source_type": "detect",
            "roi_size": [512, 512],
            "crop_storage_mode": "materialized",
            "status": "completed",
        }
    )
    crop_geometry = crop_parent.create_group("crop_geometry")
    crop_geometry.attrs.update(
        {
            "detection_source_path": "detect_runs/detect_001",
            "detection_source_type": "detect",
            "roi_size": [512, 512],
            "crop_storage_mode": "geometry_only",
            "status": "completed",
        }
    )

    plan = mod._build_plan(
        zarr_path=zarr_path,
        config={},
        source_type="detect",
        source_path=None,
        selection_policy=None,
        force_new=False,
        crop_storage_mode="geometry_only",
    )

    assert plan.status == "skipped"
    assert plan.reason == "matches latest crop run 'crop_geometry'"
    assert plan.latest_crop == "crop_geometry"
    assert plan.latest_pointer == "latest_any"
    assert plan.crop_storage_mode == "geometry_only"


def test_build_plan_materialized_compares_against_latest_materialized(tmp_path: Path) -> None:
    zarr_path = _make_archive(
        tmp_path,
        "rec_a",
        "rec_a_analysis.zarr",
        zarr_purpose="analysis",
    )
    root = zarr.open_group(str(zarr_path), mode="a")
    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_materialized"
    crop_parent.attrs["latest_materialized"] = "crop_materialized"
    crop_parent.attrs["latest_any"] = "crop_geometry"

    crop_materialized = crop_parent.create_group("crop_materialized")
    crop_materialized.attrs.update(
        {
            "detection_source_path": "detect_runs/detect_001",
            "detection_source_type": "detect",
            "roi_size": [512, 512],
            "crop_storage_mode": "materialized",
            "status": "completed",
        }
    )
    crop_geometry = crop_parent.create_group("crop_geometry")
    crop_geometry.attrs.update(
        {
            "detection_source_path": "detect_runs/detect_001",
            "detection_source_type": "detect",
            "roi_size": [512, 512],
            "crop_storage_mode": "geometry_only",
            "status": "completed",
        }
    )

    plan = mod._build_plan(
        zarr_path=zarr_path,
        config={},
        source_type="detect",
        source_path=None,
        selection_policy=None,
        force_new=False,
        crop_storage_mode="materialized",
    )

    assert plan.status == "skipped"
    assert plan.reason == "matches latest crop run 'crop_materialized'"
    assert plan.latest_crop == "crop_materialized"
    assert plan.latest_pointer == "latest_materialized"
    assert plan.crop_storage_mode == "materialized"


def test_build_plan_rejects_geometry_only_for_training_archive(tmp_path: Path) -> None:
    zarr_path = _make_archive(
        tmp_path,
        "rec_training",
        "rec_training_training.zarr",
        zarr_purpose="training",
    )

    plan = mod._build_plan(
        zarr_path=zarr_path,
        config={},
        source_type="detect",
        source_path=None,
        selection_policy=None,
        force_new=False,
        crop_storage_mode="geometry_only",
    )

    assert plan.status == "invalid"
    assert plan.reason == "training zarrs require materialized crop runs"
    assert plan.crop_storage_mode == "geometry_only"


def test_main_forwards_crop_storage_mode(monkeypatch, tmp_path: Path) -> None:
    plans = [
        mod.CropPlan(
            zarr_path=Path("/tmp/a.zarr"),
            status="ok",
            source_type="detect",
            source_path="detect_runs/x",
            crop_storage_mode="geometry_only",
        ),
    ]

    monkeypatch.setattr(mod, "_resolve_root", lambda _paths: [tmp_path])
    monkeypatch.setattr(mod, "_resolve_targets", lambda _roots, _recursive: [Path("/tmp/a.zarr")])
    monkeypatch.setattr(mod, "_load_config", lambda _path: {})
    monkeypatch.setattr(mod, "_build_plans", lambda **_kwargs: plans)

    seen_kwargs: dict = {}

    def _fake_crop(*args, **kwargs):  # noqa: ANN002, ANN003
        seen_kwargs.update(kwargs)
        return {
            "detection_source_type": "detect",
            "detection_source_path": "detect_runs/x",
            "crop_storage_mode": kwargs.get("crop_storage_mode"),
        }

    monkeypatch.setattr(mod, "crop_detections", _fake_crop)

    rc = mod.main(
        [
            "--apply",
            "--log-dir",
            str(tmp_path),
            "--crop-storage-mode",
            "geometry_only",
        ]
    )

    assert rc == 0
    assert seen_kwargs["crop_storage_mode"] == "geometry_only"


def test_main_reports_invalid_plans_in_summary(monkeypatch, tmp_path: Path, capsys) -> None:
    plans = [
        mod.CropPlan(
            zarr_path=Path("/tmp/a.zarr"),
            status="invalid",
            reason="training zarrs require materialized crop runs",
            crop_storage_mode="geometry_only",
        ),
    ]

    monkeypatch.setattr(mod, "_resolve_root", lambda _paths: [tmp_path])
    monkeypatch.setattr(mod, "_resolve_targets", lambda _roots, _recursive: [Path("/tmp/a.zarr")])
    monkeypatch.setattr(mod, "_load_config", lambda _path: {})
    monkeypatch.setattr(mod, "_build_plans", lambda **_kwargs: plans)

    rc = mod.main(["--log-dir", str(tmp_path)])
    out = capsys.readouterr().out

    assert rc == 0
    assert "invalid: 1" in out

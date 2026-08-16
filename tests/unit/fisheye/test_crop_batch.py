from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from fisheye.utils import crop_batch as mod
from fisheye.utils import crop_flat_roi_cache_batch as flat_mod


@pytest.fixture(autouse=True)
def _stub_canonical_crop_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep plan-mechanics fixtures focused; dedicated tests exercise the gate."""

    monkeypatch.setattr(
        mod,
        "get_video_source",
        lambda *_args, **_kwargs: ("zarr", None),
    )
    monkeypatch.setattr(
        mod,
        "_preflight_ordinary_crop_coordinates",
        lambda *_args, **_kwargs: SimpleNamespace(row_count=2),
    )


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
    assert by_name[analysis.name].crop_storage_mode == "materialized"
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
    assert by_name[analysis.name].crop_storage_mode == "materialized"
    assert by_name[training.name].status == "ok"
    assert by_name[training.name].crop_storage_mode == "materialized"


def test_build_plan_honors_configured_materialized_mode_for_analysis(tmp_path: Path) -> None:
    zarr_path = _make_archive(
        tmp_path,
        "rec_a",
        "rec_a_analysis.zarr",
        zarr_purpose="analysis",
    )

    plan = mod._build_plan(
        zarr_path=zarr_path,
        config={"crop": {"crop_storage_mode": "materialized"}},
        source_type="detect",
        source_path=None,
        selection_policy=None,
        force_new=False,
        crop_storage_mode=None,
    )

    assert plan.status == "ok"
    assert plan.crop_storage_mode == "materialized"


def test_build_plan_runs_full_canonical_preflight_before_reporting_ok(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    zarr_path = _make_archive(
        tmp_path,
        "rec_a",
        "rec_a_analysis.zarr",
        zarr_purpose="analysis",
    )
    seen: dict[str, object] = {}

    def preflight(root, **kwargs):  # noqa: ANN001, ANN202
        seen.update(kwargs)
        assert root["detect_runs/detect_001"].path == kwargs["source_group"].path
        return SimpleNamespace(row_count=2, padded_row_count=1)

    monkeypatch.setattr(mod, "_preflight_ordinary_crop_coordinates", preflight)

    plan = mod._build_plan(
        zarr_path=zarr_path,
        config={"crop": {"roi_sz": [64, 48]}},
        source_type="detect",
        source_path=None,
        selection_policy=None,
        force_new=False,
        crop_storage_mode="materialized",
    )

    assert plan.status == "ok"
    assert seen["source_path"] == "detect_runs/detect_001"
    assert seen["policy"].fixed_size_wh == (48, 64)
    assert seen["policy"].padding_mode.value == "zero_outside_source_frame"
    assert seen["video_source_type"] == "zarr"
    assert plan.padding_mode == "zero_outside_source_frame"
    assert plan.padded_row_count == 1


def test_build_plan_fails_closed_when_canonical_preflight_rejects_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    zarr_path = _make_archive(
        tmp_path,
        "rec_a",
        "rec_a_analysis.zarr",
        zarr_purpose="analysis",
    )
    monkeypatch.setattr(
        mod,
        "_preflight_ordinary_crop_coordinates",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("stale acquisition lineage")
        ),
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

    assert plan.status == "invalid"
    assert "canonical preflight failed" in (plan.reason or "")
    assert "stale acquisition lineage" in (plan.reason or "")


def test_build_plan_reports_validated_empty_source_without_scheduling_crop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    zarr_path = _make_archive(
        tmp_path,
        "rec_a",
        "rec_a_analysis.zarr",
        zarr_purpose="analysis",
    )
    monkeypatch.setattr(
        mod,
        "_preflight_ordinary_crop_coordinates",
        lambda *_args, **_kwargs: SimpleNamespace(row_count=0),
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

    assert plan.status == "missing"
    assert "validated canonical" in (plan.reason or "")


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
        return {
            "run_name": "crop_y",
            "total_crops": 1,
            "detection_source_type": "detect",
            "detection_source_path": "detect_runs/y",
        }

    monkeypatch.setattr(mod, "crop_detections", _fake_crop)

    rc = mod.main(["--apply", "--log-dir", str(tmp_path)])
    assert rc == 1
    assert len(calls) == 2


def test_main_treats_zero_row_result_as_failure(monkeypatch, tmp_path: Path) -> None:
    plans = [
        mod.CropPlan(
            zarr_path=Path("/tmp/a.zarr"),
            status="ok",
            source_type="detect",
            source_path="detect_runs/x",
        ),
    ]
    monkeypatch.setattr(mod, "_resolve_root", lambda _paths: [tmp_path])
    monkeypatch.setattr(
        mod,
        "_resolve_targets",
        lambda _roots, _recursive: [Path("/tmp/a.zarr")],
    )
    monkeypatch.setattr(mod, "_load_config", lambda _path: {})
    monkeypatch.setattr(mod, "_build_plans", lambda **_kwargs: plans)
    monkeypatch.setattr(
        mod,
        "crop_detections",
        lambda **_kwargs: {"total_crops": 0, "run_name": None},
    )

    assert mod.main(["--apply", "--log-dir", str(tmp_path)]) == 1


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
        return {
            "run_name": "crop_x",
            "total_crops": 1,
            "detection_source_type": "detect",
            "detection_source_path": "detect_runs/x",
        }

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


def test_build_plan_rejects_geometry_only_instead_of_reusing_latest_any(tmp_path: Path) -> None:
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

    policy = mod.ordinary_crop_geometry_policy_from_parameters(
        {"roi_sz": [348, 348], "padding_mode": "zero_outside_source_frame"}
    )
    crop_materialized = crop_parent.create_group("crop_materialized")
    crop_materialized.attrs.update(
        {
            "detection_source_path": "detect_runs/detect_001",
            "detection_source_type": "detect",
            "roi_size": [348, 348],
            "crop_geometry_policy_digest": policy.payload_digest,
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

    assert plan.status == "invalid"
    assert "requires crop_storage_mode=materialized" in (plan.reason or "")
    assert plan.latest_crop is None
    assert plan.latest_pointer is None
    assert plan.crop_storage_mode == "geometry_only"


def test_build_plan_materialized_compares_against_validated_latest_materialized(
    monkeypatch,
    tmp_path: Path,
) -> None:
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

    policy = mod.ordinary_crop_geometry_policy_from_parameters(
        {"roi_sz": [384, 384], "padding_mode": "zero_outside_source_frame"}
    )
    crop_materialized = crop_parent.create_group("crop_materialized")
    crop_materialized.attrs.update(
        {
            "detection_source_path": "detect_runs/detect_001",
            "detection_source_type": "detect",
            "roi_size": [384, 384],
            "crop_geometry_policy_digest": policy.payload_digest,
            "crop_storage_mode": "materialized",
            "status": "completed",
            "coordinate_contract": "canonical_v2",
            "stage_selector_eligible": True,
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
    monkeypatch.setattr(
        mod,
        "load_persisted_ordinary_crop_observation_geometry",
        lambda _root, _path: object(),
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
    assert plan.reason == "matches validated canonical crop run 'crop_materialized'"
    assert plan.latest_crop == "crop_materialized"
    assert plan.latest_pointer == "latest_materialized"
    assert plan.crop_storage_mode == "materialized"


def test_build_plan_replaces_canonical_run_without_explicit_padding_policy(
    monkeypatch,
    tmp_path: Path,
) -> None:
    zarr_path = _make_archive(
        tmp_path,
        "rec_a",
        "rec_a_analysis.zarr",
        zarr_purpose="analysis",
    )
    root = zarr.open_group(str(zarr_path), mode="a")
    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_old"
    crop_parent.attrs["latest_materialized"] = "crop_old"
    crop_old = crop_parent.create_group("crop_old")
    crop_old.attrs.update(
        {
            "detection_source_path": "detect_runs/detect_001",
            "detection_source_type": "detect",
            "roi_size": [384, 384],
            "crop_storage_mode": "materialized",
            "status": "completed",
            "coordinate_contract": "canonical_v2",
            "stage_selector_eligible": True,
        }
    )
    monkeypatch.setattr(
        mod,
        "load_persisted_ordinary_crop_observation_geometry",
        lambda _root, _path: object(),
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

    assert plan.status == "ok"
    assert plan.reason == "differs: crop_geometry_policy"


def test_build_plan_does_not_reuse_legacy_materialized_run(tmp_path: Path) -> None:
    zarr_path = _make_archive(
        tmp_path,
        "rec_a",
        "rec_a_analysis.zarr",
        zarr_purpose="analysis",
    )
    root = zarr.open_group(str(zarr_path), mode="a")
    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs.update(
        {
            "latest": "crop_legacy",
            "latest_materialized": "crop_legacy",
        }
    )
    legacy = crop_parent.create_group("crop_legacy")
    legacy.attrs.update(
        {
            "detection_source_path": "detect_runs/detect_001",
            "detection_source_type": "detect",
            "roi_size": [512, 512],
            "crop_storage_mode": "materialized",
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
        crop_storage_mode=None,
    )

    assert plan.status == "ok"
    assert "not an eligible canonical_v2 run" in (plan.reason or "")
    assert plan.latest_crop == "crop_legacy"


def test_build_plan_does_not_trust_declared_canonical_attrs_without_validation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    zarr_path = _make_archive(
        tmp_path,
        "rec_a",
        "rec_a_analysis.zarr",
        zarr_purpose="analysis",
    )
    root = zarr.open_group(str(zarr_path), mode="a")
    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest_materialized"] = "crop_bad"
    bad = crop_parent.create_group("crop_bad")
    bad.attrs.update(
        {
            "detection_source_path": "detect_runs/detect_001",
            "detection_source_type": "detect",
            "roi_size": [512, 512],
            "crop_storage_mode": "materialized",
            "status": "completed",
            "coordinate_contract": "canonical_v2",
            "stage_selector_eligible": True,
        }
    )
    monkeypatch.setattr(
        mod,
        "load_persisted_ordinary_crop_observation_geometry",
        lambda _root, _path: (_ for _ in ()).throw(RuntimeError("invalid lineage")),
    )

    plan = mod._build_plan(
        zarr_path=zarr_path,
        config={},
        source_type="detect",
        source_path=None,
        selection_policy=None,
        force_new=False,
        crop_storage_mode=None,
    )

    assert plan.status == "ok"
    assert "failed canonical validation" in (plan.reason or "")


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
    assert "requires crop_storage_mode=materialized" in (plan.reason or "")
    assert plan.crop_storage_mode == "geometry_only"


def test_main_forwards_crop_storage_mode(monkeypatch, tmp_path: Path) -> None:
    plans = [
        mod.CropPlan(
            zarr_path=Path("/tmp/a.zarr"),
            status="ok",
            source_type="detect",
            source_path="detect_runs/x",
            crop_storage_mode="materialized",
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
            "run_name": "crop_x",
            "total_crops": 1,
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
            "materialized",
        ]
    )

    assert rc == 0
    assert seen_kwargs["crop_storage_mode"] == "materialized"


def test_apply_result_json_carries_exact_committed_crop_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "recording.zarr"
    plan = mod.CropPlan(
        zarr_path=zarr_path,
        status="ok",
        source_type="detect",
        source_path="detect_runs/d1",
        crop_storage_mode="materialized",
    )
    monkeypatch.setattr(mod, "_resolve_root", lambda _paths: [tmp_path])
    monkeypatch.setattr(mod, "_resolve_targets", lambda *_args: [zarr_path])
    monkeypatch.setattr(mod, "_load_config", lambda _path: {})
    monkeypatch.setattr(mod, "_build_plans", lambda **_kwargs: [plan])
    monkeypatch.setattr(
        mod,
        "crop_detections",
        lambda **_kwargs: {
            "run_name": "crop_exact_001",
            "total_crops": 2,
            "crop_storage_mode": "materialized",
        },
    )
    result_path = tmp_path / "crop-result.json"

    assert (
        mod.main(
            [
                "--apply",
                "--log-dir",
                str(tmp_path / "logs"),
                "--result-json",
                str(result_path),
            ]
        )
        == 0
    )

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "palette.crop_batch_result.v1"
    assert payload["outcomes"] == [
        {
            "crop_run": "crop_exact_001",
            "reason": None,
            "status": "ok",
            "zarr_path": str(zarr_path),
        }
    ]


def test_main_reports_invalid_plans_in_summary(monkeypatch, tmp_path: Path, capsys) -> None:
    plans = [
        mod.CropPlan(
            zarr_path=Path("/tmp/a.zarr"),
            status="invalid",
            reason="future-canonical ordinary crop requires materialized crop runs",
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
    assert (
        mod.main(
            [
                "--log-dir",
                str(tmp_path),
                "--fail-on-invalid-plan",
            ]
        )
        == 1
    )


def test_build_plan_rejects_refined_source_before_latest_selection(
    monkeypatch,
    tmp_path: Path,
) -> None:
    zarr_path = _make_archive(
        tmp_path,
        "rec_a",
        "rec_a_analysis.zarr",
        zarr_purpose="analysis",
    )
    monkeypatch.setattr(
        mod,
        "get_detection_source_info",
        lambda **_kwargs: (
            "refined_detect_runs/refined_001/instances",
            object(),
            None,
            "refined",
        ),
    )

    plan = mod._build_plan(
        zarr_path=zarr_path,
        config={},
        source_type="refined",
        source_path=None,
        selection_policy=None,
        force_new=False,
        crop_storage_mode=None,
    )

    assert plan.status == "invalid"
    assert "exact detect_runs/<run> source" in (plan.reason or "")
    assert plan.latest_crop is None


def test_batch_cli_defaults_to_exact_detect_source(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(mod, "_resolve_root", lambda _paths: [tmp_path])
    monkeypatch.setattr(
        mod,
        "_resolve_targets",
        lambda _roots, _recursive: [tmp_path / "recording.zarr"],
    )
    monkeypatch.setattr(mod, "_load_config", lambda _path: {})

    def _capture_plans(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(mod, "_build_plans", _capture_plans)

    assert mod.main(["--log-dir", str(tmp_path)]) == 0
    assert captured["source_type"] == "detect"


def test_flat_cache_workflow_never_falls_back_after_failed_new_crop() -> None:
    plan = mod.CropPlan(
        zarr_path=Path("/tmp/a.zarr"),
        status="ok",
        latest_crop="legacy_crop",
    )

    assert (
        flat_mod._resolve_crop_run_after_crop(
            plan,
            {"total_crops": 0, "run_name": None},
        )
        is None
    )


def test_flat_cache_workflow_reuses_only_validated_skipped_crop() -> None:
    plan = mod.CropPlan(
        zarr_path=Path("/tmp/a.zarr"),
        status="skipped",
        latest_crop="canonical_crop",
    )

    assert flat_mod._resolve_crop_run_after_crop(plan, None) == "canonical_crop"

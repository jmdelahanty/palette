from __future__ import annotations

from pathlib import Path


class _FakeRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.conn = _FakeConn()

    def close(self) -> None:
        pass

    def refresh_eye_mask_performance_for_dataset(self, *args, **kwargs) -> int:  # type: ignore[no-untyped-def]
        return 0

    def refresh_eye_mask_quality_for_dataset(self, *args, **kwargs) -> int:  # type: ignore[no-untyped-def]
        return 0

    def refresh_subject_mask_performance_for_dataset(self, *args, **kwargs) -> int:  # type: ignore[no-untyped-def]
        return 0

    def refresh_subject_mask_component_quality_for_dataset(self, *args, **kwargs) -> int:  # type: ignore[no-untyped-def]
        return 0


class _FakeConn:
    def execute(self, sql: str, params=()):  # type: ignore[no-untyped-def]
        if "SELECT recording_id, zarr_use" in sql:
            return _FakeCursor({"recording_id": "rec", "zarr_use": "analysis"})
        if "SELECT details_json" in sql:
            return _FakeCursor(None)
        return _FakeCursor(None)


class _FakeCursor:
    def __init__(self, row) -> None:  # type: ignore[no-untyped-def]
        self._row = row

    def fetchone(self):  # type: ignore[no-untyped-def]
        return self._row


class _Console:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def print(self, *parts, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.messages.append(" ".join(str(part) for part in parts))


def test_predict_pose_status_opens_root_for_ok_run(monkeypatch, tmp_path: Path) -> None:
    from fisheye.inference import predict_pose

    sentinel_root = object()
    captured: dict[str, object] = {}

    monkeypatch.setattr(predict_pose, "open_zarr_group_direct", lambda path, mode: sentinel_root)
    monkeypatch.setattr(predict_pose, "Registry", _FakeRegistry)

    def _emit(root, zarr_path, **kwargs):  # type: ignore[no-untyped-def]
        captured["root"] = root
        captured.update(kwargs)
        return True

    monkeypatch.setattr(predict_pose, "emit_stage_completion", _emit)

    context = predict_pose._StatusContext(
        registry_path=tmp_path / "registry.sqlite",
        dataset_id="dataset",
        recording_id="rec",
        zarr_path=tmp_path / "archive.zarr",
    )

    predict_pose._emit_keypoint_status(
        context=context,
        status="ok",
        run_name="keypoints_001",
        method="yolo",
        coverage_pct=90.0,
        details={"reason": "completed"},
        console=_Console(),  # type: ignore[arg-type]
    )

    assert captured["root"] is sentinel_root
    assert captured["status"] == "ok"
    assert captured["run_name"] == "keypoints_001"


def test_refine_keypoints_status_opens_root_for_ok_run(monkeypatch, tmp_path: Path) -> None:
    from fisheye.refinement import refine_keypoints

    sentinel_root = object()
    captured: dict[str, object] = {}

    monkeypatch.setattr(refine_keypoints, "open_zarr_group_direct", lambda path, mode: sentinel_root)
    monkeypatch.setattr(refine_keypoints, "Registry", _FakeRegistry)

    def _emit(root, zarr_path, **kwargs):  # type: ignore[no-untyped-def]
        captured["root"] = root
        captured.update(kwargs)
        return True

    monkeypatch.setattr(refine_keypoints, "emit_stage_completion", _emit)

    context = refine_keypoints._StatusContext(
        registry_path=tmp_path / "registry.sqlite",
        dataset_id="dataset",
        recording_id="rec",
        zarr_path=tmp_path / "archive.zarr",
    )

    wrote = refine_keypoints._emit_refined_keypoint_status(
        context=context,
        status="ok",
        run_name="refined_keypoints_001",
        method="refine_keypoints",
        coverage_pct=91.0,
        review_status_json={"status": "approved"},
        details={"reason": "present"},
        console=_Console(),  # type: ignore[arg-type]
    )

    assert wrote is True
    assert captured["root"] is sentinel_root
    assert captured["status"] == "ok"
    assert captured["run_name"] == "refined_keypoints_001"


def test_refine_keypoints_status_skips_when_registry_writes_disabled(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from fisheye.refinement import refine_keypoints

    monkeypatch.setenv("PALETTE_DISABLE_REGISTRY_WRITES", "1")
    monkeypatch.setattr(
        refine_keypoints,
        "open_zarr_group_direct",
        lambda path, mode: (_ for _ in ()).throw(AssertionError("opened zarr root")),
    )
    monkeypatch.setattr(
        refine_keypoints,
        "Registry",
        lambda path: (_ for _ in ()).throw(AssertionError("opened registry")),
    )

    context = refine_keypoints._StatusContext(
        registry_path=tmp_path / "registry.sqlite",
        dataset_id="dataset",
        recording_id="rec",
        zarr_path=tmp_path / "archive.zarr",
    )
    console = _Console()

    wrote = refine_keypoints._emit_refined_keypoint_status(
        context=context,
        status="ok",
        run_name="refined_keypoints_001",
        method="refine_keypoints",
        coverage_pct=91.0,
        review_status_json={"status": "approved"},
        details={"reason": "present"},
        console=console,  # type: ignore[arg-type]
    )

    assert wrote is False
    assert any("Registry writes disabled" in message for message in console.messages)


def test_keypoints_batch_auto_review_sync_opens_root_for_ok_run(monkeypatch, tmp_path: Path) -> None:
    from fisheye.utils import run_keypoints_batch

    sentinel_root = object()
    captured: dict[str, object] = {}

    monkeypatch.setattr(run_keypoints_batch, "open_zarr_group_direct", lambda path, mode: sentinel_root)
    monkeypatch.setattr(run_keypoints_batch, "Registry", _FakeRegistry)
    monkeypatch.setattr(
        run_keypoints_batch,
        "_resolve_step_status_context",
        lambda zarr_path, registry_path=None: {
            "registry_path": tmp_path / "registry.sqlite",
            "dataset_id": "dataset",
            "recording_id": "rec",
        },
    )

    def _emit(root, zarr_path, **kwargs):  # type: ignore[no-untyped-def]
        captured["root"] = root
        captured.update(kwargs)
        return True

    monkeypatch.setattr(run_keypoints_batch, "emit_stage_completion", _emit)

    result = run_keypoints_batch._sync_refined_keypoint_step_status_after_auto_review(
        zarr_path=tmp_path / "archive.zarr",
        refined_run="refined_keypoints_001",
        stage_payload={"method": "refine_keypoints", "summary_statistics": {"pass_rate_percent": 92.0}},
        review_status={"status": "approved"},
    )

    assert result["synced"] is True
    assert captured["root"] is sentinel_root
    assert captured["status"] == "ok"
    assert captured["step_name"] == "refined_keypoints"


def test_eye_masks_batch_sync_opens_root_for_each_ok_run(monkeypatch, tmp_path: Path) -> None:
    from fisheye.utils import run_eye_masks_batch

    sentinel_root = object()
    roots: list[object] = []

    monkeypatch.setattr(run_eye_masks_batch, "open_zarr_group_direct", lambda path, mode: sentinel_root)
    monkeypatch.setattr(run_eye_masks_batch, "Registry", _FakeRegistry)
    monkeypatch.setattr(
        run_eye_masks_batch,
        "_resolve_step_status_context",
        lambda zarr_path, registry_path=None: {
            "registry_path": tmp_path / "registry.sqlite",
            "dataset_id": "dataset",
            "recording_id": "rec",
        },
    )

    def _emit(root, zarr_path, **kwargs):  # type: ignore[no-untyped-def]
        roots.append(root)
        return True

    monkeypatch.setattr(run_eye_masks_batch, "emit_stage_completion", _emit)

    result = run_eye_masks_batch._sync_eye_mask_registry_rows_after_run(
        zarr_path=tmp_path / "archive.zarr",
        stage_payloads={
            "eye_masks": {"run_name": "eye_masks_001", "method": "unet"},
            "refined_eye_masks": {"run_name": "refined_eye_masks_001", "method": "refine_eye_masks"},
        },
    )

    assert result["synced"] is True
    assert result["step_status_written"] == ["eye_masks", "refined_eye_masks"]
    assert roots == [sentinel_root, sentinel_root]


def test_detect_quality_status_opens_root_directly_for_fresh_run(monkeypatch, tmp_path: Path) -> None:
    from fisheye.refinement import detect_quality

    sentinel_root = object()
    captured: dict[str, object] = {}

    monkeypatch.setattr(detect_quality, "open_zarr_group_direct", lambda path, mode: sentinel_root)

    def _emit(root, zarr_path, **kwargs):  # type: ignore[no-untyped-def]
        captured["root"] = root
        captured["zarr_path"] = zarr_path
        captured.update(kwargs)
        return True

    monkeypatch.setattr(detect_quality, "emit_stage_completion", _emit)

    detect_quality._emit_detect_quality_status(
        zarr_path=str(tmp_path / "archive.zarr"),
        quality_run_name="detect_quality_001",
        source_detect_run="detect_001",
        source_detect_family_path="detect_runs",
        quality_score={"grade": "A", "overall_score": 99.0, "coverage_score": 98.0},
    )

    assert captured["root"] is sentinel_root
    assert captured["step_name"] == "detect_quality"
    assert captured["status"] == "ok"
    assert captured["run_name"] == "detect_quality_001"
    assert captured["details_json"]["source_detect_run"] == "detect_001"  # type: ignore[index]


def test_predict_pose_non_ok_status_still_bypasses_root_open(monkeypatch, tmp_path: Path) -> None:
    from fisheye.inference import predict_pose

    captured: dict[str, object] = {}

    def _open(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("non-ok status should not open root for run validation")

    monkeypatch.setattr(predict_pose, "open_zarr_group_direct", _open)
    monkeypatch.setattr(predict_pose, "Registry", _FakeRegistry)

    def _emit(root, zarr_path, **kwargs):  # type: ignore[no-untyped-def]
        captured["root"] = root
        captured.update(kwargs)
        return True

    monkeypatch.setattr(predict_pose, "emit_stage_completion", _emit)

    context = predict_pose._StatusContext(
        registry_path=tmp_path / "registry.sqlite",
        dataset_id="dataset",
        recording_id="rec",
        zarr_path=tmp_path / "archive.zarr",
    )

    predict_pose._emit_keypoint_status(
        context=context,
        status="missing",
        run_name="keypoints_001",
        method=None,
        coverage_pct=None,
        details={"reason": "no_rois"},
        console=_Console(),  # type: ignore[arg-type]
    )

    assert captured["root"] is None
    assert captured["status"] == "missing"

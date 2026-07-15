from __future__ import annotations

import json
from pathlib import Path
import sys
import types
from fisheye.utils import run_detections_batch as mod


def _patch_detect_yolo(monkeypatch, func) -> None:
    fake_module = types.ModuleType("fisheye.detection.detect_yolo")
    fake_module.detect_yolo = func
    monkeypatch.setitem(sys.modules, "fisheye.detection.detect_yolo", fake_module)


def _write_root_metadata(zarr_path: Path, attrs: dict[str, object] | None = None) -> None:
    zarr_path.mkdir(parents=True, exist_ok=True)
    (zarr_path / "zarr.json").write_text(
        json.dumps(
            {
                "attributes": attrs or {},
                "zarr_format": 3,
                "consolidated_metadata": None,
                "node_type": "group",
            }
        ),
        encoding="utf-8",
    )


def _write_group_attrs(zarr_path: Path, group_name: str, attrs: dict[str, object]) -> None:
    group_dir = zarr_path / group_name
    group_dir.mkdir(parents=True, exist_ok=True)
    (group_dir / "zarr.json").write_text(
        json.dumps(
            {
                "attributes": attrs,
                "zarr_format": 3,
                "consolidated_metadata": None,
                "node_type": "group",
            }
        ),
        encoding="utf-8",
    )


def test_discover_analysis_zarrs_is_deterministic(tmp_path: Path) -> None:
    rec_b = tmp_path / "rec_b"
    rec_a = tmp_path / "rec_a"
    (rec_b / "zarr" / "rec_b_analysis.zarr").mkdir(parents=True, exist_ok=True)
    (rec_a / "zarr" / "rec_a_analysis.zarr").mkdir(parents=True, exist_ok=True)

    found = mod._discover_analysis_zarrs(  # noqa: SLF001
        [tmp_path, rec_b / "zarr" / "rec_b_analysis.zarr"],
        recursive=True,
    )

    found_str = [str(path) for path in found]
    assert found_str == sorted(found_str)
    assert found_str == [
        str((rec_a / "zarr" / "rec_a_analysis.zarr").resolve()),
        str((rec_b / "zarr" / "rec_b_analysis.zarr").resolve()),
    ]


def test_build_plans_applies_status_reason_taxonomy(tmp_path: Path) -> None:
    rec_ok = tmp_path / "rec_ok"
    rec_skip = tmp_path / "rec_skip"
    rec_no_video = tmp_path / "rec_no_video"

    zarr_ok = rec_ok / "zarr" / "rec_ok_analysis.zarr"
    zarr_skip = rec_skip / "zarr" / "rec_skip_analysis.zarr"
    zarr_no_video = rec_no_video / "zarr" / "rec_no_video_analysis.zarr"

    _write_root_metadata(zarr_ok)
    _write_root_metadata(zarr_skip)
    _write_root_metadata(zarr_no_video)

    _write_group_attrs(zarr_ok, "detect_runs", {"latest": None})
    _write_group_attrs(zarr_ok, "background_runs", {"latest": "bg_1"})
    _write_group_attrs(zarr_ok, "analysis_metadata", {"detection_tuning": {"enabled": True}})

    _write_group_attrs(zarr_skip, "detect_runs", {"latest": "detect_1"})
    _write_group_attrs(zarr_skip, "background_runs", {"latest": "bg_1"})
    _write_group_attrs(zarr_skip, "analysis_metadata", {"detection_tuning": {"enabled": True}})

    _write_group_attrs(zarr_no_video, "detect_runs", {"latest": None})
    _write_group_attrs(zarr_no_video, "background_runs", {"latest": "bg_1"})
    _write_group_attrs(zarr_no_video, "analysis_metadata", {"detection_tuning": {"enabled": True}})

    (rec_ok / "cams").mkdir(parents=True, exist_ok=True)
    (rec_skip / "cams").mkdir(parents=True, exist_ok=True)
    (rec_ok / "cams" / "cam_1.mp4").write_bytes(b"")
    (rec_skip / "cams" / "cam_1.mp4").write_bytes(b"")

    plans = mod._build_plans(  # noqa: SLF001
        [zarr_no_video, zarr_skip, zarr_ok],
        skip_existing=True,
        require_background=False,
        require_tuning=False,
    )

    by_path = {str(plan.zarr_path): plan for plan in plans}

    ok_plan = by_path[str(zarr_ok.resolve())]
    assert ok_plan.status == mod.STATUS_OK
    assert ok_plan.reason is None

    skip_plan = by_path[str(zarr_skip.resolve())]
    assert skip_plan.status == mod.STATUS_SKIPPED
    assert skip_plan.reason == mod.REASON_DETECT_ALREADY_PRESENT

    missing_plan = by_path[str(zarr_no_video.resolve())]
    assert missing_plan.status == mod.STATUS_MISSING
    assert missing_plan.reason == mod.REASON_CAMS_DIR_MISSING


def test_main_apply_uses_pre_resolved_model_and_returns_failure(monkeypatch, tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    registry_path.write_text("", encoding="utf-8")

    zarr_path = tmp_path / "rec_a" / "zarr" / "rec_a_analysis.zarr"
    zarr_path.mkdir(parents=True, exist_ok=True)
    recording_dir = zarr_path.parent.parent
    video_path = recording_dir / "cams" / "cam_1.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"")

    plan = mod.DetectPlan(
        zarr_path=zarr_path.resolve(),
        recording_dir=recording_dir.resolve(),
        video_path=video_path.resolve(),
        status=mod.STATUS_OK,
    )

    monkeypatch.setattr(mod, "_resolve_input_paths", lambda *_args, **_kwargs: [tmp_path])
    monkeypatch.setattr(mod, "_discover_analysis_zarrs", lambda *_args, **_kwargs: [zarr_path.resolve()])
    monkeypatch.setattr(mod, "_build_plans", lambda *_args, **_kwargs: [plan])

    monkeypatch.setattr(
        mod,
        "_resolve_registry_models_for_plans",
        lambda **_kwargs: (  # noqa: ARG005
            {
                str(zarr_path.resolve()): mod.ResolvedModel(
                    model_path="/tmp/model.pt",
                    payload={
                        "selected": {
                            "run_id": "run_1",
                            "set_id": "set_1",
                            "model_path": "/tmp/model.pt",
                        }
                    },
                )
            },
            {},
        ),
    )

    calls: list[dict[str, object]] = []

    def _fake_run_detect_plan(**kwargs):  # noqa: ANN003
        calls.append(kwargs)
        return mod.DetectRegistryResult(
            ok=False,
            status="failed",
            recording_dir=str(recording_dir.resolve()),
            output_zarr=str(zarr_path.resolve()),
            registry_path=str(registry_path.resolve()),
            reason="detect_inference_failed",
            error="oom",
            remediation="retry",
            selected_model_path="/tmp/model.pt",
            selected_run_id="run_1",
            selected_set_id="set_1",
            resolved_at_utc=None,
            detect_run=None,
        )

    monkeypatch.setattr(mod, "_run_detect_plan", _fake_run_detect_plan)

    rc = mod.main(["--apply", "--json", "--no-log", "--registry", str(registry_path), str(tmp_path)])

    assert rc == 1
    assert len(calls) == 1
    assert calls[0]["plan"].recording_dir == recording_dir.resolve()
    assert calls[0]["plan"].video_path == video_path.resolve()
    assert calls[0]["plan"].zarr_path == zarr_path.resolve()
    assert calls[0]["registry_path"] == registry_path.resolve()
    assert calls[0]["resolved_model"].model_path == "/tmp/model.pt"
    assert calls[0]["detect_row_shard_rows"] == 131_072
    assert calls[0]["detect_frame_shard_rows"] == 131_072


def test_build_plan_for_zarr_uses_source_video_path_from_copied_archive(tmp_path: Path) -> None:
    source_recording = tmp_path / "source_recording"
    source_video = source_recording / "cams" / "cam_1.mp4"
    source_video.parent.mkdir(parents=True, exist_ok=True)
    source_video.write_bytes(b"")

    zarr_path = tmp_path / "smoke" / "copied_analysis.zarr"
    _write_root_metadata(zarr_path, attrs={"source_video_path": str(source_video.resolve())})
    _write_group_attrs(zarr_path, "analysis_metadata", {"detection_tuning": {"enabled": True}})
    _write_group_attrs(zarr_path, "detect_runs", {"latest": None})

    plan = mod._build_plan_for_zarr(  # noqa: SLF001
        zarr_path=zarr_path,
        skip_existing=False,
        require_background=False,
        require_tuning=False,
    )

    assert plan.status == mod.STATUS_OK
    assert plan.recording_dir == source_recording.resolve()
    assert plan.video_path == source_video.resolve()


def test_build_plan_for_zarr_prefers_copied_cams_over_stale_source_attrs(tmp_path: Path) -> None:
    stale_source = tmp_path / "stale_source" / "cams" / "cam_old.mp4"
    stale_source.parent.mkdir(parents=True, exist_ok=True)
    stale_source.write_bytes(b"")

    copied_recording = tmp_path / "copied_recording"
    copied_video = copied_recording / "cams" / "cam_copied.mp4"
    copied_video.parent.mkdir(parents=True, exist_ok=True)
    copied_video.write_bytes(b"")

    zarr_path = copied_recording / "zarr" / "copied_recording_analysis.zarr"
    _write_root_metadata(zarr_path, attrs={"source_video_path": str(stale_source.resolve())})
    _write_group_attrs(zarr_path, "analysis_metadata", {"detection_tuning": {"enabled": True}})
    _write_group_attrs(zarr_path, "detect_runs", {"latest": None})

    plan = mod._build_plan_for_zarr(  # noqa: SLF001
        zarr_path=zarr_path,
        skip_existing=False,
        require_background=False,
        require_tuning=False,
    )

    assert plan.status == mod.STATUS_OK
    assert plan.recording_dir == copied_recording.resolve()
    assert plan.video_path == copied_video.resolve()


def test_main_dry_run_marks_registry_missing(monkeypatch, tmp_path: Path, capsys) -> None:
    zarr_path = tmp_path / "rec_a" / "zarr" / "rec_a_analysis.zarr"
    zarr_path.mkdir(parents=True, exist_ok=True)
    recording_dir = zarr_path.parent.parent
    video_path = recording_dir / "cams" / "cam_1.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"")

    plan = mod.DetectPlan(
        zarr_path=zarr_path.resolve(),
        recording_dir=recording_dir.resolve(),
        video_path=video_path.resolve(),
        status=mod.STATUS_OK,
    )

    monkeypatch.setattr(mod, "_resolve_input_paths", lambda *_args, **_kwargs: [tmp_path])
    monkeypatch.setattr(mod, "_discover_analysis_zarrs", lambda *_args, **_kwargs: [zarr_path.resolve()])
    monkeypatch.setattr(mod, "_build_plans", lambda *_args, **_kwargs: [plan])

    def _unexpected_resolve(**_kwargs):
        raise AssertionError("_resolve_registry_models_for_plans should not be called in dry-run")

    monkeypatch.setattr(mod, "_resolve_registry_models_for_plans", _unexpected_resolve)

    missing_registry = tmp_path / "missing_registry.sqlite"
    rc = mod.main(["--dry-run", "--json", "--no-log", "--registry", str(missing_registry), str(tmp_path)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "registry_missing" in out


def test_main_dry_run_resolve_models_emits_selected_model(monkeypatch, tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    registry_path.write_text("", encoding="utf-8")

    zarr_path = tmp_path / "rec_a" / "zarr" / "rec_a_analysis.zarr"
    zarr_path.mkdir(parents=True, exist_ok=True)
    recording_dir = zarr_path.parent.parent
    video_path = recording_dir / "cams" / "cam_1.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"")

    plan = mod.DetectPlan(
        zarr_path=zarr_path.resolve(),
        recording_dir=recording_dir.resolve(),
        video_path=video_path.resolve(),
        status=mod.STATUS_OK,
    )

    monkeypatch.setattr(mod, "_resolve_input_paths", lambda *_args, **_kwargs: [tmp_path])
    monkeypatch.setattr(mod, "_discover_analysis_zarrs", lambda *_args, **_kwargs: [zarr_path.resolve()])
    monkeypatch.setattr(mod, "_build_plans", lambda *_args, **_kwargs: [plan])
    monkeypatch.setattr(
        mod,
        "_resolve_registry_models_for_plans",
        lambda **_kwargs: (
            {
                str(zarr_path.resolve()): mod.ResolvedModel(
                    model_path="/tmp/registry_model.pt",
                    payload={
                        "selected": {
                            "run_id": "run_1",
                            "set_id": "set_1",
                            "model_path": "/tmp/registry_model.pt",
                        }
                    },
                )
            },
            {},
        ),
    )

    rc = mod.main(
        [
            "--dry-run",
            "--json",
            "--resolve-models",
            "--no-log",
            "--registry",
            str(registry_path),
            str(tmp_path),
        ]
    )

    assert rc == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    assert rows == [
        {
            "background_present": False,
            "detect_present": False,
            "model_resolution_status": "ok",
            "reason": None,
            "recording": str(recording_dir.resolve()),
            "selected_model": "/tmp/registry_model.pt",
            "selected_run_id": "run_1",
            "selected_set_id": "set_1",
            "status": "ok",
            "tuning_present": False,
            "video": str(video_path.resolve()),
            "zarr": str(zarr_path.resolve()),
        }
    ]


def test_main_dry_run_with_explicit_model_does_not_require_registry(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    zarr_path = tmp_path / "rec_a" / "zarr" / "rec_a_analysis.zarr"
    zarr_path.mkdir(parents=True, exist_ok=True)
    recording_dir = zarr_path.parent.parent
    video_path = recording_dir / "cams" / "cam_1.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"")
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"model")

    plan = mod.DetectPlan(
        zarr_path=zarr_path.resolve(),
        recording_dir=recording_dir.resolve(),
        video_path=video_path.resolve(),
        status=mod.STATUS_OK,
    )

    monkeypatch.setattr(mod, "_resolve_input_paths", lambda *_args, **_kwargs: [tmp_path])
    monkeypatch.setattr(mod, "_discover_analysis_zarrs", lambda *_args, **_kwargs: [zarr_path.resolve()])
    monkeypatch.setattr(mod, "_build_plans", lambda *_args, **_kwargs: [plan])

    def _unexpected_resolve(**_kwargs):
        raise AssertionError("_resolve_registry_models_for_plans should not be called with --model")

    monkeypatch.setattr(mod, "_resolve_registry_models_for_plans", _unexpected_resolve)

    missing_registry = tmp_path / "missing_registry.sqlite"
    rc = mod.main([
        "--dry-run", "--json", "--no-log",
        "--registry", str(missing_registry),
        "--model", str(model_path),
        str(tmp_path),
    ])

    assert rc == 0
    out = capsys.readouterr().out
    assert "registry_missing" not in out
    assert '"status": "ok"' in out


def test_main_apply_with_explicit_model_skips_registry_resolution(monkeypatch, tmp_path: Path) -> None:
    zarr_path = tmp_path / "rec_a" / "zarr" / "rec_a_analysis.zarr"
    zarr_path.mkdir(parents=True, exist_ok=True)
    recording_dir = zarr_path.parent.parent
    video_path = recording_dir / "cams" / "cam_1.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"")
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"model")

    plan = mod.DetectPlan(
        zarr_path=zarr_path.resolve(),
        recording_dir=recording_dir.resolve(),
        video_path=video_path.resolve(),
        status=mod.STATUS_OK,
    )

    monkeypatch.setattr(mod, "_resolve_input_paths", lambda *_args, **_kwargs: [tmp_path])
    monkeypatch.setattr(mod, "_discover_analysis_zarrs", lambda *_args, **_kwargs: [zarr_path.resolve()])
    monkeypatch.setattr(mod, "_build_plans", lambda *_args, **_kwargs: [plan])

    def _unexpected_resolve(**_kwargs):
        raise AssertionError("_resolve_registry_models_for_plans should not be called with --model")

    monkeypatch.setattr(mod, "_resolve_registry_models_for_plans", _unexpected_resolve)

    calls: list[dict[str, object]] = []

    def _fake_run_detect_plan(**kwargs):  # noqa: ANN003
        calls.append(kwargs)
        resolved_model = kwargs["resolved_model"]
        return mod.DetectRegistryResult(
            ok=True,
            status="ok",
            recording_dir=str(recording_dir.resolve()),
            output_zarr=str(zarr_path.resolve()),
            registry_path=str((tmp_path / "missing_registry.sqlite").resolve()),
            video_path=str(video_path.resolve()),
            selected_model_path=resolved_model.model_path,
            detect_run="detect_explicit",
            resolved_at_utc=None,
            resolution_payload=resolved_model.payload,
        )

    monkeypatch.setattr(mod, "_run_detect_plan", _fake_run_detect_plan)

    rc = mod.main([
        "--apply", "--json", "--no-log",
        "--registry", str(tmp_path / "missing_registry.sqlite"),
        "--model", str(model_path),
        "--decode-backend", "pynvvc_luma_rgb",
        str(tmp_path),
    ])

    assert rc == 0
    assert len(calls) == 1
    resolved_model = calls[0]["resolved_model"]
    assert resolved_model.model_path == str(model_path.resolve())
    assert resolved_model.payload["mode"] == "explicit"
    assert resolved_model.payload["selected"]["model_path"] == str(model_path.resolve())
    assert calls[0]["decode_backend"] == "pynvvc_luma_rgb"
    assert resolved_model.payload["parameters"]["decode_backend"] == "pynvvc_luma_rgb"


def test_run_detect_plan_skips_registry_provenance_for_explicit_model(monkeypatch, tmp_path: Path) -> None:
    zarr_path = tmp_path / "rec_a" / "zarr" / "rec_a_analysis.zarr"
    recording_dir = zarr_path.parent.parent
    video_path = recording_dir / "cams" / "cam_1.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"")
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"model")

    plan = mod.DetectPlan(
        zarr_path=zarr_path.resolve(),
        recording_dir=recording_dir.resolve(),
        video_path=video_path.resolve(),
        status=mod.STATUS_OK,
    )

    _patch_detect_yolo(monkeypatch, lambda **_kwargs: "detect_explicit")

    def _unexpected_provenance(**_kwargs):
        raise AssertionError("registry model-resolution provenance should be skipped for explicit models")

    monkeypatch.setattr(mod, "write_detect_model_resolution_provenance", _unexpected_provenance)

    result = mod._run_detect_plan(  # noqa: SLF001
        plan=plan,
        resolved_model=mod.ResolvedModel(
            model_path=str(model_path.resolve()),
            payload={
                "mode": "explicit",
                "selected": {"model_path": str(model_path.resolve()), "run_id": None, "set_id": None},
            },
        ),
        write_raw_video_metadata=False,
        overwrite_raw_video_metadata=False,
        config=None,
        conf=None,
        iou=None,
        max_det=None,
        batch_size=None,
        resize_dims=None,
        imgsz=None,
        decode_backend=None,
        detect_row_shard_rows=262_144,
        detect_frame_shard_rows=262_144,
        cpu=False,
        registry_path=tmp_path / "missing_registry.sqlite",
    )

    assert result.ok is True
    assert result.detect_run == "detect_explicit"


# ---------------------------------------------------------------------------
# Registry discovery tests
# ---------------------------------------------------------------------------


def _make_fake_row(zarr_path: str, status=None):
    """Build a dict-like object that mimics sqlite3.Row for query_datasets output."""

    class _FakeRow(dict):
        def __getitem__(self, key):
            return super().__getitem__(key)

    return _FakeRow(zarr_path=zarr_path, status=status)


def test_discover_from_registry_returns_sorted_paths(monkeypatch, tmp_path: Path) -> None:
    """Registry discovery returns sorted, deduplicated Path list."""
    registry_path = tmp_path / "registry.sqlite"
    registry_path.write_text("", encoding="utf-8")

    rows = [
        _make_fake_row("/data/rec_c/zarr/rec_c_analysis.zarr"),
        _make_fake_row("/data/rec_a/zarr/rec_a_analysis.zarr"),
        _make_fake_row("/data/rec_b/zarr/rec_b_analysis.zarr"),
        _make_fake_row("/data/rec_a/zarr/rec_a_analysis.zarr"),  # duplicate
    ]

    class _FakeRegistry:
        def __init__(self, _path):
            pass

        def query_datasets(self, **_kwargs):
            return rows

        def close(self):
            pass

    monkeypatch.setattr("fisheye.utils.run_detections_batch.Registry", _FakeRegistry, raising=False)
    # Also patch the import inside the function
    import fisheye.utils.run_detections_batch as _mod

    original = _mod._discover_analysis_zarrs_from_registry.__code__
    # Use monkeypatch on the module-level lazy import
    monkeypatch.setattr("fisheye.registry.db.Registry", _FakeRegistry)

    result = mod._discover_analysis_zarrs_from_registry(
        registry_path=registry_path,
        scope_paths=[],
    )

    result_str = [str(p) for p in result]
    assert result_str == sorted(result_str)
    assert len(result) == 3  # duplicate removed


def test_discover_from_registry_scope_filter(monkeypatch, tmp_path: Path) -> None:
    """When scope_paths given, only registry zarrs under those paths are returned."""
    registry_path = tmp_path / "registry.sqlite"
    registry_path.write_text("", encoding="utf-8")

    rows = [
        _make_fake_row("/data/project_a/rec_1/zarr/rec_1_analysis.zarr"),
        _make_fake_row("/data/project_b/rec_2/zarr/rec_2_analysis.zarr"),
        _make_fake_row("/data/project_a/rec_3/zarr/rec_3_analysis.zarr"),
    ]

    class _FakeRegistry:
        def __init__(self, _path):
            pass

        def query_datasets(self, **_kwargs):
            return rows

        def close(self):
            pass

    monkeypatch.setattr("fisheye.registry.db.Registry", _FakeRegistry)

    result = mod._discover_analysis_zarrs_from_registry(
        registry_path=registry_path,
        scope_paths=[Path("/data/project_a")],
    )

    result_str = [str(p) for p in result]
    assert len(result) == 2
    assert all("project_a" in s for s in result_str)
    assert not any("project_b" in s for s in result_str)


def test_discover_from_registry_excludes_missing(monkeypatch, tmp_path: Path) -> None:
    """Rows with status='missing' are excluded by the exclude_status param."""
    registry_path = tmp_path / "registry.sqlite"
    registry_path.write_text("", encoding="utf-8")

    captured_kwargs: list[dict] = []

    class _FakeRegistry:
        def __init__(self, _path):
            pass

        def query_datasets(self, **kwargs):
            captured_kwargs.append(kwargs)
            return [_make_fake_row("/data/rec_a/zarr/rec_a_analysis.zarr")]

        def close(self):
            pass

    monkeypatch.setattr("fisheye.registry.db.Registry", _FakeRegistry)

    mod._discover_analysis_zarrs_from_registry(
        registry_path=registry_path,
        scope_paths=[],
    )

    assert len(captured_kwargs) == 1
    assert captured_kwargs[0]["exclude_status"] == "missing"
    assert captured_kwargs[0]["zarr_use"] == "analysis"
    assert captured_kwargs[0]["require_recording"] is True


def test_main_source_registry_dry_run(monkeypatch, tmp_path: Path, capsys) -> None:
    """--source registry uses registry discovery instead of filesystem."""
    registry_path = tmp_path / "registry.sqlite"
    registry_path.write_text("", encoding="utf-8")

    zarr_path = tmp_path / "rec_a" / "zarr" / "rec_a_analysis.zarr"
    zarr_path.mkdir(parents=True, exist_ok=True)
    recording_dir = zarr_path.parent.parent
    video_path = recording_dir / "cams" / "cam_1.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"")

    _write_root_metadata(zarr_path)
    _write_group_attrs(zarr_path, "detect_runs", {"latest": None})
    _write_group_attrs(zarr_path, "background_runs", {"latest": "bg_1"})
    _write_group_attrs(zarr_path, "analysis_metadata", {"detection_tuning": {"enabled": True}})

    monkeypatch.setattr(
        mod,
        "_discover_analysis_zarrs_from_registry",
        lambda **_kw: [zarr_path.resolve()],
    )

    # Filesystem discovery should NOT be called.
    def _unexpected_fs(*_a, **_kw):
        raise AssertionError("_discover_analysis_zarrs should not be called with --source registry")

    monkeypatch.setattr(mod, "_discover_analysis_zarrs", _unexpected_fs)

    rc = mod.main([
        "--source", "registry", "--dry-run", "--json", "--no-log",
        "--registry", str(registry_path), str(tmp_path),
    ])

    assert rc == 0
    out = capsys.readouterr().out
    assert "rec_a_analysis.zarr" in out


def test_main_source_registry_missing_registry_fails(tmp_path: Path, capsys) -> None:
    """--source registry with a nonexistent registry returns exit code 1."""
    missing_registry = tmp_path / "nonexistent.sqlite"

    rc = mod.main([
        "--source", "registry", "--dry-run", "--no-log",
        "--registry", str(missing_registry),
    ])

    assert rc == 1
    err = capsys.readouterr().err
    assert "Registry not found" in err


def test_discover_from_registry_skip_existing_passes_exclude_step_ok(
    monkeypatch, tmp_path: Path
) -> None:
    """When skip_existing=True, exclude_step_ok='detect' is passed to query_datasets."""
    registry_path = tmp_path / "registry.sqlite"
    registry_path.write_text("", encoding="utf-8")

    captured_kwargs: list[dict] = []

    class _FakeRegistry:
        def __init__(self, _path):
            pass

        def query_datasets(self, **kwargs):
            captured_kwargs.append(kwargs)
            return []

        def close(self):
            pass

    monkeypatch.setattr("fisheye.registry.db.Registry", _FakeRegistry)

    mod._discover_analysis_zarrs_from_registry(
        registry_path=registry_path,
        scope_paths=[],
        skip_existing=True,
    )

    assert len(captured_kwargs) == 1
    assert captured_kwargs[0]["exclude_step_ok"] == "detect"


def test_discover_from_registry_no_skip_existing_omits_exclude_step_ok(
    monkeypatch, tmp_path: Path
) -> None:
    """When skip_existing=False (default), exclude_step_ok is not passed."""
    registry_path = tmp_path / "registry.sqlite"
    registry_path.write_text("", encoding="utf-8")

    captured_kwargs: list[dict] = []

    class _FakeRegistry:
        def __init__(self, _path):
            pass

        def query_datasets(self, **kwargs):
            captured_kwargs.append(kwargs)
            return []

        def close(self):
            pass

    monkeypatch.setattr("fisheye.registry.db.Registry", _FakeRegistry)

    mod._discover_analysis_zarrs_from_registry(
        registry_path=registry_path,
        scope_paths=[],
        skip_existing=False,
    )

    assert len(captured_kwargs) == 1
    assert "exclude_step_ok" not in captured_kwargs[0]


def test_main_emit_paths(monkeypatch, tmp_path: Path, capsys) -> None:
    """--emit-paths prints paths to stdout and exits 0."""
    registry_path = tmp_path / "registry.sqlite"
    registry_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        mod,
        "_discover_analysis_zarrs_from_registry",
        lambda **_kw: [Path("/data/rec_a_analysis.zarr"), Path("/data/rec_b_analysis.zarr")],
    )

    rc = mod.main([
        "--source", "registry", "--emit-paths", "--no-log",
        "--registry", str(registry_path),
    ])

    assert rc == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 2
    assert "/data/rec_a_analysis.zarr" in out[0]
    assert "/data/rec_b_analysis.zarr" in out[1]

from __future__ import annotations

import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import zarr

import fisheye.registry.recording_identity_authority as authority_module
from fisheye.registry.db import Registry
from fisheye.registry.recording_identity_authority import (
    RecordingIdentityAuthorityError,
    RecordingIdentityProjectionConflict,
    collect_regular_source_recording_identity,
)
from fisheye.shared.recording_import_receipt import (
    RecordingImportReceipt,
    publish_recording_import_receipt,
)
from fisheye.shared import run_provenance
from fisheye.shared.source_recording_identity import (
    SOURCE_RECORDING_IDENTITY_PROFILE,
    SOURCE_RECORDING_IDENTITY_PROFILE_ATTR,
)
from fisheye.utils import import_recording_analysis as importer


def _publish_current_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[importer.RecordingAnalysisPlan, RecordingImportReceipt]:
    recording_dir = tmp_path / "recordings" / "recording-a"
    video = recording_dir / "cams" / "Cam2010093.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"bounded-video-fixture")
    (recording_dir / "recording_manifest.json").write_text(
        json.dumps(
            {
                SOURCE_RECORDING_IDENTITY_PROFILE_ATTR: (
                    SOURCE_RECORDING_IDENTITY_PROFILE
                ),
                "recording_id": "recording-a",
                "session_uuid": "session-a",
                "camera_id": "2010093",
                "recording_name": "recording-a",
                "recording_type": "behavior",
                "recording_subtype": "free",
                "behavior_mode": "free",
                "artifact_schema_id": "behavior_v1",
            }
        ),
        encoding="utf-8",
    )
    plan = importer.RecordingAnalysisPlan(
        recording_dir=recording_dir,
        h5_path=None,
        cam_video=video,
        zarr_path=recording_dir / "zarr" / "recording-a_analysis.zarr",
    )
    options = importer.RecordingImportOptions(
        import_video_metadata=True,
        video_metadata_overwrite=False,
        import_stimulus=False,
        stimulus_always=False,
        stimulus_run_name=None,
        stimulus_overwrite=False,
        stimulus_quiet=True,
    )

    def probe(_path: Path, **_kwargs: object) -> dict[str, object]:
        return {
            "source_video": video.name,
            "source_path": str(video),
            "width": 4512,
            "height": 4512,
            "total_frames": 100,
            "fps": 100.0,
            "duration_seconds": 1.0,
            "codec": "hevc",
            "pix_fmt": "yuv420p",
        }

    monkeypatch.setattr(importer, "probe_video_metadata", probe)
    # Exercise the real clock producer as well as authority/receipt publication.
    video.with_name(f"{video.stem}_meta.csv").write_text(
        "recording_frame_id,timestamp,timestamp_sys\n"
        + "".join(f"{i + 1},{1_000_000_000 + i * 10_000_000},{2_000_000_000 + i * 10_000_000}\n" for i in range(100))
    )
    monkeypatch.setattr(
        importer,
        "git_identity",
        lambda **_kwargs: {"git_sha": "1" * 40, "git_dirty": False},
    )
    monkeypatch.setattr(
        run_provenance,
        "git_identity",
        lambda **_kwargs: {"git_sha": "1" * 40, "git_dirty": False},
    )
    result = importer.process_recording_import(plan, options)
    assert result.ok is True, result
    assert result.receipt is not None
    return plan, result.receipt


@pytest.mark.parametrize("damage", [
    "none", "missing_receipt", "missing_authority", "stale_metadata",
    "missing_clock", "tampered_clock", "changed_clock_source",
    "missing_manifest_context",
    "conflicting_root_context",
])
def test_batch_skip_requires_live_receipt_and_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str,
) -> None:
    from fisheye.utils.import_organized_recordings_analysis import _existing_analysis_complete

    plan, receipt = _publish_current_import(tmp_path, monkeypatch)
    if damage == "missing_receipt":
        (plan.zarr_path / ".imports" / f"{receipt.receipt_sha256}.json").unlink()
    elif damage == "missing_authority":
        root = zarr.open_group(str(plan.zarr_path), mode="r+", use_consolidated=False)
        del root["analysis/acquisition_camera_frames/2010093"]
    elif damage == "stale_metadata":
        root = zarr.open_group(str(plan.zarr_path), mode="r+", use_consolidated=False)
        root["raw_video"].attrs["stale_generation_test"] = True
    elif damage == "missing_clock":
        root = zarr.open_group(str(plan.zarr_path), mode="r+", use_consolidated=False)
        del root["analysis/acquisition_frame_clock_runs"]
    elif damage == "tampered_clock":
        root = zarr.open_group(str(plan.zarr_path), mode="r+", use_consolidated=False)
        clock = root[str(root.attrs["acquisition_frame_clock_ref"])]
        clock["camera_timestamp_ns"][1] = 123
    elif damage == "changed_clock_source":
        source = plan.cam_video.with_name(f"{plan.cam_video.stem}_meta.csv")
        source.write_text(source.read_text().replace("1010000000", "1010000001"))
    elif damage == "missing_manifest_context":
        manifest_path = plan.recording_dir / "recording_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        del manifest["recording_type"]
        manifest_path.write_text(json.dumps(manifest))
    elif damage == "conflicting_root_context":
        root = zarr.open_group(str(plan.zarr_path), mode="r+", use_consolidated=False)
        root.attrs["recording_type"] = "microscopy"
        from fisheye.shared.zarr_helpers import consolidate_metadata_capture_expected_warnings
        consolidate_metadata_capture_expected_warnings(str(plan.zarr_path))
    complete, reason = _existing_analysis_complete(plan.zarr_path, require_stimulus=False)
    assert complete is (damage == "none"), reason
    if damage in {"missing_manifest_context", "conflicting_root_context"}:
        assert "recording_type" in reason


def _authority_row_counts(registry: Registry) -> dict[str, int]:
    tables = (
        "recordings",
        "datasets",
        "recording_identity_evidence",
        "recording_identity_revisions",
        "recording_identity_current",
        "dataset_recording_identity_current",
        "recording_import_receipt_bindings",
    )
    return {
        table: int(
            registry.conn.execute(f"SELECT COUNT(*) FROM {table};").fetchone()[0]
        )
        for table in tables
    }


@pytest.mark.parametrize("damage", ["failed_optional_diagnostic", "missing_clock"])
def test_sealed_pipeline_replay_refuses_failed_intake_before_registry_or_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str,
) -> None:
    from fisheye.utils import run_recording_analysis_pipeline as pipeline

    plan, _receipt = _publish_current_import(tmp_path, monkeypatch)
    if damage == "failed_optional_diagnostic":
        path = plan.recording_dir / "recording_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["preflight"] = {"status": "pass", "h5": {"optional_status": "fail"}}
        path.write_text(json.dumps(manifest))
    else:
        root = zarr.open_group(str(plan.zarr_path), mode="r+", use_consolidated=False)
        del root["analysis/acquisition_frame_clock_runs"]
    opts = SimpleNamespace(refine_keypoints=False, register=True, registry_path=tmp_path / "registry.sqlite")
    monkeypatch.setattr(pipeline, "_sync_pipeline_registry", lambda **_: pytest.fail("invalid replay reached registry"))
    monkeypatch.setattr(pipeline, "run_detect_registry_model", lambda *_: pytest.fail("invalid replay reached detection"))
    result = pipeline.process_recording_analysis_pipeline(plan, opts)
    assert result.ok is False
    assert result.failed_step == "recording_import_preflight"


def test_valid_sealed_pipeline_replay_verifies_before_detection_without_import_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fisheye.utils import run_recording_analysis_pipeline as pipeline

    plan, receipt = _publish_current_import(tmp_path, monkeypatch)
    registry_path = tmp_path / "registry.sqlite"
    Registry(registry_path).close()
    opts = SimpleNamespace(refine_keypoints=False, register=True, registry_path=registry_path)
    monkeypatch.setattr(pipeline, "process_recording_import", lambda *_args, **_kwargs: pytest.fail("sealed replay invoked an import writer"))

    def detect(*_):
        registry = Registry(registry_path)
        try:
            rows = registry.conn.execute("SELECT receipt_sha256 FROM recording_import_receipt_bindings").fetchall()
            assert [row[0] for row in rows] == [receipt.receipt_sha256]
        finally:
            registry.close()
        # Stop outside the ingestion scope without running inference.
        return False, 7, ["fixture-detect"]

    monkeypatch.setattr(pipeline, "run_detect_registry_model", detect)
    result = pipeline.process_recording_analysis_pipeline(plan, opts)
    assert result.failed_step == "detect_yolo", result
    assert result.returncode == 7


@pytest.mark.parametrize("damage", ["none", "no_log", "no_acknowledgment", "wrong_output", "missing_receipt"])
def test_citrus_acknowledgment_requires_real_live_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str,
) -> None:
    from fisheye.utils import run_citrus_session_import as citrus

    plan, receipt = _publish_current_import(tmp_path, monkeypatch)
    log = tmp_path / "import.jsonl"
    output = tmp_path / "wrong.zarr" if damage == "wrong_output" else plan.zarr_path
    event = "recording_plan" if damage == "no_acknowledgment" else "recording_ok"
    log.write_text(json.dumps({"event": event, "zarr_path": str(output)}))
    if damage == "missing_receipt":
        (plan.zarr_path / ".imports" / f"{receipt.receipt_sha256}.json").unlink()
    paths = citrus._read_zarr_paths_from_import_log(log)
    kwargs = dict(import_log=None if damage == "no_log" else log,
                  recording_dirs=[plan.recording_dir], zarr_paths=paths, recording_only=True)
    if damage == "none":
        citrus._verify_import_acknowledgments(**kwargs)
    else:
        with pytest.raises(ValueError):
            citrus._verify_import_acknowledgments(**kwargs)


def test_real_import_authority_receipt_binds_and_reopens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, receipt = _publish_current_import(tmp_path, monkeypatch)
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        finalized = registry.finalize_current_source_import(
            zarr_path=plan.zarr_path,
            decided_by="pytest",
            receipt=receipt,
        )
        dataset_id = finalized.identity.dataset_id
        row = registry.conn.execute(
            """
            SELECT artifact_kind, zarr_origin, zarr_use, status
            FROM datasets WHERE dataset_id = ?;
            """,
            (dataset_id,),
        ).fetchone()
        assert tuple(row) == ("source_recording", "source", "analysis", "active")
        registry.close()

        registry = Registry(tmp_path / "registry.sqlite")
        verified = registry.read_verified_recording_import(dataset_id)
        assert verified.identity.recording_id == "recording-a"
        assert verified.identity.session_uuid == "session-a"
        assert verified.receipt.receipt_sha256 == receipt.receipt_sha256
        assert verified.acquisition_frame.record.camera_id == "2010093"
        assert verified.acquisition_frame.record.recording_id == "recording-a"
    finally:
        registry.close()


def test_stale_acquisition_receipt_cannot_mint_registry_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, receipt = _publish_current_import(tmp_path, monkeypatch)
    stale = RecordingImportReceipt.create(
        producer_id=receipt.producer_id,
        producer_git_sha=receipt.producer_git_sha,
        producer_git_dirty=receipt.producer_git_dirty,
        config_sha256=receipt.config_sha256,
        target_relative_path=receipt.target_relative_path,
        identity_claim=receipt.identity_claim,
        acquisition_ownership_ref=receipt.acquisition_ownership_ref,
        acquisition_ownership_sha256=receipt.acquisition_ownership_sha256,
        acquisition_frame_ref=receipt.acquisition_frame_ref,
        acquisition_frame_sha256="f" * 64,
    )
    publish_recording_import_receipt(plan.zarr_path, stale)
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        with pytest.raises(
            RecordingIdentityProjectionConflict,
            match="acquisition references differ",
        ):
            registry.project_regular_source_recording_identity(
                zarr_path=plan.zarr_path,
                decided_by="pytest",
                import_receipt=stale,
            )
        for table in (
            "recording_identity_revisions",
            "recording_identity_current",
            "dataset_recording_identity_current",
            "recording_import_receipt_bindings",
        ):
            assert registry.conn.execute(
                f"SELECT COUNT(*) FROM {table};"
            ).fetchone()[0] == 0
    finally:
        registry.close()


@pytest.mark.parametrize(
    "entrypoint",
    ("register_from_root", "scan_zarr", "reconcile_dataset_from_root"),
)
def test_profiled_current_source_never_enters_generic_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
) -> None:
    plan, _receipt = _publish_current_import(tmp_path, monkeypatch)
    registry = Registry(tmp_path / "registry.sqlite")
    root = zarr.open_group(
        str(plan.zarr_path),
        mode="r",
        zarr_format=3,
        use_consolidated=False,
    )

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("legacy identity upsert was reached")

    monkeypatch.setattr(registry, "upsert_dataset", forbidden)
    monkeypatch.setattr(registry, "upsert_recording", forbidden)
    try:
        with pytest.raises(
            RecordingIdentityAuthorityError,
            match="receipt-bound",
        ):
            if entrypoint == "scan_zarr":
                registry.scan_zarr(plan.zarr_path)
            elif entrypoint == "register_from_root":
                registry.register_from_root(root, plan.zarr_path)
            else:
                registry.reconcile_dataset_from_root(root, plan.zarr_path)
        assert set(_authority_row_counts(registry).values()) == {0}
    finally:
        registry.close()


def test_bound_current_source_reconcile_uses_verified_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, receipt = _publish_current_import(tmp_path, monkeypatch)
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        finalized = registry.finalize_current_source_import(
            zarr_path=plan.zarr_path,
            receipt=receipt,
            decided_by="pytest",
        )
        before = _authority_row_counts(registry)
        root = zarr.open_group(
            str(plan.zarr_path),
            mode="r",
            zarr_format=3,
            use_consolidated=False,
        )

        def forbidden(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("legacy identity upsert was reached")

        monkeypatch.setattr(registry, "upsert_dataset", forbidden)
        monkeypatch.setattr(registry, "upsert_recording", forbidden)
        result = registry.reconcile_dataset_from_root(root, plan.zarr_path)

        assert result["dataset_id"] == finalized.identity.dataset_id
        assert _authority_row_counts(registry) == before
    finally:
        registry.close()


def test_bound_current_source_scan_refreshes_step_status_without_legacy_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fisheye.registry import maintenance

    plan, receipt = _publish_current_import(tmp_path, monkeypatch)
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        finalized = registry.finalize_current_source_import(
            zarr_path=plan.zarr_path,
            receipt=receipt,
            decided_by="pytest",
        )
        calls: list[dict[str, object]] = []

        def reconcile_step_status(
            _registry: Registry,
            **kwargs: object,
        ) -> dict[str, object]:
            calls.append(kwargs)
            return {"status": "ok"}

        def forbidden(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("legacy identity upsert was reached")

        monkeypatch.setattr(
            maintenance,
            "reconcile_recording_step_status_for_dataset",
            reconcile_step_status,
        )
        monkeypatch.setattr(registry, "upsert_dataset", forbidden)
        monkeypatch.setattr(registry, "upsert_recording", forbidden)

        dataset_id = registry.scan_zarr(
            plan.zarr_path,
            include_step_status=True,
        )

        assert dataset_id == finalized.identity.dataset_id
        assert len(calls) == 1
        assert calls[0]["dataset_id"] == finalized.identity.dataset_id
        assert calls[0]["zarr_path"] == plan.zarr_path
        assert calls[0]["recording_id"] == "recording-a"
        assert calls[0]["zarr_use"] == "analysis"
    finally:
        registry.close()


def test_unprofiled_v3_root_keeps_generic_registration(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "recordings" / "legacy" / "zarr" / "analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w", zarr_format=3)
    root.attrs.update(
        {
            "recording_id": "legacy-recording",
            "session_uuid": "legacy-session",
            "zarr_purpose": "analysis",
        }
    )
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        dataset_id = registry.register_from_root(root, zarr_path)
        row = registry.conn.execute(
            "SELECT recording_id, session_uuid FROM datasets WHERE dataset_id = ?;",
            (dataset_id,),
        ).fetchone()
        assert tuple(row) == ("legacy-recording", "legacy-session")
    finally:
        registry.close()


def test_current_source_finalizer_rolls_back_downstream_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, receipt = _publish_current_import(tmp_path, monkeypatch)
    registry = Registry(tmp_path / "registry.sqlite")

    def fail_after_identity(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected non-identity projection failure")

    monkeypatch.setattr(registry, "_project_nonidentity_from_root", fail_after_identity)
    try:
        with pytest.raises(RuntimeError, match="injected non-identity"):
            registry.finalize_current_source_import(
                zarr_path=plan.zarr_path,
                receipt=receipt,
                decided_by="pytest",
            )
        assert set(_authority_row_counts(registry).values()) == {0}
    finally:
        registry.close()


def test_current_source_finalizer_requires_returned_receipt_before_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, _receipt = _publish_current_import(tmp_path, monkeypatch)
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        with pytest.raises(
            RecordingIdentityAuthorityError,
            match="exact in-memory import receipt",
        ):
            registry.finalize_current_source_import(
                zarr_path=plan.zarr_path,
                receipt=None,
                decided_by="pytest",
            )
        assert set(_authority_row_counts(registry).values()) == {0}
    finally:
        registry.close()


def test_synchronize_current_source_with_receipt_dispatches_to_finalizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The central boundary must use the receipt-bearing finalizer exactly once."""

    zarr_path = tmp_path / "recording" / "analysis.zarr"
    receipt = object()
    verified = SimpleNamespace(identity=SimpleNamespace(dataset_id="current-id"))
    calls: list[tuple[str, dict[str, object]]] = []
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        monkeypatch.setattr(
            authority_module,
            "load_source_recording_identity_profile",
            lambda _path: SOURCE_RECORDING_IDENTITY_PROFILE,
        )

        def finalize(**kwargs: object) -> object:
            calls.append(("finalize", kwargs))
            return verified

        def forbidden_refresh(**_kwargs: object) -> object:
            raise AssertionError("current receipt dispatch used refresh")

        def forbidden_scan(**_kwargs: object) -> object:
            raise AssertionError("current receipt dispatch used legacy scan")

        monkeypatch.setattr(registry, "finalize_current_source_import", finalize)
        monkeypatch.setattr(
            registry,
            "refresh_bound_current_source_import",
            forbidden_refresh,
        )
        monkeypatch.setattr(registry, "scan_zarr", forbidden_scan)

        assert (
            registry.synchronize_recording_import(
                zarr_path=zarr_path,
                receipt=receipt,
                decided_by="pytest",
            )
            == "current-id"
        )
        assert calls == [
            (
                "finalize",
                {
                    "zarr_path": zarr_path,
                    "receipt": receipt,
                    "decided_by": "pytest",
                },
            )
        ]
    finally:
        registry.close()


def test_synchronize_current_source_without_receipt_dispatches_to_bound_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bound current source may refresh metadata without minting a receipt."""

    zarr_path = tmp_path / "recording" / "analysis.zarr"
    verified = SimpleNamespace(identity=SimpleNamespace(dataset_id="bound-id"))
    calls: list[dict[str, object]] = []
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        monkeypatch.setattr(
            authority_module,
            "load_source_recording_identity_profile",
            lambda _path: SOURCE_RECORDING_IDENTITY_PROFILE,
        )

        def refresh(**kwargs: object) -> object:
            calls.append(kwargs)
            return verified

        def forbidden_finalize(**_kwargs: object) -> object:
            raise AssertionError("receipt-less dispatch used finalizer")

        def forbidden_scan(**_kwargs: object) -> object:
            raise AssertionError("current receipt-less dispatch used legacy scan")

        monkeypatch.setattr(registry, "refresh_bound_current_source_import", refresh)
        monkeypatch.setattr(
            registry,
            "finalize_current_source_import",
            forbidden_finalize,
        )
        monkeypatch.setattr(registry, "scan_zarr", forbidden_scan)

        assert (
            registry.synchronize_recording_import(
                zarr_path=zarr_path,
                receipt=None,
                decided_by="pytest",
            )
            == "bound-id"
        )
        assert calls == [{"zarr_path": zarr_path}]
    finally:
        registry.close()


def test_synchronize_unprofiled_with_receipt_rejects_before_legacy_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A receipt cannot be attached to an artifact outside the current profile."""

    zarr_path = tmp_path / "legacy" / "analysis.zarr"
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        monkeypatch.setattr(
            authority_module,
            "load_source_recording_identity_profile",
            lambda _path: None,
        )

        def forbidden_scan(**_kwargs: object) -> object:
            raise AssertionError("legacy scan ran before receipt rejection")

        monkeypatch.setattr(registry, "scan_zarr", forbidden_scan)
        with pytest.raises(
            RecordingIdentityAuthorityError,
            match="receipt cannot be applied to an unprofiled artifact",
        ):
            registry.synchronize_recording_import(
                zarr_path=zarr_path,
                receipt=object(),
                decided_by="pytest",
            )
    finally:
        registry.close()


def test_synchronize_unprofiled_without_receipt_delegates_to_legacy_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Historical/unprofiled artifacts retain the explicit scan compatibility path."""

    zarr_path = tmp_path / "legacy" / "analysis.zarr"
    calls: list[dict[str, object]] = []
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        monkeypatch.setattr(
            authority_module,
            "load_source_recording_identity_profile",
            lambda _path: None,
        )

        def scan(path: Path) -> str:
            calls.append({"path": path})
            return "legacy-id"

        monkeypatch.setattr(registry, "scan_zarr", scan)
        assert (
            registry.synchronize_recording_import(
                zarr_path=zarr_path,
                receipt=None,
                decided_by="pytest",
            )
            == "legacy-id"
        )
        assert calls == [{"path": zarr_path}]
    finally:
        registry.close()


def test_bound_refresh_uses_only_registry_selected_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, receipt = _publish_current_import(tmp_path, monkeypatch)
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        finalized = registry.finalize_current_source_import(
            zarr_path=plan.zarr_path,
            receipt=receipt,
            decided_by="pytest",
        )
        decoy = RecordingImportReceipt.create(
            producer_id=receipt.producer_id,
            producer_git_sha=receipt.producer_git_sha,
            producer_git_dirty=receipt.producer_git_dirty,
            config_sha256="f" * 64,
            target_relative_path=receipt.target_relative_path,
            identity_claim=receipt.identity_claim,
            acquisition_ownership_ref=receipt.acquisition_ownership_ref,
            acquisition_ownership_sha256=receipt.acquisition_ownership_sha256,
            acquisition_frame_ref=receipt.acquisition_frame_ref,
            acquisition_frame_sha256=receipt.acquisition_frame_sha256,
        )
        publish_recording_import_receipt(plan.zarr_path, decoy)

        refreshed = registry.refresh_bound_current_source_import(
            zarr_path=plan.zarr_path
        )
        assert refreshed.receipt.receipt_sha256 == receipt.receipt_sha256
        assert registry.scan_zarr(plan.zarr_path) == finalized.identity.dataset_id
        before = _authority_row_counts(registry)
        finalized.receipt_path.unlink()
        with pytest.raises(
            RecordingIdentityAuthorityError,
            match="bound recording import receipt",
        ):
            registry.refresh_bound_current_source_import(zarr_path=plan.zarr_path)
        assert _authority_row_counts(registry) == before
    finally:
        registry.close()


def test_current_source_identity_cannot_bind_a_second_locator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, receipt = _publish_current_import(tmp_path, monkeypatch)
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        registry.finalize_current_source_import(
            zarr_path=plan.zarr_path,
            receipt=receipt,
            decided_by="pytest",
        )
        copied_recording = tmp_path / "recordings" / "recording-a-copy"
        shutil.copytree(plan.recording_dir, copied_recording)
        copied_zarr = copied_recording / plan.zarr_path.relative_to(plan.recording_dir)
        copied_receipt = RecordingImportReceipt.create(
            producer_id=receipt.producer_id,
            producer_git_sha=receipt.producer_git_sha,
            producer_git_dirty=receipt.producer_git_dirty,
            config_sha256=receipt.config_sha256,
            target_relative_path=receipt.target_relative_path,
            identity_claim=collect_regular_source_recording_identity(copied_zarr),
            acquisition_ownership_ref=receipt.acquisition_ownership_ref,
            acquisition_ownership_sha256=receipt.acquisition_ownership_sha256,
            acquisition_frame_ref=receipt.acquisition_frame_ref,
            acquisition_frame_sha256=receipt.acquisition_frame_sha256,
        )
        publish_recording_import_receipt(copied_zarr, copied_receipt)
        before = _authority_row_counts(registry)

        with pytest.raises(
            RecordingIdentityProjectionConflict,
            match="immutable locator|recording_path conflicts",
        ):
            registry.finalize_current_source_import(
                zarr_path=copied_zarr,
                receipt=copied_receipt,
                decided_by="pytest",
            )
        assert _authority_row_counts(registry) == before
    finally:
        registry.close()

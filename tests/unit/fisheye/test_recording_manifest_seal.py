"""B6: sealed transfer-v2 parent manifests are immutable after organize.

Every in-place ``recording_manifest.json`` mutator must refuse a manifest that
carries the organizer's ``source_transfer`` binding, and must keep updating
legacy manifests exactly as before. Each case runs the tool's real entry point.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any, Callable

import h5py
import pytest
import zarr

from fisheye.shared.recording_preflight import build_video_preflight_payload
from fisheye.shared.source_recording_identity import (
    SOURCE_RECORDING_IDENTITY_PROFILE,
    SOURCE_RECORDING_IDENTITY_PROFILE_ATTR,
)
from fisheye.utils import backfill_hevc_keyframe_flags as hevc
from fisheye.utils import backfill_video_only_sidecars as sidecars
from fisheye.utils import intake_video_only_recording as intake
from fisheye.utils import recording_manifest_import_status as import_status
from fisheye.utils import refresh_recording_manifest_metadata as refresh_metadata
from fisheye.utils import refresh_recording_preflight as refresh_preflight
from fisheye.utils import set_recording_subject_metadata as subject_setter

SEALED_MESSAGE = "sealed by transfer-v2 intake"
# The binding the transfer-v2 organizer writes (organize_transfer_recordings._parent_manifest).
SOURCE_TRANSFER = {
    "schema_id": "palette.organized_recording_transfer.v1",
    "snapshot_id": "sha256:" + "0" * 64,
    "transfer_attempt_id": "attempt-1",
    "organization_plan_sha256": "1" * 64,
    "snapshot_path": "raw/acquisition/_citrus_transfer/snapshot.json",
    "marker_path": "raw/acquisition/_citrus_transfer_complete.json",
    "source_to_parent_files": {},
}


@dataclass
class Case:
    recording_dir: Path
    run: Callable[[], str | None]
    """Run the tool; return its refusal/error text (None if it reported none)."""
    untouched: Callable[[], bool] = lambda: True
    """For a refusal: no other side effect happened before the manifest write."""


def _write_manifest(recording_dir: Path, payload: dict[str, Any], sealed: bool) -> Path:
    if sealed:
        payload = {**payload, "source_transfer": SOURCE_TRANSFER}
    recording_dir.mkdir(parents=True, exist_ok=True)
    path = recording_dir / "recording_manifest.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _error_text(call: Callable[[], Any]) -> str | None:
    try:
        call()
    except Exception as exc:  # the refusal type is the subject under test
        return f"{type(exc).__name__}: {exc}"
    return None


# --- one harness per mutator -------------------------------------------------


def _refresh_recording_manifest_metadata(tmp_path, monkeypatch, capsys, sealed) -> Case:
    recording_dir = tmp_path / "2026-06-14T21-12-08Z_arena_1_GoodCopBadCop"
    raw = recording_dir / "raw"
    raw.mkdir(parents=True)
    h5_path = raw / f"{recording_dir.name}.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.attrs["session_uuid"] = "2026-06-14T21-12-08Z_arena_1"
        protocol = h5.create_group("protocol_snapshot")
        protocol.attrs["protocol_name"] = "GoodCopBadCop"
    _write_manifest(
        recording_dir,
        {
            "recording_name": recording_dir.name,
            "files": {"raw": [f"raw/{h5_path.name}"]},
            "protocol_name": None,
        },
        sealed,
    )
    return Case(
        recording_dir,
        lambda: _error_text(lambda: refresh_metadata.main([str(recording_dir), "--apply"])),
    )


def _backfill_hevc_keyframe_flags(tmp_path, monkeypatch, capsys, sealed) -> Case:
    recording_dir = tmp_path / "recording_1"
    (recording_dir / "cams").mkdir(parents=True)
    (recording_dir / "cams" / "cam.mp4").write_bytes(b"cams")
    _write_manifest(
        recording_dir,
        {"recording_name": "recording_1", "files": {"cams": ["cams/cam.mp4"]}},
        sealed,
    )
    monkeypatch.setattr(
        hevc,
        "check_hevc_keyframe_flags",
        lambda path: {"codec": "hevc", "has_stss": True, "needs_fix": False, "message": "ok"},
    )

    def run() -> str | None:
        rc = hevc.main([str(recording_dir), "--apply"])
        out = capsys.readouterr().out
        return None if rc == 0 else out

    return Case(recording_dir, run)


def _backfill_video_only_sidecars(tmp_path, monkeypatch, capsys, sealed) -> Case:
    source_root = tmp_path / "staging"
    source_root.mkdir()
    (source_root / "ptp_sync_summary.json").write_text("{}", encoding="utf-8")
    metadata_csv = tmp_path / "video_only_manifest.csv"
    name = "sleepyfish_2026_05_05_17_45_30_cam2010093"
    with metadata_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_video", "source_camera_metadata_csv",
                SOURCE_RECORDING_IDENTITY_PROFILE_ATTR, "camera_id", "session_uuid",
                "recording_id", "recording_name", "dish_design",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "source_video": "Cam2010093.mp4",
            "source_camera_metadata_csv": "Cam2010093_meta.csv",
            SOURCE_RECORDING_IDENTITY_PROFILE_ATTR: SOURCE_RECORDING_IDENTITY_PROFILE,
            "camera_id": "2010093",
            "session_uuid": name,
            "recording_id": "2026_05_05_17_45_30",
            "recording_name": name,
            "dish_design": "palm",
        })
    recording_dir = tmp_path / "recordings" / name
    _write_manifest(
        recording_dir, {"recording_name": name, "files": {"raw": [], "cams": []}}, sealed
    )
    args = [
        str(source_root), "--metadata-csv", str(metadata_csv),
        "--dest-root", str(tmp_path / "recordings"), "--apply",
    ]
    return Case(
        recording_dir,
        lambda: _error_text(lambda: sidecars.main(args)),
        untouched=lambda: not (recording_dir / "raw").exists(),
    )


def _recording_manifest_import_status(tmp_path, monkeypatch, capsys, sealed) -> Case:
    recording_dir = tmp_path / "rec"
    _write_manifest(recording_dir, {"recording_name": "rec", "import_status": None}, sealed)
    update = import_status.ManifestImportStatusUpdate(
        recording_dir=recording_dir,
        zarr_path=recording_dir / "zarr" / "rec_analysis.zarr",
        status="ok",
        import_log=tmp_path / "import.jsonl",
        imported_at_utc="2026-06-21T23:18:53Z",
        import_run_id="run-1",
    )

    def run() -> str | None:
        result = import_status.write_manifest_import_status(update)
        return result.error

    return Case(recording_dir, run)


def _refresh_recording_preflight(tmp_path, monkeypatch, capsys, sealed) -> Case:
    recording_dir = tmp_path / "rec"
    manifest_path = _write_manifest(recording_dir, {"camera_id": "2010093"}, sealed)
    diagnostics_ran: list[str] = []

    class _VideoResult:
        manifest_payload = build_video_preflight_payload(
            status="pass", media_status="pass", tooling_status="pass",
            videos_scanned=1, finding_codes=[],
        )

    def fake_video(plan, logger):
        diagnostics_ran.append(plan.name)
        return _VideoResult()

    monkeypatch.setattr(refresh_preflight, "_run_video_diagnostics_for_plan", fake_video)

    def run() -> str | None:
        status, error = refresh_preflight.refresh_manifest_preflight(
            manifest_path, run_video=True, run_h5=False, apply=True
        )
        return error if status != "updated" else None

    return Case(recording_dir, run, untouched=lambda: diagnostics_ran == [])


def _set_recording_subject_metadata(tmp_path, monkeypatch, capsys, sealed) -> Case:
    recording_dir = tmp_path / "Cam2010093"
    zarr_path = recording_dir / "zarr" / "Cam2010093_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w", zarr_format=3)
    root.attrs.update({
        "session_uuid": recording_dir.name,
        "recording_id": recording_dir.name,
        "recording_name": recording_dir.name,
        "recording_type": "behavior",
        "experiment_context_status": "absent",
        "zarr_purpose": "analysis",
    })
    _write_manifest(recording_dir, {"recording_name": recording_dir.name}, sealed)

    def run() -> str | None:
        plan = subject_setter.plan_recording(
            recording_dir, species="Danionella cerebrum", dpf=7, subject_count=5
        )
        return _error_text(
            lambda: subject_setter.apply_plan(
                plan, repair_id="test_repair", reason="test", registry=None
            )
        )

    def untouched() -> bool:
        reopened = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
        return "analysis/subject_metadata_runs" not in reopened

    return Case(recording_dir, run, untouched=untouched)


class _FakeGroup:
    def __init__(self) -> None:
        self.attrs: dict[str, Any] = _FakeAttrs()
        self._groups: dict[str, _FakeGroup] = {}

    def require_group(self, name: str) -> "_FakeGroup":
        return self._groups.setdefault(name, _FakeGroup())


class _FakeAttrs(dict):
    def put(self, payload: dict[str, Any]) -> None:
        self.clear()
        self.update(payload)


def _intake_video_only_recording(tmp_path, monkeypatch, capsys, sealed) -> Case:
    recording_dir = tmp_path / "recordings" / "colleague_set_001"
    video_path = recording_dir / "cams" / "Cam2010093.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"fake")
    zarr_path = recording_dir / "zarr" / "colleague_set_001_training.zarr"
    zarr_path.mkdir(parents=True)
    root = _FakeGroup()
    root.attrs["zarr_purpose"] = "training"
    monkeypatch.setattr(intake.zarr, "open_group", lambda *args, **kwargs: root)
    _write_manifest(
        recording_dir,
        {
            "recording_name": "colleague_set_001",
            "recording_type": "behavior",
            "recording_subtype": "free",
            "behavior_mode": "free",
            "artifact_schema_id": "video_only_v1",
        },
        sealed,
    )
    args = [
        str(video_path), "--recording-dir", str(recording_dir),
        "--zarr-path", str(zarr_path), "--metadata-only",
        "--session-uuid", "2026-03-09_colleague_set_001",
        "--recording-id", "recording-camera-2010093",
        "--camera-id", "2010093", "--protocol-name", "ManualProtocol",
        "--write-manifest", "--overwrite-manifest",
    ]

    def run() -> str | None:
        rc = intake.main(args)
        out = capsys.readouterr().out
        return None if rc == 0 else out

    # Refusal happens before the training Zarr is touched.
    return Case(recording_dir, run, untouched=lambda: "recording_id" not in root.attrs)


WRITERS = {
    "refresh_recording_manifest_metadata": _refresh_recording_manifest_metadata,
    "backfill_hevc_keyframe_flags": _backfill_hevc_keyframe_flags,
    "backfill_video_only_sidecars": _backfill_video_only_sidecars,
    "recording_manifest_import_status": _recording_manifest_import_status,
    "refresh_recording_preflight": _refresh_recording_preflight,
    "set_recording_subject_metadata": _set_recording_subject_metadata,
    "intake_video_only_recording": _intake_video_only_recording,
}


@pytest.mark.parametrize("tool", sorted(WRITERS))
def test_mutator_refuses_sealed_transfer_v2_manifest(tool, tmp_path, monkeypatch, capsys):
    case = WRITERS[tool](tmp_path, monkeypatch, capsys, sealed=True)
    manifest_path = case.recording_dir / "recording_manifest.json"
    before = manifest_path.read_bytes()

    error = case.run()

    assert manifest_path.read_bytes() == before
    assert error is not None and SEALED_MESSAGE in error, error
    assert tool in error
    assert case.untouched()


@pytest.mark.parametrize("tool", sorted(WRITERS))
def test_mutator_still_updates_legacy_manifest(tool, tmp_path, monkeypatch, capsys):
    case = WRITERS[tool](tmp_path, monkeypatch, capsys, sealed=False)
    manifest_path = case.recording_dir / "recording_manifest.json"
    before = manifest_path.read_bytes()

    error = case.run()

    assert error is None
    after = json.loads(manifest_path.read_bytes())
    assert manifest_path.read_bytes() != before
    assert "source_transfer" not in after


def test_guard_refuses_unreadable_manifest_and_allows_missing(tmp_path):
    from fisheye.shared.recording_manifest_seal import (
        SealedRecordingManifestError,
        require_unsealed_recording_manifest,
    )

    path = tmp_path / "recording_manifest.json"
    require_unsealed_recording_manifest(path, tool="t")  # missing: nothing to protect
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(SealedRecordingManifestError, match="cannot read"):
        require_unsealed_recording_manifest(path, tool="t")
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(SealedRecordingManifestError, match="not a JSON object"):
        require_unsealed_recording_manifest(path, tool="t")


# --- the organizer's own write-once path and resume -------------------------

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures/recording_transfer_v2"


@pytest.fixture
def _placeholder_media_sync_assessment(monkeypatch):
    from fisheye.diagnostics.video import container

    monkeypatch.setattr(
        container,
        "check_hevc_keyframe_flags",
        lambda path, **_: {
            "schema_id": "palette.video.sync_sample_assessment.v1",
            "codec": "h264",
            "container_inspection_status": "ok",
            "sync_sample_proof": "container_declared",
            "message": "placeholder media",
        },
    )


@pytest.mark.usefixtures("_placeholder_media_sync_assessment")
def test_organizer_writes_sealed_manifest_and_resumes_after_refused_mutations(
    tmp_path, monkeypatch
):
    from fisheye.shared.recording_manifest_seal import (
        SEALED_MANIFEST_MARKER,
        is_sealed_recording_manifest,
    )
    from fisheye.utils import organize_transfer_recordings as organizer

    source = Path(shutil.copytree(FIXTURES / "rolling", tmp_path / "staging"))
    plan = organizer.build_transfer_organization_plan(
        source, destination_root=tmp_path / "recordings"
    )
    organizer.prepare_transfer_parent_recordings(plan, batch_rows=1)

    manifests = {}
    for parent in plan["parents"]:
        path = Path(parent["destination_dir"]) / "recording_manifest.json"
        payload = json.loads(path.read_bytes())
        assert SEALED_MANIFEST_MARKER == "source_transfer"
        assert is_sealed_recording_manifest(payload)
        manifests[path] = path.read_bytes()

    # Real mutators against the organizer's own manifests are refused ...
    for path in manifests:
        result = import_status.write_manifest_import_status(
            import_status.ManifestImportStatusUpdate(
                recording_dir=path.parent,
                zarr_path=path.parent / "zarr" / "x_analysis.zarr",
                status="ok",
                import_log=None,
                imported_at_utc=None,
            )
        )
        assert SEALED_MESSAGE in (result.error or "")
        status, error = refresh_preflight.refresh_manifest_preflight(
            path, run_video=True, run_h5=False, apply=True
        )
        assert status == "failed" and SEALED_MESSAGE in (error or "")
        assert path.read_bytes() == manifests[path]

    # ... so the organizer's byte-exact resume still verifies.
    organizer.prepare_transfer_parent_recordings(plan, batch_rows=1)
    assert {path: path.read_bytes() for path in manifests} == manifests

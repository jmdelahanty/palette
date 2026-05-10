from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.utils import intake_video_only_recording as mod


class FakeAttrs(dict):
    def put(self, payload: dict[str, object]) -> None:
        self.clear()
        self.update(payload)


class FakeGroup:
    def __init__(self) -> None:
        self.attrs = FakeAttrs()
        self._groups: dict[str, "FakeGroup"] = {}

    def require_group(self, name: str) -> "FakeGroup":
        group = self._groups.get(name)
        if group is None:
            group = FakeGroup()
            self._groups[name] = group
        return group

    def __getitem__(self, name: str) -> "FakeGroup":
        return self._groups[name]


def _install_fake_zarr(monkeypatch) -> tuple[dict[str, FakeGroup], object]:
    roots: dict[str, FakeGroup] = {}
    original_open_group = mod.zarr.open_group

    def fake_open_group(path: str, mode: str = "r+") -> FakeGroup:
        key = str(path)
        if key not in roots:
            roots[key] = FakeGroup()
        return roots[key]

    monkeypatch.setattr(mod.zarr, "open_group", fake_open_group)
    return roots, original_open_group


def _make_metadata(**overrides: object) -> mod.VideoOnlyRecordingMetadata:
    payload = {
        "session_uuid": "2026-03-09_colleague_set_001",
        "recording_id": "2026-03-09_colleague_set_001",
        "recording_name": "colleague_set_001",
        "session_start_iso8601_utc": None,
        "recording_type": "behavior",
        "recording_subtype": "free",
        "behavior_mode": "free",
        "artifact_schema_id": "video_only_v1",
        "dish_design": "cedar",
        "rig_id": "omnifin0",
        "arena_id": "arena_1",
        "camera_id": "2010093",
        "canvas_name": "DefaultScreen",
        "protocol_name": "ManualProtocol",
        "genotype": None,
        "dpf_at_acquisition": None,
        "num_dishes": None,
        "fish_per_dish": None,
    }
    payload.update(overrides)
    return mod.VideoOnlyRecordingMetadata(**payload)


def test_apply_manual_metadata_preserves_existing_fields_without_overwrite(tmp_path: Path, monkeypatch) -> None:
    roots, _ = _install_fake_zarr(monkeypatch)
    zarr_path = tmp_path / "recording.zarr"
    zarr_path.mkdir(parents=True)
    root = roots.setdefault(str(zarr_path), FakeGroup())
    root.attrs["dish_design"] = "existing_dish"
    analysis_meta = root.require_group("analysis_metadata")
    analysis_meta.attrs["session_context"] = json.dumps(
        {
            "rig_id": "existing_rig",
            "camera_id": "existing_camera",
        },
        sort_keys=True,
    )

    mod.apply_manual_metadata(
        zarr_path=zarr_path,
        metadata=_make_metadata(),
        overwrite=False,
    )

    root = roots[str(zarr_path)]
    assert root.attrs["dish_design"] == "existing_dish"
    assert root.attrs["session_uuid"] == "2026-03-09_colleague_set_001"
    assert root.attrs["protocol_name"] == "ManualProtocol"
    assert root.attrs["experiment_context_status"] == "absent"
    assert root.attrs["experiment_context_source"] == "none"
    assert root.attrs["stimulus_runs_available"] is False

    analysis_meta = root["analysis_metadata"]
    session_context = json.loads(str(analysis_meta.attrs["session_context"]))
    assert session_context["rig_id"] == "existing_rig"
    assert session_context["camera_id"] == "existing_camera"
    assert session_context["arena_id"] == "arena_1"
    assert session_context["protocol_name"] == "ManualProtocol"
    assert session_context["experiment_context_status"] == "absent"
    assert session_context["experiment_context_source"] == "none"
    assert session_context["stimulus_runs_available"] is False


def test_build_manifest_payload_uses_relative_camera_video_path(tmp_path: Path) -> None:
    recording_dir = tmp_path / "recordings" / "colleague_set_001"
    video_path = recording_dir / "cams" / "Cam2010093.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"fake")

    payload = mod.build_manifest_payload(
        recording_dir=recording_dir,
        video_path=video_path,
        metadata=_make_metadata(),
    )

    assert payload["artifact_schema_id"] == "video_only_v1"
    assert payload["experiment_context_status"] == "absent"
    assert payload["experiment_context_source"] == "none"
    assert payload["stimulus_runs_available"] is False
    assert payload["dish_design"] == "cedar"
    assert payload["protocol_name_from_definition"] == "ManualProtocol"
    assert payload["files"]["cams"] == ["cams/Cam2010093.mp4"]


def test_main_runs_import_and_writes_manifest_and_experiment_setup(
    tmp_path: Path,
    monkeypatch,
) -> None:
    roots, _ = _install_fake_zarr(monkeypatch)
    recording_dir = tmp_path / "recordings" / "colleague_set_001"
    video_path = recording_dir / "cams" / "Cam2010093.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"fake")

    zarr_path = recording_dir / "zarr" / "colleague_set_001_training.zarr"

    calls: list[list[str]] = []

    def fake_run_import(command: list[str]) -> None:
        calls.append(command)
        zarr_path.mkdir(parents=True, exist_ok=True)
        root = roots.setdefault(str(zarr_path), FakeGroup())
        root.attrs["zarr_purpose"] = "training"
        root.require_group("analysis_metadata")

    monkeypatch.setattr(mod, "_run_import_command", fake_run_import)

    rc = mod.main(
        [
            str(video_path),
            "--recording-dir",
            str(recording_dir),
            "--zarr-path",
            str(zarr_path),
            "--frame-step",
            "100",
            "--session-uuid",
            "2026-03-09_colleague_set_001",
            "--dish-design",
            "cedar",
            "--rig-id",
            "omnifin0",
            "--arena-id",
            "arena_1",
            "--camera-id",
            "2010093",
            "--protocol-name",
            "ManualProtocol",
            "--write-manifest",
            "--num-dishes",
            "1",
            "--fish-per-dish",
            "1",
        ]
    )

    assert rc == 0
    assert calls
    assert "--training-data" in calls[0]
    assert "--frame-step" in calls[0]

    root = roots[str(zarr_path)]
    assert root.attrs["dish_design"] == "cedar"
    assert root.attrs["recording_id"] == "2026-03-09_colleague_set_001"
    assert root.attrs["artifact_schema_id"] == "video_only_v1"
    assert root.attrs["experiment_context_status"] == "absent"
    assert root.attrs["experiment_context_source"] == "none"
    assert root.attrs["stimulus_runs_available"] is False
    assert root.attrs["experiment_setup"]["setup_type"] == "single_dish"

    analysis_meta = root["analysis_metadata"]
    session_context = json.loads(str(analysis_meta.attrs["session_context"]))
    assert session_context["rig_id"] == "omnifin0"
    assert session_context["arena_id"] == "arena_1"
    assert session_context["camera_id"] == "2010093"
    assert session_context["protocol_name_from_definition"] == "ManualProtocol"

    manifest_path = recording_dir / "recording_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifact_schema_id"] == "video_only_v1"
    assert manifest["files"]["cams"] == ["cams/Cam2010093.mp4"]

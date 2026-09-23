import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from fisheye.analysis_workflows.execution_profiles import (
    SELECTOR_INELIGIBLE_CANARY_EXECUTION_PROFILE_ID,
)
from fisheye.analysis_workflows.materializers.subject_position import (
    SubjectPositionPreparedInput,
    plan_subject_position_run,
    publish_subject_position_run,
)
from fisheye.analysis_workflows.protocol_semantic_chaser_selection import (
    compile_protocol_semantic_chaser_selections,
    load_protocol_semantic_selection_evidence,
)
from fisheye.analysis_workflows.protocol_semantic_chaser_selection_publication import (
    build_protocol_semantic_chaser_selection_publication_plan,
    publish_protocol_semantic_chaser_selection_run,
)
from fisheye.analysis_workflows.resolved_epoch_selection import (
    resolve_exact_stimulus_epoch_selection,
)
from fisheye.analysis_workflows.runtime_verification import (
    verify_persisted_stage_output,
)
from fisheye.shared.coordinate_descriptor import (
    CanonicalFrameRecord,
    DigestBoundCoordinateRecordRef,
    PIXEL_FRAME_AUTHORITY_RECORD_KIND,
    build_canonical_coordinate_descriptor,
)
from fisheye.shared.coordinate_identity import (
    OBSERVATION_INSTANCE_DOMAIN,
    build_row_identity_contract,
)
from fisheye.shared.coordinate_surface_contract import SOURCE_CAMERA_POINT_XY
from fisheye.shared.pixel_frame_authority import (
    stamp_acquisition_camera_frame,
    stamp_acquisition_import_ownership,
)
from fisheye.shared.selected_calibration import (
    build_selected_camera_source_evidence_from_h5_values,
)
from fisheye.shared.source_camera_physical_authority import (
    publish_source_camera_physical_authority,
)
from fisheye.shared.subject_position_expression import (
    DETECTION_BBOX_CENTROID_ESTIMATOR_ID,
    estimator_profile_digest,
    get_estimator_profile,
)
from fisheye.shared.subject_position_storage import (
    canonical_source_camera_coordinate_metadata,
)
from fisheye.shared.subject_position_types import POSITION_FAILURE_REASON_CODES
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr_helpers import (
    consolidate_metadata_capture_expected_warnings,
)
from fisheye.utils import materialize_provider_behavior_chain as mod
from tests.unit.fisheye.test_provider_swim_bout_binding import (
    _fixture as _provider_swim_bout_fixture,
)
from tests.unit.fisheye.test_body_frame_source_handle import _published_fixture
from tests.unit.fisheye.test_protocol_semantic_chaser_selection import (
    _chaser_bindings,
    _evidence,
    _semantic_selection,
)


def _task_payload(archive: Path, *, schema_version: int) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_id": mod.TASK_SCHEMA_ID,
        "schema_version": schema_version,
        "recording_id": "recording",
        "analysis_zarr": str(archive),
        "source_runs": {
            "keypoint": "keypoints_v1",
            "body_frame": "body_frame_v1",
            "stimulus": "stimulus_v1",
        },
        "output_runs": {
            "position": "position_v1",
            "tracking": "tracking_v1",
            "motion": "motion_v1",
            "swim_bouts": "bouts_v1",
            "stimulus_epochs_v1": "epochs_v1",
            "stimulus_epochs_v2": "epochs_v2",
            "epoch_summary": "summary_v1",
        },
        "fps": 250.0,
        "metric_disposition": mod.LINEAR_ONLY_DISPOSITION,
    }
    if schema_version in {
        mod.EXPLICIT_POSITION_TASK_SCHEMA_VERSION,
        mod.RECEIPTED_TASK_SCHEMA_VERSION,
    }:
        payload["source_runs"] = {
            "position": {
                "run_path": (
                    "analysis/subject_position_runs/observation/position_detection_v1"
                ),
                "manifest_sha256": "a" * 64,
            },
            "body_frame": "body_frame_v1",
            "stimulus": "stimulus_v1",
        }
        payload["output_runs"] = {
            key: value
            for key, value in dict(payload["output_runs"]).items()
            if key != "position"
        }
    return payload


def _write_task(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "task.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_task_v1_preserves_legacy_summary_contract(tmp_path: Path) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    archive.mkdir()
    payload = _task_payload(archive, schema_version=1)

    loaded = mod.load_task(_write_task(tmp_path, payload))

    assert "protocol_semantic_selection_run" not in loaded


def test_task_v1_rejects_semantic_successor_field(tmp_path: Path) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    archive.mkdir()
    payload = _task_payload(archive, schema_version=1)
    payload["protocol_semantic_selection_run"] = "semantic_v2"

    with pytest.raises(mod.ProviderBehaviorChainError, match="task v2"):
        mod.load_task(_write_task(tmp_path, payload))


def test_task_v2_requires_and_accepts_exact_semantic_run(tmp_path: Path) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    archive.mkdir()
    payload = _task_payload(archive, schema_version=2)
    path = _write_task(tmp_path, payload)
    with pytest.raises(mod.ProviderBehaviorChainError, match="requires"):
        mod.load_task(path)

    payload["protocol_semantic_selection_run"] = "semantic_v2"
    loaded = mod.load_task(_write_task(tmp_path, payload))

    assert loaded["protocol_semantic_selection_run"] == "semantic_v2"


def test_task_v3_requires_exact_manifest_pinned_position_input(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    archive.mkdir()
    payload = _task_payload(
        archive,
        schema_version=mod.EXPLICIT_POSITION_TASK_SCHEMA_VERSION,
    )
    payload["protocol_semantic_selection_run"] = "semantic_v2"

    loaded = mod.load_task(_write_task(tmp_path, payload))

    assert loaded["source_runs"]["position"] == {
        "run_path": (
            "analysis/subject_position_runs/observation/position_detection_v1"
        ),
        "manifest_sha256": "a" * 64,
    }
    assert "keypoint" not in loaded["source_runs"]
    assert "position" not in loaded["output_runs"]


@pytest.mark.parametrize(
    ("section", "field"),
    [("source_runs", "keypoint"), ("output_runs", "position")],
)
def test_task_v3_rejects_obsolete_keypoint_position_fields(
    tmp_path: Path,
    section: str,
    field: str,
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    archive.mkdir()
    payload = _task_payload(
        archive,
        schema_version=mod.EXPLICIT_POSITION_TASK_SCHEMA_VERSION,
    )
    payload["protocol_semantic_selection_run"] = "semantic_v2"
    payload[section][field] = f"obsolete_{field}_v1"

    with pytest.raises(mod.ProviderBehaviorChainError, match=field):
        mod.load_task(_write_task(tmp_path, payload))


@pytest.mark.parametrize(
    ("run_path", "manifest_sha256"),
    [
        ("analysis/subject_position_runs/observation/latest", "a" * 64),
        ("analysis/subject_position_runs/track_sample/position_v1", "a" * 64),
        ("analysis/subject_position_runs/observation/position_v1/extra", "a" * 64),
        ("analysis/subject_position_runs/observation/position_v1", "A" * 64),
        ("analysis/subject_position_runs/observation/position_v1", "a" * 63),
    ],
)
def test_task_v3_rejects_ambiguous_or_unpinned_position_input(
    tmp_path: Path,
    run_path: str,
    manifest_sha256: str,
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    archive.mkdir()
    payload = _task_payload(
        archive,
        schema_version=mod.EXPLICIT_POSITION_TASK_SCHEMA_VERSION,
    )
    payload["protocol_semantic_selection_run"] = "semantic_v2"
    payload["source_runs"]["position"] = {
        "run_path": run_path,
        "manifest_sha256": manifest_sha256,
    }

    with pytest.raises(mod.ProviderBehaviorChainError, match="source_runs.position"):
        mod.load_task(_write_task(tmp_path, payload))


def _receipted_task(archive: Path) -> dict[str, object]:
    payload = _task_payload(
        archive,
        schema_version=mod.RECEIPTED_TASK_SCHEMA_VERSION,
    )
    payload["protocol_semantic_selection_run"] = "semantic_v2"
    payload["arena_id"] = 0
    return mod.build_receipted_task(payload)


def _receipt_stages() -> dict[str, object]:
    stages: dict[str, object] = {}
    for stage_name in mod.RECEIPT_STAGE_NAMES:
        digest_fields = {
            field: canonical_json_sha256({"stage": stage_name, "field": field})
            for field in mod.RECEIPT_STAGE_DIGEST_FIELDS[stage_name]
        }
        stages[stage_name] = {
            "status": "reused",
            "run_path": f"analysis/test_runs/{stage_name}",
            **digest_fields,
        }
    return stages


def test_task_v4_is_canonical_self_digested_and_loadable(tmp_path: Path) -> None:
    archive = (tmp_path / "recording_analysis.zarr").resolve()
    archive.mkdir()

    task = _receipted_task(archive)
    loaded = mod.load_task(_write_task(tmp_path, task))

    body = {key: value for key, value in task.items() if key != "task_sha256"}
    assert task["schema_version"] == mod.RECEIPTED_TASK_SCHEMA_VERSION
    assert task["analysis_zarr"] == str(archive)
    assert task["fps"] == 250.0
    assert task["arena_id"] == 0
    assert task["task_sha256"] == canonical_json_sha256(body)
    assert loaded == task


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda task: task.update(fps=125.0), "task_sha256"),
        (
            lambda task: task["source_runs"]["position"].update(
                manifest_sha256="b" * 64
            ),
            "task_sha256",
        ),
        (lambda task: task.update(uncontracted=True), "field set"),
    ],
)
def test_task_v4_rejects_tampering_or_uncontracted_fields(
    tmp_path: Path,
    mutation,
    match: str,
) -> None:
    archive = (tmp_path / "recording_analysis.zarr").resolve()
    archive.mkdir()
    task = _receipted_task(archive)
    mutation(task)

    with pytest.raises(mod.ProviderBehaviorChainError, match=match):
        mod.load_task(_write_task(tmp_path, task))


def test_task_v4_rejects_noncanonical_persisted_values(tmp_path: Path) -> None:
    archive = (tmp_path / "recording_analysis.zarr").resolve()
    archive.mkdir()
    task = _receipted_task(archive)
    task["fps"] = 250
    task["task_sha256"] = canonical_json_sha256(
        {key: value for key, value in task.items() if key != "task_sha256"}
    )

    with pytest.raises(mod.ProviderBehaviorChainError, match="canonical"):
        mod.load_task(_write_task(tmp_path, task))


def test_materialize_v4_revalidates_task_before_any_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = (tmp_path / "recording_analysis.zarr").resolve()
    archive.mkdir()
    task = _receipted_task(archive)
    task["fps"] = 125.0
    monkeypatch.setattr(
        mod,
        "_position",
        lambda *args, **kwargs: pytest.fail("tampered task must fail before stages"),
    )

    with pytest.raises(mod.ProviderBehaviorChainError, match="task_sha256"):
        mod.materialize_chain(task, scratch_root=tmp_path / "scratch")


def test_chain_receipt_binds_task_each_stage_and_itself(tmp_path: Path) -> None:
    archive = (tmp_path / "recording_analysis.zarr").resolve()
    archive.mkdir()
    task = _receipted_task(archive)
    stages = _receipt_stages()

    receipt = mod.build_chain_receipt(
        task,
        stages,
        completed_at_utc="2026-09-10T12:00:00+00:00",
    )
    validated = mod.validate_chain_receipt(
        receipt,
        expected_task_sha256=task["task_sha256"],
    )

    assert validated == receipt
    assert receipt["schema_id"] == mod.RECEIPT_SCHEMA_ID
    assert receipt["task_sha256"] == task["task_sha256"]
    assert receipt["stage_sha256s"] == {
        name: canonical_json_sha256(stages[name]) for name in mod.RECEIPT_STAGE_NAMES
    }
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    assert receipt["receipt_sha256"] == canonical_json_sha256(body)


def test_chain_receipt_requires_each_stage_artifact_binding(tmp_path: Path) -> None:
    archive = (tmp_path / "recording_analysis.zarr").resolve()
    archive.mkdir()
    task = _receipted_task(archive)
    stages = _receipt_stages()
    stages["motion"].pop("verification_digest")

    with pytest.raises(mod.ProviderBehaviorChainError, match="motion.*binding"):
        mod.build_chain_receipt(task, stages)


@pytest.mark.parametrize("field", ["stages", "stage_sha256s", "receipt_sha256"])
def test_chain_receipt_refuses_tampering(tmp_path: Path, field: str) -> None:
    archive = (tmp_path / "recording_analysis.zarr").resolve()
    archive.mkdir()
    task = _receipted_task(archive)
    receipt = mod.build_chain_receipt(
        task,
        _receipt_stages(),
        completed_at_utc="2026-09-10T12:00:00+00:00",
    )
    if field == "stages":
        receipt[field]["motion"]["run_path"] = "analysis/test_runs/tampered"
    elif field == "stage_sha256s":
        receipt[field]["motion"] = "f" * 64
    else:
        receipt[field] = "f" * 64

    with pytest.raises(mod.ProviderBehaviorChainError, match="receipt|stage"):
        mod.validate_chain_receipt(receipt)


def test_main_writes_and_reloads_v4_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    archive = (tmp_path / "recording_analysis.zarr").resolve()
    archive.mkdir()
    task = _receipted_task(archive)
    task_path = _write_task(tmp_path, task)
    receipt_path = tmp_path / "receipt.json"

    monkeypatch.setattr(
        mod,
        "materialize_chain",
        lambda loaded, *, scratch_root: mod.build_chain_receipt(
            loaded,
            _receipt_stages(),
            completed_at_utc="2026-09-10T12:00:00+00:00",
        ),
    )

    assert (
        mod.main(
            [
                "--task-json",
                str(task_path),
                "--scratch-root",
                str(tmp_path / "scratch"),
                "--receipt-json",
                str(receipt_path),
                "--apply",
                "--json",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (
        mod.validate_chain_receipt(
            receipt_path,
            expected_task_sha256=task["task_sha256"],
        )["status"]
        == "complete"
    )


def test_position_v3_loads_exact_input_without_materializing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    archive.mkdir()
    payload = _task_payload(
        archive,
        schema_version=mod.EXPLICIT_POSITION_TASK_SCHEMA_VERSION,
    )
    payload["protocol_semantic_selection_run"] = "semantic_v2"
    task = mod.load_task(_write_task(tmp_path, payload))
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    handle = type(
        "PositionHandle",
        (),
        {
            "run_path": task["source_runs"]["position"]["run_path"],
            "manifest_sha256": "a" * 64,
            "decoded_content_sha256": "b" * 64,
        },
    )()

    def load(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        return handle

    monkeypatch.setattr(mod, "load_subject_position_source_handle", load)
    monkeypatch.setattr(
        mod,
        "validate_direct_consolidated_subtree",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        mod,
        "plan_subject_position_run",
        lambda *args, **kwargs: pytest.fail("v3 must not materialize a position run"),
    )

    actual, evidence = mod._position(task, tmp_path / "scratch")  # noqa: SLF001

    assert actual is handle
    assert evidence == {
        "status": "reused",
        "run_path": handle.run_path,
        "manifest_sha256": handle.manifest_sha256,
        "decoded_content_sha256": handle.decoded_content_sha256,
    }
    assert calls == [
        (
            (archive.resolve(), handle.run_path),
            {
                "expected_selector_eligible": False,
                "expected_manifest_sha256": "a" * 64,
                "use_consolidated": True,
            },
        ),
        (
            (archive.resolve(), handle.run_path),
            {
                "expected_selector_eligible": False,
                "expected_manifest_sha256": "a" * 64,
                "use_consolidated": False,
            },
        ),
    ]


def test_position_v3_rejects_direct_consolidated_disagreement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    archive.mkdir()
    payload = _task_payload(
        archive,
        schema_version=mod.EXPLICIT_POSITION_TASK_SCHEMA_VERSION,
    )
    payload["protocol_semantic_selection_run"] = "semantic_v2"
    task = mod.load_task(_write_task(tmp_path, payload))
    monkeypatch.setattr(
        mod,
        "validate_direct_consolidated_subtree",
        lambda *args, **kwargs: None,
    )

    def load(*args: object, **kwargs: object) -> object:
        return SimpleNamespace(
            run_path=task["source_runs"]["position"]["run_path"],
            manifest_sha256="a" * 64,
            decoded_content_sha256=(
                "b" * 64 if kwargs["use_consolidated"] else "c" * 64
            ),
        )

    monkeypatch.setattr(mod, "load_subject_position_source_handle", load)

    with pytest.raises(
        mod.ProviderBehaviorChainError,
        match="direct content digest differs",
    ):
        mod._position(task, tmp_path / "scratch")  # noqa: SLF001


def test_tracking_reuse_rejects_another_position_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    target = archive / "tracking_runs" / "tracking_v1"
    target.mkdir(parents=True)
    task = {
        "analysis_zarr": str(archive),
        "output_runs": {"tracking": "tracking_v1"},
    }
    position = type(
        "PositionHandle",
        (),
        {
            "run_path": (
                "analysis/subject_position_runs/observation/position_detection_v1"
            ),
            "manifest_sha256": "a" * 64,
            "decoded_content_sha256": "b" * 64,
        },
    )()
    tracking = type(
        "TrackingHandle",
        (),
        {
            "run_path": "tracking_runs/tracking_v1",
            "manifest_sha256": "c" * 64,
            "manifest": {
                "payload": {
                    "source": {
                        "source_authority_kind": "subject_position_run",
                        "source_subject_position_run": (
                            "analysis/subject_position_runs/observation/position_other"
                        ),
                        "source_subject_position_manifest_sha256": "d" * 64,
                        "source_subject_position_decoded_content_sha256": "e" * 64,
                    }
                }
            },
        },
    )()
    monkeypatch.setattr(
        mod,
        "load_tracking_source_handle",
        lambda *args, **kwargs: tracking,
    )
    monkeypatch.setattr(
        mod,
        "require_subject_position_source_handle",
        lambda value: value,
    )

    with pytest.raises(
        mod.ProviderBehaviorChainError, match="another subject-position"
    ):
        mod._tracking(task, position, tmp_path / "scratch")  # noqa: SLF001


def test_tracking_reuse_accepts_exact_position_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    target = archive / "tracking_runs" / "tracking_v1"
    target.mkdir(parents=True)
    task = {
        "analysis_zarr": str(archive),
        "output_runs": {"tracking": "tracking_v1"},
    }
    position = SimpleNamespace(
        run_path=("analysis/subject_position_runs/observation/position_detection_v1"),
        manifest_sha256="a" * 64,
        decoded_content_sha256="b" * 64,
    )
    tracking = SimpleNamespace(
        run_path="tracking_runs/tracking_v1",
        manifest_sha256="c" * 64,
        verification_digest="d" * 64,
        manifest={
            "payload": {
                "source": {
                    "source_authority_kind": "subject_position_run",
                    "source_subject_position_run": position.run_path,
                    "source_subject_position_manifest_sha256": (
                        position.manifest_sha256
                    ),
                    "source_subject_position_decoded_content_sha256": (
                        position.decoded_content_sha256
                    ),
                }
            }
        },
    )
    monkeypatch.setattr(
        mod,
        "load_tracking_source_handle",
        lambda *args, **kwargs: tracking,
    )
    monkeypatch.setattr(
        mod,
        "require_subject_position_source_handle",
        lambda value: value,
    )
    monkeypatch.setattr(
        mod,
        "plan_single_subject_tracking_run",
        lambda *args, **kwargs: pytest.fail("matching reuse must not republish"),
    )

    actual, evidence = mod._tracking(  # noqa: SLF001
        task,
        position,
        tmp_path / "scratch",
    )

    assert actual is tracking
    assert evidence == {
        "status": "reused",
        "run_path": tracking.run_path,
        "manifest_sha256": tracking.manifest_sha256,
        "verification_digest": tracking.verification_digest,
    }


@pytest.mark.parametrize(
    "mismatch_field",
    ["source_authority_sha256", "tracked_input_sha256"],
)
def test_motion_reuse_rejects_another_composed_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mismatch_field: str,
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    target = archive / "analysis" / "track_kinematics_runs" / "provider" / "motion_v1"
    target.mkdir(parents=True)
    task = {
        "analysis_zarr": str(archive),
        "source_runs": {"body_frame": "body_frame_v1"},
        "output_runs": {"motion": "motion_v1"},
        "fps": 250.0,
    }
    authority = SimpleNamespace(authority_sha256="a" * 64)
    tracked = SimpleNamespace(authority_sha256="b" * 64)
    motion_values = {
        "run_path": "analysis/track_kinematics_runs/provider/motion_v1",
        "source_authority_sha256": authority.authority_sha256,
        "tracked_input_sha256": tracked.authority_sha256,
        "temporal_authority_sha256": "f" * 64,
    }
    motion_values[mismatch_field] = "c" * 64
    motion = SimpleNamespace(**motion_values)
    timing = SimpleNamespace(nominal_fps=250.0, sha256="f" * 64)
    monkeypatch.setattr(mod, "load_body_frame_source_handle", lambda *a, **k: object())
    monkeypatch.setattr(
        mod,
        "compose_position_body_frame_motion_authority",
        lambda *a, **k: authority,
    )
    monkeypatch.setattr(
        mod,
        "bind_position_body_frame_to_tracking",
        lambda *a, **k: tracked,
    )
    monkeypatch.setattr(
        mod,
        "load_provider_track_motion_source_handle",
        lambda *a, **k: motion,
    )
    monkeypatch.setattr(
        mod,
        "load_provider_recording_timing_authority",
        lambda *a, **k: timing,
    )

    with pytest.raises(
        mod.ProviderBehaviorChainError, match="another composed authority"
    ):
        mod._motion(task, object(), object(), tmp_path / "scratch")  # noqa: SLF001


def test_motion_reuse_accepts_exact_composed_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    target = archive / "analysis" / "track_kinematics_runs" / "provider" / "motion_v1"
    target.mkdir(parents=True)
    task = {
        "analysis_zarr": str(archive),
        "source_runs": {"body_frame": "body_frame_v1"},
        "output_runs": {"motion": "motion_v1"},
        "fps": 250.0,
    }
    authority = SimpleNamespace(authority_sha256="a" * 64)
    tracked = SimpleNamespace(authority_sha256="b" * 64)
    motion = SimpleNamespace(
        run_path="analysis/track_kinematics_runs/provider/motion_v1",
        source_authority_sha256=authority.authority_sha256,
        tracked_input_sha256=tracked.authority_sha256,
        temporal_authority_sha256="f" * 64,
        provider_manifest_sha256="c" * 64,
        verification_digest="d" * 64,
    )
    timing = SimpleNamespace(nominal_fps=250.0, sha256="f" * 64)
    monkeypatch.setattr(mod, "load_body_frame_source_handle", lambda *a, **k: object())
    monkeypatch.setattr(
        mod,
        "compose_position_body_frame_motion_authority",
        lambda *a, **k: authority,
    )
    monkeypatch.setattr(
        mod,
        "bind_position_body_frame_to_tracking",
        lambda *a, **k: tracked,
    )
    monkeypatch.setattr(
        mod,
        "load_provider_track_motion_source_handle",
        lambda *a, **k: motion,
    )
    monkeypatch.setattr(
        mod,
        "load_provider_recording_timing_authority",
        lambda *a, **k: timing,
    )
    monkeypatch.setattr(
        mod,
        "prepare_provider_track_motion",
        lambda *args, **kwargs: pytest.fail("matching reuse must not republish"),
    )

    actual, evidence = mod._motion(  # noqa: SLF001
        task,
        object(),
        object(),
        tmp_path / "scratch",
    )

    assert actual is motion
    assert evidence == {
        "status": "reused",
        "run_path": motion.run_path,
        "manifest_sha256": motion.provider_manifest_sha256,
        "verification_digest": motion.verification_digest,
    }


def test_epoch_summary_uses_physical_filtered_speed(
    monkeypatch, tmp_path: Path
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    archive.mkdir()
    captured: dict[str, object] = {}

    def fake_materialize(source_zarr: Path, **kwargs: object) -> dict[str, object]:
        captured["source_zarr"] = source_zarr
        captured.update(kwargs)
        return {"status": "published"}

    monkeypatch.setattr(
        mod,
        "materialize_provider_epoch_behavior_summary",
        fake_materialize,
    )
    monkeypatch.setattr(
        mod,
        "_load_matching_summary",
        lambda task: SimpleNamespace(
            run_path="analysis/stimulus_epoch_behavior_summary_runs/summary_v1",
            manifest_sha256="a" * 64,
        ),
    )

    result = mod._summary(  # noqa: SLF001 - focused orchestration contract test
        {
            "analysis_zarr": str(archive),
            "protocol_semantic_selection_run": "semantic_v2",
            "output_runs": {
                "epoch_summary": "summary_v1",
                "stimulus_epochs_v2": "epochs_v2",
                "motion": "motion_v1",
                "swim_bouts": "bouts_v1",
            },
        },
        tmp_path / "scratch",
    )

    assert captured["source_zarr"] == archive
    assert captured["speed_level"] == mod.SUMMARY_SPEED_LEVEL == "filtered"
    assert captured["protocol_semantic_selection_run_name"] == "semantic_v2"
    assert result["status"] == "published"


def test_swim_bout_reuse_rejects_another_motion_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    archive.mkdir()
    motion, tables = _provider_swim_bout_fixture()
    tables.run_attrs["source_track_motion_manifest_sha256"] = "d" * 64
    monkeypatch.setattr(
        mod,
        "validate_direct_consolidated_subtree",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        mod,
        "open_zarr_root",
        lambda *args, **kwargs: SimpleNamespace(attrs={"fps": 100.0}),
    )
    monkeypatch.setattr(
        mod,
        "load_exact_selector_ineligible_default_swim_bout_tables",
        lambda *args, **kwargs: tables,
    )
    monkeypatch.setattr(
        mod,
        "_swim_bout_content_sha256",
        lambda value: "e" * 64,
    )

    with pytest.raises(mod.ProviderBehaviorChainError, match="another provider-motion"):
        mod._load_matching_swim_bout(  # noqa: SLF001
            archive,
            run_name="bouts_v1",
            motion=motion,
        )


def test_legacy_epoch_reuse_rejects_another_stimulus_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    archive.mkdir()
    lineage_payload_json = mod.canonical_lineage_json({"source": "fixture"})
    run = SimpleNamespace(
        attrs={
            mod.RUN_COMPLETION_CONTRACT_ATTR: mod.RUN_COMPLETION_CONTRACT,
            mod.RUN_COMPLETION_STATUS_ATTR: mod.RUN_STATUS_COMPLETE,
            "stage_selector_eligible": False,
            "source_stimulus_run": "stimulus_other",
            "source_stimulus_path": "analysis/stimulus_runs/stimulus_other",
            "lineage_payload_json": lineage_payload_json,
            "lineage_hash": mod.compute_run_lineage_hash({"source": "fixture"}),
        }
    )
    monkeypatch.setattr(
        mod,
        "validate_direct_consolidated_subtree",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        mod,
        "validate_legacy_stimulus_epoch_source",
        lambda value: [],
    )
    monkeypatch.setattr(
        mod,
        "stimulus_epoch_logical_content_sha256",
        lambda value: "a" * 64,
    )
    monkeypatch.setattr(
        mod,
        "open_zarr_root",
        lambda *args, **kwargs: {"analysis/stimulus_epoch_runs/epochs_v1": run},
    )

    with pytest.raises(mod.ProviderBehaviorChainError, match="another stimulus"):
        mod._validate_legacy_epoch_reuse(  # noqa: SLF001
            archive,
            run_name="epochs_v1",
            source_stimulus_run="stimulus_v1",
        )


def test_epoch_candidate_reuse_rejects_another_legacy_epoch_digest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    archive.mkdir()
    source_record = {
        "logical_content_sha256": "a" * 64,
        "lineage_hash": "b" * 64,
        "lineage_payload_sha256": "c" * 64,
    }
    run = SimpleNamespace(
        attrs={
            "source_stimulus_epoch_run": "epochs_v1",
            "source_stimulus_epoch_path": "analysis/stimulus_epoch_runs/epochs_v1",
            "source_stimulus_run": "stimulus_v1",
            "source_stimulus_epoch_logical_content_sha256": "d" * 64,
            "source_stimulus_epoch_lineage_hash": source_record["lineage_hash"],
            "source_stimulus_epoch_lineage_payload_sha256": source_record[
                "lineage_payload_sha256"
            ],
        }
    )
    monkeypatch.setattr(
        mod,
        "resolve_exact_stimulus_epoch_selection",
        lambda *args, **kwargs: SimpleNamespace(selection_digest="e" * 64),
    )
    monkeypatch.setattr(
        mod,
        "open_zarr_root",
        lambda *args, **kwargs: {"analysis/stimulus_epoch_runs/epochs_v2": run},
    )

    with pytest.raises(mod.ProviderBehaviorChainError, match="another legacy epoch"):
        mod._validate_epoch_candidate_reuse(  # noqa: SLF001
            archive,
            run_name="epochs_v2",
            source_run_name="epochs_v1",
            source_record=source_record,
            source_stimulus_run="stimulus_v1",
        )


def test_summary_reuse_rejects_another_swim_bout_lineage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    archive.mkdir()
    task = {
        "analysis_zarr": str(archive),
        "recording_id": "recording",
        "protocol_semantic_selection_run": "semantic_v2",
        "output_runs": {
            "epoch_summary": "summary_v1",
            "stimulus_epochs_v2": "epochs_v2",
            "motion": "motion_v1",
            "swim_bouts": "bouts_v1",
        },
    }
    motion = SimpleNamespace(
        run_path="analysis/track_kinematics_runs/provider/motion_v1",
        provider_manifest_sha256="a" * 64,
        verification_digest="b" * 64,
    )
    bouts = SimpleNamespace(
        run_path="analysis/swim_bout_runs/bouts_v1",
        run_attrs={"lineage_hash": "c" * 64},
    )
    selection = SimpleNamespace(selection_digest="d" * 64)
    semantic = SimpleNamespace(source_binding=lambda: {"sha256": "e" * 64})
    summary = SimpleNamespace(
        manifest={
            "sources": {
                "provider_motion": {
                    "run_path": motion.run_path,
                    "manifest_sha256": motion.provider_manifest_sha256,
                    "verification_digest": motion.verification_digest,
                },
                "swim_bouts": {
                    "run_path": bouts.run_path,
                    "lineage_hash": "f" * 64,
                    "source_track_motion_manifest_sha256": (
                        motion.provider_manifest_sha256
                    ),
                },
                "epoch_selection": {"sha256": selection.selection_digest},
            }
        }
    )
    monkeypatch.setattr(
        mod,
        "load_provider_track_motion_source_handle",
        lambda *args, **kwargs: motion,
    )
    monkeypatch.setattr(mod, "_load_matching_swim_bout", lambda *a, **k: bouts)
    monkeypatch.setattr(
        mod,
        "resolve_exact_stimulus_epoch_selection",
        lambda *args, **kwargs: selection,
    )
    monkeypatch.setattr(
        mod,
        "load_protocol_semantic_chaser_selection_source_handle",
        lambda *args, **kwargs: semantic,
    )
    monkeypatch.setattr(
        mod,
        "load_provider_epoch_behavior_summary_source_handle",
        lambda *args, **kwargs: summary,
    )

    with pytest.raises(mod.ProviderBehaviorChainError, match="another motion, bout"):
        mod._load_matching_summary(task)  # noqa: SLF001


def _install_source_camera_physical_authority(
    archive: Path,
    tmp_path: Path,
):  # type: ignore[no-untyped-def]
    root = zarr.open_group(
        str(archive), mode="a", zarr_format=3, use_consolidated=False
    )
    camera_id = str(root.attrs["camera_id"])
    authority = root.require_group(f"analysis/acquisition_camera_frames/{camera_id}")
    ownership = stamp_acquisition_import_ownership(root, authority)
    stamp_acquisition_camera_frame(
        root,
        authority,
        import_ownership=ownership,
    )
    arena = {
        "active_camera_id": camera_id,
        "calculated_z_eff_mm": 20.0,
        "camera_calibrations": [
            {
                "camera_id": camera_id,
                "native_width_px": 640,
                "native_height_px": 640,
                "pixels_per_mm_camera": 10.0,
                "pixels_per_mm_projector": 4.0,
                "real_world_ref_mm": 10.0,
            }
        ],
    }
    camera_attrs = {
        "pixels_per_mm_camera": 10.0,
        "pixels_per_mm_projector": 4.0,
        "real_world_ref_mm": 10.0,
    }
    evidence = build_selected_camera_source_evidence_from_h5_values(
        source_h5_path=str((tmp_path / "calibration.h5").resolve()),
        arena_config_raw=json.dumps(arena, sort_keys=True).encode("utf-8"),
        camera_group_path=f"/calibration_snapshot/{camera_id}",
        camera_group_attrs=camera_attrs,
        expected_camera_id=camera_id,
    )
    return publish_source_camera_physical_authority(
        root,
        source_camera_evidence=evidence,
        source_kind="stimulus_h5_calibration_snapshot",
        provenance={"source": "provider behavior chain real-Zarr fixture"},
    )


def _publish_detection_position(
    archive: Path,
    tmp_path: Path,
    *,
    source_camera_frame: object,
):  # type: ignore[no-untyped-def]
    keys = np.asarray([101, 102, 103], dtype=np.uint64)
    frames = np.asarray([0, 1, 2], dtype=np.int64)
    positions = np.asarray(
        [[20.0, 20.0], [21.0, 20.0], [22.0, 20.0]],
        dtype=np.float32,
    )
    valid = np.ones(3, dtype=bool)
    reasons = np.full(
        3,
        POSITION_FAILURE_REASON_CODES["ok"],
        dtype=np.uint16,
    )
    reference = DigestBoundCoordinateRecordRef(
        record_ref=source_camera_frame.record_ref,
        record_sha256=source_camera_frame.record_sha256,
    )
    descriptor = build_canonical_coordinate_descriptor(
        **SOURCE_CAMERA_POINT_XY.descriptor_kwargs(),
        reference_width=640,
        reference_height=640,
        reference_authority=reference,
        reference_selector="record",
        row_identity_contract=build_row_identity_contract(
            domain=OBSERVATION_INSTANCE_DOMAIN,
            values=keys,
        ),
        row_identity_record_ref=(
            "/detect_runs/detect_authorized@row_identity_contract"
        ),
        overlay_transform_refs=(),
        frame_record=CanonicalFrameRecord(
            kind=PIXEL_FRAME_AUTHORITY_RECORD_KIND,
            record_ref=reference.record_ref,
            record_sha256=reference.record_sha256,
        ),
    )
    coordinate = canonical_source_camera_coordinate_metadata(descriptor)
    estimator = get_estimator_profile(DETECTION_BBOX_CENTROID_ESTIMATOR_ID)
    records = {
        "anatomy": {"anatomy_profile_id": None, "record_id": "none.v1"},
        "source": {
            "run_path": "detect_runs/detect_authorized",
            "array_paths": ["instance_key", "bbox_img_xyxy"],
            "row_axis": "observation_instance",
        },
        "policy": {
            "provider_selection": "explicit_source_adapter",
            "validity": "upstream_authority_required.v1",
        },
        "software": {"package": "palette", "commit": "c" * 40},
    }
    prepared = SubjectPositionPreparedInput(
        arrays={
            "position_xy": positions,
            "valid": valid,
            "failure_reason_codes": reasons,
            "instance_key": keys,
            "source_acquisition_frame_index": frames,
            "source_row_index": np.arange(3, dtype=np.int64),
        },
        estimator_record=estimator,
        estimator_sha256=estimator_profile_digest(estimator),
        anatomy_record=records["anatomy"],
        anatomy_sha256=canonical_json_sha256(records["anatomy"]),
        source_record=records["source"],
        source_sha256=canonical_json_sha256(records["source"]),
        policy_record=records["policy"],
        policy_sha256=canonical_json_sha256(records["policy"]),
        software_record=records["software"],
        software_sha256=canonical_json_sha256(records["software"]),
        coordinate_record=coordinate,
        coordinate_sha256=canonical_json_sha256(coordinate),
    )
    plan = plan_subject_position_run(
        archive,
        prepared,
        run_name="position_detection_v1",
        scratch_root=tmp_path / "scratch-position",
    )
    publish_subject_position_run(plan, keep_scratch=False)
    return plan


def test_v4_real_zarr_chain_receipts_reuse_and_refuse_tampered_position(
    tmp_path: Path,
) -> None:
    semantic_archive, _selection, _protocol_evidence = _semantic_selection(tmp_path)
    direct = zarr.open_group(
        str(semantic_archive), mode="a", zarr_format=3, use_consolidated=False
    )
    metadata = dict(direct.attrs["source_video_metadata"])
    source_video = Path(str(metadata["source_path"]))
    source_stat = source_video.stat()
    metadata["file_fingerprint"] = {
        "strategy": "sha256_test_fixture_v1",
        "value": hashlib.sha256(source_video.read_bytes()).hexdigest(),
        "size_bytes": source_stat.st_size,
        "mtime_ns": source_stat.st_mtime_ns,
        "relocation_stable": False,
    }
    direct.attrs["source_video_metadata"] = metadata
    consolidate_metadata_capture_expected_warnings(semantic_archive)
    selection = resolve_exact_stimulus_epoch_selection(
        semantic_archive,
        run_name="candidate",
    )
    protocol_evidence = load_protocol_semantic_selection_evidence(
        semantic_archive,
        selection,
    )
    selections = compile_protocol_semantic_chaser_selections(
        selection,
        timeline_evidence=_evidence(semantic_archive, selection),
        protocol_evidence=protocol_evidence,
        role_bindings=_chaser_bindings(selection),
    )
    semantic_plan = build_protocol_semantic_chaser_selection_publication_plan(
        semantic_archive,
        selections=selections,
        source_selection=selection,
        run_name="semantic_v2",
    )
    publish_protocol_semantic_chaser_selection_run(
        semantic_plan,
        scratch_root=tmp_path / "scratch-semantic",
    )
    archive = tmp_path / "recording_1_analysis.zarr"
    semantic_archive.rename(archive)

    body_archive, _publication = _published_fixture(tmp_path / "body-frame")
    shutil.copytree(
        body_archive / "analysis" / "body_frame_runs",
        archive / "analysis" / "body_frame_runs",
    )
    physical = _install_source_camera_physical_authority(archive, tmp_path)
    position_plan = _publish_detection_position(
        archive,
        tmp_path,
        source_camera_frame=physical.physical_frame.source_camera_pixels,
    )
    direct = zarr.open_group(
        str(archive), mode="a", zarr_format=3, use_consolidated=False
    )
    direct["analysis/stimulus_epoch_runs/source"].attrs[
        "stage_selector_eligible"
    ] = False
    consolidate_metadata_capture_expected_warnings(archive)

    payload = {
        "schema_id": mod.TASK_SCHEMA_ID,
        "schema_version": mod.RECEIPTED_TASK_SCHEMA_VERSION,
        "recording_id": "recording_1",
        "analysis_zarr": str(archive),
        "source_runs": {
            "position": {
                "run_path": position_plan.run_path,
                "manifest_sha256": position_plan.final_manifest_sha256,
            },
            "body_frame": "body_frame_v1_001",
            "stimulus": "stimulus_1",
        },
        "output_runs": {
            "tracking": "tracking_v1",
            "motion": "motion_v1",
            "swim_bouts": "bouts_v1",
            "stimulus_epochs_v1": "source",
            "stimulus_epochs_v2": "candidate",
            "epoch_summary": "summary_v1",
        },
        "protocol_semantic_selection_run": "semantic_v2",
        "fps": 10.0,
        "arena_id": 0,
        "metric_disposition": mod.LINEAR_ONLY_DISPOSITION,
    }
    payload = mod.build_receipted_task(payload)
    task = mod.load_task(_write_task(tmp_path, payload))

    first = mod.materialize_chain(task, scratch_root=tmp_path / "scratch-chain-first")
    assert first["status"] == "complete"
    assert first["schema_id"] == mod.RECEIPT_SCHEMA_ID
    assert first["task_sha256"] == task["task_sha256"]
    assert (
        mod.validate_chain_receipt(
            first,
            expected_task_sha256=task["task_sha256"],
        )
        == first
    )
    assert first["stages"]["tracking"]["status"] == "published"
    assert first["stages"]["motion"]["status"] == "published"
    assert first["stages"]["swim_bouts"]["status"] == "published"
    assert first["stages"]["epoch_summary"]["status"] == "published"
    motion_verification = verify_persisted_stage_output(
        archive,
        "track_kinematics",
        requested_run="provider/motion_v1",
        dependency_runs={},
        run_scope="provider",
        execution_profile_id=SELECTOR_INELIGIBLE_CANARY_EXECUTION_PROFILE_ID,
    )
    bout_verification = verify_persisted_stage_output(
        archive,
        "swim_bouts",
        requested_run="bouts_v1",
        dependency_runs={"track_kinematics": "provider/motion_v1"},
        execution_profile_id=SELECTOR_INELIGIBLE_CANARY_EXECUTION_PROFILE_ID,
    )
    assert motion_verification.available, motion_verification.reason
    assert bout_verification.available, bout_verification.reason

    second = mod.materialize_chain(
        task,
        scratch_root=tmp_path / "scratch-chain-second",
    )
    assert set(second["stages"]) == {
        "position",
        "tracking",
        "motion",
        "swim_bouts",
        "stimulus_epochs_v1",
        "stimulus_epochs_v2",
        "epoch_summary",
    }
    assert all(stage["status"] == "reused" for stage in second["stages"].values())
    assert second["task_sha256"] == first["task_sha256"]
    assert second["receipt_sha256"] != first["receipt_sha256"]
    assert (
        mod.validate_chain_receipt(
            second,
            expected_task_sha256=task["task_sha256"],
        )
        == second
    )

    direct = zarr.open_group(
        str(archive), mode="r+", zarr_format=3, use_consolidated=False
    )
    position_xy = direct[f"{position_plan.run_path}/position_xy"]
    position_xy[0, 0] = float(position_xy[0, 0]) + 1.0
    with pytest.raises(ValueError, match="position_xy|content|digest"):
        mod.materialize_chain(
            task,
            scratch_root=tmp_path / "scratch-chain-tampered",
        )

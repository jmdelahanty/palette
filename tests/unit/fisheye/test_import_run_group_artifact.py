from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from fisheye.utils import import_run_group_artifact as mod
from fisheye.utils.run_detection_artifact import (
    ARTIFACT_SCHEMA,
    DETECTION_ARTIFACT_FAMILY_CONTRACT,
    DETECTION_ARTIFACT_LAYOUT,
    REQUIRED_DETECT_ARRAYS,
    RUN_FAMILY,
    UNBOUND_DETECTION_ARTIFACT_COORDINATE_CONTRACT,
    tree_hash,
)


def _write_group(path: Path, *, attributes: dict[str, object] | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "zarr.json").write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": attributes or {},
            }
        ),
        encoding="utf-8",
    )


def _write_array(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "zarr.json").write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "node_type": "array",
                "shape": [0],
                "data_type": "int32",
                "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [1]}},
                "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
                "fill_value": 0,
                "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
                "attributes": {},
            }
        ),
        encoding="utf-8",
    )


def _write_artifact(
    tmp_path: Path,
    *,
    target_zarr: Path,
    corrupt_hash: bool = False,
    latest_policy: str = "do_not_set_latest",
    intended_target_group_path: str | None = None,
    stage_selector_eligible: bool = False,
) -> Path:
    source_video = tmp_path / "camera.mp4"
    source_video.write_bytes(b"fake")
    artifact_root = tmp_path / "palette_run_group_artifact"
    run_group = artifact_root / "run_group"
    _write_group(
        run_group,
        attributes={
            "coordinate_contract_mode": "artifact_unbound",
            "coordinate_contract": UNBOUND_DETECTION_ARTIFACT_COORDINATE_CONTRACT,
            "stage_selector_eligible": stage_selector_eligible,
            "palette_run_completion_contract": "palette.zarr_run_completion.v1",
            "palette_run_completion_status": "complete",
            "palette_run_stage": "detection_artifact",
        },
    )
    for name in REQUIRED_DETECT_ARRAYS:
        _write_array(run_group / name)

    digest = tree_hash(run_group)
    manifest = {
        "artifact_schema": ARTIFACT_SCHEMA,
        "created_at": "2026-05-15T00:00:00+00:00",
        "target_archive_path": str(target_zarr),
        "target_group_path": f"{RUN_FAMILY}/detect_fake",
        "run_family": RUN_FAMILY,
        "run_name": "detect_fake",
        "layout": DETECTION_ARTIFACT_LAYOUT,
        "schema_version": 1,
        "latest_policy": latest_policy,
        "selector_policy": "never_select_or_promote_v1",
        "artifact_family_contract": DETECTION_ARTIFACT_FAMILY_CONTRACT,
        "stage_selector_eligible": stage_selector_eligible,
        "source_inputs": [
            {"path": str(source_video), "role": "source_video"},
            {"path": str(target_zarr), "role": "target_analysis_archive"},
        ],
        "provenance": {"command": "scripts/py -m fake"},
        "timing": {},
        "checksums": {"run_group_tree_hash": "bad" if corrupt_hash else digest},
        "validation": {
            "strict_json": "pass",
            "required_arrays": "pass",
            "canonical_write": "not_performed",
        },
    }
    if intended_target_group_path is not None:
        manifest["intended_target_group_path"] = intended_target_group_path
    (artifact_root / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (artifact_root / "validation").mkdir()
    (artifact_root / "validation" / "strict_json_report.json").write_text("{}", encoding="utf-8")
    (artifact_root / "validation" / "array_presence_report.json").write_text("{}", encoding="utf-8")

    tarball = tmp_path / "artifact.tar.gz"
    with tarfile.open(tarball, "w:gz") as tar:
        tar.add(artifact_root, arcname=artifact_root.name)
    return tarball


def test_build_import_plan_validates_artifact_without_mutating_target(tmp_path: Path) -> None:
    target_zarr = tmp_path / "recording_analysis.zarr"
    _write_group(target_zarr)
    tarball = _write_artifact(tmp_path, target_zarr=target_zarr)

    plan = mod.build_import_plan(tarball_path=tarball)

    assert plan["status"] == "ok"
    assert plan["apply"] is False
    assert plan["target_group_path"] == f"{RUN_FAMILY}/detect_fake"
    assert plan["final_path"] == str(target_zarr / RUN_FAMILY / "detect_fake")
    assert plan["incoming_path"] == str(
        target_zarr / RUN_FAMILY / ".incoming" / "detect_fake"
    )
    assert plan["validations"]["strict_json"]["status"] == "pass"
    assert plan["validations"]["required_arrays"]["status"] == "pass"
    assert plan["validations"]["run_group_tree_hash"]["status"] == "pass"
    assert not (target_zarr / RUN_FAMILY).exists()


def test_build_import_plan_can_target_clip_local_intended_path(tmp_path: Path) -> None:
    target_zarr = tmp_path / "recording_analysis.zarr"
    _write_group(target_zarr)
    tarball = _write_artifact(
        tmp_path,
        target_zarr=target_zarr,
        intended_target_group_path=(
            f"clips/clip_000000/cameras/2010093/{RUN_FAMILY}/detect_fake"
        ),
    )

    plan = mod.build_import_plan(tarball_path=tarball, use_intended_target=True)

    assert plan["status"] == "ok"
    assert plan["target_group_path"] == (
        f"clips/clip_000000/cameras/2010093/{RUN_FAMILY}/detect_fake"
    )
    assert plan["target_group_path_source"] == "intended_target_group_path"
    assert plan["run_family_path"] == f"clips/clip_000000/cameras/2010093/{RUN_FAMILY}"
    assert plan["final_path"] == str(
        target_zarr / "clips" / "clip_000000" / "cameras" / "2010093" / RUN_FAMILY / "detect_fake"
    )
    assert plan["incoming_path"] == str(
        target_zarr
        / "clips"
        / "clip_000000"
        / "cameras"
        / "2010093"
        / RUN_FAMILY
        / ".incoming"
        / "detect_fake"
    )


def test_build_import_plan_fails_when_final_target_exists(tmp_path: Path) -> None:
    target_zarr = tmp_path / "recording_analysis.zarr"
    _write_group(target_zarr)
    _write_group(target_zarr / RUN_FAMILY / "detect_fake")
    tarball = _write_artifact(tmp_path, target_zarr=target_zarr)

    plan = mod.build_import_plan(tarball_path=tarball)

    assert plan["status"] == "failed"
    assert plan["validations"]["target_paths"]["status"] == "fail"
    assert "final target already exists" in "\n".join(plan["errors"])


def test_build_import_plan_fails_on_hash_mismatch(tmp_path: Path) -> None:
    target_zarr = tmp_path / "recording_analysis.zarr"
    _write_group(target_zarr)
    tarball = _write_artifact(tmp_path, target_zarr=target_zarr, corrupt_hash=True)

    plan = mod.build_import_plan(tarball_path=tarball)

    assert plan["status"] == "failed"
    assert plan["validations"]["run_group_tree_hash"]["status"] == "fail"
    assert "run_group_tree_hash mismatch" in "\n".join(plan["errors"])


def test_build_import_plan_rejects_selector_eligible_artifact(tmp_path: Path) -> None:
    target_zarr = tmp_path / "recording_analysis.zarr"
    _write_group(target_zarr)
    tarball = _write_artifact(
        tmp_path,
        target_zarr=target_zarr,
        stage_selector_eligible=True,
    )

    plan = mod.build_import_plan(tarball_path=tarball)

    assert plan["status"] == "failed"
    assert plan["validations"]["manifest"]["status"] == "fail"
    assert plan["validations"]["artifact_run_metadata"]["status"] == "fail"
    assert "stage_selector_eligible must be false" in "\n".join(plan["errors"])


def test_build_import_plan_rejects_unsafe_tar_member(tmp_path: Path) -> None:
    tarball = tmp_path / "unsafe.tar.gz"
    payload = tmp_path / "payload.txt"
    payload.write_text("bad", encoding="utf-8")
    with tarfile.open(tarball, "w:gz") as tar:
        tar.add(payload, arcname="../escape.txt")

    plan = mod.build_import_plan(tarball_path=tarball)

    assert plan["status"] == "failed"
    assert "unsafe tar member path" in "\n".join(plan["errors"])


def test_apply_import_promotes_run_group_and_writes_receipt_sidecar(tmp_path: Path) -> None:
    target_zarr = tmp_path / "recording_analysis.zarr"
    _write_group(target_zarr)
    tarball = _write_artifact(tmp_path, target_zarr=target_zarr)

    result = mod.apply_import(tarball_path=tarball)

    final_path = target_zarr / RUN_FAMILY / "detect_fake"
    incoming_path = target_zarr / RUN_FAMILY / ".incoming" / "detect_fake"
    receipt_path = target_zarr / RUN_FAMILY / ".imports" / "detect_fake_import_receipt.json"
    assert result["status"] == "ok"
    assert result["apply"] is True
    assert result["applied"] is True
    assert final_path.exists()
    assert not incoming_path.exists()
    assert (final_path / "frame_indices" / "zarr.json").exists()
    imported_attrs = json.loads(
        (final_path / "zarr.json").read_text(encoding="utf-8")
    )["attributes"]
    assert imported_attrs["stage_selector_eligible"] is False
    assert receipt_path.exists()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["run_name"] == "detect_fake"
    assert receipt["latest_updated"] is False
    parent_attrs = json.loads((target_zarr / RUN_FAMILY / "zarr.json").read_text(encoding="utf-8"))[
        "attributes"
    ]
    assert "latest" not in parent_attrs
    assert parent_attrs["stage_selector_eligible"] is False


def test_apply_import_can_promote_to_clip_local_intended_path(tmp_path: Path) -> None:
    target_zarr = tmp_path / "recording_analysis.zarr"
    _write_group(target_zarr)
    intended = f"clips/clip_000000/cameras/2010093/{RUN_FAMILY}/detect_fake"
    tarball = _write_artifact(
        tmp_path,
        target_zarr=target_zarr,
        intended_target_group_path=intended,
    )

    result = mod.apply_import(tarball_path=tarball, use_intended_target=True)

    family = target_zarr / "clips" / "clip_000000" / "cameras" / "2010093" / RUN_FAMILY
    final_path = family / "detect_fake"
    receipt_path = family / ".imports" / "detect_fake_import_receipt.json"
    assert result["status"] == "ok"
    assert result["applied"] is True
    assert result["target_group_path"] == intended
    assert result["run_family_path"] == f"clips/clip_000000/cameras/2010093/{RUN_FAMILY}"
    assert final_path.exists()
    assert receipt_path.exists()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["target_group_path"] == intended
    assert receipt["target_group_path_source"] == "intended_target_group_path"
    assert receipt["run_family"] == RUN_FAMILY
    assert receipt["run_family_path"] == f"clips/clip_000000/cameras/2010093/{RUN_FAMILY}"


def test_importer_rejects_artifact_selector_promotion(tmp_path: Path) -> None:
    target_zarr = tmp_path / "recording_analysis.zarr"
    _write_group(target_zarr)
    tarball = _write_artifact(
        tmp_path,
        target_zarr=target_zarr,
        latest_policy="set_latest_explicit",
    )

    result = mod.apply_import(tarball_path=tarball)

    assert result["status"] == "failed"
    assert result["applied"] is False
    assert "forbids latest or authoritative promotion" in "\n".join(
        result["errors"]
    )
    assert not (target_zarr / RUN_FAMILY).exists()


def test_apply_import_moves_incoming_to_failed_on_apply_validation_failure(
    tmp_path: Path, monkeypatch
) -> None:
    target_zarr = tmp_path / "recording_analysis.zarr"
    _write_group(target_zarr)
    tarball = _write_artifact(tmp_path, target_zarr=target_zarr)
    original_required_arrays_report = mod.required_arrays_report
    call_count = {"n": 0}

    def fail_second_required_arrays_report(path: Path):
        call_count["n"] += 1
        if call_count["n"] == 2:
            return {
                "status": "fail",
                "run_group_zarr_json_present": True,
                "arrays": [],
                "missing_arrays": ["frame_indices"],
            }
        return original_required_arrays_report(path)

    monkeypatch.setattr(mod, "required_arrays_report", fail_second_required_arrays_report)

    result = mod.apply_import(tarball_path=tarball)

    assert result["status"] == "failed"
    assert result["applied"] is False
    assert result["failed_path"] is not None
    assert Path(result["failed_path"]).exists()
    assert not (target_zarr / RUN_FAMILY / ".incoming" / "detect_fake").exists()
    assert not (target_zarr / RUN_FAMILY / "detect_fake").exists()


def test_overwrite_validation_failure_preserves_existing_final_byte_for_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_zarr = tmp_path / "recording_analysis.zarr"
    _write_group(target_zarr)
    existing = target_zarr / RUN_FAMILY / "detect_fake"
    _write_group(existing, attributes={"sentinel": "old"})
    (existing / "payload.bin").write_bytes(b"old-payload")
    before = tree_hash(existing)
    tarball = _write_artifact(tmp_path, target_zarr=target_zarr)
    original_required_arrays_report = mod.required_arrays_report
    call_count = {"n": 0}

    def fail_second_required_arrays_report(path: Path):
        call_count["n"] += 1
        if call_count["n"] == 2:
            return {
                "status": "fail",
                "run_group_zarr_json_present": True,
                "arrays": [],
                "missing_arrays": ["frame_indices"],
            }
        return original_required_arrays_report(path)

    monkeypatch.setattr(
        mod,
        "required_arrays_report",
        fail_second_required_arrays_report,
    )

    result = mod.apply_import(tarball_path=tarball, overwrite=True)

    assert result["status"] == "failed"
    assert result["applied"] is False
    assert result["imported"] is False
    assert result["rollback_errors"] == []
    assert tree_hash(existing) == before
    assert (existing / "payload.bin").read_bytes() == b"old-payload"


def test_receipt_build_failure_never_replaces_existing_final_or_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_zarr = tmp_path / "recording_analysis.zarr"
    _write_group(target_zarr)
    family = target_zarr / RUN_FAMILY
    existing = family / "detect_fake"
    _write_group(existing, attributes={"sentinel": "old"})
    (existing / "payload.bin").write_bytes(b"old-payload")
    receipt = family / ".imports" / "detect_fake_import_receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_bytes(b"old-receipt")
    before = tree_hash(existing)
    tarball = _write_artifact(tmp_path, target_zarr=target_zarr)

    def fail_git_info():
        raise RuntimeError("injected receipt provenance failure")

    monkeypatch.setattr(mod, "get_git_info", fail_git_info)

    result = mod.apply_import(tarball_path=tarball, overwrite=True)

    assert result["status"] == "failed"
    assert result["applied"] is False
    assert result["imported"] is False
    assert result["rollback_errors"] == []
    assert "injected receipt provenance failure" in "\n".join(result["errors"])
    assert tree_hash(existing) == before
    assert receipt.read_bytes() == b"old-receipt"


def test_receipt_commit_failure_restores_previous_final_and_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_zarr = tmp_path / "recording_analysis.zarr"
    _write_group(target_zarr)
    family = target_zarr / RUN_FAMILY
    existing = family / "detect_fake"
    _write_group(existing, attributes={"sentinel": "old"})
    (existing / "payload.bin").write_bytes(b"old-payload")
    receipt = family / ".imports" / "detect_fake_import_receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_bytes(b"old-receipt")
    before = tree_hash(existing)
    tarball = _write_artifact(tmp_path, target_zarr=target_zarr)
    original_atomic_move = mod._atomic_move

    def fail_receipt_commit(source: Path, destination: Path) -> None:
        if source.name.endswith(".json.pending"):
            raise RuntimeError("injected receipt commit failure")
        original_atomic_move(source, destination)

    monkeypatch.setattr(mod, "_atomic_move", fail_receipt_commit)

    result = mod.apply_import(tarball_path=tarball, overwrite=True)

    assert result["status"] == "failed"
    assert result["applied"] is False
    assert result["imported"] is False
    assert result["rollback_errors"] == []
    assert "injected receipt commit failure" in "\n".join(result["errors"])
    assert tree_hash(existing) == before
    assert (existing / "payload.bin").read_bytes() == b"old-payload"
    assert receipt.read_bytes() == b"old-receipt"
    assert result["failed_path"] is not None
    assert Path(result["failed_path"]).exists()


@pytest.mark.parametrize("move_number", [1, 2, 3, 4])
def test_move_that_succeeds_then_raises_is_reconciled_at_every_commit_edge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    move_number: int,
) -> None:
    target_zarr = tmp_path / "recording_analysis.zarr"
    _write_group(target_zarr)
    family = target_zarr / RUN_FAMILY
    existing = family / "detect_fake"
    _write_group(existing, attributes={"sentinel": "old"})
    receipt = family / ".imports" / "detect_fake_import_receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_bytes(b"old-receipt")
    tarball = _write_artifact(tmp_path, target_zarr=target_zarr)
    original_atomic_move = mod._atomic_move
    calls = {"count": 0}

    def move_then_raise(source: Path, destination: Path) -> None:
        calls["count"] += 1
        original_atomic_move(source, destination)
        if calls["count"] == move_number:
            raise RuntimeError(f"injected post-move failure {move_number}")

    monkeypatch.setattr(mod, "_atomic_move", move_then_raise)

    result = mod.apply_import(tarball_path=tarball, overwrite=True)

    assert result["status"] == "ok"
    assert result["applied"] is True
    assert calls["count"] == 4
    assert (existing / "frame_indices" / "zarr.json").exists()
    assert json.loads(receipt.read_text(encoding="utf-8"))["run_name"] == "detect_fake"


def test_quarantine_failure_never_reports_clean_not_applied_with_replacement_at_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_zarr = tmp_path / "recording_analysis.zarr"
    _write_group(target_zarr)
    family = target_zarr / RUN_FAMILY
    existing = family / "detect_fake"
    _write_group(existing, attributes={"sentinel": "old"})
    receipt = family / ".imports" / "detect_fake_import_receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_bytes(b"old-receipt")
    tarball = _write_artifact(tmp_path, target_zarr=target_zarr)
    original_atomic_move = mod._atomic_move

    def fail_commit_and_quarantine(source: Path, destination: Path) -> None:
        if source.name.endswith(".json.pending"):
            raise RuntimeError("injected receipt commit failure")
        if ".failed" in destination.parts:
            raise RuntimeError("injected quarantine failure")
        original_atomic_move(source, destination)

    monkeypatch.setattr(mod, "_atomic_move", fail_commit_and_quarantine)

    result = mod.apply_import(tarball_path=tarball, overwrite=True)

    assert result["status"] == "rollback_incomplete"
    assert result["applied"] is None
    assert result["imported"] is None
    assert result["final_state"] == "ambiguous_manual_recovery_required"
    assert (existing / "frame_indices" / "zarr.json").exists()
    assert any("quarantine attempted import" in error for error in result["rollback_errors"])
    assert any("final path does not match" in error for error in result["rollback_errors"])


def test_failed_quarantine_uses_unique_container_despite_legacy_name_collisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_zarr = tmp_path / "recording_analysis.zarr"
    _write_group(target_zarr)
    family = target_zarr / RUN_FAMILY
    failed = family / ".failed"
    (failed / "detect_fake_fixed").mkdir(parents=True)
    (failed / "detect_fake_fixed_1").mkdir()
    tarball = _write_artifact(tmp_path, target_zarr=target_zarr)
    original_required_arrays_report = mod.required_arrays_report
    calls = {"count": 0}

    def fail_apply_validation(path: Path):
        calls["count"] += 1
        if calls["count"] == 2:
            return {
                "status": "fail",
                "run_group_zarr_json_present": True,
                "arrays": [],
                "missing_arrays": ["frame_indices"],
            }
        return original_required_arrays_report(path)

    monkeypatch.setattr(mod, "required_arrays_report", fail_apply_validation)
    monkeypatch.setattr(mod, "_utc_now_label", lambda: "fixed")

    result = mod.apply_import(tarball_path=tarball)

    assert result["status"] == "failed"
    assert result["applied"] is False
    quarantined = Path(result["failed_path"])
    assert quarantined.exists()
    assert quarantined.parent.name not in {"detect_fake_fixed", "detect_fake_fixed_1"}


def test_orphan_previous_final_backup_fails_closed_when_final_is_absent(
    tmp_path: Path,
) -> None:
    target_zarr = tmp_path / "recording_analysis.zarr"
    _write_group(target_zarr)
    family = target_zarr / RUN_FAMILY
    orphan = family / ".incoming" / ".detect_fake_previous_orphan"
    _write_group(orphan, attributes={"sentinel": "recover-me"})
    tarball = _write_artifact(tmp_path, target_zarr=target_zarr)

    result = mod.apply_import(tarball_path=tarball)

    assert result["status"] == "failed"
    assert result["applied"] is False
    assert "orphaned import transaction paths" in "\n".join(result["errors"])
    assert orphan.exists()
    assert not (family / "detect_fake").exists()


def test_exact_timestamp_orphans_are_never_claimed_or_mutated_by_new_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_zarr = tmp_path / "recording_analysis.zarr"
    _write_group(target_zarr)
    family = target_zarr / RUN_FAMILY
    incoming = family / ".incoming"
    previous_final = incoming / ".detect_fake_previous_fixed"
    previous_receipt = incoming / ".detect_fake_previous_receipt_fixed.json"
    pending_receipt = incoming / ".detect_fake_import_receipt_fixed.json.pending"
    _write_group(previous_final, attributes={"sentinel": "orphan-final"})
    previous_receipt.write_bytes(b"orphan-receipt")
    pending_receipt.write_bytes(b"orphan-pending")
    previous_final_hash = tree_hash(previous_final)
    tarball = _write_artifact(tmp_path, target_zarr=target_zarr)
    monkeypatch.setattr(mod, "_utc_now_label", lambda: "fixed")

    result = mod.apply_import(tarball_path=tarball)

    assert result["status"] == "failed"
    assert result["applied"] is False
    assert result["rollback_errors"] == []
    assert tree_hash(previous_final) == previous_final_hash
    assert previous_receipt.read_bytes() == b"orphan-receipt"
    assert pending_receipt.read_bytes() == b"orphan-pending"
    assert not (family / "detect_fake").exists()
    assert not (family / ".imports" / "detect_fake_import_receipt.json").exists()


def test_incoming_path_appearing_after_plan_is_never_quarantined_as_attempt_owned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_zarr = tmp_path / "recording_analysis.zarr"
    _write_group(target_zarr)
    tarball = _write_artifact(tmp_path, target_zarr=target_zarr)
    original_build_import_plan = mod.build_import_plan
    concurrent_payload = b"concurrent-writer"

    def plan_then_create_incoming(**kwargs):
        plan = original_build_import_plan(**kwargs)
        incoming = Path(plan["incoming_path"])
        _write_group(incoming, attributes={"sentinel": "not-this-attempt"})
        (incoming / "payload.bin").write_bytes(concurrent_payload)
        return plan

    monkeypatch.setattr(mod, "build_import_plan", plan_then_create_incoming)

    result = mod.apply_import(tarball_path=tarball)

    incoming = target_zarr / RUN_FAMILY / ".incoming" / "detect_fake"
    assert result["status"] == "failed"
    assert result["applied"] is False
    assert result["rollback_errors"] == []
    assert (incoming / "payload.bin").read_bytes() == concurrent_payload
    assert not (target_zarr / RUN_FAMILY / "detect_fake").exists()
    assert list((target_zarr / RUN_FAMILY / ".failed").iterdir()) == []


def test_concurrent_incoming_and_exact_pending_are_both_left_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_zarr = tmp_path / "recording_analysis.zarr"
    _write_group(target_zarr)
    tarball = _write_artifact(tmp_path, target_zarr=target_zarr)
    original_stamp = mod._stamp_artifact_family_metadata
    concurrent_payload = b"concurrent-writer"
    pending_payload = b"concurrent-pending"

    def stamp_then_publish_concurrent_paths(family: Path) -> None:
        original_stamp(family)
        incoming = family / ".incoming" / "detect_fake"
        _write_group(incoming, attributes={"sentinel": "not-this-attempt"})
        (incoming / "payload.bin").write_bytes(concurrent_payload)
        pending = family / ".incoming" / ".detect_fake_import_receipt_fixed.json.pending"
        pending.write_bytes(pending_payload)

    monkeypatch.setattr(
        mod,
        "_stamp_artifact_family_metadata",
        stamp_then_publish_concurrent_paths,
    )
    monkeypatch.setattr(mod, "_utc_now_label", lambda: "fixed")

    result = mod.apply_import(tarball_path=tarball)

    family = target_zarr / RUN_FAMILY
    incoming = family / ".incoming" / "detect_fake"
    pending = family / ".incoming" / ".detect_fake_import_receipt_fixed.json.pending"
    assert result["status"] == "failed"
    assert result["applied"] is False
    assert result["rollback_errors"] == []
    assert (incoming / "payload.bin").read_bytes() == concurrent_payload
    assert pending.read_bytes() == pending_payload
    assert not (family / "detect_fake").exists()
    assert list((family / ".failed").iterdir()) == []


def test_successful_overwrite_removes_previous_final_and_receipt_backups(
    tmp_path: Path,
) -> None:
    target_zarr = tmp_path / "recording_analysis.zarr"
    _write_group(target_zarr)
    family = target_zarr / RUN_FAMILY
    _write_group(family / "detect_fake", attributes={"sentinel": "old"})
    receipt = family / ".imports" / "detect_fake_import_receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_bytes(b"old-receipt")
    tarball = _write_artifact(tmp_path, target_zarr=target_zarr)

    result = mod.apply_import(tarball_path=tarball, overwrite=True)

    assert result["status"] == "ok"
    assert result["cleanup_warnings"] == []
    transaction_paths = list((family / ".incoming").iterdir())
    assert transaction_paths == []


def test_cleanup_failure_keeps_success_and_recoverable_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_zarr = tmp_path / "recording_analysis.zarr"
    _write_group(target_zarr)
    family = target_zarr / RUN_FAMILY
    _write_group(family / "detect_fake", attributes={"sentinel": "old"})
    receipt = family / ".imports" / "detect_fake_import_receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_bytes(b"old-receipt")
    tarball = _write_artifact(tmp_path, target_zarr=target_zarr)
    original_remove_path = mod._remove_path

    def fail_previous_final_cleanup(path: Path) -> None:
        if "_previous_" in path.name and "previous_receipt" not in path.name:
            raise RuntimeError("injected cleanup failure")
        original_remove_path(path)

    monkeypatch.setattr(mod, "_remove_path", fail_previous_final_cleanup)

    result = mod.apply_import(tarball_path=tarball, overwrite=True)

    assert result["status"] == "ok"
    assert result["applied"] is True
    assert any("previous final" in warning for warning in result["cleanup_warnings"])
    backups = [
        path
        for path in (family / ".incoming").iterdir()
        if "_previous_" in path.name and "previous_receipt" not in path.name
    ]
    assert len(backups) == 1
    assert json.loads((backups[0] / "zarr.json").read_text(encoding="utf-8"))[
        "attributes"
    ]["sentinel"] == "old"


def test_receipt_commit_failure_without_prior_receipt_restores_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_zarr = tmp_path / "recording_analysis.zarr"
    _write_group(target_zarr)
    existing = target_zarr / RUN_FAMILY / "detect_fake"
    _write_group(existing, attributes={"sentinel": "old"})
    before = tree_hash(existing)
    tarball = _write_artifact(tmp_path, target_zarr=target_zarr)
    original_atomic_move = mod._atomic_move

    def fail_receipt_commit(source: Path, destination: Path) -> None:
        if source.name.endswith(".json.pending"):
            raise RuntimeError("injected receipt commit failure")
        original_atomic_move(source, destination)

    monkeypatch.setattr(mod, "_atomic_move", fail_receipt_commit)

    result = mod.apply_import(tarball_path=tarball, overwrite=True)

    receipt = (
        target_zarr
        / RUN_FAMILY
        / ".imports"
        / "detect_fake_import_receipt.json"
    )
    assert result["status"] == "failed"
    assert result["applied"] is False
    assert result["rollback_errors"] == []
    assert tree_hash(existing) == before
    assert not receipt.exists()


def test_tarball_mutation_between_plan_and_apply_fails_before_target_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_zarr = tmp_path / "recording_analysis.zarr"
    _write_group(target_zarr)
    tarball = _write_artifact(tmp_path, target_zarr=target_zarr)
    original_build_import_plan = mod.build_import_plan

    def plan_then_mutate(**kwargs):
        plan = original_build_import_plan(**kwargs)
        with Path(kwargs["tarball_path"]).open("ab") as handle:
            handle.write(b"mutated-after-plan")
        return plan

    monkeypatch.setattr(mod, "build_import_plan", plan_then_mutate)

    result = mod.apply_import(tarball_path=tarball)

    assert result["status"] == "failed"
    assert result["applied"] is False
    assert "source tarball changed after import planning" in "\n".join(result["errors"])
    assert not (target_zarr / RUN_FAMILY / "detect_fake").exists()

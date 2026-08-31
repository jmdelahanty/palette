from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

import fisheye.utils.materialize_eye_gaze_prerequisite_cohort as cohort
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

REPO = Path(__file__).resolve().parents[3]
SUBMIT_SCRIPT = REPO / "scripts/submit_eye_gaze_prerequisites_bsub.sh"


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _task(tmp_path: Path) -> tuple[dict[str, object], Path]:
    recording_id = "recording_001"
    archive = tmp_path / f"{recording_id}_analysis.zarr"
    input_path = archive / "zarr.json"
    _write(input_path, {"node_type": "group", "attributes": {}})
    entry: dict[str, object] = {
        "task_index": 1,
        "recording_id": recording_id,
        "analysis_zarr": str(archive),
        "subject_mask": {
            "bundle_id": "bundle_001",
            "bundle_manifest_payload_digest": "1" * 64,
            "assignment_keypoint_group": "refined_keypoints_runs",
            "assignment_keypoint_run": "refined_001",
            "assignment_success_dataset": "usable_keypoints",
        },
        "canonical_keypoints": {
            "run_name": "keypoints_coordinate_001",
            "manifest_payload_digest": "2" * 64,
            "coordinate_successor_authority_sha256": "3" * 64,
            "active_bundle_authority_sha256": "4" * 64,
        },
        "outputs": {
            "rebinding_run": cohort.REBINDING_RUN,
            "subject_shape_run": cohort.SUBJECT_SHAPE_RUN,
            "eye_angle_run": cohort.EYE_ANGLE_RUN,
        },
        "input_files": [
            {
                "relative_path": "zarr.json",
                "sha256": cohort._sha256_file(input_path),
            }
        ],
        "status": "metadata_ready_for_exhaustive_proof",
    }
    entry["entry_sha256"] = cohort._entry_digest(entry)
    task: dict[str, object] = {
        "schema_id": cohort.TASK_SCHEMA_ID,
        "schema_version": cohort.TASK_SCHEMA_VERSION,
        "created_at_utc": "2026-08-31T00:00:00+00:00",
        "source_chaser_task": {
            "path": "/tmp/source.json",
            "task_sha256": "5" * 64,
            "recording_count": 1,
        },
        "recording_count": 1,
        "entries": [entry],
        "safety": cohort.EXPECTED_SAFETY,
    }
    task["task_sha256"] = cohort._task_digest(task)
    return task, archive


def _clean_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "palette-deployment"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    py = scripts / "py"
    shutil.copy2(REPO / "scripts/py", py)
    os.chmod(py, 0o755)
    (repo / "src").symlink_to(REPO / "src", target_is_directory=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Palette Tests",
            "-c",
            "user.email=palette-tests@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    return repo, commit


def test_task_loader_is_closed_and_digest_bound(tmp_path: Path) -> None:
    task, _archive = _task(tmp_path)

    assert cohort.load_task(task) == task

    expanded = copy.deepcopy(task)
    expanded["entries"][0]["unexpected"] = True
    expanded["entries"][0]["entry_sha256"] = cohort._entry_digest(
        expanded["entries"][0]
    )
    expanded["task_sha256"] = cohort._task_digest(expanded)
    with pytest.raises(ValueError, match="entry is invalid"):
        cohort.load_task(expanded)

    unsafe = copy.deepcopy(task)
    unsafe["entries"][0]["input_files"][0]["relative_path"] = "../zarr.json"
    unsafe["entries"][0]["entry_sha256"] = cohort._entry_digest(unsafe["entries"][0])
    unsafe["task_sha256"] = cohort._task_digest(unsafe)
    with pytest.raises(ValueError, match="unsafe or duplicated"):
        cohort.load_task(unsafe)


def test_input_revalidation_rejects_changed_source_or_existing_target(
    tmp_path: Path,
) -> None:
    task, archive = _task(tmp_path)
    loaded = cohort.load_task(task)
    entry = loaded["entries"][0]

    cohort._revalidate_input_files(entry)

    (archive / "zarr.json").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="input changed"):
        cohort._revalidate_input_files(entry)

    task, archive = _task(tmp_path / "second")
    entry = cohort.load_task(task)["entries"][0]
    target = (
        archive
        / "subject_mask_assignment_keypoint_rebinding_runs"
        / cohort.REBINDING_RUN
    )
    target.mkdir(parents=True)
    with pytest.raises(ValueError, match="target already exists"):
        cohort._revalidate_input_files(entry)


def test_proof_receipt_binds_task_entry_commit_and_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task, _archive = _task(tmp_path)
    loaded = cohort.load_task(task)
    manifest = {
        "schema_id": "palette.subject_mask.assignment_keypoint_rebinding_manifest",
        "schema_version": 1,
        "payload_digest": "6" * 64,
    }
    commit = "a" * 40
    monkeypatch.setattr(cohort, "_git_identity", lambda *_args: commit)
    monkeypatch.setattr(
        cohort,
        "inspect_assignment_keypoint_rebinding",
        lambda **_kwargs: manifest,
    )
    monkeypatch.setattr(
        cohort,
        "validate_assignment_keypoint_rebinding_manifest",
        lambda _manifest: (),
    )
    proof_root = tmp_path / "proofs"

    proof = cohort.prove_one(
        loaded,
        task_index=1,
        palette_repo=tmp_path,
        palette_commit=commit,
        proof_root=proof_root,
        block_rows=8,
    )

    entry = loaded["entries"][0]
    persisted = cohort._load_proof(
        cohort._proof_path(proof_root, entry),
        task=loaded,
        entry=entry,
        palette_commit=commit,
    )
    assert persisted == proof
    assert proof["rebinding_manifest_sha256"] == canonical_json_sha256(manifest)
    assert proof["zarr_writes"] is False

    tampered = copy.deepcopy(proof)
    tampered["palette_commit"] = "b" * 40
    _write(cohort._proof_path(tmp_path / "tampered", entry), tampered)
    with pytest.raises(ValueError, match="proof receipt is invalid"):
        cohort._load_proof(
            cohort._proof_path(tmp_path / "tampered", entry),
            task=loaded,
            entry=entry,
            palette_commit=commit,
        )


def test_materialization_wires_exact_ineligible_candidates_without_acceptance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fisheye.analysis import gaze_convention_validation
    from fisheye.analysis_workflows.materializers import eye_angles, subject_shape

    task, _archive = _task(tmp_path)
    loaded = cohort.load_task(task)
    manifest = {
        "schema_id": "palette.subject_mask.assignment_keypoint_rebinding_manifest",
        "schema_version": 1,
        "payload_digest": "6" * 64,
    }
    commit = "a" * 40
    monkeypatch.setattr(cohort, "_git_identity", lambda *_args: commit)
    monkeypatch.setattr(
        cohort,
        "inspect_assignment_keypoint_rebinding",
        lambda **_kwargs: manifest,
    )
    monkeypatch.setattr(
        cohort,
        "validate_assignment_keypoint_rebinding_manifest",
        lambda _manifest: (),
    )
    proof_root = tmp_path / "proofs"
    cohort.prove_one(
        loaded,
        task_index=1,
        palette_repo=tmp_path,
        palette_commit=commit,
        proof_root=proof_root,
        block_rows=8,
    )

    calls: dict[str, dict[str, object]] = {}

    def fake_publish(**kwargs: object) -> dict[str, object]:
        calls["rebinding"] = kwargs
        return {"status": "complete", "manifest": manifest}

    def fake_shape(*args: object, **kwargs: object) -> dict[str, object]:
        calls["shape"] = {"args": args, **kwargs}
        return {"status": "complete", "kind": "subject_shape"}

    def fake_eye(*args: object, **kwargs: object) -> dict[str, object]:
        calls["eye"] = {"args": args, **kwargs}
        return {"status": "complete", "kind": "eye_angles"}

    def fake_validate(*args: object, **kwargs: object) -> dict[str, object]:
        calls["validation"] = {"args": args, **kwargs}
        Path(str(kwargs["review_png"])).write_bytes(b"review")
        return {"status": "pass", "kind": "numeric_only"}

    monkeypatch.setattr(cohort, "publish_assignment_keypoint_rebinding", fake_publish)
    monkeypatch.setattr(subject_shape, "materialize_subject_shape", fake_shape)
    monkeypatch.setattr(eye_angles, "materialize_eye_angles", fake_eye)
    monkeypatch.setattr(
        gaze_convention_validation,
        "validate_eye_angle_run",
        fake_validate,
    )

    result = cohort.materialize_one(
        loaded,
        task_index=1,
        palette_repo=tmp_path,
        palette_commit=commit,
        proof_root=proof_root,
        scratch_root=tmp_path / "scratch",
        receipt_root=tmp_path / "receipts",
        copy_backend="python",
        num_workers=4,
        block_rows=8,
        apply=True,
    )

    assert calls["shape"]["refined_run"] is None
    assert calls["shape"]["allow_inactive_subject_mask_bundle"] is True
    assert (
        calls["shape"]["assignment_keypoint_rebinding_run_id"] == cohort.REBINDING_RUN
    )
    assert (
        calls["shape"]["storage_profile"]
        == cohort.SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_PROFILE_ID
    )
    assert calls["shape"]["apply"] is True
    assert "stage_selector_eligible" not in calls["shape"]
    assert calls["eye"]["subject_shape_run"] == cohort.SUBJECT_SHAPE_RUN
    assert calls["eye"]["keypoint_run"] == "keypoints_coordinate_001"
    assert (
        calls["eye"]["storage_profile"]
        == cohort.EYE_ANGLE_ACCESS_AWARE_CANDIDATE_PROFILE_ID
    )
    assert calls["eye"]["apply"] is True
    assert calls["validation"]["allow_ineligible_candidate"] is True
    assert result["human_gaze_direction_acceptance"] is False
    assert result["selector_eligible"] is False
    assert result["production_authority"] is False
    assert result["registry_update"] is False
    assert result["selector_activation"] is False


def test_bsub_submitter_renders_pinned_proof_and_materialization_workers(
    tmp_path: Path,
) -> None:
    task, _archive = _task(tmp_path)
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(task), encoding="utf-8")
    repo, commit = _clean_repo(tmp_path)
    run_root = tmp_path / "runs"
    environment = {
        **os.environ,
        "PALETTE_PYTHON": os.environ.get("PALETTE_PYTHON", ""),
    }

    proof = subprocess.run(
        [
            "bash",
            str(SUBMIT_SCRIPT),
            "--phase",
            "prove",
            "--task",
            str(task_path),
            "--palette-repo",
            str(repo),
            "--palette-commit",
            commit,
            "--run-root",
            str(run_root),
            "--run-id",
            "proof_fixture",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=environment,
    )
    assert "mode=dry_run_no_submission" in proof.stdout
    assert "array_indices=1-1" in proof.stdout
    proof_dir = run_root / "eye_gaze_prerequisites_prove_proof_fixture"
    proof_script = (proof_dir / "run_one_recording.sh").read_text(encoding="utf-8")
    assert 'TASK_INDEX="${LSB_JOBINDEX}"' in proof_script
    assert "materialize_eye_gaze_prerequisite_cohort prove-one" in proof_script
    assert f"--palette-commit {commit}" in proof_script

    proof_root = proof_dir / "proofs"
    materialize = subprocess.run(
        [
            "bash",
            str(SUBMIT_SCRIPT),
            "--phase",
            "materialize",
            "--task",
            str(task_path),
            "--palette-repo",
            str(repo),
            "--palette-commit",
            commit,
            "--run-root",
            str(run_root),
            "--proof-root",
            str(proof_root),
            "--array-indices",
            "1",
            "--run-id",
            "materialize_fixture",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=environment,
    )
    assert "mode=dry_run_no_submission" in materialize.stdout
    assert "selected_recording_count=1" in materialize.stdout
    materialize_dir = (
        run_root / "eye_gaze_prerequisites_materialize_materialize_fixture"
    )
    materialize_script = (materialize_dir / "run_one_recording.sh").read_text(
        encoding="utf-8"
    )
    assert "materialize_eye_gaze_prerequisite_cohort run-one" in materialize_script
    assert '--task-index "${TASK_INDEX}"' in materialize_script
    assert "--apply" in materialize_script
    submission = (materialize_dir / "submission.env").read_text(encoding="utf-8")
    assert f"palette_commit={commit}" in submission
    assert "selector_eligible=false" in submission
    assert "production_authority=false" in submission
    assert "registry_update=false" in submission
    assert "selector_activation=false" in submission

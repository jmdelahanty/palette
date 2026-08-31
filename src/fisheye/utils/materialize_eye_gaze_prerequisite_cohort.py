"""Freeze, prove, and materialize historical eye-gaze prerequisites by cohort.

The planner consumes an already frozen composable-chaser cohort task.  It
selects no mutable scientific aliases: each recording must expose exactly one
inactive coordinate subject-mask bundle and exactly one keypoint coordinate
successor bound to the active keypoint-bundle authority.  A separate proof
phase exhaustively compares every assignment-relevant row before any write.

The materialization phase requires that commit-bound proof, rechecks every
frozen metadata file, republishes the proof as an immutable selector-ineligible
Zarr record, and then creates exact subject-shape and eye-angle candidates.
It finishes with numeric gaze-convention validation and a bounded review PNG;
it never accepts the biological direction assumption on a reviewer's behalf.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from fisheye.analysis.eye_angle_storage import (
    EYE_ANGLE_ACCESS_AWARE_CANDIDATE_PROFILE_ID,
)
from fisheye.analysis.subject_shape_storage import (
    SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_PROFILE_ID,
)
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.zarr.assignment_keypoint_rebinding import (
    _assignment_collection_source,
    inspect_assignment_keypoint_rebinding,
    publish_assignment_keypoint_rebinding,
    validate_assignment_keypoint_rebinding_manifest,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.utils.materialize_composable_chaser_successor_cohort import (
    load_cohort_task,
)

TASK_SCHEMA_ID = "palette.eye_gaze_prerequisite_cohort_task"
TASK_SCHEMA_VERSION = 1
PROOF_SCHEMA_ID = "palette.eye_gaze_prerequisite_assignment_proof"
PROOF_SCHEMA_VERSION = 1
RECEIPT_SCHEMA_ID = "palette.eye_gaze_prerequisite_materialization_receipt"
RECEIPT_SCHEMA_VERSION = 1

REBINDING_RUN = "assignment_keypoint_rebinding_goodbatbadbat_gaze_20260831_v3"
SUBJECT_SHAPE_RUN = "subject_shape_goodbatbadbat_gaze_20260831_v3"
EYE_ANGLE_RUN = "eye_angles_goodbatbadbat_gaze_20260831_v3"

EXPECTED_SAFETY = {
    "selector_eligible": False,
    "production_authority": False,
    "registry_update": False,
    "selector_activation": False,
    "one_writer_per_analysis_zarr": True,
    "human_gaze_direction_acceptance": False,
}


class EyeGazePrerequisiteCohortError(ValueError):
    """Raised when cohort prerequisite work cannot remain fail-closed."""


def _fail(message: str) -> None:
    raise EyeGazePrerequisiteCohortError(message)


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be one mapping.")
    return value


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        _fail(f"{field} must be non-empty normalized text.")
    return value


def _digest(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{field} must be one lowercase SHA-256 digest.")
    return value


def _run_name(value: object, *, field: str) -> str:
    name = _text(value, field=field)
    if (
        "/" in name
        or "\\" in name
        or name in {"latest", "latest_complete", "selected", "current", ".", ".."}
        or any(character.isspace() for character in name)
    ):
        _fail(f"{field} must be one exact immutable child name.")
    return name


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata(group: Path, *, field: str) -> tuple[dict[str, Any], str]:
    path = group / "zarr.json"
    if not path.is_file():
        _fail(f"{field} metadata is absent: {path}")
    try:
        document = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise EyeGazePrerequisiteCohortError(
            f"{field} metadata is unreadable: {path}"
        ) from exc
    attrs = _mapping(document.get("attributes"), field=f"{field} attributes")
    return dict(attrs), _sha256_file(path)


def _children(parent: Path) -> tuple[Path, ...]:
    if not parent.is_dir():
        return ()
    return tuple(
        sorted(
            child
            for child in parent.iterdir()
            if child.is_dir() and (child / "zarr.json").is_file()
        )
    )


def _manifest(value: object, *, field: str) -> Mapping[str, Any]:
    manifest = _mapping(value, field=field)
    payload = _mapping(manifest.get("payload"), field=f"{field} payload")
    if manifest.get("payload_digest") != canonical_json_sha256(payload):
        _fail(f"{field} payload digest is stale.")
    return manifest


def _file_binding(archive: Path, path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(archive)
    except ValueError as exc:
        raise EyeGazePrerequisiteCohortError(
            f"Frozen input leaves its analysis archive: {resolved}"
        ) from exc
    if not resolved.is_file():
        _fail(f"Frozen input is absent: {resolved}")
    return {"relative_path": relative.as_posix(), "sha256": _sha256_file(resolved)}


def _candidate_keypoint_run(
    archive: Path,
    *,
    authority: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    authority_sha = canonical_json_sha256(authority)
    candidates: list[tuple[str, dict[str, Any]]] = []
    for child in _children(archive / "keypoints_runs"):
        attrs, _metadata_sha = _metadata(child, field="keypoint run")
        record = attrs.get("coordinate_successor_authority")
        payload = record.get("payload") if isinstance(record, Mapping) else None
        source = (
            payload.get("source_authority") if isinstance(payload, Mapping) else None
        )
        successor = payload.get("successor") if isinstance(payload, Mapping) else None
        manifest = attrs.get("run_manifest")
        if (
            attrs.get("palette_run_completion_status") == "complete"
            and attrs.get("stage_selector_eligible") is False
            and isinstance(record, Mapping)
            and attrs.get("coordinate_successor_authority_sha256")
            == canonical_json_sha256(record)
            and isinstance(source, Mapping)
            and source.get("record") == authority
            and source.get("record_sha256") == authority_sha
            and isinstance(successor, Mapping)
            and successor.get("run_path") == f"keypoints_runs/{child.name}"
            and isinstance(manifest, Mapping)
        ):
            _manifest(manifest, field="coordinate-successor run manifest")
            candidates.append((child.name, attrs))
    if len(candidates) != 1:
        _fail(
            "Recording must expose exactly one complete ineligible coordinate "
            f"successor bound to the active keypoint bundle; observed {len(candidates)}."
        )
    return candidates[0]


def _candidate_subject_mask_bundle(
    archive: Path,
    *,
    recording_id: str,
    keypoint_authority: Mapping[str, Any],
) -> tuple[str, dict[str, Any], tuple[str, str, str], Mapping[str, Any]]:
    candidates: list[
        tuple[str, dict[str, Any], tuple[str, str, str], Mapping[str, Any]]
    ] = []
    for child in _children(archive / "subject_mask_bundle_runs"):
        attrs, _metadata_sha = _metadata(child, field="subject-mask bundle")
        if (
            attrs.get("palette_run_completion_status") != "complete"
            or attrs.get("subject_mask_bundle_selector_eligible") is not False
        ):
            continue
        manifest = _manifest(attrs.get("run_manifest"), field="subject-mask bundle")
        payload = _mapping(manifest["payload"], field="subject-mask bundle payload")
        if payload.get("recording_identity") != recording_id:
            continue
        members = _mapping(payload.get("members"), field="subject-mask members")
        refined = _mapping(members.get("refined"), field="refined mask member")
        refined_path = _text(refined.get("run_path"), field="refined mask path")
        refined_attrs, _refined_sha = _metadata(
            archive / refined_path,
            field="refined mask member",
        )
        refined_manifest = _manifest(
            refined_attrs.get("run_manifest"), field="refined mask manifest"
        )
        refined_payload = _mapping(
            refined_manifest["payload"], field="refined mask payload"
        )
        dependencies = _mapping(
            refined_payload.get("coordinate_dependencies"),
            field="refined mask coordinate dependencies",
        )
        document = _mapping(
            dependencies.get("document"), field="refined mask coordinate document"
        )
        collection = _mapping(
            document.get("assignment_keypoints"),
            field="assignment-keypoint collection",
        )
        profile = _assignment_collection_source(collection)
        if profile[0] == "refined_keypoints_runs":
            active_members = _mapping(
                keypoint_authority.get("members"), field="active keypoint members"
            )
            active_refined = _mapping(
                active_members.get("refined_keypoints"),
                field="active refined-keypoint member",
            )
            if active_refined.get("run_path") != f"{profile[0]}/{profile[1]}":
                continue
        candidates.append((child.name, attrs, profile, manifest))
    if len(candidates) != 1:
        _fail(
            "Recording must expose exactly one compatible inactive subject-mask "
            f"bundle; observed {len(candidates)}."
        )
    return candidates[0]


def _input_files(
    archive: Path,
    *,
    root_attrs: Mapping[str, Any],
    bundle_manifest: Mapping[str, Any],
    assignment_profile: tuple[str, str, str],
    keypoint_run: str,
) -> list[dict[str, str]]:
    paths = {archive / "zarr.json"}
    payload = bundle_manifest["payload"]
    paths.add(
        archive / "subject_mask_bundle_runs" / str(payload["bundle_id"]) / "zarr.json"
    )
    members = _mapping(payload.get("members"), field="subject-mask members")
    for role in ("raw", "refined"):
        member = _mapping(members.get(role), field=f"subject-mask {role} member")
        member_path = archive / _text(member.get("run_path"), field=f"{role} path")
        paths.add(member_path / "zarr.json")
        member_attrs, _sha = _metadata(member_path, field=f"{role} mask member")
        manifest = _manifest(member_attrs.get("run_manifest"), field=f"{role} manifest")
        source = _mapping(manifest["payload"].get("source"), field=f"{role} source")
        receipt = _mapping(
            source.get("validation_receipt"), field=f"{role} validation receipt"
        )
        paths.add(archive / _text(receipt.get("relative_path"), field="receipt path"))
    coordinate = _mapping(
        payload.get("cross_binding", {}).get("coordinate_contract"),
        field="mask bundle coordinate contract",
    )
    crop = _mapping(coordinate.get("crop"), field="mask bundle crop")
    paths.add(archive / _text(crop.get("run_path"), field="crop path") / "zarr.json")
    paths.add(archive / assignment_profile[0] / assignment_profile[1] / "zarr.json")
    active_members = _mapping(
        root_attrs.get("keypoint_bundle_authority", {}).get("members"),
        field="active keypoint members",
    )
    for role in ("raw_keypoints", "refined_keypoints"):
        member = _mapping(active_members.get(role), field=f"active {role}")
        paths.add(
            archive / _text(member.get("run_path"), field=f"{role} path") / "zarr.json"
        )
    paths.add(archive / "keypoints_runs" / keypoint_run / "zarr.json")
    return [
        _file_binding(archive, path)
        for path in sorted(paths, key=lambda value: str(value))
    ]


def _entry_digest(entry: Mapping[str, Any]) -> str:
    return canonical_json_sha256(
        {key: value for key, value in entry.items() if key != "entry_sha256"}
    )


def _task_digest(task: Mapping[str, Any]) -> str:
    return canonical_json_sha256(
        {key: value for key, value in task.items() if key != "task_sha256"}
    )


def plan_cohort(source_task: str | Path) -> dict[str, Any]:
    """Build one immutable metadata-bound prerequisite cohort task."""

    source_path = Path(source_task).expanduser().resolve()
    source = load_cohort_task(source_path)
    entries: list[dict[str, Any]] = []
    for index, raw in enumerate(source["entries"], start=1):
        source_entry = _mapping(raw, field="source cohort entry")
        archive = (
            Path(_text(source_entry.get("analysis_zarr"), field="analysis Zarr"))
            .expanduser()
            .resolve()
        )
        recording_id = _text(source_entry.get("recording_id"), field="recording ID")
        root_attrs, _root_sha = _metadata(archive, field="analysis root")
        if (
            root_attrs.get("recording_id") != recording_id
            or archive.name != f"{recording_id}_analysis.zarr"
        ):
            _fail("Source task recording identity differs from its analysis archive.")
        authority = _mapping(
            root_attrs.get("keypoint_bundle_authority"),
            field="active keypoint-bundle authority",
        )
        keypoint_run, keypoint_attrs = _candidate_keypoint_run(
            archive,
            authority=authority,
        )
        bundle_id, _bundle_attrs, profile, bundle_manifest = (
            _candidate_subject_mask_bundle(
                archive,
                recording_id=recording_id,
                keypoint_authority=authority,
            )
        )
        output_paths = {
            "rebinding": f"subject_mask_assignment_keypoint_rebinding_runs/{REBINDING_RUN}",
            "subject_shape": f"analysis/subject_shape_runs/{SUBJECT_SHAPE_RUN}",
            "eye_angles": f"analysis/eye_angle_runs/{EYE_ANGLE_RUN}",
        }
        present = [path for path in output_paths.values() if (archive / path).exists()]
        if present:
            _fail(
                f"Prerequisite output names already exist for {recording_id}: {present}."
            )
        keypoint_manifest = _manifest(
            keypoint_attrs.get("run_manifest"), field="keypoint successor manifest"
        )
        entry = json_attr_safe(
            {
                "task_index": index,
                "recording_id": recording_id,
                "analysis_zarr": str(archive),
                "subject_mask": {
                    "bundle_id": bundle_id,
                    "bundle_manifest_payload_digest": bundle_manifest["payload_digest"],
                    "assignment_keypoint_group": profile[0],
                    "assignment_keypoint_run": profile[1],
                    "assignment_success_dataset": profile[2],
                },
                "canonical_keypoints": {
                    "run_name": keypoint_run,
                    "manifest_payload_digest": keypoint_manifest["payload_digest"],
                    "coordinate_successor_authority_sha256": keypoint_attrs[
                        "coordinate_successor_authority_sha256"
                    ],
                    "active_bundle_authority_sha256": canonical_json_sha256(authority),
                },
                "outputs": {
                    "rebinding_run": REBINDING_RUN,
                    "subject_shape_run": SUBJECT_SHAPE_RUN,
                    "eye_angle_run": EYE_ANGLE_RUN,
                },
                "input_files": _input_files(
                    archive,
                    root_attrs=root_attrs,
                    bundle_manifest=bundle_manifest,
                    assignment_profile=profile,
                    keypoint_run=keypoint_run,
                ),
                "status": "metadata_ready_for_exhaustive_proof",
            }
        )
        entry["entry_sha256"] = _entry_digest(entry)
        entries.append(entry)
    task = json_attr_safe(
        {
            "schema_id": TASK_SCHEMA_ID,
            "schema_version": TASK_SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_chaser_task": {
                "path": str(source_path),
                "task_sha256": source["task_sha256"],
                "recording_count": source["recording_count"],
            },
            "recording_count": len(entries),
            "entries": entries,
            "safety": EXPECTED_SAFETY,
        }
    )
    task["task_sha256"] = _task_digest(task)
    return task


def load_task(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, Mapping):
        task = dict(source)
    else:
        path = Path(source).expanduser().resolve()
        task = dict(_mapping(json.loads(path.read_bytes()), field="cohort task"))
    expected_fields = {
        "schema_id",
        "schema_version",
        "created_at_utc",
        "source_chaser_task",
        "recording_count",
        "entries",
        "safety",
        "task_sha256",
    }
    source_binding = task.get("source_chaser_task")
    if (
        set(task) != expected_fields
        or task.get("schema_id") != TASK_SCHEMA_ID
        or task.get("schema_version") != TASK_SCHEMA_VERSION
        or task.get("safety") != EXPECTED_SAFETY
        or task.get("task_sha256") != _task_digest(task)
        or not isinstance(task.get("created_at_utc"), str)
        or not task["created_at_utc"]
        or not isinstance(source_binding, Mapping)
        or set(source_binding) != {"path", "task_sha256", "recording_count"}
    ):
        _fail("Eye-gaze prerequisite cohort task envelope is invalid or stale.")
    _text(source_binding.get("path"), field="source chaser task path")
    _digest(source_binding.get("task_sha256"), field="source chaser task digest")
    if (
        type(source_binding.get("recording_count")) is not int
        or source_binding["recording_count"] <= 0
    ):
        _fail("Source chaser task recording count is invalid.")
    entries = task.get("entries")
    recording_count = task.get("recording_count")
    if (
        type(recording_count) is not int
        or recording_count <= 0
        or not isinstance(entries, list)
        or recording_count != len(entries)
        or source_binding["recording_count"] != recording_count
    ):
        _fail("Eye-gaze prerequisite cohort dimensions are invalid.")
    archives: set[str] = set()
    for index, raw in enumerate(entries, start=1):
        entry = _mapping(raw, field="cohort entry")
        expected_entry_fields = {
            "task_index",
            "recording_id",
            "analysis_zarr",
            "subject_mask",
            "canonical_keypoints",
            "outputs",
            "input_files",
            "status",
            "entry_sha256",
        }
        if (
            set(entry) != expected_entry_fields
            or type(entry.get("task_index")) is not int
            or entry.get("task_index") != index
            or entry.get("entry_sha256") != _entry_digest(entry)
            or entry.get("status") != "metadata_ready_for_exhaustive_proof"
        ):
            _fail("Eye-gaze prerequisite cohort entry is invalid or stale.")
        recording_id = _text(entry.get("recording_id"), field="recording ID")
        archive = _text(entry.get("analysis_zarr"), field="analysis Zarr")
        archive_path = Path(archive)
        if (
            not archive_path.is_absolute()
            or archive != str(archive_path.resolve())
            or archive_path.name != f"{recording_id}_analysis.zarr"
        ):
            _fail("Cohort entry analysis path and recording identity disagree.")
        if archive in archives:
            _fail("Cohort task contains duplicate analysis archives.")
        archives.add(archive)
        subject = _mapping(entry.get("subject_mask"), field="subject-mask source")
        if set(subject) != {
            "bundle_id",
            "bundle_manifest_payload_digest",
            "assignment_keypoint_group",
            "assignment_keypoint_run",
            "assignment_success_dataset",
        }:
            _fail("Subject-mask source binding fields are not exact.")
        _run_name(subject.get("bundle_id"), field="subject-mask bundle")
        _digest(
            subject.get("bundle_manifest_payload_digest"),
            field="subject-mask bundle digest",
        )
        assignment = (
            subject.get("assignment_keypoint_group"),
            subject.get("assignment_success_dataset"),
        )
        if assignment not in {
            ("keypoints_runs", "detection_success"),
            ("refined_keypoints_runs", "usable_keypoints"),
        }:
            _fail("Subject-mask assignment profile is unsupported.")
        _run_name(
            subject.get("assignment_keypoint_run"),
            field="assignment keypoint run",
        )
        keypoints = _mapping(
            entry.get("canonical_keypoints"), field="canonical keypoint source"
        )
        if set(keypoints) != {
            "run_name",
            "manifest_payload_digest",
            "coordinate_successor_authority_sha256",
            "active_bundle_authority_sha256",
        }:
            _fail("Canonical keypoint source fields are not exact.")
        _run_name(keypoints.get("run_name"), field="canonical keypoint run")
        for field in (
            "manifest_payload_digest",
            "coordinate_successor_authority_sha256",
            "active_bundle_authority_sha256",
        ):
            _digest(keypoints.get(field), field=f"canonical keypoint {field}")
        outputs = _mapping(entry.get("outputs"), field="output identities")
        if outputs != {
            "rebinding_run": REBINDING_RUN,
            "subject_shape_run": SUBJECT_SHAPE_RUN,
            "eye_angle_run": EYE_ANGLE_RUN,
        }:
            _fail("Cohort output identities differ from the closed task version.")
        bindings = entry.get("input_files")
        if not isinstance(bindings, list) or not bindings:
            _fail("Cohort input-file bindings are absent.")
        relative_paths: set[str] = set()
        for raw_binding in bindings:
            binding = _mapping(raw_binding, field="input-file binding")
            if set(binding) != {"relative_path", "sha256"}:
                _fail("Input-file binding fields are not exact.")
            relative = _text(binding.get("relative_path"), field="input relative path")
            if (
                relative.startswith("/")
                or "\\" in relative
                or ".." in Path(relative).parts
                or relative != Path(relative).as_posix()
                or relative in relative_paths
            ):
                _fail("Input-file paths are unsafe or duplicated.")
            relative_paths.add(relative)
            _digest(binding.get("sha256"), field="input-file digest")
    return task


def _entry(task: Mapping[str, Any], task_index: int) -> Mapping[str, Any]:
    entries = task["entries"]
    if task_index < 1 or task_index > len(entries):
        _fail(f"Task index {task_index} is outside the frozen cohort.")
    return _mapping(entries[task_index - 1], field="cohort entry")


def _revalidate_input_files(entry: Mapping[str, Any]) -> None:
    archive = Path(_text(entry.get("analysis_zarr"), field="analysis Zarr"))
    bindings = entry.get("input_files")
    if not isinstance(bindings, list) or not bindings:
        _fail("Frozen input-file inventory is absent.")
    for raw in bindings:
        binding = _mapping(raw, field="input-file binding")
        relative = _text(binding.get("relative_path"), field="input relative path")
        if relative.startswith("/") or ".." in Path(relative).parts:
            _fail("Frozen input relative path leaves its archive.")
        path = archive / relative
        if not path.is_file() or _sha256_file(path) != binding.get("sha256"):
            _fail(f"Frozen prerequisite input changed: {relative}")
    outputs = _mapping(entry.get("outputs"), field="output identities")
    for family, name in (
        ("subject_mask_assignment_keypoint_rebinding_runs", outputs["rebinding_run"]),
        ("analysis/subject_shape_runs", outputs["subject_shape_run"]),
        ("analysis/eye_angle_runs", outputs["eye_angle_run"]),
    ):
        if (archive / family / str(name)).exists():
            _fail(f"Frozen prerequisite target already exists: {family}/{name}")


def _git_identity(repository: Path, expected_commit: str) -> str:
    if len(expected_commit) != 40 or any(
        character not in "0123456789abcdef" for character in expected_commit
    ):
        _fail("Expected Palette commit must be one full lowercase Git SHA.")
    repo = repository.expanduser().resolve()
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != expected_commit or dirty:
        _fail("Palette execution repository is not the exact clean requested commit.")
    return commit


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(json_attr_safe(dict(value)), indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)


def _proof_path(root: Path, entry: Mapping[str, Any]) -> Path:
    return root / f"{int(entry['task_index']):03d}_{entry['recording_id']}.proof.json"


def prove_one(
    task: Mapping[str, Any],
    *,
    task_index: int,
    palette_repo: Path,
    palette_commit: str,
    proof_root: Path,
    block_rows: int,
) -> dict[str, Any]:
    if type(block_rows) is not int or block_rows <= 0:
        _fail("Proof block_rows must be one positive integer.")
    entry = _entry(task, task_index)
    commit = _git_identity(palette_repo, palette_commit)
    _revalidate_input_files(entry)
    archive = Path(entry["analysis_zarr"])
    subject = _mapping(entry["subject_mask"], field="subject-mask source")
    keypoints = _mapping(entry["canonical_keypoints"], field="keypoint source")
    outputs = _mapping(entry["outputs"], field="output identities")
    manifest = inspect_assignment_keypoint_rebinding(
        analysis_zarr=archive,
        subject_mask_bundle_id=str(subject["bundle_id"]),
        keypoint_run_id=str(keypoints["run_name"]),
        rebinding_run_id=str(outputs["rebinding_run"]),
        block_rows=block_rows,
    )
    payload = json_attr_safe(
        {
            "schema_id": PROOF_SCHEMA_ID,
            "schema_version": PROOF_SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "task_sha256": task["task_sha256"],
            "entry_sha256": entry["entry_sha256"],
            "task_index": task_index,
            "recording_id": entry["recording_id"],
            "palette_commit": commit,
            "input_files_revalidated": True,
            "zarr_writes": False,
            "rebinding_manifest": manifest,
            "rebinding_manifest_sha256": canonical_json_sha256(manifest),
        }
    )
    result = {**payload, "receipt_sha256": canonical_json_sha256(payload)}
    _write_exclusive(_proof_path(proof_root, entry), result)
    return result


def _load_proof(
    path: Path,
    *,
    task: Mapping[str, Any],
    entry: Mapping[str, Any],
    palette_commit: str,
) -> dict[str, Any]:
    proof = dict(_mapping(json.loads(path.read_bytes()), field="proof receipt"))
    body = dict(proof)
    persisted = body.pop("receipt_sha256", None)
    manifest = proof.get("rebinding_manifest")
    manifest_errors = (
        validate_assignment_keypoint_rebinding_manifest(manifest)
        if isinstance(manifest, Mapping)
        else ("assignment rebinding manifest is absent",)
    )
    expected_fields = {
        "schema_id",
        "schema_version",
        "created_at_utc",
        "task_sha256",
        "entry_sha256",
        "task_index",
        "recording_id",
        "palette_commit",
        "input_files_revalidated",
        "zarr_writes",
        "rebinding_manifest",
        "rebinding_manifest_sha256",
        "receipt_sha256",
    }
    if (
        set(proof) != expected_fields
        or proof.get("schema_id") != PROOF_SCHEMA_ID
        or proof.get("schema_version") != PROOF_SCHEMA_VERSION
        or persisted != canonical_json_sha256(body)
        or proof.get("task_sha256") != task["task_sha256"]
        or proof.get("entry_sha256") != entry["entry_sha256"]
        or proof.get("task_index") != entry["task_index"]
        or proof.get("recording_id") != entry["recording_id"]
        or proof.get("palette_commit") != palette_commit
        or proof.get("input_files_revalidated") is not True
        or proof.get("zarr_writes") is not False
        or not isinstance(manifest, Mapping)
        or bool(manifest_errors)
        or proof.get("rebinding_manifest_sha256") != canonical_json_sha256(manifest)
    ):
        _fail("Assignment-keypoint proof receipt is invalid or belongs elsewhere.")
    return proof


def materialize_one(
    task: Mapping[str, Any],
    *,
    task_index: int,
    palette_repo: Path,
    palette_commit: str,
    proof_root: Path,
    scratch_root: Path,
    receipt_root: Path,
    copy_backend: str,
    num_workers: int,
    block_rows: int,
    apply: bool,
) -> dict[str, Any]:
    from fisheye.analysis.gaze_convention_validation import validate_eye_angle_run
    from fisheye.analysis_workflows.materializers.eye_angles import (
        materialize_eye_angles,
    )
    from fisheye.analysis_workflows.materializers.subject_shape import (
        materialize_subject_shape,
    )

    if copy_backend not in {"rsync", "python"}:
        _fail("Materialization copy backend is unsupported.")
    if type(num_workers) is not int or num_workers <= 0:
        _fail("Materialization num_workers must be one positive integer.")
    if type(block_rows) is not int or block_rows <= 0:
        _fail("Materialization block_rows must be one positive integer.")
    entry = _entry(task, task_index)
    commit = _git_identity(palette_repo, palette_commit)
    _revalidate_input_files(entry)
    proof = _load_proof(
        _proof_path(proof_root, entry),
        task=task,
        entry=entry,
        palette_commit=commit,
    )
    archive = Path(entry["analysis_zarr"])
    subject = _mapping(entry["subject_mask"], field="subject-mask source")
    keypoints = _mapping(entry["canonical_keypoints"], field="keypoint source")
    outputs = _mapping(entry["outputs"], field="output identities")
    inspected = inspect_assignment_keypoint_rebinding(
        analysis_zarr=archive,
        subject_mask_bundle_id=str(subject["bundle_id"]),
        keypoint_run_id=str(keypoints["run_name"]),
        rebinding_run_id=str(outputs["rebinding_run"]),
        block_rows=block_rows,
    )
    if inspected != proof["rebinding_manifest"]:
        _fail("Live exhaustive rebinding proof differs from the frozen proof receipt.")
    if not apply:
        return {
            "status": "ready",
            "mode": "dry_run",
            "task_index": task_index,
            "recording_id": entry["recording_id"],
            "zarr_writes": False,
        }

    scratch = scratch_root.expanduser().resolve()
    receipt_dir = receipt_root.expanduser().resolve() / str(entry["recording_id"])
    review_png = receipt_dir / "gaze_convention_review.png"
    for path in (
        receipt_dir / "subject_shape_result.json",
        receipt_dir / "eye_angle_result.json",
        receipt_dir / "gaze_convention_numeric_validation.json",
        receipt_dir / "materialization_receipt.json",
        review_png,
    ):
        if path.exists():
            _fail(f"Refusing to replace prerequisite output receipt: {path}")

    rebinding = publish_assignment_keypoint_rebinding(
        analysis_zarr=archive,
        subject_mask_bundle_id=str(subject["bundle_id"]),
        keypoint_run_id=str(keypoints["run_name"]),
        rebinding_run_id=str(outputs["rebinding_run"]),
        block_rows=block_rows,
    )
    if rebinding.get("status") != "complete" or rebinding.get("manifest") != inspected:
        _fail("Published rebinding differs from the exhaustive proof.")
    shape = materialize_subject_shape(
        archive,
        scratch_root=scratch / "subject_shape",
        refined_run=None,
        subject_mask_bundle_id=str(subject["bundle_id"]),
        allow_inactive_subject_mask_bundle=True,
        assignment_keypoint_rebinding_run_id=str(outputs["rebinding_run"]),
        run_name=str(outputs["subject_shape_run"]),
        storage_profile=SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_PROFILE_ID,
        execution_backend="dask_worker_chunks",
        scheduler="processes",
        num_workers=num_workers,
        native_threads=1,
        copy_backend=copy_backend,
        apply=True,
    )
    if not isinstance(shape, Mapping) or shape.get("status") != "complete":
        _fail("Subject-shape candidate did not complete exactly.")
    shape_publish = _mapping(
        shape.get("publish"),
        field="subject-shape candidate publication result",
    )
    subject_shape_candidate_owner = _text(
        shape_publish.get("publication_owner_uuid"),
        field="subject-shape candidate publication owner",
    )
    _write_exclusive(receipt_dir / "subject_shape_result.json", shape)
    eye = materialize_eye_angles(
        archive,
        scratch_root=scratch / "eye_angles",
        subject_shape_run=str(outputs["subject_shape_run"]),
        keypoint_run=str(keypoints["run_name"]),
        run_name=str(outputs["eye_angle_run"]),
        subject_shape_candidate_owner=subject_shape_candidate_owner,
        storage_profile=EYE_ANGLE_ACCESS_AWARE_CANDIDATE_PROFILE_ID,
        execution_backend="serial_driver",
        scheduler="single-threaded",
        num_workers=1,
        native_threads=1,
        copy_backend=copy_backend,
        apply=True,
    )
    if not isinstance(eye, Mapping) or eye.get("status") != "complete":
        _fail("Eye-angle candidate did not complete exactly.")
    _write_exclusive(receipt_dir / "eye_angle_result.json", eye)
    numeric = validate_eye_angle_run(
        archive,
        eye_angle_run=str(outputs["eye_angle_run"]),
        review_png=review_png,
        allow_ineligible_candidate=True,
    )
    if not isinstance(numeric, Mapping) or numeric.get("status") != "pass":
        _fail("Numeric gaze-convention validation failed after materialization.")
    if not review_png.is_file() or review_png.stat().st_size <= 0:
        _fail("Numeric gaze-convention validation did not produce a review PNG.")
    _write_exclusive(
        receipt_dir / "gaze_convention_numeric_validation.json",
        numeric,
    )
    body = json_attr_safe(
        {
            "schema_id": RECEIPT_SCHEMA_ID,
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "status": "complete",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "task_sha256": task["task_sha256"],
            "entry_sha256": entry["entry_sha256"],
            "task_index": task_index,
            "recording_id": entry["recording_id"],
            "palette_commit": commit,
            "rebinding_manifest_sha256": canonical_json_sha256(inspected),
            "subject_shape_result_sha256": canonical_json_sha256(shape),
            "eye_angle_result_sha256": canonical_json_sha256(eye),
            "numeric_validation_sha256": canonical_json_sha256(numeric),
            "review_png": str(review_png),
            "review_png_sha256": _sha256_file(review_png),
            "human_gaze_direction_acceptance": False,
            "selector_eligible": False,
            "production_authority": False,
            "registry_update": False,
            "selector_activation": False,
        }
    )
    result = {**body, "receipt_sha256": canonical_json_sha256(body)}
    _write_exclusive(receipt_dir / "materialization_receipt.json", result)
    return result


def _write_task(path: Path, task: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to replace cohort task: {path}")
    _write_exclusive(path, task)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("source_task", type=Path)
    plan.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("task", type=Path)
    for name in ("prove-one", "run-one"):
        child = subparsers.add_parser(name)
        child.add_argument("task", type=Path)
        child.add_argument("--task-index", type=int, required=True)
        child.add_argument("--palette-repo", type=Path, required=True)
        child.add_argument("--palette-commit", required=True)
        child.add_argument("--proof-root", type=Path, required=True)
        child.add_argument("--block-rows", type=int, default=65_536)
    run = subparsers.choices["run-one"]
    run.add_argument("--scratch-root", type=Path, required=True)
    run.add_argument("--receipt-root", type=Path, required=True)
    run.add_argument("--copy-backend", choices=("rsync", "python"), default="rsync")
    run.add_argument("--num-workers", type=int, default=8)
    run.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "plan":
        task = plan_cohort(args.source_task)
        _write_task(args.output.expanduser().resolve(), task)
        result: Mapping[str, Any] = {
            "status": "planned",
            "output": str(args.output.expanduser().resolve()),
            "recording_count": task["recording_count"],
            "task_sha256": task["task_sha256"],
        }
    elif args.command == "validate":
        task = load_task(args.task)
        result = {
            "status": "valid",
            "recording_count": task["recording_count"],
            "task_sha256": task["task_sha256"],
        }
    elif args.command == "prove-one":
        task = load_task(args.task)
        result = prove_one(
            task,
            task_index=args.task_index,
            palette_repo=args.palette_repo,
            palette_commit=args.palette_commit,
            proof_root=args.proof_root,
            block_rows=args.block_rows,
        )
    else:
        task = load_task(args.task)
        result = materialize_one(
            task,
            task_index=args.task_index,
            palette_repo=args.palette_repo,
            palette_commit=args.palette_commit,
            proof_root=args.proof_root,
            scratch_root=args.scratch_root,
            receipt_root=args.receipt_root,
            copy_backend=args.copy_backend,
            num_workers=args.num_workers,
            block_rows=args.block_rows,
            apply=bool(args.apply),
        )
    print(json.dumps(json_attr_safe(dict(result)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EYE_ANGLE_RUN",
    "REBINDING_RUN",
    "SUBJECT_SHAPE_RUN",
    "load_task",
    "materialize_one",
    "plan_cohort",
    "prove_one",
]

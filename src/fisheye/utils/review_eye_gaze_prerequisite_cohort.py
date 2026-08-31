"""Freeze and apply explicit human review of cohort gaze conventions.

The prerequisite materializer intentionally stops before accepting the
biological direction of a directionless ellipse axis.  This module bridges
that deliberate gate without resolving selectors or mutating an analysis
archive:

* ``plan`` closes the exact materialization receipts, numeric validations, and
  review PNGs into one immutable review task;
* ``template`` creates a decision document with every entry still pending;
* ``accept`` requires an explicit decision and exact PNG digest for every
  recording before exclusively publishing per-run convention receipts and the
  binding list consumed by the composable chaser cohort planner; and
* ``validate`` reopens the whole accepted bundle and fails closed on drift.

No command infers or supplies a human decision.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from fisheye.analysis_workflows.eye_gaze_source_handle import (
    build_gaze_convention_review_receipt,
    validate_gaze_convention_review_receipt,
)
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.utils.materialize_eye_gaze_prerequisite_cohort import (
    EXPECTED_SAFETY as PREREQUISITE_SAFETY,
    RECEIPT_SCHEMA_ID as MATERIALIZATION_RECEIPT_SCHEMA_ID,
    RECEIPT_SCHEMA_VERSION as MATERIALIZATION_RECEIPT_SCHEMA_VERSION,
    load_task as load_prerequisite_task,
)


REVIEW_TASK_SCHEMA_ID = "palette.eye_gaze_convention_review_cohort_task"
REVIEW_TASK_SCHEMA_VERSION = 1
DECISIONS_SCHEMA_ID = "palette.eye_gaze_convention_review_cohort_decisions"
DECISIONS_SCHEMA_VERSION = 1
ACCEPTANCE_SCHEMA_ID = "palette.eye_gaze_convention_review_cohort_acceptance"
ACCEPTANCE_SCHEMA_VERSION = 1

PENDING = "pending_human_biological_direction_review"
ACCEPTED = "accepted"
REJECTED = "rejected"
EYE_RUN_PARENT = "analysis/eye_angle_runs"

REVIEW_SAFETY = {
    "human_decision_inferred": False,
    "selector_eligible": False,
    "production_authority": False,
    "registry_update": False,
    "selector_activation": False,
    "analysis_zarr_mutation": False,
}

_MATERIALIZATION_RECEIPT_FIELDS = {
    "schema_id",
    "schema_version",
    "status",
    "completed_at_utc",
    "task_sha256",
    "entry_sha256",
    "task_index",
    "recording_id",
    "palette_commit",
    "rebinding_manifest_sha256",
    "subject_shape_result_sha256",
    "eye_angle_result_sha256",
    "numeric_validation_sha256",
    "review_png",
    "review_png_sha256",
    "human_gaze_direction_acceptance",
    "selector_eligible",
    "production_authority",
    "registry_update",
    "selector_activation",
    "receipt_sha256",
}

_REVIEW_ENTRY_FIELDS = {
    "task_index",
    "recording_id",
    "analysis_zarr",
    "eye_angle_run",
    "eye_angle_run_path",
    "eye_channel_variant",
    "source_eye_logical_sha256",
    "materialization_palette_commit",
    "materialization_receipt",
    "subject_shape_result",
    "eye_angle_result",
    "numeric_validation",
    "review_png",
    "review_status",
}
_FILE_BINDING_FIELDS = {"path", "file_sha256", "document_sha256"}
_RECEIPT_BINDING_FIELDS = {"path", "file_sha256", "receipt_sha256"}
_PNG_BINDING_FIELDS = {"path", "file_sha256", "review_row_indices"}
_ACCEPTANCE_FIELDS = {
    "schema_id",
    "schema_version",
    "status",
    "review_task_sha256",
    "decisions_sha256",
    "reviewer",
    "reviewed_at_utc",
    "recording_count",
    "entries",
    "eye_gaze_bindings",
    "eye_gaze_bindings_file_sha256",
    "safety",
    "acceptance_sha256",
}
_ACCEPTANCE_ENTRY_FIELDS = {
    "task_index",
    "recording_id",
    "convention_receipt",
    "convention_receipt_file_sha256",
    "convention_receipt_sha256",
}


class EyeGazeCohortReviewError(ValueError):
    """Raised when the human-review boundary cannot remain fail-closed."""


def _fail(message: str) -> None:
    raise EyeGazeCohortReviewError(message)


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


def _utc_timestamp(value: object, *, field: str) -> str:
    text = _text(value, field=field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EyeGazeCohortReviewError(f"{field} must be ISO-8601 UTC.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        _fail(f"{field} must use UTC.")
    return text


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path, *, field: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{field} is absent: {path}")
    try:
        value = json.loads(path.read_bytes())
    except json.JSONDecodeError as exc:
        raise EyeGazeCohortReviewError(f"{field} is not strict JSON: {path}") from exc
    return dict(_mapping(value, field=field))


def _write_new_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(json_attr_safe(value), indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)


def _self_digest(value: Mapping[str, Any], *, digest_field: str) -> str:
    return canonical_json_sha256(
        {key: item for key, item in value.items() if key != digest_field}
    )


def _exact_run_name(value: object, *, field: str) -> str:
    name = _text(value, field=field)
    if (
        name in {"latest", "latest_complete", "selected", "current", ".", ".."}
        or "/" in name
        or "\\" in name
        or any(character.isspace() for character in name)
    ):
        _fail(f"{field} must be one exact immutable run name.")
    return name


def _review_task_digest(task: Mapping[str, Any]) -> str:
    return _self_digest(task, digest_field="review_task_sha256")


def _acceptance_digest(manifest: Mapping[str, Any]) -> str:
    return _self_digest(manifest, digest_field="acceptance_sha256")


def _resolved_existing_directory(value: str | Path, *, field: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{field} is not an existing directory: {path}")
    return path


def _load_materialization_member(
    *,
    task: Mapping[str, Any],
    entry: Mapping[str, Any],
    receipt_roots: Sequence[Path],
    eye_channel_variant: str,
) -> dict[str, Any]:
    recording_id = _text(entry.get("recording_id"), field="recording identity")
    task_index = entry.get("task_index")
    if type(task_index) is not int or task_index <= 0:
        _fail("Prerequisite task index is invalid.")
    candidates = [
        root / recording_id / "materialization_receipt.json"
        for root in receipt_roots
        if (root / recording_id / "materialization_receipt.json").is_file()
    ]
    if len(candidates) != 1:
        _fail(
            f"Recording {recording_id!r} requires exactly one materialization "
            f"receipt across the supplied roots; observed {candidates!r}."
        )
    receipt_path = candidates[0].resolve()
    receipt = _read_object(receipt_path, field="materialization receipt")
    if set(receipt) != _MATERIALIZATION_RECEIPT_FIELDS:
        _fail(f"Materialization receipt fields are not exact: {receipt_path}")
    receipt_digest = _digest(
        receipt.get("receipt_sha256"), field="materialization receipt digest"
    )
    if receipt_digest != _self_digest(receipt, digest_field="receipt_sha256"):
        _fail(f"Materialization receipt self-digest is stale: {receipt_path}")
    expected = {
        "schema_id": MATERIALIZATION_RECEIPT_SCHEMA_ID,
        "schema_version": MATERIALIZATION_RECEIPT_SCHEMA_VERSION,
        "status": "complete",
        "task_sha256": task["task_sha256"],
        "entry_sha256": entry["entry_sha256"],
        "task_index": task_index,
        "recording_id": recording_id,
        "human_gaze_direction_acceptance": False,
        "selector_eligible": False,
        "production_authority": False,
        "registry_update": False,
        "selector_activation": False,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            _fail(f"Materialization receipt {key!r} differs from the frozen task.")
    palette_commit = _text(
        receipt.get("palette_commit"), field="materialization Palette commit"
    )
    if len(palette_commit) != 40 or any(
        c not in "0123456789abcdef" for c in palette_commit
    ):
        _fail("Materialization Palette commit is not one full lowercase Git SHA.")

    receipt_dir = receipt_path.parent
    companion_specs = (
        ("subject_shape_result.json", "subject_shape_result_sha256"),
        ("eye_angle_result.json", "eye_angle_result_sha256"),
        ("gaze_convention_numeric_validation.json", "numeric_validation_sha256"),
    )
    companions: dict[str, tuple[Path, dict[str, Any], str]] = {}
    for filename, digest_field in companion_specs:
        path = receipt_dir / filename
        document = _read_object(path, field=filename)
        document_digest = canonical_json_sha256(document)
        if document_digest != _digest(receipt.get(digest_field), field=digest_field):
            _fail(f"{filename} differs from its materialization receipt.")
        companions[filename] = (path, document, document_digest)

    shape = companions["subject_shape_result.json"][1]
    eye = companions["eye_angle_result.json"][1]
    numeric = companions["gaze_convention_numeric_validation.json"][1]
    if shape.get("status") != "complete" or eye.get("status") != "complete":
        _fail("Prerequisite shape and eye-angle results must both be complete.")
    logical_digest = _digest(
        eye.get("published_logical_manifest_sha256"),
        field="published eye-angle logical manifest digest",
    )
    if eye.get("local_logical_manifest_sha256") != logical_digest:
        _fail("Local and published eye-angle logical manifests differ.")

    archive = (
        Path(_text(entry.get("analysis_zarr"), field="analysis Zarr"))
        .expanduser()
        .resolve()
    )
    outputs = _mapping(entry.get("outputs"), field="prerequisite output identities")
    run_name = _exact_run_name(outputs.get("eye_angle_run"), field="eye-angle run name")
    run_path = f"{EYE_RUN_PARENT}/{run_name}"
    if (
        numeric.get("schema_id") != "palette.gaze_convention_validation.v1"
        or numeric.get("schema_version") != 1
        or numeric.get("status") != "pass"
        or numeric.get("read_only") is not True
        or Path(_text(numeric.get("zarr_path"), field="numeric validation Zarr"))
        .expanduser()
        .resolve()
        != archive
        or numeric.get("eye_angle_run") != run_name
        or numeric.get("eye_angle_run_path") != run_path
    ):
        _fail("Numeric gaze-convention validation is not bound to the exact eye run.")
    checks = numeric.get("checks")
    if (
        not isinstance(checks, list)
        or not checks
        or any(
            not isinstance(check, Mapping) or check.get("passed") is not True
            for check in checks
        )
    ):
        _fail("Every numeric gaze-convention check must explicitly pass.")

    png = Path(_text(receipt.get("review_png"), field="review PNG")).resolve()
    if (
        png != (receipt_dir / "gaze_convention_review.png").resolve()
        or numeric.get("review_png") != str(png)
        or not png.is_file()
        or png.stat().st_size <= 0
    ):
        _fail("Review PNG identity is absent or differs across prerequisite evidence.")
    png_digest = _sha256_file(png)
    if png_digest != _digest(
        receipt.get("review_png_sha256"), field="review PNG digest"
    ):
        _fail("Review PNG differs from its materialization receipt.")
    review_rows = numeric.get("review_row_indices")
    if (
        not isinstance(review_rows, list)
        or not review_rows
        or any(type(value) is not int or value < 0 for value in review_rows)
    ):
        _fail("Review PNG lacks exact non-empty sampled row identities.")

    return json_attr_safe(
        {
            "task_index": task_index,
            "recording_id": recording_id,
            "analysis_zarr": str(archive),
            "eye_angle_run": run_name,
            "eye_angle_run_path": run_path,
            "eye_channel_variant": eye_channel_variant,
            "source_eye_logical_sha256": logical_digest,
            "materialization_palette_commit": palette_commit,
            "materialization_receipt": {
                "path": str(receipt_path),
                "file_sha256": _sha256_file(receipt_path),
                "receipt_sha256": receipt_digest,
            },
            "subject_shape_result": {
                "path": str(companions["subject_shape_result.json"][0]),
                "file_sha256": _sha256_file(companions["subject_shape_result.json"][0]),
                "document_sha256": companions["subject_shape_result.json"][2],
            },
            "eye_angle_result": {
                "path": str(companions["eye_angle_result.json"][0]),
                "file_sha256": _sha256_file(companions["eye_angle_result.json"][0]),
                "document_sha256": companions["eye_angle_result.json"][2],
            },
            "numeric_validation": {
                "path": str(companions["gaze_convention_numeric_validation.json"][0]),
                "file_sha256": _sha256_file(
                    companions["gaze_convention_numeric_validation.json"][0]
                ),
                "document_sha256": companions[
                    "gaze_convention_numeric_validation.json"
                ][2],
            },
            "review_png": {
                "path": str(png),
                "file_sha256": png_digest,
                "review_row_indices": list(review_rows),
            },
            "review_status": PENDING,
        }
    )


def build_review_task(
    prerequisite_task: str | Path | Mapping[str, Any],
    *,
    receipt_roots: Sequence[str | Path],
    eye_channel_variant: str = "smoothed",
) -> dict[str, Any]:
    """Close one complete prerequisite cohort into a pending review task."""

    if eye_channel_variant not in {"raw", "smoothed"}:
        _fail("Eye channel variant must be exact 'raw' or 'smoothed'.")
    if not receipt_roots:
        _fail("At least one materialization receipt root is required.")
    roots = tuple(
        _resolved_existing_directory(root, field="materialization receipt root")
        for root in receipt_roots
    )
    if len(set(roots)) != len(roots):
        _fail("Materialization receipt roots are duplicated.")
    task = load_prerequisite_task(prerequisite_task)
    if task.get("safety") != PREREQUISITE_SAFETY:
        _fail("Prerequisite cohort safety envelope changed.")
    entries = [
        _load_materialization_member(
            task=task,
            entry=_mapping(raw, field="prerequisite cohort entry"),
            receipt_roots=roots,
            eye_channel_variant=eye_channel_variant,
        )
        for raw in task["entries"]
    ]
    commits = {entry["materialization_palette_commit"] for entry in entries}
    if len(commits) != 1:
        _fail("Prerequisite cohort was materialized by mixed Palette commits.")
    if isinstance(prerequisite_task, Mapping):
        source_path: str | None = None
        source_file_sha256: str | None = None
    else:
        path = Path(prerequisite_task).expanduser().resolve()
        source_path = str(path)
        source_file_sha256 = _sha256_file(path)
    body = json_attr_safe(
        {
            "schema_id": REVIEW_TASK_SCHEMA_ID,
            "schema_version": REVIEW_TASK_SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_prerequisite_task": {
                "path": source_path,
                "file_sha256": source_file_sha256,
                "task_sha256": task["task_sha256"],
            },
            "materialization_receipt_roots": [str(root) for root in roots],
            "materialization_palette_commit": next(iter(commits)),
            "recording_count": len(entries),
            "entries": entries,
            "review_status": PENDING,
            "safety": REVIEW_SAFETY,
        }
    )
    return {**body, "review_task_sha256": canonical_json_sha256(body)}


def load_review_task(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, Mapping):
        task = dict(source)
    else:
        task = _read_object(
            Path(source).expanduser().resolve(), field="gaze convention review task"
        )
    expected = {
        "schema_id",
        "schema_version",
        "created_at_utc",
        "source_prerequisite_task",
        "materialization_receipt_roots",
        "materialization_palette_commit",
        "recording_count",
        "entries",
        "review_status",
        "safety",
        "review_task_sha256",
    }
    if (
        set(task) != expected
        or task.get("schema_id") != REVIEW_TASK_SCHEMA_ID
        or task.get("schema_version") != REVIEW_TASK_SCHEMA_VERSION
        or task.get("review_status") != PENDING
        or task.get("safety") != REVIEW_SAFETY
        or task.get("review_task_sha256") != _review_task_digest(task)
    ):
        _fail("Gaze convention review task envelope is invalid or stale.")
    _utc_timestamp(task.get("created_at_utc"), field="created_at_utc")
    source_binding = _mapping(
        task.get("source_prerequisite_task"), field="source prerequisite task"
    )
    if set(source_binding) != {"path", "file_sha256", "task_sha256"}:
        _fail("Source prerequisite task binding fields are not exact.")
    _digest(source_binding.get("task_sha256"), field="source task semantic digest")
    source_path = source_binding.get("path")
    source_file_digest = source_binding.get("file_sha256")
    if (source_path is None) != (source_file_digest is None):
        _fail("Source prerequisite task path and file digest must be paired.")
    if source_path is not None:
        source_path = _text(source_path, field="source prerequisite task path")
        if not Path(source_path).is_absolute() or source_path != str(
            Path(source_path).resolve()
        ):
            _fail("Source prerequisite task path must be canonical and absolute.")
        _digest(source_file_digest, field="source prerequisite task file digest")
    roots = task.get("materialization_receipt_roots")
    if (
        not isinstance(roots, list)
        or not roots
        or any(
            type(root) is not str
            or not Path(root).is_absolute()
            or root != str(Path(root).resolve())
            for root in roots
        )
        or len(set(roots)) != len(roots)
    ):
        _fail("Review task materialization receipt roots are invalid or duplicated.")
    palette_commit = _text(
        task.get("materialization_palette_commit"),
        field="materialization Palette commit",
    )
    if len(palette_commit) != 40 or any(
        c not in "0123456789abcdef" for c in palette_commit
    ):
        _fail("Review task materialization commit is not one full Git SHA.")
    count = task.get("recording_count")
    entries = task.get("entries")
    if (
        type(count) is not int
        or count <= 0
        or not isinstance(entries, list)
        or len(entries) != count
    ):
        _fail("Gaze convention review task dimensions are invalid.")
    seen_indices: set[int] = set()
    seen_recordings: set[str] = set()
    for position, raw in enumerate(entries, start=1):
        entry = _mapping(raw, field="review task entry")
        if set(entry) != _REVIEW_ENTRY_FIELDS:
            _fail("Review task entry fields are not exact.")
        index = entry.get("task_index")
        recording_id = _text(entry.get("recording_id"), field="review recording")
        if (
            index != position
            or index in seen_indices
            or recording_id in seen_recordings
        ):
            _fail("Review task entries are unordered or duplicated.")
        seen_indices.add(index)
        seen_recordings.add(recording_id)
        if entry.get("review_status") != PENDING:
            _fail("Review task contains an inferred human decision.")
        archive = _text(entry.get("analysis_zarr"), field="analysis Zarr")
        if not Path(archive).is_absolute() or archive != str(Path(archive).resolve()):
            _fail("Review task analysis Zarr path must be canonical and absolute.")
        _digest(entry.get("source_eye_logical_sha256"), field="eye logical digest")
        _exact_run_name(entry.get("eye_angle_run"), field="eye-angle run")
        if (
            entry.get("eye_angle_run_path")
            != f"{EYE_RUN_PARENT}/{entry['eye_angle_run']}"
        ):
            _fail("Review task eye run path is not exact.")
        if entry.get("eye_channel_variant") not in {"raw", "smoothed"}:
            _fail("Review task eye channel variant is unsupported.")
        if entry.get("materialization_palette_commit") != palette_commit:
            _fail("Review entry materialization commit differs from its cohort.")
        receipt_binding = _mapping(
            entry.get("materialization_receipt"), field="materialization receipt"
        )
        if set(receipt_binding) != _RECEIPT_BINDING_FIELDS:
            _fail("Materialization receipt binding fields are not exact.")
        for field in ("subject_shape_result", "eye_angle_result", "numeric_validation"):
            binding = _mapping(entry.get(field), field=field)
            if set(binding) != _FILE_BINDING_FIELDS:
                _fail(f"{field} binding fields are not exact.")
        png = _mapping(entry.get("review_png"), field="review PNG")
        if set(png) != _PNG_BINDING_FIELDS:
            _fail("Review PNG binding fields are not exact.")
        for binding in (
            receipt_binding,
            entry["subject_shape_result"],
            entry["eye_angle_result"],
            entry["numeric_validation"],
            png,
        ):
            path = _text(binding.get("path"), field="review evidence path")
            if not Path(path).is_absolute() or path != str(Path(path).resolve()):
                _fail("Review evidence paths must be canonical and absolute.")
            _digest(binding.get("file_sha256"), field="review evidence file digest")
        _digest(receipt_binding.get("receipt_sha256"), field="receipt semantic digest")
        _digest(
            entry["subject_shape_result"].get("document_sha256"),
            field="subject-shape result document digest",
        )
        _digest(
            entry["eye_angle_result"].get("document_sha256"),
            field="eye result document digest",
        )
        _digest(
            entry["numeric_validation"].get("document_sha256"),
            field="numeric validation document digest",
        )
        review_rows = png.get("review_row_indices")
        if (
            not isinstance(review_rows, list)
            or not review_rows
            or any(type(row) is not int or row < 0 for row in review_rows)
        ):
            _fail("Review PNG binding lacks exact sampled row identities.")
    return task


def build_decision_template(
    review_task: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    """Return an operator-editable document with every decision pending."""

    task = load_review_task(review_task)
    return {
        "schema_id": DECISIONS_SCHEMA_ID,
        "schema_version": DECISIONS_SCHEMA_VERSION,
        "review_task_sha256": task["review_task_sha256"],
        "reviewer": "",
        "reviewed_at_utc": "",
        "entries": [
            {
                "task_index": entry["task_index"],
                "recording_id": entry["recording_id"],
                "review_png_sha256": entry["review_png"]["file_sha256"],
                "decision": "pending",
            }
            for entry in task["entries"]
        ],
    }


def _validate_current_review_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    bindings = (
        ("materialization_receipt", "receipt_sha256"),
        ("subject_shape_result", "document_sha256"),
        ("eye_angle_result", "document_sha256"),
        ("numeric_validation", "document_sha256"),
    )
    documents: dict[str, dict[str, Any]] = {}
    for field, document_digest_field in bindings:
        binding = _mapping(entry.get(field), field=field)
        path = Path(_text(binding.get("path"), field=f"{field} path")).resolve()
        if _sha256_file(path) != _digest(
            binding.get("file_sha256"), field=f"{field} file digest"
        ):
            _fail(f"Review source changed after planning: {path}")
        document = _read_object(path, field=field)
        if field == "materialization_receipt":
            observed = _self_digest(document, digest_field="receipt_sha256")
            if document.get("receipt_sha256") != observed:
                _fail(f"Materialization receipt self-digest changed: {path}")
        else:
            observed = canonical_json_sha256(document)
        if observed != _digest(
            binding.get(document_digest_field), field=f"{field} document digest"
        ):
            _fail(f"Review source document digest changed: {path}")
        documents[field] = document
    png_binding = _mapping(entry.get("review_png"), field="review PNG binding")
    png = Path(_text(png_binding.get("path"), field="review PNG path")).resolve()
    if _sha256_file(png) != _digest(
        png_binding.get("file_sha256"), field="review PNG digest"
    ):
        _fail(f"Review PNG changed after planning: {png}")
    return documents


def _load_decisions(
    source: str | Path | Mapping[str, Any], *, review_task: Mapping[str, Any]
) -> tuple[dict[int, dict[str, Any]], str, str, str]:
    if isinstance(source, Mapping):
        decisions = dict(source)
    else:
        path = Path(source).expanduser().resolve()
        decisions = _read_object(path, field="gaze convention decisions")
    source_sha256 = canonical_json_sha256(decisions)
    expected = {
        "schema_id",
        "schema_version",
        "review_task_sha256",
        "reviewer",
        "reviewed_at_utc",
        "entries",
    }
    if (
        set(decisions) != expected
        or decisions.get("schema_id") != DECISIONS_SCHEMA_ID
        or decisions.get("schema_version") != DECISIONS_SCHEMA_VERSION
        or decisions.get("review_task_sha256") != review_task["review_task_sha256"]
    ):
        _fail("Gaze convention decision envelope is invalid or belongs elsewhere.")
    reviewer = _text(decisions.get("reviewer"), field="reviewer")
    reviewed_at = _utc_timestamp(
        decisions.get("reviewed_at_utc"), field="reviewed_at_utc"
    )
    rows = decisions.get("entries")
    if not isinstance(rows, list) or len(rows) != review_task["recording_count"]:
        _fail("Decision rows do not cover the exact review cohort.")
    by_index: dict[int, dict[str, Any]] = {}
    for raw in rows:
        row = dict(_mapping(raw, field="decision row"))
        if set(row) != {"task_index", "recording_id", "review_png_sha256", "decision"}:
            _fail("Decision row fields are not exact.")
        index = row.get("task_index")
        if type(index) is not int or index in by_index:
            _fail("Decision task indices are invalid or duplicated.")
        by_index[index] = row
    expected_indices = set(range(1, review_task["recording_count"] + 1))
    if set(by_index) != expected_indices:
        _fail("Decision rows do not cover every exact task index.")
    return by_index, reviewer, reviewed_at, source_sha256


def _acceptance_paths(output_root: Path, entry: Mapping[str, Any]) -> tuple[Path, Path]:
    filename = f"{entry['task_index']:03d}_{entry['recording_id']}.json"
    return output_root / "receipts" / filename, output_root / "eye_gaze_bindings.json"


def accept_reviewed_cohort(
    review_task: str | Path | Mapping[str, Any],
    *,
    decisions: str | Path | Mapping[str, Any],
    output_root: str | Path,
) -> dict[str, Any]:
    """Publish a manifest-last bundle only when every decision is accepted."""

    task = load_review_task(review_task)
    decision_rows, reviewer, reviewed_at, decisions_sha256 = _load_decisions(
        decisions, review_task=task
    )
    rejected: list[str] = []
    receipts: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for raw_entry in task["entries"]:
        entry = dict(_mapping(raw_entry, field="review task entry"))
        row = decision_rows[entry["task_index"]]
        if (
            row.get("recording_id") != entry["recording_id"]
            or row.get("review_png_sha256") != entry["review_png"]["file_sha256"]
        ):
            _fail("A decision does not bind its exact recording and review PNG.")
        decision = row.get("decision")
        if decision == REJECTED:
            rejected.append(entry["recording_id"])
            continue
        if decision != ACCEPTED:
            _fail("Every decision must be exact 'accepted' or 'rejected'.")
        documents = _validate_current_review_entry(entry)
        convention = build_gaze_convention_review_receipt(
            numeric_validation=documents["numeric_validation"],
            source_eye_logical_sha256=entry["source_eye_logical_sha256"],
            reviewer=reviewer,
            reviewed_at_utc=reviewed_at,
            review_artifact_sha256=entry["review_png"]["file_sha256"],
        )
        validate_gaze_convention_review_receipt(
            convention,
            expected_run_path=entry["eye_angle_run_path"],
            expected_logical_sha256=entry["source_eye_logical_sha256"],
        )
        receipts.append((entry, convention))
    if rejected:
        _fail(
            "Cohort contains rejected biological-direction reviews; no acceptance "
            f"bundle was written: {rejected!r}."
        )

    target = Path(output_root).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to replace acceptance bundle: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        manifest_entries = []
        bindings = []
        for entry, convention in receipts:
            final_receipt, final_bindings = _acceptance_paths(target, entry)
            temporary_receipt, _ = _acceptance_paths(temporary, entry)
            _write_new_json(temporary_receipt, convention)
            receipt_file_sha256 = _sha256_file(temporary_receipt)
            manifest_entries.append(
                {
                    "task_index": entry["task_index"],
                    "recording_id": entry["recording_id"],
                    "convention_receipt": str(final_receipt),
                    "convention_receipt_file_sha256": receipt_file_sha256,
                    "convention_receipt_sha256": convention["receipt_sha256"],
                }
            )
            bindings.append(
                {
                    "recording_id": entry["recording_id"],
                    "analysis_zarr": entry["analysis_zarr"],
                    "eye_run_name": entry["eye_angle_run"],
                    "eye_channel_variant": entry["eye_channel_variant"],
                    "eye_convention_receipt": str(final_receipt),
                }
            )
        _write_new_json(temporary / final_bindings.name, bindings)
        body = json_attr_safe(
            {
                "schema_id": ACCEPTANCE_SCHEMA_ID,
                "schema_version": ACCEPTANCE_SCHEMA_VERSION,
                "status": "complete",
                "review_task_sha256": task["review_task_sha256"],
                "decisions_sha256": decisions_sha256,
                "reviewer": reviewer,
                "reviewed_at_utc": reviewed_at,
                "recording_count": len(receipts),
                "entries": manifest_entries,
                "eye_gaze_bindings": str(target / final_bindings.name),
                "eye_gaze_bindings_file_sha256": _sha256_file(
                    temporary / final_bindings.name
                ),
                "safety": REVIEW_SAFETY,
            }
        )
        manifest = {**body, "acceptance_sha256": canonical_json_sha256(body)}
        _write_new_json(temporary / "acceptance_manifest.json", manifest)

        # Reserve the final name without replacement, then move the already
        # complete children.  The manifest is moved last and is the sole
        # completeness marker; an interrupted partial root is never resumed.
        target.mkdir()
        (temporary / "receipts").rename(target / "receipts")
        (temporary / final_bindings.name).rename(target / final_bindings.name)
        (temporary / "acceptance_manifest.json").rename(
            target / "acceptance_manifest.json"
        )
        temporary.rmdir()
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    validate_acceptance_bundle(task, acceptance_root=target)
    return manifest


def validate_acceptance_bundle(
    review_task: str | Path | Mapping[str, Any], *, acceptance_root: str | Path
) -> dict[str, Any]:
    """Reopen a complete acceptance bundle and all immutable review sources."""

    task = load_review_task(review_task)
    root = _resolved_existing_directory(acceptance_root, field="acceptance root")
    manifest = _read_object(
        root / "acceptance_manifest.json", field="acceptance manifest"
    )
    if (
        set(manifest) != _ACCEPTANCE_FIELDS
        or manifest.get("schema_id") != ACCEPTANCE_SCHEMA_ID
        or manifest.get("schema_version") != ACCEPTANCE_SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("review_task_sha256") != task["review_task_sha256"]
        or manifest.get("recording_count") != task["recording_count"]
        or manifest.get("safety") != REVIEW_SAFETY
        or manifest.get("acceptance_sha256") != _acceptance_digest(manifest)
    ):
        _fail("Acceptance manifest is invalid, stale, or belongs elsewhere.")
    _digest(manifest.get("decisions_sha256"), field="decisions semantic digest")
    reviewer = _text(manifest.get("reviewer"), field="acceptance reviewer")
    reviewed_at = _utc_timestamp(
        manifest.get("reviewed_at_utc"), field="acceptance reviewed_at_utc"
    )
    raw_entries = manifest.get("entries")
    if not isinstance(raw_entries, list) or len(raw_entries) != task["recording_count"]:
        _fail("Acceptance manifest does not cover the exact review cohort.")
    manifest_by_index = {
        item["task_index"]: item
        for item in (_mapping(raw, field="acceptance entry") for raw in raw_entries)
    }
    if set(manifest_by_index) != set(range(1, task["recording_count"] + 1)):
        _fail("Acceptance manifest task indices are incomplete or duplicated.")
    expected_root_children = {
        "acceptance_manifest.json",
        "eye_gaze_bindings.json",
        "receipts",
    }
    if {child.name for child in root.iterdir()} != expected_root_children:
        _fail("Acceptance root has missing or unexpected children.")
    expected_receipt_names = {
        _acceptance_paths(root, entry)[0].name for entry in task["entries"]
    }
    receipt_root = root / "receipts"
    if (
        not receipt_root.is_dir()
        or {child.name for child in receipt_root.iterdir()} != expected_receipt_names
    ):
        _fail("Acceptance receipt roster is incomplete or contains extras.")
    expected_bindings = []
    for raw_entry in task["entries"]:
        entry = dict(_mapping(raw_entry, field="review task entry"))
        _validate_current_review_entry(entry)
        accepted = _mapping(
            manifest_by_index[entry["task_index"]], field="acceptance entry"
        )
        if set(accepted) != _ACCEPTANCE_ENTRY_FIELDS:
            _fail("Acceptance entry fields are not exact.")
        if accepted.get("recording_id") != entry["recording_id"]:
            _fail("Acceptance entry recording identity changed.")
        path = Path(
            _text(accepted.get("convention_receipt"), field="convention receipt")
        ).resolve()
        expected_receipt, _ = _acceptance_paths(root, entry)
        if path != expected_receipt:
            _fail("Convention receipt path is not the exact acceptance child.")
        if _sha256_file(path) != _digest(
            accepted.get("convention_receipt_file_sha256"),
            field="convention receipt file digest",
        ):
            _fail("Convention receipt file changed after acceptance.")
        receipt = _read_object(path, field="convention receipt")
        validated = validate_gaze_convention_review_receipt(
            receipt,
            expected_run_path=entry["eye_angle_run_path"],
            expected_logical_sha256=entry["source_eye_logical_sha256"],
        )
        if validated.get("receipt_sha256") != accepted.get("convention_receipt_sha256"):
            _fail("Convention receipt semantic identity changed.")
        if (
            validated.get("numeric_validation_sha256")
            != entry["numeric_validation"]["document_sha256"]
        ):
            _fail("Convention receipt differs from the frozen numeric validation.")
        biological_review = _mapping(
            validated.get("biological_direction_review"),
            field="biological direction review",
        )
        if (
            biological_review.get("reviewer") != reviewer
            or biological_review.get("reviewed_at_utc") != reviewed_at
        ):
            _fail("Convention receipt reviewer identity differs from its cohort.")
        if (
            biological_review.get("review_artifact_sha256")
            != entry["review_png"]["file_sha256"]
            or biological_review.get("review_row_indices")
            != entry["review_png"]["review_row_indices"]
        ):
            _fail("Convention receipt differs from the frozen review PNG evidence.")
        expected_bindings.append(
            {
                "recording_id": entry["recording_id"],
                "analysis_zarr": entry["analysis_zarr"],
                "eye_run_name": entry["eye_angle_run"],
                "eye_channel_variant": entry["eye_channel_variant"],
                "eye_convention_receipt": str(path),
            }
        )
    bindings_path = Path(
        _text(manifest.get("eye_gaze_bindings"), field="eye-gaze bindings path")
    ).resolve()
    if bindings_path != root / "eye_gaze_bindings.json":
        _fail("Eye-gaze bindings path is not the exact acceptance child.")
    if _sha256_file(bindings_path) != _digest(
        manifest.get("eye_gaze_bindings_file_sha256"),
        field="eye-gaze bindings file digest",
    ):
        _fail("Eye-gaze bindings file changed after acceptance.")
    try:
        bindings = json.loads(bindings_path.read_bytes())
    except json.JSONDecodeError as exc:
        raise EyeGazeCohortReviewError(
            "Eye-gaze bindings are not strict JSON."
        ) from exc
    if bindings != expected_bindings:
        _fail("Eye-gaze bindings differ from the accepted exact review set.")
    return {
        "status": "valid",
        "recording_count": task["recording_count"],
        "review_task_sha256": task["review_task_sha256"],
        "acceptance_sha256": manifest["acceptance_sha256"],
        "eye_gaze_bindings": str(bindings_path),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="freeze complete pending review evidence")
    plan.add_argument("--task", type=Path, required=True)
    plan.add_argument("--receipt-root", type=Path, action="append", required=True)
    plan.add_argument(
        "--eye-channel-variant", choices=("raw", "smoothed"), default="smoothed"
    )
    plan.add_argument("--output", type=Path, required=True)
    template = subparsers.add_parser("template", help="write all-pending decisions")
    template.add_argument("review_task", type=Path)
    template.add_argument("--output", type=Path, required=True)
    accept = subparsers.add_parser(
        "accept", help="publish explicitly accepted receipts"
    )
    accept.add_argument("review_task", type=Path)
    accept.add_argument("--decisions", type=Path, required=True)
    accept.add_argument("--output-root", type=Path, required=True)
    validate = subparsers.add_parser("validate", help="validate an accepted bundle")
    validate.add_argument("review_task", type=Path)
    validate.add_argument("--acceptance-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        result = build_review_task(
            args.task,
            receipt_roots=args.receipt_root,
            eye_channel_variant=args.eye_channel_variant,
        )
        _write_new_json(args.output.expanduser().resolve(), result)
    elif args.command == "template":
        result = build_decision_template(args.review_task)
        _write_new_json(args.output.expanduser().resolve(), result)
    elif args.command == "accept":
        result = accept_reviewed_cohort(
            args.review_task,
            decisions=args.decisions,
            output_root=args.output_root,
        )
    else:
        result = validate_acceptance_bundle(
            args.review_task, acceptance_root=args.acceptance_root
        )
    print(json.dumps(json_attr_safe(result), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACCEPTANCE_SCHEMA_ID",
    "DECISIONS_SCHEMA_ID",
    "EyeGazeCohortReviewError",
    "REVIEW_TASK_SCHEMA_ID",
    "accept_reviewed_cohort",
    "build_decision_template",
    "build_review_task",
    "load_review_task",
    "validate_acceptance_bundle",
]

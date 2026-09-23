"""Verified node-local staging for immutable merged training datasets.

This module owns an execution optimization only.  It never changes the source
artifact, its logical identity, split, labels, or storage declaration.  A staged
dataset is admitted only after its complete physical tree matches the source by
relative path, size, and SHA-256 content digest.
"""

from __future__ import annotations

import atexit
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

from fisheye.shared.atomic_run_publisher import TreeInventory, tree_inventory

STAGING_RECEIPT_SCHEMA_ID = "palette.training_dataset_local_staging_receipt"
STAGING_RECEIPT_SCHEMA_VERSION = 1
STAGING_POLICY = "verified_node_local_merged_training_copy_v1"
DEFAULT_MAX_TOTAL_BYTES = 8 * 1024**3
DEFAULT_MIN_FREE_BYTES_AFTER_STAGE = 8 * 1024**3


def _is_shared_storage(path: Path) -> bool:
    text = path.expanduser().resolve().as_posix()
    return (
        text == "/groups"
        or text.startswith("/groups/")
        or text == "/nrs"
        or text.startswith("/nrs/")
    )


def _require_scratch_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(
            f"Training staging scratch root is not a directory: {resolved}"
        )
    if _is_shared_storage(resolved):
        raise ValueError(
            f"Training staging scratch root must be node-local, got {resolved}"
        )
    if not os.access(resolved, os.W_OK | os.X_OK):
        raise ValueError(f"Training staging scratch root is not writable: {resolved}")
    return resolved


def _default_scratch_root() -> Path:
    user = str(os.environ.get("USER") or "").strip()
    job_id = str(os.environ.get("LSB_JOBID") or "").strip()
    if user and job_id:
        user_root = Path("/scratch") / user
        if user_root.is_dir() and os.access(user_root, os.W_OK | os.X_OK):
            job_root = user_root / job_id
            job_root.mkdir(parents=True, exist_ok=True)
            return _require_scratch_root(job_root)

    tmpdir = str(os.environ.get("TMPDIR") or "").strip()
    if tmpdir:
        candidate = Path(tmpdir)
        if candidate.is_dir() and not _is_shared_storage(candidate):
            return _require_scratch_root(candidate)
    return _require_scratch_root(Path("/tmp"))


def _read_root_attrs(path: Path) -> Mapping[str, Any]:
    metadata_path = path / "zarr.json"
    if not metadata_path.is_file():
        return {}
    try:
        document = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    if not isinstance(document, Mapping):
        return {}
    attrs = document.get("attributes")
    return attrs if isinstance(attrs, Mapping) else {}


def _training_export_declaration(path: Path) -> dict[str, Any]:
    attrs = _read_root_attrs(path)
    training_export = attrs.get("training_export")
    if not isinstance(training_export, Mapping):
        return {
            "eligibility": "not_merged_training_export",
            "training_set_id": None,
            "logical_dataset_hash": None,
        }
    task = str(training_export.get("task") or "").strip().lower()
    if attrs.get("zarr_purpose") != "training" or task not in {"pose", "detect"}:
        return {
            "eligibility": "not_merged_training_export",
            "training_set_id": training_export.get("set_id"),
            "logical_dataset_hash": training_export.get("logical_dataset_hash"),
        }

    publication = attrs.get("immutable_training_publication")
    if (
        attrs.get("training_artifact_status") == "complete"
        and attrs.get("training_artifact_mutability") == "immutable"
        and isinstance(publication, Mapping)
        and str(publication.get("schema_id") or "").startswith(
            "palette.immutable_merged_"
        )
    ):
        eligibility = "explicit_immutable_merged_training_export"
    elif (
        path.name.endswith("_merged.zarr")
        and (path / "crop_runs").is_dir()
        and (path / "splits").is_dir()
    ):
        # Compatibility for the closed pre-publication-contract merged exports
        # that supplied the established production pose models.  Exact physical
        # verification still binds the copy used by this execution.
        eligibility = "legacy_closed_merged_training_export"
    else:
        eligibility = "not_merged_training_export"
    return {
        "eligibility": eligibility,
        "task": task,
        "training_set_id": training_export.get("set_id"),
        "logical_dataset_hash": training_export.get("logical_dataset_hash"),
    }


def _same_physical_tree(source: TreeInventory, staged: TreeInventory) -> bool:
    return (
        source.files == staged.files
        and source.inventory_sha256 == staged.inventory_sha256
        and source.content_sha256 is not None
        and source.content_sha256 == staged.content_sha256
    )


def _safe_component(value: str) -> str:
    normalized = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in value
    ).strip("._")
    return normalized or "dataset"


@dataclass
class StagedTrainingDatasets:
    """Effective paths and provenance for one ephemeral staging decision."""

    effective_paths: dict[str, Path]
    receipt: dict[str, Any]
    stage_root: Path | None = None
    _cleaned: bool = False

    def __enter__(self) -> "StagedTrainingDatasets":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        """Remove only the unique temporary directory owned by this object."""

        if self._cleaned:
            return
        self._cleaned = True
        if self.stage_root is not None and self.stage_root.exists():
            shutil.rmtree(self.stage_root)


def stage_training_datasets(
    datasets: Mapping[str, str | Path],
    *,
    mode: str = "auto",
    scratch_root: str | Path | None = None,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    min_free_bytes_after_stage: int = DEFAULT_MIN_FREE_BYTES_AFTER_STAGE,
) -> StagedTrainingDatasets:
    """Return exact effective dataset paths under a bounded staging policy.

    ``auto`` stages eligible merged exports only when they reside on shared
    storage.  ``required`` requires every configured dataset to be an eligible
    merged export and creates a verified copy even when the source is local.
    ``disabled`` preserves every configured source path.  The returned object
    is a context manager and also registers process-exit cleanup; callers with
    a longer lifecycle may call ``cleanup()`` explicitly.
    """

    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {"auto", "required", "disabled"}:
        raise ValueError(
            "Training dataset staging mode must be auto, required, or disabled"
        )
    if type(max_total_bytes) is not int or max_total_bytes <= 0:
        raise ValueError("max_total_bytes must be a positive integer")
    if type(min_free_bytes_after_stage) is not int or min_free_bytes_after_stage < 0:
        raise ValueError("min_free_bytes_after_stage must be a non-negative integer")
    if not datasets:
        raise ValueError("Training dataset staging requires at least one dataset")

    effective_paths = {
        str(name): Path(path).expanduser().resolve() for name, path in datasets.items()
    }
    records: dict[str, dict[str, Any]] = {}
    candidates: list[str] = []
    total_candidate_bytes = 0
    for name, source in effective_paths.items():
        if not source.is_dir():
            raise ValueError(f"Training dataset path is not a directory: {source}")
        declaration = _training_export_declaration(source)
        eligibility = str(declaration["eligibility"])
        record = {
            "source_path": str(source),
            "effective_path": str(source),
            **declaration,
        }
        if normalized_mode == "disabled":
            record["action"] = "disabled"
        elif eligibility == "not_merged_training_export":
            if normalized_mode == "required":
                raise ValueError(
                    f"Required local staging accepts only merged training exports: {source}"
                )
            record["action"] = "not_eligible"
        elif (
            normalized_mode == "auto"
            and not _is_shared_storage(source)
        ):
            record["action"] = "already_node_local"
        else:
            inventory = tree_inventory(source, hash_content=False)
            record["physical_bytes"] = int(inventory.physical_bytes)
            record["action"] = "planned"
            candidates.append(name)
            total_candidate_bytes += int(inventory.physical_bytes)
        records[name] = record

    receipt: dict[str, Any] = {
        "schema_id": STAGING_RECEIPT_SCHEMA_ID,
        "schema_version": STAGING_RECEIPT_SCHEMA_VERSION,
        "policy": {
            "policy_id": STAGING_POLICY,
            "mode": normalized_mode,
            "max_total_bytes": int(max_total_bytes),
            "min_free_bytes_after_stage": int(min_free_bytes_after_stage),
            "source_scope": (
                "shared_storage_merged_exports"
                if normalized_mode == "auto"
                else "all_eligible_merged_exports"
            ),
            "verification": "relative_path_size_and_sha256_all_files",
            "cleanup": "remove_ephemeral_stage_root_after_training",
        },
        "datasets": records,
        "summary": {
            "configured_dataset_count": len(records),
            "candidate_dataset_count": len(candidates),
            "candidate_total_bytes": int(total_candidate_bytes),
            "staged_dataset_count": 0,
        },
    }
    if not candidates:
        return StagedTrainingDatasets(effective_paths, receipt)

    if total_candidate_bytes > max_total_bytes:
        if normalized_mode == "required":
            raise ValueError(
                "Required training staging exceeds max_total_bytes "
                f"({total_candidate_bytes} > {max_total_bytes})"
            )
        for name in candidates:
            records[name]["action"] = "size_gate_skipped"
        return StagedTrainingDatasets(effective_paths, receipt)

    base = (
        _require_scratch_root(Path(scratch_root))
        if scratch_root
        else _default_scratch_root()
    )
    free_bytes = int(shutil.disk_usage(base).free)
    receipt["scratch"] = {
        "base_path": str(base),
        "free_bytes_before_stage": free_bytes,
    }
    required_bytes = int(total_candidate_bytes + min_free_bytes_after_stage)
    if free_bytes < required_bytes:
        if normalized_mode == "required":
            raise ValueError(
                "Required training staging lacks free space after reserve "
                f"({free_bytes} < {required_bytes})"
            )
        for name in candidates:
            records[name]["action"] = "free_space_gate_skipped"
        return StagedTrainingDatasets(effective_paths, receipt)

    stage_root_path = Path(
        tempfile.mkdtemp(prefix="palette-training-datasets-", dir=str(base))
    )
    receipt["scratch"]["stage_root"] = str(stage_root_path)
    result = StagedTrainingDatasets(
        effective_paths=effective_paths,
        receipt=receipt,
        stage_root=stage_root_path,
    )
    atexit.register(result.cleanup)
    try:
        payload_root = stage_root_path / "datasets"
        payload_root.mkdir()
        for ordinal, name in enumerate(candidates):
            source = effective_paths[name]
            target = payload_root / (
                f"{ordinal:03d}-{_safe_component(name)}-{_safe_component(source.name)}"
            )
            source_inventory = tree_inventory(source, hash_content=True)
            shutil.copytree(source, target, copy_function=shutil.copyfile)
            staged_inventory = tree_inventory(target, hash_content=True)
            if not _same_physical_tree(source_inventory, staged_inventory):
                raise RuntimeError(
                    f"Staged training dataset differs from source: {source}"
                )
            effective_paths[name] = target
            records[name].update(
                {
                    "action": "staged",
                    "effective_path": str(target),
                    "verification": {
                        "status": "exact_physical_tree_match",
                        **source_inventory.to_json(),
                    },
                }
            )
        receipt["summary"]["staged_dataset_count"] = len(candidates)
        return result
    except Exception:
        result.cleanup()
        raise


__all__ = [
    "DEFAULT_MAX_TOTAL_BYTES",
    "DEFAULT_MIN_FREE_BYTES_AFTER_STAGE",
    "STAGING_POLICY",
    "STAGING_RECEIPT_SCHEMA_ID",
    "STAGING_RECEIPT_SCHEMA_VERSION",
    "StagedTrainingDatasets",
    "stage_training_datasets",
]

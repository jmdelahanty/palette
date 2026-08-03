"""Optional Megabouts bout-classifier execution over Palette-derived windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional, Sequence
import uuid

import numpy as np
import zarr

from fisheye.shared.zarr.columnar import store_array, write_columnar_dataset
from fisheye.analysis.bout_classification_runs import (
    BOUT_CLASSIFICATION_SCHEMA_ID,
    BOUT_CLASSIFICATION_SCHEMA_VERSION,
    PER_BOUT_SCHEMA_ID,
    validate_staged_bout_classification_run,
)
from fisheye.analysis.bout_classification_schema import (
    BOUT_CLASSIFICATION_ACCESS_UNIT_SEMANTICS,
    BOUT_CLASSIFICATION_CANDIDATE_ARRAY_DECLARATIONS,
    BOUT_CLASSIFICATION_FIELD_DTYPES,
    BOUT_CLASSIFICATION_FIELD_NAMES,
    BOUT_CLASSIFICATION_FILL_VALUES,
    CATEGORY_LABEL_BYTES_WIDTH,
    FAILURE_REASON_BYTES_WIDTH,
    BoutClassificationDimensions,
    validate_bout_classification_arrays,
    write_bout_classification_array_schema_manifest,
)
from fisheye.analysis.direct_writer_storage import (
    ANALYSIS_STORAGE_PROFILE_ROLE,
    build_direct_writer_storage_receipt,
    create_direct_writer_arrays_from_receipt,
    persist_direct_writer_storage_receipt,
)
from fisheye.shared.coordinate_frame_record import array_payload_sha256
from fisheye.shared.coordinate_reference import canonical_node_path
from fisheye.analysis.megabouts_classifier_inputs import (
    DEFAULT_BOUT_DURATION_S,
    DEFAULT_ALIGN_TRAJ_TO_ONSET,
    DEFAULT_HEADING_SOURCE,
    DEFAULT_MAX_CONSECUTIVE_INVALID_FRAMES,
    DEFAULT_MIN_TAIL_VALID_FRACTION,
    DEFAULT_MIN_TRAJ_VALID_FRACTION,
    DEFAULT_TRAJ_REFERENCE_INDEX,
    MegaboutsClassifierInputPack,
    build_megabouts_classifier_input_pack,
    summarize_input_pack,
)
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.run_lineage_fingerprint import write_best_effort_run_lineage_attrs
from fisheye.shared.run_provenance import build_run_provenance_from_stage_record
from fisheye.shared.selector_activation import (
    SelectorActivationError,
    activate_selector_eligible_run,
)
from fisheye.shared.stage_provenance import (
    build_stage_provenance,
    write_stage_provenance,
)
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_CONTRACT,
    RUN_COMPLETION_CONTRACT_ATTR,
    RUN_COMPLETED_AT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_NAME_ATTR,
    RUN_STAGE_ATTR,
    RUN_STARTED_AT_ATTR,
    RUN_STATUS_COMPLETE,
    RUN_STATUS_FAILED,
    RUN_STATUS_RUNNING,
    mark_run_complete,
    mark_run_failed,
    require_runs_parent,
)
from fisheye.shared.system_metadata import get_environment_info, get_git_info
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr.storage_profiles import StorageProfile, get_storage_profile

SCHEMA_ID = BOUT_CLASSIFICATION_SCHEMA_ID
SCHEMA_VERSION = BOUT_CLASSIFICATION_SCHEMA_VERSION
ADAPTER_METHOD = "palette_megabouts_direct_classifier"
ADAPTER_METHOD_VERSION = 1
CLASSIFIER_FAMILY = "megabouts"
CLASSIFIER_NAME = "megabouts_transformer"
SOURCE_MODE = "palette_bouts"
INVALID_WINDOW_POLICY = "skip_invalid_windows"
PALETTE_PREPARED_INPUT_MODE = "palette_prepared_fixed_windows"
MEGABOUTS_PREPROCESSED_INPUT_MODE = "megabouts_preprocessed_full_timeseries"
BOUT_CLASSIFICATION_PUBLICATION_OWNER_ATTR = (
    "bout_classification_publication_owner_uuid"
)
BOUT_CLASSIFICATION_PARENT_PUBLICATION_LEASE_ATTR = (
    "bout_classification_publication_lease"
)
BOUT_CLASSIFICATION_PUBLICATION_GENERATION_ATTR = "publication_generation"
BOUT_CLASSIFICATION_PUBLICATION_POLICY_ATTR = "publication_policy"
BOUT_CLASSIFICATION_PUBLICATION_POLICY = (
    "owner_generation_guarded_selectors_then_eligibility_v1"
)
BOUT_CLASSIFICATION_PUBLICATION_TOMBSTONE_ATTR = (
    "bout_classification_publication_tombstone"
)


def _fixed_text_matrix(values: np.ndarray, *, width: int) -> np.ndarray:
    """Encode one fixed-string field to its exact cross-language uint8 surface."""

    source = np.asarray(values)
    if source.ndim != 1 or source.dtype.kind != "S":
        raise ValueError(
            "Exact text fields must be one-dimensional fixed byte strings."
        )
    encoded = np.zeros((int(source.shape[0]), int(width)), dtype=np.uint8)
    for row_index, value in enumerate(source):
        payload = bytes(value).split(b"\x00", 1)[0]
        if len(payload) >= int(width):
            raise ValueError(
                f"Text row {row_index} does not leave room for a NUL terminator."
            )
        encoded[row_index, : len(payload)] = np.frombuffer(payload, dtype=np.uint8)
    return encoded


@dataclass(frozen=True)
class MegaboutsRuntime:
    """Resolved optional Megabouts runtime objects."""

    classifier_class: object
    tracking_config_class: object
    segmentation_config_class: object
    category_names: tuple[str, ...]
    package_version: str
    package_path: str
    source_repo: Optional[str]
    git_commit: Optional[str]


@dataclass(frozen=True)
class MegaboutsClassificationResult:
    """Megabouts classification outputs for the valid-window subset."""

    classified_indices: np.ndarray
    classif_results: Mapping[str, np.ndarray]
    runtime: Optional[MegaboutsRuntime]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_run_name() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"megabouts_classifier_{stamp}"


_json_safe = json_attr_safe


def _resolve_megabouts_repo(megabouts_repo: Optional[str | Path]) -> Optional[Path]:
    raw = (
        megabouts_repo
        if megabouts_repo is not None
        else os.environ.get("MEGABOUTS_REPO")
    )
    if raw is None or str(raw).strip() == "":
        return None
    path = Path(raw).expanduser().resolve()
    if not (path / "megabouts").is_dir():
        raise ValueError(
            f"Megabouts repo {path} does not contain a top-level 'megabouts' package directory."
        )
    return path


def _git_commit_for_repo(repo_path: Optional[Path]) -> Optional[str]:
    if repo_path is None:
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None


def _load_megabouts_runtime(
    megabouts_repo: Optional[str | Path] = None,
) -> MegaboutsRuntime:
    repo_path = _resolve_megabouts_repo(megabouts_repo)
    if repo_path is not None and str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    try:
        import megabouts
        from megabouts.classification import BoutClassifier
        from megabouts.config.segmentation_config import TailSegmentationConfig
        from megabouts.tracking_data import TrackingConfig
    except Exception as exc:  # pragma: no cover - depends on optional external package
        raise RuntimeError(
            "Megabouts is required for classifier execution but is not importable. "
            "Install/configure Megabouts outside Palette, or run "
            "`fisheye.analysis.megabouts_classifier_inputs` for a dependency-free dry run."
        ) from exc

    try:
        from megabouts.utils.plots_utils import bouts_category_name

        category_names = tuple(str(value) for value in bouts_category_name)
    except Exception:  # pragma: no cover - optional label helper
        category_names = ()

    package_path = str(Path(getattr(megabouts, "__file__", "") or "").resolve())
    return MegaboutsRuntime(
        classifier_class=BoutClassifier,
        tracking_config_class=TrackingConfig,
        segmentation_config_class=TailSegmentationConfig,
        category_names=category_names,
        package_version=str(getattr(megabouts, "__version__", "unknown")),
        package_path=package_path,
        source_repo=None if repo_path is None else str(repo_path),
        git_commit=_git_commit_for_repo(repo_path),
    )


def _runtime_attrs(runtime: Optional[MegaboutsRuntime]) -> dict[str, object]:
    if runtime is None:
        return {
            "classifier_version": None,
            "megabouts_package_version": None,
            "megabouts_package_path": None,
            "megabouts_category_labels": [],
        }
    return {
        "classifier_version": runtime.package_version,
        "megabouts_package_version": runtime.package_version,
        "megabouts_package_path": runtime.package_path,
        "megabouts_source_repo": runtime.source_repo,
        "megabouts_git_commit": runtime.git_commit,
        "megabouts_category_labels": list(runtime.category_names),
    }


def _resolve_fps(pack: MegaboutsClassifierInputPack) -> int:
    fps = float(pack.parameters.get("fps", 0.0))
    rounded = int(round(fps))
    if rounded <= 0 or not math.isclose(fps, float(rounded), rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            "Megabouts TrackingConfig requires integer fps in [20, 700]; "
            f"Palette resolved fps={fps!r}."
        )
    if not 20 <= rounded <= 700:
        raise ValueError(
            "Megabouts TrackingConfig requires integer fps in [20, 700]; "
            f"Palette resolved fps={rounded}."
        )
    return rounded


def classify_megabouts_input_pack(
    pack: MegaboutsClassifierInputPack,
    *,
    exclude_cs: bool = False,
    device: str = "auto",
    megabouts_repo: Optional[str | Path] = None,
    runtime: Optional[MegaboutsRuntime] = None,
) -> MegaboutsClassificationResult:
    """Run Megabouts on valid source windows only.

    Invalid Palette windows are intentionally excluded from the Megabouts call.
    They are still represented in the persisted result table as skipped rows.
    """

    classified_indices = np.flatnonzero(np.asarray(pack.valid_bout, dtype=bool))
    if classified_indices.size == 0:
        return MegaboutsClassificationResult(
            classified_indices=classified_indices.astype(np.int64, copy=False),
            classif_results={
                "cat": np.asarray([], dtype=np.int32),
                "subcat": np.asarray([], dtype=np.int32),
                "sign": np.asarray([], dtype=np.int32),
                "proba": np.asarray([], dtype=np.float32),
                "first_half_beat": np.asarray([], dtype=np.int32),
            },
            runtime=runtime,
        )

    resolved_runtime = (
        runtime if runtime is not None else _load_megabouts_runtime(megabouts_repo)
    )
    fps = _resolve_fps(pack)
    window_frames = int(pack.tail_array.shape[2])
    # Add a tiny epsilon because Megabouts converts milliseconds with int(),
    # and we need the segmentation mask length to match our fixed window.
    bout_duration_ms = (float(window_frames) + 1e-6) * 1000.0 / float(fps)
    tracking_cfg = resolved_runtime.tracking_config_class(
        fps=fps, tracking="full_tracking"
    )
    segmentation_cfg = resolved_runtime.segmentation_config_class(
        fps=fps,
        bout_duration_ms=bout_duration_ms,
    )
    if int(segmentation_cfg.bout_duration) != window_frames:
        raise ValueError(
            "Megabouts segmentation config duration does not match Palette classifier window: "
            f"{segmentation_cfg.bout_duration} != {window_frames}."
        )

    device_obj = None
    if str(device) != "auto":
        try:
            import torch
        except Exception as exc:  # pragma: no cover - optional external package path
            raise RuntimeError(
                "A Megabouts device was requested but torch is not importable."
            ) from exc
        device_obj = torch.device(str(device))

    classifier = resolved_runtime.classifier_class(
        tracking_cfg,
        segmentation_cfg,
        exclude_CS=bool(exclude_cs),
        device=device_obj,
    )
    classif_results = classifier.run_classification(
        tail_array=pack.tail_array[classified_indices],
        traj_array=pack.traj_array[classified_indices],
    )
    normalized_results = {
        "cat": np.asarray(classif_results["cat"], dtype=np.int32),
        "subcat": np.asarray(classif_results["subcat"], dtype=np.int32),
        "sign": np.asarray(classif_results["sign"], dtype=np.int32),
        "proba": np.asarray(classif_results["proba"], dtype=np.float32),
        "first_half_beat": np.asarray(
            classif_results["first_half_beat"], dtype=np.int32
        ),
    }
    expected = int(classified_indices.size)
    for name, values in normalized_results.items():
        if int(values.shape[0]) != expected:
            raise ValueError(
                f"Megabouts returned {values.shape[0]} values for {name!r}; expected {expected}."
            )
    return MegaboutsClassificationResult(
        classified_indices=classified_indices.astype(np.int64, copy=False),
        classif_results=normalized_results,
        runtime=resolved_runtime,
    )


def _category_label(category_id: int, category_names: Sequence[str]) -> str:
    if 0 <= int(category_id) < len(category_names):
        return str(category_names[int(category_id)])
    if int(category_id) < 0:
        return "skipped_invalid_window"
    return f"category_{int(category_id)}"


def _as_bytes(value: object, *, width: int) -> bytes:
    return str(value or "").encode("utf-8", errors="replace")[: int(width)]


def build_per_bout_classification_table(
    pack: MegaboutsClassifierInputPack,
    result: MegaboutsClassificationResult,
) -> np.ndarray:
    """Build a row-aligned classification table for every source bout."""

    n_bouts = int(pack.source_bout_id.shape[0])
    dtype = np.dtype(
        [
            ("source_bout_id", "i8"),
            ("start_frame", "i8"),
            ("end_frame", "i8"),
            ("window_start_frame", "i8"),
            ("window_end_frame", "i8"),
            ("HB1_frame", "i8"),
            ("HB1_offset_frames", "i4"),
            ("category_id", "i4"),
            ("category_label_bytes", f"S{CATEGORY_LABEL_BYTES_WIDTH}"),
            ("subcategory_id", "i4"),
            ("sign", "i4"),
            ("probability", "f4"),
            ("tail_valid_fraction", "f4"),
            ("traj_valid_fraction", "f4"),
            ("max_consecutive_tail_invalid", "i4"),
            ("max_consecutive_traj_invalid", "i4"),
            ("source_window_valid", "?"),
            ("classified", "?"),
            ("valid", "?"),
            ("failure_reason_bytes", f"S{FAILURE_REASON_BYTES_WIDTH}"),
        ]
    )
    table = np.zeros((n_bouts,), dtype=dtype)
    table["source_bout_id"] = np.asarray(pack.source_bout_id, dtype=np.int64)
    table["start_frame"] = np.asarray(pack.source_start_frame, dtype=np.int64)
    table["end_frame"] = np.asarray(pack.source_end_frame, dtype=np.int64)
    table["window_start_frame"] = np.asarray(pack.window_start_frame, dtype=np.int64)
    table["window_end_frame"] = np.asarray(pack.window_end_frame, dtype=np.int64)
    table["HB1_frame"] = -1
    table["HB1_offset_frames"] = -1
    table["category_id"] = -1
    table["category_label_bytes"] = _as_bytes(
        "skipped_invalid_window", width=CATEGORY_LABEL_BYTES_WIDTH
    )
    table["subcategory_id"] = -1
    table["sign"] = 0
    table["probability"] = np.nan
    table["tail_valid_fraction"] = np.asarray(
        pack.tail_valid_fraction, dtype=np.float32
    )
    table["traj_valid_fraction"] = np.asarray(
        pack.traj_valid_fraction, dtype=np.float32
    )
    table["max_consecutive_tail_invalid"] = np.asarray(
        pack.max_consecutive_tail_invalid, dtype=np.int32
    )
    table["max_consecutive_traj_invalid"] = np.asarray(
        pack.max_consecutive_traj_invalid, dtype=np.int32
    )
    source_valid = np.asarray(pack.valid_bout, dtype=bool)
    table["source_window_valid"] = source_valid
    table["classified"] = False
    table["valid"] = False
    for idx, reason in enumerate(
        np.asarray(pack.failure_reason, dtype=object).tolist()
    ):
        table["failure_reason_bytes"][idx] = _as_bytes(
            reason, width=FAILURE_REASON_BYTES_WIDTH
        )

    classified_indices = np.asarray(result.classified_indices, dtype=np.int64)
    if classified_indices.size == 0:
        return table

    category = np.asarray(result.classif_results["cat"], dtype=np.int32)
    subcategory = np.asarray(result.classif_results["subcat"], dtype=np.int32)
    sign = np.asarray(result.classif_results["sign"], dtype=np.int32)
    proba = np.asarray(result.classif_results["proba"], dtype=np.float32)
    hb1_offset = np.asarray(result.classif_results["first_half_beat"], dtype=np.int32)
    category_names = () if result.runtime is None else result.runtime.category_names

    table["category_id"][classified_indices] = category
    table["subcategory_id"][classified_indices] = subcategory
    table["sign"][classified_indices] = sign
    table["probability"][classified_indices] = proba
    table["HB1_offset_frames"][classified_indices] = hb1_offset
    table["HB1_frame"][classified_indices] = np.asarray(
        pack.window_start_frame, dtype=np.int64
    )[classified_indices] + hb1_offset.astype(np.int64)
    table["classified"][classified_indices] = True
    table["valid"][classified_indices] = True
    for source_idx, cat_id in zip(classified_indices.tolist(), category.tolist()):
        table["category_label_bytes"][source_idx] = _as_bytes(
            _category_label(int(cat_id), category_names),
            width=CATEGORY_LABEL_BYTES_WIDTH,
        )
        table["failure_reason_bytes"][source_idx] = _as_bytes(
            "ok", width=FAILURE_REASON_BYTES_WIDTH
        )

    return table


def _resolve_parent(root: zarr.Group) -> zarr.Group:
    analysis = root["analysis"] if "analysis" in root else root.create_group("analysis")
    return require_runs_parent(analysis, "bout_classification_runs")


def _fresh_bout_classification_candidate(
    root: zarr.Group,
    *,
    run_name: str,
    expected_publication_owner_uuid: str,
) -> zarr.Group:
    """Return one fresh exact owner-bound public child or fail closed."""

    run_path = f"analysis/bout_classification_runs/{run_name}"
    candidate = root.get(run_path)
    if (
        not isinstance(candidate, zarr.Group)
        or canonical_node_path(candidate) != run_path
        or candidate.attrs.get(BOUT_CLASSIFICATION_PUBLICATION_OWNER_ATTR)
        != expected_publication_owner_uuid
    ):
        raise RuntimeError(
            "Bout-classification public candidate changed path or exact ownership."
        )
    return candidate


def _require_fresh_candidate_initialization(
    root: zarr.Group,
    *,
    run_name: str,
    expected_publication_owner_uuid: str,
    expected_started_at_utc: str,
) -> zarr.Group:
    """Verify the atomic public-child initialization before payload writes."""

    candidate = _fresh_bout_classification_candidate(
        root,
        run_name=run_name,
        expected_publication_owner_uuid=expected_publication_owner_uuid,
    )
    expected = {
        BOUT_CLASSIFICATION_PUBLICATION_OWNER_ATTR: (expected_publication_owner_uuid),
        "stage_selector_eligible": False,
        RUN_COMPLETION_CONTRACT_ATTR: RUN_COMPLETION_CONTRACT,
        RUN_COMPLETION_STATUS_ATTR: RUN_STATUS_RUNNING,
        RUN_STARTED_AT_ATTR: expected_started_at_utc,
        RUN_NAME_ATTR: run_name,
        RUN_STAGE_ATTR: "bout_classification",
    }
    if any(candidate.attrs.get(key) != value for key, value in expected.items()):
        raise RuntimeError(
            "Bout-classification public candidate initialization did not persist exactly."
        )
    if tuple(candidate.array_keys()) or tuple(candidate.group_keys()):
        raise RuntimeError(
            "Bout-classification public candidate was not empty after initialization."
        )
    return candidate


def _failed_publication_tombstone(
    *,
    run_name: str,
    publication_owner_uuid: str,
    failed_at_utc: str,
    failure: BaseException,
) -> dict[str, object]:
    return {
        "schema_id": "palette.bout_classification_publication_tombstone",
        "schema_version": 1,
        "failed_at_utc": failed_at_utc,
        "publication_owner_uuid": publication_owner_uuid,
        "run_name": run_name,
        "run_path": f"analysis/bout_classification_runs/{run_name}",
        "public_path_retained": True,
        "selector_eligible": False,
        "retry_policy": "new_immutable_run_name_required",
        "failure_type": type(failure).__name__,
        "failure": str(failure),
    }


def _persist_failed_bout_classification_tombstone(
    root: zarr.Group,
    *,
    run_name: str,
    publication_owner_uuid: str,
    failure: BaseException,
) -> list[str]:
    """Disarm and fresh-verify one exact owned immutable public tombstone."""

    run_path = f"analysis/bout_classification_runs/{run_name}"
    candidate = root.get(run_path)
    if candidate is None:
        return []
    if not isinstance(candidate, zarr.Group):
        return [f"failed public path {run_path!r} is not a group"]
    if (
        canonical_node_path(candidate) != run_path
        or candidate.attrs.get(BOUT_CLASSIFICATION_PUBLICATION_OWNER_ATTR)
        != publication_owner_uuid
    ):
        return ["failed public candidate is not owned by this publication attempt"]

    errors: list[str] = []
    failed_at_utc = _utc_now()
    tombstone = _json_safe(
        _failed_publication_tombstone(
            run_name=run_name,
            publication_owner_uuid=publication_owner_uuid,
            failed_at_utc=failed_at_utc,
            failure=failure,
        )
    )

    try:
        owned = _fresh_bout_classification_candidate(
            root,
            run_name=run_name,
            expected_publication_owner_uuid=publication_owner_uuid,
        )
        owned.attrs["stage_selector_eligible"] = False
    except BaseException as exc:  # pragma: no cover - hostile store
        errors.append(f"selector disarm: {exc}")
    try:
        owned = _fresh_bout_classification_candidate(
            root,
            run_name=run_name,
            expected_publication_owner_uuid=publication_owner_uuid,
        )
        if RUN_COMPLETED_AT_ATTR in owned.attrs:
            del owned.attrs[RUN_COMPLETED_AT_ATTR]
    except BaseException as exc:  # pragma: no cover - hostile store
        errors.append(f"clear completed timestamp: {exc}")
    try:
        owned = _fresh_bout_classification_candidate(
            root,
            run_name=run_name,
            expected_publication_owner_uuid=publication_owner_uuid,
        )
        parent = root["analysis/bout_classification_runs"]
        mark_run_failed(
            owned,
            parent_group=parent,
            run_name=run_name,
            failed_at_utc=failed_at_utc,
            error=f"{type(failure).__name__}: {failure}",
        )
    except BaseException as exc:  # pragma: no cover - hostile store
        errors.append(f"mark failed: {exc}")
    try:
        owned = _fresh_bout_classification_candidate(
            root,
            run_name=run_name,
            expected_publication_owner_uuid=publication_owner_uuid,
        )
        owned.attrs[BOUT_CLASSIFICATION_PUBLICATION_TOMBSTONE_ATTR] = tombstone
    except BaseException as exc:  # pragma: no cover - hostile store
        errors.append(f"persist tombstone: {exc}")

    try:
        fresh = _fresh_bout_classification_candidate(
            root,
            run_name=run_name,
            expected_publication_owner_uuid=publication_owner_uuid,
        )
        parent = root["analysis/bout_classification_runs"]
        if (
            fresh.attrs.get("stage_selector_eligible") is not False
            or fresh.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_FAILED
            or RUN_COMPLETED_AT_ATTR in fresh.attrs
            or fresh.attrs.get(BOUT_CLASSIFICATION_PUBLICATION_TOMBSTONE_ATTR)
            != tombstone
            or parent.attrs.get("latest") == run_name
            or parent.attrs.get("latest_complete") == run_name
        ):
            raise RuntimeError("failed public tombstone did not persist exactly")
    except BaseException as exc:  # pragma: no cover - hostile store
        errors.append(f"verify tombstone: {exc}")
    return errors


def _populate_megabouts_classification_run(
    root: zarr.Group,
    *,
    parent: zarr.Group,
    run_group: zarr.Group,
    resolved_run_name: str,
    pack: MegaboutsClassifierInputPack,
    result: MegaboutsClassificationResult,
    exclude_cs: bool = False,
    command: Optional[str] = None,
    storage_profile: StorageProfile | None = None,
) -> str:
    """Populate one already-owned, selector-ineligible public candidate."""

    created_at_utc = _utc_now()
    runtime_attrs = _runtime_attrs(result.runtime)
    table = build_per_bout_classification_table(pack, result)
    source_refs = dict(pack.source_refs)
    classifier_input_mode = str(
        pack.parameters.get("classifier_input_mode") or PALETTE_PREPARED_INPUT_MODE
    )
    megabouts_preprocessing = bool(
        pack.parameters.get("megabouts_preprocessing", False)
    )
    megabouts_segmentation = bool(pack.parameters.get("megabouts_segmentation", False))
    source_fps = float(pack.parameters.get("fps", math.nan))
    window_frames = int(pack.tail_array.shape[2])
    window_duration_s = float(pack.parameters.get("bout_duration_s", math.nan))
    parameters = {
        **dict(pack.parameters),
        "adapter_method": ADAPTER_METHOD,
        "adapter_method_version": ADAPTER_METHOD_VERSION,
        "classifier_family": CLASSIFIER_FAMILY,
        "classifier_name": CLASSIFIER_NAME,
        "classifier_input_mode": classifier_input_mode,
        "megabouts_preprocessing": megabouts_preprocessing,
        "megabouts_segmentation": megabouts_segmentation,
        "source_fps": source_fps,
        "window_duration_s": window_duration_s,
        "window_frames": window_frames,
        "megabouts_time_sampling": True,
        "source_mode": SOURCE_MODE,
        "invalid_window_policy": INVALID_WINDOW_POLICY,
        "exclude_capture_swims": bool(exclude_cs),
        "calls_megabouts": bool(result.classified_indices.size > 0),
        "classified_bout_count": int(result.classified_indices.size),
        "source_bout_count": int(table.shape[0]),
        "valid_source_window_count": int(np.count_nonzero(pack.valid_bout)),
        "invalid_source_window_count": int(
            table.shape[0] - np.count_nonzero(pack.valid_bout)
        ),
    }
    tail_angle_conversion = {
        "source_array": source_refs.get("tail_angle_rad"),
        "source_valid_array": source_refs.get("tail_valid"),
        "convention": "megabouts_cumulative_segment_angle",
        "channels": int(pack.tail_array.shape[1]),
        "units": "radians",
    }
    trajectory_conversion = {
        "source_positions_array": source_refs.get("positions_mm"),
        "source_heading_array": source_refs.get("heading"),
        "source_valid_array": source_refs.get("sample_valid"),
        "channels": ["x_mm", "y_mm", "heading_radians"],
        "alignment": pack.parameters.get("traj_alignment"),
        "reference_index": pack.parameters.get("traj_reference_index"),
        "heading_reference": "classifier_window_reference_sample",
    }
    invalid_frame_policy = {
        "policy": INVALID_WINDOW_POLICY,
        "min_tail_valid_fraction": pack.parameters.get("min_tail_valid_fraction"),
        "min_traj_valid_fraction": pack.parameters.get("min_traj_valid_fraction"),
        "max_consecutive_invalid_frames": pack.parameters.get(
            "max_consecutive_invalid_frames"
        ),
        "requires_traj_reference_valid": pack.parameters.get(
            "requires_traj_reference_valid"
        ),
    }
    attrs = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "method": ADAPTER_METHOD,
        "method_version": ADAPTER_METHOD_VERSION,
        "adapter_method": ADAPTER_METHOD,
        "adapter_method_version": ADAPTER_METHOD_VERSION,
        "classifier_family": CLASSIFIER_FAMILY,
        "classifier_name": CLASSIFIER_NAME,
        "classifier_input_mode": classifier_input_mode,
        "megabouts_preprocessing": megabouts_preprocessing,
        "megabouts_segmentation": megabouts_segmentation,
        "source_fps": source_fps,
        "window_duration_s": window_duration_s,
        "window_frames": window_frames,
        "megabouts_time_sampling": True,
        "source_mode": SOURCE_MODE,
        "row_axis": "swim_bout_rows",
        "invalid_window_policy": INVALID_WINDOW_POLICY,
        "tail_angle_conversion": _json_safe(tail_angle_conversion),
        "trajectory_conversion": _json_safe(trajectory_conversion),
        "invalid_frame_policy": _json_safe(invalid_frame_policy),
        "source_refs": _json_safe(source_refs),
        "parameters": _json_safe(parameters),
        "source_bout_count": int(table.shape[0]),
        "valid_source_window_count": int(np.count_nonzero(pack.valid_bout)),
        "invalid_source_window_count": int(
            table.shape[0] - np.count_nonzero(pack.valid_bout)
        ),
        "classified_bout_count": int(result.classified_indices.size),
        **runtime_attrs,
    }
    for key, value in attrs.items():
        run_group.attrs[key] = _json_safe(value)

    per_bout_attrs = {
        "schema_id": PER_BOUT_SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "storage_semantics": "one row per source swim-bout row",
        "invalid_window_policy": INVALID_WINDOW_POLICY,
        "category_label_encoding": "utf8-null-terminated",
        "category_label_bytes_width": CATEGORY_LABEL_BYTES_WIDTH,
        "failure_reason_encoding": "utf8-null-terminated",
        "failure_reason_bytes_width": FAILURE_REASON_BYTES_WIDTH,
    }
    if storage_profile is None:
        per_bout = write_columnar_dataset(
            run_group,
            "per_bout",
            table,
            attrs=per_bout_attrs,
        )
        for field_name, width in (
            ("category_label_bytes", CATEGORY_LABEL_BYTES_WIDTH),
            ("failure_reason_bytes", FAILURE_REASON_BYTES_WIDTH),
        ):
            store_array(
                per_bout,
                field_name,
                _fixed_text_matrix(table[field_name], width=width),
            )
    else:
        candidate_arrays = {
            f"per_bout/{field_name}": (
                _fixed_text_matrix(table[field_name], width=width)
                if (
                    width := {
                        "category_label_bytes": CATEGORY_LABEL_BYTES_WIDTH,
                        "failure_reason_bytes": FAILURE_REASON_BYTES_WIDTH,
                    }.get(field_name)
                )
                is not None
                else np.asarray(table[field_name])
            )
            for field_name in BOUT_CLASSIFICATION_FIELD_NAMES
        }
        storage_receipt = build_direct_writer_storage_receipt(
            declarations=BOUT_CLASSIFICATION_CANDIDATE_ARRAY_DECLARATIONS,
            arrays_by_path=candidate_arrays,
            access_unit_semantics=BOUT_CLASSIFICATION_ACCESS_UNIT_SEMANTICS,
            profile=storage_profile,
            dimensions={"n_bouts": int(table.shape[0])},
        )
        persist_direct_writer_storage_receipt(run_group, storage_receipt)
        per_bout = run_group.create_group(
            "per_bout",
            attributes={
                "storage_layout": "columnar",
                "field_names": list(BOUT_CLASSIFICATION_FIELD_NAMES),
                "field_dtypes": dict(BOUT_CLASSIFICATION_FIELD_DTYPES),
                "physical_layout": "analysis_storage_plan_receipt_v1",
                **per_bout_attrs,
            },
        )
        create_direct_writer_arrays_from_receipt(
            run_group,
            receipt=storage_receipt,
            arrays_by_path=candidate_arrays,
            fill_values=BOUT_CLASSIFICATION_FILL_VALUES,
        )
    per_bout.attrs["source_swim_bout_path"] = source_refs.get("swim_bout_level")
    write_bout_classification_array_schema_manifest(
        run_group,
        n_bouts=int(table.shape[0]),
        byte_planner_adopted=storage_profile is not None,
    )
    if storage_profile is not None:
        storage_issues = validate_bout_classification_arrays(
            run_group,
            dimensions=BoutClassificationDimensions(n_bouts=int(table.shape[0])),
        )
        if storage_issues:
            detail = "; ".join(
                f"{issue.code}:{issue.path}:{issue.message}" for issue in storage_issues
            )
            raise RuntimeError(
                "Bout-classification candidate storage validation failed: " + detail
            )

    zarr_path = getattr(root, "_palette_fs_path", None)
    env_info = get_environment_info(
        disk_path=str(zarr_path) if zarr_path is not None else None,
        capture_env_vars=False,
    )
    provenance = build_stage_provenance(
        stage="bout_classification",
        created_at_utc=created_at_utc,
        parameters=_json_safe(parameters),
        inputs=_json_safe(source_refs),
        command=command,
        version=str(ADAPTER_METHOD_VERSION),
        git=get_git_info(),
        environment=env_info.get("environment"),
        platform=env_info.get("platform"),
        artifacts={
            "run_path": f"analysis/bout_classification_runs/{resolved_run_name}",
            "per_bout_path": f"analysis/bout_classification_runs/{resolved_run_name}/per_bout",
        },
    )
    write_stage_provenance(run_group, provenance)
    write_best_effort_run_lineage_attrs(run_group, run_family="bout_classification_run")
    mark_run_complete(
        run_group,
        parent_group=parent,
        run_name=resolved_run_name,
        run_provenance=build_run_provenance_from_stage_record(provenance),
    )
    return resolved_run_name


def _activate_megabouts_classification_run(
    root: zarr.Group,
    parent: zarr.Group,
    run_group: zarr.Group,
    *,
    run_name: str,
    expected_publication_owner_uuid: str,
) -> None:
    """Expose one immutable, freshly revalidated classification run."""

    run_path = f"analysis/bout_classification_runs/{run_name}"

    def proof() -> tuple[object, ...]:
        validation = validate_staged_bout_classification_run(
            root,
            run_name,
            strict=True,
        )
        if validation.get("ok") is not True:
            raise RuntimeError(
                "Bout-classification candidate failed strict activation validation: "
                f"{validation.get('errors')!r}; {validation.get('warnings')!r}."
            )
        candidate = root[run_path]
        if (
            candidate.attrs.get(BOUT_CLASSIFICATION_PUBLICATION_OWNER_ATTR)
            != expected_publication_owner_uuid
            or run_group.attrs.get(BOUT_CLASSIFICATION_PUBLICATION_OWNER_ATTR)
            != expected_publication_owner_uuid
            or candidate.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
            or candidate.attrs.get("stage_selector_eligible") is not False
        ):
            raise RuntimeError(
                "Bout-classification candidate changed ownership, completion, or "
                "selector eligibility before activation."
            )
        per_bout = candidate["per_bout"]
        field_names = tuple(str(value) for value in per_bout.attrs["field_names"])
        run_array_names = tuple(sorted(str(name) for name in candidate.array_keys()))
        run_group_names = tuple(sorted(str(name) for name in candidate.group_keys()))
        per_bout_array_names = tuple(
            sorted(str(name) for name in per_bout.array_keys())
        )
        per_bout_group_names = tuple(
            sorted(str(name) for name in per_bout.group_keys())
        )
        if (
            run_array_names
            or run_group_names != ("per_bout",)
            or len(field_names) != len(set(field_names))
            or per_bout_array_names != tuple(sorted(field_names))
            or per_bout_group_names
        ):
            raise RuntimeError(
                "Bout-classification candidate child inventory is not exact."
            )
        field_digests = tuple(
            (name, array_payload_sha256(per_bout[name])) for name in field_names
        )
        field_attrs = {
            name: _json_safe(dict(per_bout[name].attrs)) for name in field_names
        }
        candidate_attrs = {
            str(key): value
            for key, value in candidate.attrs.items()
            if str(key) != "stage_selector_eligible"
        }
        metadata_payload = {
            "candidate_attrs": _json_safe(candidate_attrs),
            "candidate_array_names": list(run_array_names),
            "candidate_group_names": list(run_group_names),
            "per_bout_attrs": _json_safe(dict(per_bout.attrs)),
            "per_bout_array_names": list(per_bout_array_names),
            "per_bout_group_names": list(per_bout_group_names),
            "field_array_attrs": field_attrs,
        }
        metadata_digest = hashlib.sha256(
            json.dumps(
                metadata_payload,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        return (
            expected_publication_owner_uuid,
            candidate.attrs.get(RUN_COMPLETION_STATUS_ATTR),
            metadata_digest,
            field_digests,
        )

    try:
        activate_selector_eligible_run(
            root,
            parent,
            run_group,
            parent_path="analysis/bout_classification_runs",
            run_path=run_path,
            run_name=run_name,
            owner_attr=BOUT_CLASSIFICATION_PUBLICATION_OWNER_ATTR,
            expected_owner_uuid=expected_publication_owner_uuid,
            policy_attr=BOUT_CLASSIFICATION_PUBLICATION_POLICY_ATTR,
            generation_attr=BOUT_CLASSIFICATION_PUBLICATION_GENERATION_ATTR,
            lease_attr=BOUT_CLASSIFICATION_PARENT_PUBLICATION_LEASE_ATTR,
            policy=BOUT_CLASSIFICATION_PUBLICATION_POLICY,
            lease_schema_id="palette.bout_classification_publication_lease",
            proof_loader=proof,
            selector_attrs=("latest_complete", "latest"),
        )
    except SelectorActivationError as exc:
        raise RuntimeError(
            f"Bout-classification activation lost exact ownership: {exc}."
        ) from exc


def write_megabouts_classification_run(
    root: zarr.Group,
    *,
    run_name: Optional[str],
    pack: MegaboutsClassifierInputPack,
    result: MegaboutsClassificationResult,
    overwrite: bool = False,
    exclude_cs: bool = False,
    command: Optional[str] = None,
    storage_profile: StorageProfile | None = None,
) -> str:
    """Publish one immutable, owner-guarded Megabouts classification run."""

    parent = _resolve_parent(root)
    resolved_run_name = str(run_name or _default_run_name()).strip()
    if not resolved_run_name or "/" in resolved_run_name:
        raise ValueError(
            "Bout classification run name must be one non-empty child token."
        )
    if resolved_run_name in parent:
        suffix = (
            " --overwrite cannot replace an immutable public run." if overwrite else ""
        )
        raise ValueError(
            f"Bout classification run {resolved_run_name!r} already exists; "
            f"choose a new --run-name.{suffix}"
        )

    publication_owner_uuid = str(uuid.uuid4())
    started_at_utc = _utc_now()
    run_group: zarr.Group | None = None
    try:
        run_group = parent.create_group(
            resolved_run_name,
            attributes={
                BOUT_CLASSIFICATION_PUBLICATION_OWNER_ATTR: publication_owner_uuid,
                "stage_selector_eligible": False,
                RUN_COMPLETION_CONTRACT_ATTR: RUN_COMPLETION_CONTRACT,
                RUN_COMPLETION_STATUS_ATTR: RUN_STATUS_RUNNING,
                RUN_STARTED_AT_ATTR: started_at_utc,
                RUN_NAME_ATTR: resolved_run_name,
                RUN_STAGE_ATTR: "bout_classification",
            },
        )
        run_group = _require_fresh_candidate_initialization(
            root,
            run_name=resolved_run_name,
            expected_publication_owner_uuid=publication_owner_uuid,
            expected_started_at_utc=started_at_utc,
        )
        _populate_megabouts_classification_run(
            root,
            parent=parent,
            run_group=run_group,
            resolved_run_name=resolved_run_name,
            pack=pack,
            result=result,
            exclude_cs=exclude_cs,
            command=command,
            storage_profile=storage_profile,
        )
        if storage_profile is None:
            _activate_megabouts_classification_run(
                root,
                parent,
                run_group,
                run_name=resolved_run_name,
                expected_publication_owner_uuid=publication_owner_uuid,
            )
    except BaseException as exc:
        cleanup_errors = _persist_failed_bout_classification_tombstone(
            root,
            run_name=resolved_run_name,
            publication_owner_uuid=publication_owner_uuid,
            failure=exc,
        )
        if cleanup_errors:
            raise RuntimeError(
                "Bout-classification failure cleanup was incomplete: "
                f"{cleanup_errors!r}."
            ) from exc
        raise
    return resolved_run_name


def run_megabouts_classifier(
    zarr_path: str | Path,
    *,
    run_name: Optional[str] = None,
    overwrite: bool = False,
    tail_posture_view_run: str = "latest",
    track_kinematics_run: str = "latest",
    track_scope: str = "offline",
    track_id: int = 0,
    swim_bout_run: str = "latest",
    speed_level: str = "default",
    heading_source: str = DEFAULT_HEADING_SOURCE,
    bout_duration_s: float = DEFAULT_BOUT_DURATION_S,
    bout_duration_frames: Optional[int] = None,
    min_tail_valid_fraction: float = DEFAULT_MIN_TAIL_VALID_FRACTION,
    min_traj_valid_fraction: float = DEFAULT_MIN_TRAJ_VALID_FRACTION,
    max_consecutive_invalid_frames: int = DEFAULT_MAX_CONSECUTIVE_INVALID_FRAMES,
    align_traj_to_onset: bool = DEFAULT_ALIGN_TRAJ_TO_ONSET,
    traj_reference_index: int = DEFAULT_TRAJ_REFERENCE_INDEX,
    exclude_cs: bool = False,
    device: str = "auto",
    megabouts_repo: Optional[str | Path] = None,
    classifier_input_mode: str = PALETTE_PREPARED_INPUT_MODE,
    dry_run: bool = False,
    command: Optional[str] = None,
    storage_profile: StorageProfile | None = None,
) -> dict[str, object]:
    """Run or dry-run the optional Megabouts classifier adapter."""

    root = open_zarr_root(zarr_path, mode="r" if dry_run else "a")
    mode = str(classifier_input_mode or PALETTE_PREPARED_INPUT_MODE)
    pack_kwargs = {
        "tail_posture_view_run": tail_posture_view_run,
        "track_kinematics_run": track_kinematics_run,
        "track_scope": track_scope,
        "track_id": track_id,
        "swim_bout_run": swim_bout_run,
        "speed_level": speed_level,
        "heading_source": heading_source,
        "bout_duration_s": bout_duration_s,
        "bout_duration_frames": bout_duration_frames,
        "min_tail_valid_fraction": min_tail_valid_fraction,
        "min_traj_valid_fraction": min_traj_valid_fraction,
        "max_consecutive_invalid_frames": max_consecutive_invalid_frames,
        "align_traj_to_onset": align_traj_to_onset,
        "traj_reference_index": traj_reference_index,
    }
    if mode == PALETTE_PREPARED_INPUT_MODE:
        pack = build_megabouts_classifier_input_pack(root, **pack_kwargs)
    elif mode == MEGABOUTS_PREPROCESSED_INPUT_MODE:
        from fisheye.analysis.megabouts_preprocessing_comparison import (
            build_megabouts_preprocessed_input_pack,
        )

        pack = build_megabouts_preprocessed_input_pack(
            root,
            megabouts_repo=megabouts_repo,
            **pack_kwargs,
        )
    else:
        raise ValueError(
            "Unsupported classifier_input_mode "
            f"{classifier_input_mode!r}; expected {PALETTE_PREPARED_INPUT_MODE!r} "
            f"or {MEGABOUTS_PREPROCESSED_INPUT_MODE!r}."
        )
    if dry_run:
        summary = summarize_input_pack(pack)
        summary_parameters = dict(summary.get("parameters", {}))
        summary_parameters.update(
            {
                "adapter_method": ADAPTER_METHOD,
                "adapter_method_version": ADAPTER_METHOD_VERSION,
                "classifier_family": CLASSIFIER_FAMILY,
                "classifier_name": CLASSIFIER_NAME,
                "classifier_input_mode": mode,
                "megabouts_preprocessing": bool(
                    pack.parameters.get("megabouts_preprocessing", False)
                ),
                "megabouts_segmentation": bool(
                    pack.parameters.get("megabouts_segmentation", False)
                ),
                "source_fps": float(pack.parameters.get("fps", math.nan)),
                "window_duration_s": float(
                    pack.parameters.get("bout_duration_s", math.nan)
                ),
                "window_frames": int(pack.tail_array.shape[2]),
                "megabouts_time_sampling": True,
                "calls_megabouts_classifier": False,
            }
        )
        summary["parameters"] = summary_parameters
        summary["would_write_run_family"] = "analysis/bout_classification_runs"
        summary["adapter_method"] = ADAPTER_METHOD
        summary["classifier_input_mode"] = mode
        summary["calls_megabouts_preprocessing"] = bool(
            pack.parameters.get("megabouts_preprocessing", False)
        )
        summary["calls_megabouts_classifier"] = False
        return summary

    result = classify_megabouts_input_pack(
        pack,
        exclude_cs=exclude_cs,
        device=device,
        megabouts_repo=megabouts_repo,
    )
    resolved_run_name = write_megabouts_classification_run(
        root,
        run_name=run_name,
        pack=pack,
        result=result,
        overwrite=overwrite,
        exclude_cs=exclude_cs,
        command=command,
        storage_profile=storage_profile,
    )
    n_bouts = int(pack.valid_bout.shape[0])
    summary = {
        "status": "ok",
        "run_name": resolved_run_name,
        "run_path": f"analysis/bout_classification_runs/{resolved_run_name}",
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "adapter_method": ADAPTER_METHOD,
        "classifier_input_mode": mode,
        "source_bout_count": n_bouts,
        "valid_source_window_count": int(np.count_nonzero(pack.valid_bout)),
        "invalid_source_window_count": int(n_bouts - np.count_nonzero(pack.valid_bout)),
        "classified_bout_count": int(result.classified_indices.shape[0]),
        "invalid_window_policy": INVALID_WINDOW_POLICY,
        "source_refs": pack.source_refs,
        "parameters": {
            **pack.parameters,
            "adapter_method": ADAPTER_METHOD,
            "adapter_method_version": ADAPTER_METHOD_VERSION,
            "calls_megabouts": True,
            "classifier_family": CLASSIFIER_FAMILY,
            "classifier_name": CLASSIFIER_NAME,
            "exclude_capture_swims": bool(exclude_cs),
            "device": str(device),
            "megabouts_repo": None if megabouts_repo is None else str(megabouts_repo),
            "align_traj_to_onset": bool(align_traj_to_onset),
            "traj_reference_index": int(traj_reference_index),
        },
        "storage_profile_id": (
            storage_profile.profile_id if storage_profile is not None else None
        ),
        "storage_profile_role": (
            ANALYSIS_STORAGE_PROFILE_ROLE if storage_profile is not None else "legacy"
        ),
        "selector_eligible": storage_profile is None,
        **_runtime_attrs(result.runtime),
    }
    return dict(_json_safe(summary))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run optional Megabouts bout classification over Palette swim-bout windows."
    )
    parser.add_argument("zarr_path", type=Path, help="Palette zarr archive.")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Output analysis/bout_classification_runs/<run> name.",
    )
    parser.add_argument(
        "--storage-profile",
        choices=("published_http_v1", "detection_regular_rollback_v1"),
        help=(
            "Explicit unpromoted byte-planned candidate profile. Supplying it "
            "writes a complete selector-ineligible candidate; omission preserves "
            "the legacy physical layout and activation behavior."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Compatibility flag only; canonical public run names are immutable, "
            "so an existing --run-name is never replaced."
        ),
    )
    parser.add_argument("--tail-posture-view-run", default="latest")
    parser.add_argument("--track-kinematics-run", default="latest")
    parser.add_argument("--track-scope", default="offline")
    parser.add_argument("--track-id", type=int, default=0)
    parser.add_argument("--swim-bout-run", default="latest")
    parser.add_argument("--speed-level", default="default")
    parser.add_argument("--heading-source", default=DEFAULT_HEADING_SOURCE)
    parser.add_argument(
        "--bout-duration-s", type=float, default=DEFAULT_BOUT_DURATION_S
    )
    parser.add_argument("--bout-duration-frames", type=int, default=None)
    parser.add_argument(
        "--min-tail-valid-fraction", type=float, default=DEFAULT_MIN_TAIL_VALID_FRACTION
    )
    parser.add_argument(
        "--min-traj-valid-fraction", type=float, default=DEFAULT_MIN_TRAJ_VALID_FRACTION
    )
    parser.add_argument(
        "--max-consecutive-invalid-frames",
        type=int,
        default=DEFAULT_MAX_CONSECUTIVE_INVALID_FRAMES,
    )
    parser.add_argument(
        "--no-align-traj-to-onset",
        action="store_true",
        help="Disable Megabouts-style onset-frame translation/rotation for trajectory windows.",
    )
    parser.add_argument(
        "--traj-reference-index", type=int, default=DEFAULT_TRAJ_REFERENCE_INDEX
    )
    parser.add_argument(
        "--exclude-CS", action="store_true", help="Pass exclude_CS=True to Megabouts."
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Megabouts torch device: auto, cpu, cuda, cuda:0, etc.",
    )
    parser.add_argument(
        "--megabouts-repo",
        default=None,
        help="Optional local Megabouts checkout to add to sys.path without installing it. Also supports MEGABOUTS_REPO.",
    )
    parser.add_argument(
        "--classifier-input-mode",
        default=PALETTE_PREPARED_INPUT_MODE,
        choices=[PALETTE_PREPARED_INPUT_MODE, MEGABOUTS_PREPROCESSED_INPUT_MODE],
        help=(
            "Input pack mode. The Megabouts-preprocessed mode runs Megabouts "
            "preprocessing before classification and records megabouts_preprocessing=true."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Build inputs and report eligibility without writing. In palette-prepared mode this "
            "does not import Megabouts; in Megabouts-preprocessed mode it imports preprocessing."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit compact JSON.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    command = (
        " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]) if argv is None else None
    )
    summary = run_megabouts_classifier(
        args.zarr_path,
        run_name=args.run_name,
        overwrite=bool(args.overwrite),
        tail_posture_view_run=args.tail_posture_view_run,
        track_kinematics_run=args.track_kinematics_run,
        track_scope=args.track_scope,
        track_id=int(args.track_id),
        swim_bout_run=args.swim_bout_run,
        speed_level=args.speed_level,
        heading_source=args.heading_source,
        bout_duration_s=float(args.bout_duration_s),
        bout_duration_frames=args.bout_duration_frames,
        min_tail_valid_fraction=float(args.min_tail_valid_fraction),
        min_traj_valid_fraction=float(args.min_traj_valid_fraction),
        max_consecutive_invalid_frames=int(args.max_consecutive_invalid_frames),
        align_traj_to_onset=not bool(args.no_align_traj_to_onset),
        traj_reference_index=int(args.traj_reference_index),
        exclude_cs=bool(args.exclude_CS),
        device=str(args.device),
        megabouts_repo=args.megabouts_repo,
        classifier_input_mode=str(args.classifier_input_mode),
        dry_run=bool(args.dry_run),
        command=command,
        storage_profile=(
            get_storage_profile(args.storage_profile)
            if args.storage_profile is not None
            else None
        ),
    )
    print(json.dumps(summary, indent=None if args.json else 2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

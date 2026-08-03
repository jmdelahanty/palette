"""Strict maintained reader for immutable stimulus-epoch v2 candidates.

The maintained path accepts one explicitly named, selector-ineligible v2
candidate and validates its complete publication contract before decoding any
rows.  Historical v1 runs remain available only through an explicit
compatibility policy; they are never selected as a fallback for a rejected v2
run.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any

import zarr

from fisheye.analysis.epoch_segments import EpochSegment, read_epoch_segments
from fisheye.analysis.exact_tabular_storage import (
    validate_exact_tabular_storage_receipt,
)
from fisheye.analysis.stimulus_epoch_schema import (
    LEGACY_STIMULUS_EPOCH_SCHEMA_ID,
    LEGACY_STIMULUS_EPOCH_SCHEMA_VERSION,
    STIMULUS_EPOCH_RUN_SCHEMA_ID,
    STIMULUS_EPOCH_RUN_SCHEMA_VERSION,
    build_stimulus_epoch_array_declarations,
    validate_legacy_stimulus_epoch_source,
    validate_stimulus_epoch_array_manifest,
    validate_stimulus_epoch_candidate_lineage,
    validate_stimulus_epoch_run_manifest,
)
from fisheye.shared.zarr.metadata_equivalence import (
    MetadataEquivalenceReceipt,
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETED_AT_ATTR,
    RUN_COMPLETION_CONTRACT,
    RUN_COMPLETION_CONTRACT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_NAME_ATTR,
    RUN_STATUS_COMPLETE,
)


PARENT_PATH = "analysis/stimulus_epoch_runs"
EXPECTED_V2_ARRAY_COUNT = 12
EXPECTED_V2_GROUP_COUNT = 2


class StimulusEpochCompatibilityPolicy(str, Enum):
    """Explicit compatibility boundary for stimulus-epoch reads."""

    EXACT_V2_ONLY = "exact_v2_only"
    ALLOW_EXPLICIT_V1 = "allow_explicit_v1"


@dataclass(frozen=True)
class StimulusEpochSnapshot:
    """Validated, eager, backend-independent stimulus-epoch rows."""

    run_name: str
    run_path: str
    schema_id: str
    schema_version: int
    compatibility_policy: StimulusEpochCompatibilityPolicy
    segments: tuple[EpochSegment, ...]
    metadata_equivalence: MetadataEquivalenceReceipt | None

    @property
    def is_legacy_compatibility_read(self) -> bool:
        return self.schema_id == LEGACY_STIMULUS_EPOCH_SCHEMA_ID


def _safe_run_name(value: str) -> str:
    if type(value) is not str:
        raise TypeError("run_name must be one exact string")
    name = value.strip()
    if (
        not name
        or name != value
        or name in {".", "..", "latest", "latest_complete"}
        or "/" in name
        or "\\" in name
        or any(character.isspace() for character in name)
    ):
        raise ValueError("run_name must be one explicit immutable run name")
    return name


def _read_direct_schema_identity(archive: Path, run_path: str) -> tuple[str, int]:
    run_directory = archive.joinpath(*run_path.split("/"))
    if run_directory.is_symlink():
        raise ValueError("Selected stimulus-epoch run cannot be a symlink")
    metadata_path = run_directory / "zarr.json"
    if metadata_path.is_symlink():
        raise ValueError("Selected stimulus-epoch run metadata cannot be a symlink")
    try:
        metadata_path.resolve().relative_to(archive.resolve())
    except ValueError as exc:
        raise ValueError("Selected stimulus-epoch run escapes the archive") from exc
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Cannot read selected stimulus-epoch run metadata: {exc}"
        ) from exc
    if type(metadata) is not dict or metadata.get("node_type") != "group":
        raise ValueError("Selected stimulus-epoch path is not one Zarr group")
    attrs = metadata.get("attributes")
    if type(attrs) is not dict:
        raise ValueError("Selected stimulus-epoch group attributes are not exact")
    schema_id = attrs.get("schema_id")
    schema_version = attrs.get("schema_version")
    if type(schema_id) is not str or type(schema_version) is not int:
        raise ValueError(
            "Selected stimulus-epoch schema identity is absent or malformed"
        )
    return schema_id, schema_version


def _require_complete_named_candidate(group: Any, *, run_name: str) -> tuple[str, ...]:
    attrs = group.attrs
    errors: list[str] = []
    if attrs.get(RUN_COMPLETION_CONTRACT_ATTR) != RUN_COMPLETION_CONTRACT:
        errors.append("completion contract is absent or unsupported")
    if attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
        errors.append("selected candidate is not complete")
    if attrs.get(RUN_NAME_ATTR) != run_name:
        errors.append("completion run-name binding does not match explicit selection")
    completed_at = attrs.get(RUN_COMPLETED_AT_ATTR)
    if type(completed_at) is not str or not completed_at.strip():
        errors.append("completion timestamp is absent or malformed")
    if attrs.get("stage_selector_eligible") is not False:
        errors.append("selected v2 candidate is not exact selector-ineligible")
    if attrs.get("storage_candidate_profile_promoted") is not False:
        errors.append("selected v2 candidate has an invalid profile-promotion state")
    return tuple(errors)


def _read_exact_v2(
    archive: Path,
    *,
    run_name: str,
    run_path: str,
    policy: StimulusEpochCompatibilityPolicy,
) -> StimulusEpochSnapshot:
    equivalence = validate_direct_consolidated_subtree(
        archive,
        subtree_path=run_path,
    )
    if (
        equivalence.array_count != EXPECTED_V2_ARRAY_COUNT
        or equivalence.group_count != EXPECTED_V2_GROUP_COUNT
        or equivalence.node_count != EXPECTED_V2_ARRAY_COUNT + EXPECTED_V2_GROUP_COUNT
    ):
        raise ValueError(
            "Stimulus-epoch v2 persisted metadata inventory is not exact: "
            f"{equivalence.to_json()}"
        )

    root = zarr.open_group(
        str(archive),
        mode="r",
        zarr_format=3,
        use_consolidated=True,
    )
    try:
        group = root[run_path]
    except KeyError as exc:
        raise ValueError(
            "Explicit stimulus-epoch candidate is absent from published "
            "consolidated metadata"
        ) from exc

    errors = list(_require_complete_named_candidate(group, run_name=run_name))
    errors.extend(
        validate_stimulus_epoch_array_manifest(
            group,
            byte_planner_adopted=True,
        )
    )
    errors.extend(validate_stimulus_epoch_candidate_lineage(group))
    errors.extend(validate_stimulus_epoch_run_manifest(group))
    try:
        declarations = build_stimulus_epoch_array_declarations(
            group,
            byte_planner_adopted=True,
        )
        errors.extend(
            validate_exact_tabular_storage_receipt(
                group,
                declarations=declarations,
            )
        )
        if len(declarations) != EXPECTED_V2_ARRAY_COUNT:
            errors.append("stimulus-epoch v2 declaration count is not exact")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(str(exc))
    if errors:
        raise ValueError("Invalid stimulus-epoch v2 candidate: " + "; ".join(errors))

    return StimulusEpochSnapshot(
        run_name=run_name,
        run_path=run_path,
        schema_id=STIMULUS_EPOCH_RUN_SCHEMA_ID,
        schema_version=STIMULUS_EPOCH_RUN_SCHEMA_VERSION,
        compatibility_policy=policy,
        segments=read_epoch_segments(group),
        metadata_equivalence=equivalence,
    )


def _read_explicit_legacy_v1(
    archive: Path,
    *,
    run_name: str,
    run_path: str,
    policy: StimulusEpochCompatibilityPolicy,
) -> StimulusEpochSnapshot:
    root = zarr.open_group(
        str(archive),
        mode="r",
        zarr_format=3,
        use_consolidated=False,
    )
    try:
        group = root[run_path]
    except KeyError as exc:
        raise ValueError("Explicit legacy stimulus-epoch run does not exist") from exc
    errors = validate_legacy_stimulus_epoch_source(group)
    if errors:
        raise ValueError(
            "Invalid explicit legacy stimulus-epoch v1 run: " + "; ".join(errors)
        )
    return StimulusEpochSnapshot(
        run_name=run_name,
        run_path=run_path,
        schema_id=LEGACY_STIMULUS_EPOCH_SCHEMA_ID,
        schema_version=LEGACY_STIMULUS_EPOCH_SCHEMA_VERSION,
        compatibility_policy=policy,
        segments=read_epoch_segments(group),
        metadata_equivalence=None,
    )


def read_stimulus_epoch_snapshot(
    archive_path: str | Path,
    *,
    run_name: str,
    compatibility_policy: StimulusEpochCompatibilityPolicy = (
        StimulusEpochCompatibilityPolicy.EXACT_V2_ONLY
    ),
) -> StimulusEpochSnapshot:
    """Read one explicitly named run without selector or schema probing.

    Exact v2 is the maintained path and requires the persisted consolidated
    generation to match every direct declaration.  A v1 read is allowed only
    when the caller explicitly opts into :attr:`ALLOW_EXPLICIT_V1`; failure of
    an explicitly selected v2 run is always terminal and never falls back.
    """

    if not isinstance(compatibility_policy, StimulusEpochCompatibilityPolicy):
        raise TypeError(
            "compatibility_policy must be a StimulusEpochCompatibilityPolicy"
        )
    archive = Path(archive_path).expanduser().resolve()
    if not archive.is_dir():
        raise FileNotFoundError(f"Zarr archive does not exist: {archive}")
    selected = _safe_run_name(run_name)
    run_path = f"{PARENT_PATH}/{selected}"
    schema_id, schema_version = _read_direct_schema_identity(archive, run_path)

    if (
        schema_id == STIMULUS_EPOCH_RUN_SCHEMA_ID
        and schema_version == STIMULUS_EPOCH_RUN_SCHEMA_VERSION
    ):
        return _read_exact_v2(
            archive,
            run_name=selected,
            run_path=run_path,
            policy=compatibility_policy,
        )
    if (
        schema_id == LEGACY_STIMULUS_EPOCH_SCHEMA_ID
        and schema_version == LEGACY_STIMULUS_EPOCH_SCHEMA_VERSION
    ):
        if (
            compatibility_policy
            is not StimulusEpochCompatibilityPolicy.ALLOW_EXPLICIT_V1
        ):
            raise ValueError(
                "Legacy stimulus-epoch v1 requires the explicit "
                "ALLOW_EXPLICIT_V1 compatibility policy"
            )
        return _read_explicit_legacy_v1(
            archive,
            run_name=selected,
            run_path=run_path,
            policy=compatibility_policy,
        )
    raise ValueError(
        "Explicit stimulus-epoch run has unsupported exact schema identity: "
        f"{schema_id!r} version {schema_version!r}"
    )


__all__ = [
    "PARENT_PATH",
    "StimulusEpochCompatibilityPolicy",
    "StimulusEpochSnapshot",
    "read_stimulus_epoch_snapshot",
]

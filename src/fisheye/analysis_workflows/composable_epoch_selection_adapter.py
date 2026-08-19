"""Adapt one verified epoch-v2 selection into GoodBatBadBat compositions.

The exact epoch loader intentionally does not assign analysis roles.  This
module is the explicit boundary where an operator or a protocol resolver
binds the three GoodBatBadBat windows to ``black_before``, ``chaser``, and
``black_after``.  It only accepts a loader-minted, verified
``ResolvedEpochSelection`` and never reads labels, window order, or protocol
mode as role authority.

The adapter also requires a complete caller-supplied evidence record for the
timeline, run, timing, and source video.  The record is checked against the
resolved selection and all declared digests are recomputed.  Missing evidence
is therefore a hard error rather than an opportunity to invent a digest.

This module is pure: it performs no Zarr writes, selector resolution, or
registry updates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from types import MappingProxyType
from typing import Any, Mapping

from fisheye.analysis_workflows.composable_stimulus_selection import (
    CompiledSelection,
    RoleMetadata,
    SelectionSpec,
    TimelineAuthority,
    canonical_json,
    canonical_sha256,
    compile_selection,
    interval_annotation_reference,
    member,
    union,
)
from fisheye.analysis_workflows.resolved_epoch_selection import (
    ResolvedEpochSelection,
    ResolvedEpochSelectionError,
)


ADAPTER_SCHEMA_ID = "palette.goodbatbadbat_composable_epoch_selection_adapter.v1"
GOODBATBADBAT_ROLE_ORDER = ("black_before", "chaser", "black_after")
SELECTION_ID_BY_ROLE = {
    "black_before": "black_before",
    "chaser": "chaser",
    "black_after": "black_after",
    "all_black": "all_black",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SELECTOR_ALIASES = {
    "latest",
    "latest_complete",
    "authoritative_run",
    "current",
    "default",
    "selected",
    "active",
}


class ComposableEpochSelectionAdapterError(ValueError):
    """Raised when an exact epoch selection cannot be composed safely."""


def _require_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ComposableEpochSelectionAdapterError(
            f"{name} must be one exact nonempty string."
        )
    return value


def _require_digest(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ComposableEpochSelectionAdapterError(
            f"{name} must be one lowercase SHA-256 digest."
        )
    return value


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _strict_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ComposableEpochSelectionAdapterError(f"{name} must be one object.")
    try:
        canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise ComposableEpochSelectionAdapterError(
            f"{name} must contain strict finite JSON values."
        ) from exc
    return value


@dataclass(frozen=True)
class TimelineAuthorityEvidence:
    """Exact external evidence needed to construct one timeline authority.

    ``source_metadata`` is the persisted evidence envelope.  Its required
    fields bind the resolved epoch run, source timeline, timing authority, and
    source-video metadata together.  The supplied digest is checked against
    that complete envelope; this class does not calculate a substitute for a
    missing producer digest.
    """

    recording_id: str
    timeline_id: str
    stimulus_authority_id: str
    acquisition_frame_domain: str
    source_video_metadata_ref: str
    source_video_metadata_sha256: str
    source_video_metadata: Mapping[str, Any]
    acquisition_clock_authority_ref: str
    acquisition_clock_authority_sha256: str
    acquisition_clock_authority: Mapping[str, Any]
    source_metadata_sha256: str
    source_metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in (
            "recording_id",
            "timeline_id",
            "stimulus_authority_id",
            "acquisition_frame_domain",
            "source_video_metadata_ref",
            "acquisition_clock_authority_ref",
        ):
            _require_text(getattr(self, name), name=name)
        for name in (
            "source_video_metadata_sha256",
            "acquisition_clock_authority_sha256",
            "source_metadata_sha256",
        ):
            _require_digest(getattr(self, name), name=name)
        for name in (
            "source_video_metadata",
            "acquisition_clock_authority",
            "source_metadata",
        ):
            mapping = _strict_mapping(getattr(self, name), name=name)
            object.__setattr__(self, name, _freeze(mapping))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": f"{ADAPTER_SCHEMA_ID}.timeline_authority_evidence",
            "schema_version": 1,
            "recording_id": self.recording_id,
            "timeline_id": self.timeline_id,
            "stimulus_authority_id": self.stimulus_authority_id,
            "acquisition_frame_domain": self.acquisition_frame_domain,
            "source_video_metadata_ref": self.source_video_metadata_ref,
            "source_video_metadata_sha256": self.source_video_metadata_sha256,
            "source_video_metadata": _thaw(self.source_video_metadata),
            "acquisition_clock_authority_ref": self.acquisition_clock_authority_ref,
            "acquisition_clock_authority_sha256": (
                self.acquisition_clock_authority_sha256
            ),
            "acquisition_clock_authority": _thaw(self.acquisition_clock_authority),
            "source_metadata_sha256": self.source_metadata_sha256,
            "source_metadata": _thaw(self.source_metadata),
        }


@dataclass(frozen=True)
class EpochRoleBinding:
    """One explicit role-to-window binding.

    Exactly one of ``window_id`` and ``source_interval_digest`` must be set.
    A label, mode, ordinal, or selector name is never accepted as a binding.
    """

    window_id: int | None = None
    source_interval_digest: str | None = None

    def __post_init__(self) -> None:
        by_window = self.window_id is not None
        by_digest = self.source_interval_digest is not None
        if by_window == by_digest:
            raise ComposableEpochSelectionAdapterError(
                "Each role binding must specify exactly one window_id or "
                "source_interval_digest."
            )
        if by_window and (type(self.window_id) is not int or self.window_id < 0):
            raise ComposableEpochSelectionAdapterError(
                "role binding window_id must be one non-negative integer."
            )
        if by_digest:
            _require_digest(
                self.source_interval_digest,
                name="role binding source_interval_digest",
            )

    @classmethod
    def by_window_id(cls, window_id: int) -> "EpochRoleBinding":
        return cls(window_id=window_id)

    @classmethod
    def by_source_interval_digest(cls, digest: str) -> "EpochRoleBinding":
        return cls(source_interval_digest=digest)

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "source_interval_digest": self.source_interval_digest,
        }


def _selection_identity(selection: ResolvedEpochSelection) -> dict[str, Any]:
    return {
        "source_epoch_run_path": selection.run_path,
        "source_epoch_run_manifest_sha256": selection.run_manifest_digest,
        "source_epoch_run_manifest_payload_sha256": (
            selection.run_manifest_payload_digest
        ),
        "source_epoch_logical_content_sha256": (
            selection.source_epoch_logical_content_digest
        ),
        "source_epoch_lineage_hash": selection.source_epoch_lineage_hash,
        "source_epoch_lineage_payload_sha256": (
            selection.source_epoch_lineage_payload_digest
        ),
        "source_timeline_digest": selection.source_timeline_digest,
        "selection_sha256": selection.selection_digest,
    }


def _require_selection(selection: object) -> ResolvedEpochSelection:
    if type(selection) is not ResolvedEpochSelection:
        raise ComposableEpochSelectionAdapterError(
            "A loader-minted ResolvedEpochSelection is required."
        )
    try:
        selection.assert_verified()  # type: ignore[union-attr]
    except ResolvedEpochSelectionError as exc:
        raise ComposableEpochSelectionAdapterError(
            f"Resolved epoch selection is not verified: {exc}"
        ) from exc
    run_name = _require_text(selection.run_name, name="resolved epoch run_name")
    if run_name.lower() in _SELECTOR_ALIASES or "/" in run_name or "\\" in run_name:
        raise ComposableEpochSelectionAdapterError(
            "Resolved epoch selection must bind one explicit non-selector run."
        )
    expected_path = f"analysis/stimulus_epoch_runs/{run_name}"
    if selection.run_path != expected_path:
        raise ComposableEpochSelectionAdapterError(
            "Resolved epoch selection run path is not bound to its exact run name."
        )
    if selection.run_schema_id != "palette.stimulus_epoch_windows.v2":
        raise ComposableEpochSelectionAdapterError(
            "Only exact stimulus-epoch v2 selections may be composed."
        )
    if selection.run_schema_version != 2:
        raise ComposableEpochSelectionAdapterError(
            "Only exact stimulus-epoch v2 selections may be composed."
        )
    if selection.recording_timing_authority_status != "bound":
        raise ComposableEpochSelectionAdapterError(
            "A bound recording timing authority is required; legacy-missing "
            "timing cannot be composed."
        )
    if selection.recording_timing_authority_sha256 is None:
        raise ComposableEpochSelectionAdapterError(
            "Resolved epoch selection lacks its exact recording timing digest."
        )
    return selection


def _validate_timeline_evidence(
    selection: ResolvedEpochSelection,
    evidence: TimelineAuthorityEvidence,
) -> TimelineAuthority:
    timeline = _strict_mapping(
        selection.source_timeline_identity,
        name="resolved source timeline identity",
    )
    source_run = _require_text(
        timeline.get("source_stimulus_run"),
        name="resolved source stimulus run",
    )
    if source_run.lower() in _SELECTOR_ALIASES:
        raise ComposableEpochSelectionAdapterError(
            "resolved source stimulus run cannot be a selector alias."
        )
    source_path = _require_text(
        timeline.get("source_stimulus_path"),
        name="resolved source stimulus path",
    )
    if source_path != f"analysis/stimulus_runs/{source_run}":
        raise ComposableEpochSelectionAdapterError(
            "resolved source stimulus path is not bound to its exact run."
        )
    recording_id = timeline.get("recording_id")
    if evidence.recording_id != recording_id:
        raise ComposableEpochSelectionAdapterError(
            "timeline evidence recording_id differs from the resolved source."
        )
    if evidence.timeline_id != selection.source_timeline_digest:
        raise ComposableEpochSelectionAdapterError(
            "timeline evidence timeline_id is not the exact source timeline digest."
        )
    if evidence.stimulus_authority_id != selection.run_path:
        raise ComposableEpochSelectionAdapterError(
            "stimulus authority must identify the exact resolved epoch run path."
        )
    if evidence.acquisition_clock_authority_sha256 != (
        selection.recording_timing_authority_sha256
    ):
        raise ComposableEpochSelectionAdapterError(
            "clock authority digest differs from the resolved recording timing "
            "authority."
        )

    video = _strict_mapping(
        evidence.source_video_metadata,
        name="source video metadata evidence",
    )
    if canonical_sha256(video) != evidence.source_video_metadata_sha256:
        raise ComposableEpochSelectionAdapterError(
            "source video metadata digest is stale."
        )
    if video.get("total_frames") != selection.native_frame_count:
        raise ComposableEpochSelectionAdapterError(
            "source video frame count differs from the resolved epoch selection."
        )
    try:
        video_fps = float(video.get("fps"))
    except (TypeError, ValueError) as exc:
        raise ComposableEpochSelectionAdapterError(
            "source video metadata FPS is missing or malformed."
        ) from exc
    if (
        isinstance(video.get("fps"), bool)
        or not math.isfinite(video_fps)
        or video_fps != selection.fps
    ):
        raise ComposableEpochSelectionAdapterError(
            "source video FPS differs from the resolved epoch selection."
        )

    clock = _strict_mapping(
        evidence.acquisition_clock_authority,
        name="acquisition clock authority evidence",
    )
    if canonical_sha256(clock) != evidence.acquisition_clock_authority_sha256:
        raise ComposableEpochSelectionAdapterError(
            "acquisition clock authority digest is stale."
        )
    if clock.get("recording_id") != selection.source_timeline_identity.get(
        "recording_id"
    ):
        raise ComposableEpochSelectionAdapterError(
            "acquisition clock recording identity differs from the source timeline."
        )
    if clock.get("frame_count") != selection.native_frame_count:
        raise ComposableEpochSelectionAdapterError(
            "acquisition clock frame count differs from the resolved epoch selection."
        )
    try:
        clock_fps = float(clock.get("nominal_fps"))
    except (TypeError, ValueError) as exc:
        raise ComposableEpochSelectionAdapterError(
            "acquisition clock FPS is missing or malformed."
        ) from exc
    if (
        isinstance(clock.get("nominal_fps"), bool)
        or not math.isfinite(clock_fps)
        or clock_fps != selection.fps
    ):
        raise ComposableEpochSelectionAdapterError(
            "acquisition clock FPS differs from the resolved epoch selection."
        )

    clock_record = clock.get("acquisition_frame_clock")
    if not isinstance(clock_record, Mapping):
        raise ComposableEpochSelectionAdapterError(
            "acquisition clock authority lacks its exact clock record."
        )
    clock_path = clock_record.get("run_path")
    if clock_path != evidence.acquisition_clock_authority_ref:
        raise ComposableEpochSelectionAdapterError(
            "acquisition clock reference is not bound to its exact record."
        )

    source_metadata = _strict_mapping(
        evidence.source_metadata,
        name="source metadata evidence",
    )
    if canonical_sha256(source_metadata) != evidence.source_metadata_sha256:
        raise ComposableEpochSelectionAdapterError("source metadata digest is stale.")
    expected_metadata: dict[str, Any] = {
        "recording_id": evidence.recording_id,
        "source_timeline_digest": selection.source_timeline_digest,
        "source_epoch_run_path": selection.run_path,
        "source_epoch_run_manifest_sha256": selection.run_manifest_digest,
        "source_epoch_run_manifest_payload_sha256": (
            selection.run_manifest_payload_digest
        ),
        "source_epoch_logical_content_sha256": (
            selection.source_epoch_logical_content_digest
        ),
        "source_epoch_lineage_hash": selection.source_epoch_lineage_hash,
        "source_epoch_lineage_payload_sha256": (
            selection.source_epoch_lineage_payload_digest
        ),
        "timing_authority": selection.timing_authority,
        "source_video_metadata_ref": evidence.source_video_metadata_ref,
        "source_video_metadata_sha256": evidence.source_video_metadata_sha256,
        "acquisition_clock_authority_ref": evidence.acquisition_clock_authority_ref,
        "acquisition_clock_authority_sha256": (
            evidence.acquisition_clock_authority_sha256
        ),
        "acquisition_frame_domain": evidence.acquisition_frame_domain,
        "frame_count": selection.native_frame_count,
        "fps": selection.fps,
    }
    for key, expected in expected_metadata.items():
        if source_metadata.get(key) != expected:
            raise ComposableEpochSelectionAdapterError(
                f"source metadata evidence field {key!r} is not bound to the "
                "resolved selection."
            )

    return TimelineAuthority(
        recording_id=evidence.recording_id,
        timeline_id=evidence.timeline_id,
        stimulus_authority_id=evidence.stimulus_authority_id,
        stimulus_authority_sha256=selection.source_epoch_logical_content_digest,
        acquisition_frame_domain=evidence.acquisition_frame_domain,
        acquisition_frame_count=selection.native_frame_count,
        source_video_metadata_ref=evidence.source_video_metadata_ref,
        source_video_metadata_sha256=evidence.source_video_metadata_sha256,
        acquisition_clock_authority_ref=evidence.acquisition_clock_authority_ref,
        acquisition_clock_authority_sha256=evidence.acquisition_clock_authority_sha256,
        source_metadata_sha256=evidence.source_metadata_sha256,
    )


def _interval_by_binding(
    selection: ResolvedEpochSelection,
    binding: EpochRoleBinding,
) -> Any:
    matches = []
    for interval in selection.intervals:
        if binding.window_id is not None and interval.window_id == binding.window_id:
            matches.append(interval)
        if (
            binding.source_interval_digest is not None
            and interval.source_interval_digest == binding.source_interval_digest
        ):
            matches.append(interval)
    if len(matches) != 1:
        raise ComposableEpochSelectionAdapterError(
            "Each explicit role binding must resolve to exactly one source interval."
        )
    return matches[0]


def _role_member(
    *,
    selection: ResolvedEpochSelection,
    authority: TimelineAuthority,
    role: str,
    binding: EpochRoleBinding,
) -> Any:
    interval = _interval_by_binding(selection, binding)
    occurrence_id = interval.occurrence_identity.get("occurrence_id")
    if type(occurrence_id) is not str or not occurrence_id:
        raise ComposableEpochSelectionAdapterError(
            "source interval occurrence identity is missing."
        )
    metadata = {
        "adapter_schema_id": ADAPTER_SCHEMA_ID,
        "role": role,
        "role_binding": binding.to_dict(),
        "source_interval_digest": interval.source_interval_digest,
        "source_metadata_identity": _thaw(interval.source_metadata_identity),
        "occurrence_identity": _thaw(interval.occurrence_identity),
        "resolved_epoch_selection": _selection_identity(selection),
    }
    reference = interval_annotation_reference(
        reference_id=interval.source_interval_digest,
        label=interval.label,
        start_frame=interval.start_frame,
        end_frame=interval.end_frame,
        authority=authority,
        occurrence_id=occurrence_id,
    )
    return member(
        reference,
        role=RoleMetadata(role=role, label=role, metadata=metadata),
    )


def _selection_spec_metadata(
    selection: ResolvedEpochSelection,
    *,
    role: str,
    binding: EpochRoleBinding | None,
    source_roles: tuple[str, ...],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "adapter_schema_id": ADAPTER_SCHEMA_ID,
        "protocol_profile": "goodbatbadbat_pre_training_post_v1",
        "role": role,
        "source_roles": list(source_roles),
        "resolved_epoch_selection": _selection_identity(selection),
    }
    if binding is not None:
        metadata["role_binding"] = binding.to_dict()
    return metadata


@dataclass(frozen=True)
class GoodBatBadBatComposableSelections:
    """Named immutable compositions produced by the explicit role adapter."""

    timeline_authority: TimelineAuthority
    source_selection_digest: str
    black_before: CompiledSelection
    chaser: CompiledSelection
    black_after: CompiledSelection
    all_black: CompiledSelection | None = None

    @property
    def pre(self) -> CompiledSelection:
        return self.black_before

    @property
    def training(self) -> CompiledSelection:
        return self.chaser

    @property
    def post(self) -> CompiledSelection:
        return self.black_after

    @property
    def named(self) -> Mapping[str, CompiledSelection]:
        result: dict[str, CompiledSelection] = {
            "black_before": self.black_before,
            "pre": self.black_before,
            "chaser": self.chaser,
            "training": self.chaser,
            "black_after": self.black_after,
            "post": self.black_after,
        }
        if self.all_black is not None:
            result["all_black"] = self.all_black
        return MappingProxyType(result)

    def __getitem__(self, name: str) -> CompiledSelection:
        try:
            return self.named[name]
        except KeyError as exc:
            raise KeyError(f"unknown GoodBatBadBat selection name: {name!r}") from exc


def compile_goodbatbadbat_selections(
    selection: ResolvedEpochSelection,
    *,
    timeline_evidence: TimelineAuthorityEvidence,
    role_bindings: Mapping[str, EpochRoleBinding],
    include_all_black: bool = False,
) -> GoodBatBadBatComposableSelections:
    """Compile exact GoodBatBadBat pre/training/post selections.

    ``role_bindings`` is intentionally keyed by the canonical role names and
    must contain exactly three ``EpochRoleBinding`` values.  The adapter does
    not inspect source labels or infer which interval is before, during, or
    after the chaser.
    """

    resolved = _require_selection(selection)
    if type(timeline_evidence) is not TimelineAuthorityEvidence:
        raise ComposableEpochSelectionAdapterError(
            "TimelineAuthorityEvidence is required; authority evidence cannot be "
            "inferred."
        )
    if type(include_all_black) is not bool:
        raise ComposableEpochSelectionAdapterError(
            "include_all_black must be one exact boolean."
        )
    if not isinstance(role_bindings, Mapping):
        raise ComposableEpochSelectionAdapterError(
            "role_bindings must be one explicit mapping."
        )
    if set(role_bindings) != set(GOODBATBADBAT_ROLE_ORDER):
        raise ComposableEpochSelectionAdapterError(
            "role_bindings must contain exactly black_before, chaser, and black_after."
        )
    bindings: dict[str, EpochRoleBinding] = {}
    for role in GOODBATBADBAT_ROLE_ORDER:
        binding = role_bindings[role]
        if type(binding) is not EpochRoleBinding:
            raise ComposableEpochSelectionAdapterError(
                f"role binding for {role!r} must be one explicit window_id or "
                "source_interval_digest binding."
            )
        bindings[role] = binding

    if len(resolved.intervals) != len(GOODBATBADBAT_ROLE_ORDER):
        raise ComposableEpochSelectionAdapterError(
            "GoodBatBadBat composition requires exactly three resolved windows."
        )
    authority = _validate_timeline_evidence(resolved, timeline_evidence)
    intervals = {
        role: _interval_by_binding(resolved, binding)
        for role, binding in bindings.items()
    }
    if len({interval.source_interval_digest for interval in intervals.values()}) != 3:
        raise ComposableEpochSelectionAdapterError(
            "GoodBatBadBat role bindings must resolve to three distinct intervals."
        )
    if len({interval.occurrence_identity.get("occurrence_id") for interval in intervals.values()}) != 3:
        raise ComposableEpochSelectionAdapterError(
            "GoodBatBadBat role bindings must preserve three distinct occurrences."
        )

    members = {
        role: _role_member(
            selection=resolved,
            authority=authority,
            role=role,
            binding=bindings[role],
        )
        for role in GOODBATBADBAT_ROLE_ORDER
    }

    compiled: dict[str, CompiledSelection] = {}
    for role in GOODBATBADBAT_ROLE_ORDER:
        compiled[role] = compile_selection(
            SelectionSpec(
                selection_id=SELECTION_ID_BY_ROLE[role],
                expression=members[role],
                aggregation_policy="keep_occurrences",
                metadata=_selection_spec_metadata(
                    resolved,
                    role=role,
                    binding=bindings[role],
                    source_roles=(role,),
                ),
            ),
            expected_authority=authority,
        )

    all_black: CompiledSelection | None = None
    if include_all_black:
        all_black = compile_selection(
            SelectionSpec(
                selection_id=SELECTION_ID_BY_ROLE["all_black"],
                expression=union(members["black_before"], members["black_after"]),
                aggregation_policy="keep_occurrences",
                metadata=_selection_spec_metadata(
                    resolved,
                    role="all_black",
                    binding=None,
                    source_roles=("black_before", "black_after"),
                ),
            ),
            expected_authority=authority,
        )

    return GoodBatBadBatComposableSelections(
        timeline_authority=authority,
        source_selection_digest=resolved.selection_digest,
        black_before=compiled["black_before"],
        chaser=compiled["chaser"],
        black_after=compiled["black_after"],
        all_black=all_black,
    )


adapt_resolved_epoch_selection = compile_goodbatbadbat_selections


__all__ = [
    "ADAPTER_SCHEMA_ID",
    "GOODBATBADBAT_ROLE_ORDER",
    "ComposableEpochSelectionAdapterError",
    "EpochRoleBinding",
    "GoodBatBadBatComposableSelections",
    "TimelineAuthorityEvidence",
    "adapt_resolved_epoch_selection",
    "compile_goodbatbadbat_selections",
]

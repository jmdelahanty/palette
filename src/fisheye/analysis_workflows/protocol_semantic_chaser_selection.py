"""Protocol-semantic successor for composable chaser epoch selections.

This pure, selector-ineligible adapter keeps the producer protocol hierarchy
separate from derived epoch windows.  It never treats the legacy
``black_before``/``chaser``/``black_after`` role names as protocol identity.

For historical Citrus snapshot-v1 sources, acquisition did not declare the
raw camera-correspondence ``STEP_END`` interval convention.  The pending v1
policy therefore takes only the boundary interpretation common to both
possibilities: ``[STEP_START, STEP_END)``.  Citrus snapshot v2 supersedes that
ambiguity with half-open ``stimulus_frame_num`` execution intervals.  V2 is
accepted only with a loader-minted, raw-H5-bound acquisition mapping; camera
frame correspondence alone remains insufficient.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import h5py
import numpy as np
import zarr

from fisheye.analysis_workflows.composable_epoch_selection_adapter import (
    ComposableEpochSelectionAdapterError,
    EpochRoleBinding,
    TimelineAuthorityEvidence,
    _interval_by_binding,
    _require_selection,
    _selection_identity,
    _thaw,
    _validate_timeline_evidence,
)
from fisheye.analysis_workflows.composable_stimulus_selection import (
    CompiledSelection,
    RoleMetadata,
    SelectionSpec,
    TimelineAuthority,
    canonical_sha256,
    compile_selection,
    interval_annotation_reference,
    member,
)
from fisheye.analysis_workflows.chaser_profile_applicability import (
    CapabilityAssessment,
    CapabilityState,
)
from fisheye.analysis_workflows.resolved_epoch_selection import (
    ResolvedEpochSelection,
)
from fisheye.analysis.stimulus_epoch_schema import (
    STIMULUS_SOURCE_FINGERPRINT_ALGORITHM,
    stimulus_group_logical_fingerprint,
)
from fisheye.shared.protocol_semantic_contract import (
    ProtocolSemanticContractError,
    ProtocolSemanticSnapshot,
    ProtocolStepIdentity,
    TRIAL_INDEX_INTEGRITY_LOCAL,
    read_materialized_protocol_semantic_snapshot,
    read_protocol_semantic_snapshot,
    validate_protocol_semantic_snapshot,
)
from fisheye.shared.protocol_execution_contract import (
    CHASER_PHASE_NAMES,
    ProtocolExecutionContractError,
    ProtocolStimulusFrameInterval,
    read_materialized_protocol_execution_index,
    read_protocol_execution_index,
)
from fisheye.shared.frame_bound_acquisition_identity import (
    PairedFrameBoundChaserSource,
    load_paired_frame_bound_chaser_source_from_binding,
)
from fisheye.shared.stimulus_coordinate_contract import (
    StimulusCoordinateContractError,
)
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr_io import open_zarr_root


ADAPTER_SCHEMA_ID = "palette.protocol_semantic_chaser_selection_adapter.v2"
CHASER_WINDOW_ROLES = (
    "chaser_pre",
    "chaser_training",
    "chaser_post",
)
CHASER_SOURCE_LABEL_BY_ROLE = MappingProxyType(
    {
        "chaser_pre": "pre_event",
        "chaser_training": "training_event",
        "chaser_post": "post_event",
    }
)
STANDALONE_SOLID_BLACK_ROLE = "standalone_solid_black"
LEGACY_ROLE_NAMES = frozenset({"black_before", "chaser", "black_after", "all_black"})

STEP_END_PENDING = "producer_contract_pending"
STEP_END_INCLUSIVE = "producer_declared_step_end_inclusive"
STEP_END_EXCLUSIVE = "producer_declared_step_end_exclusive"
STEP_END_POLICIES = frozenset(
    {STEP_END_PENDING, STEP_END_INCLUSIVE, STEP_END_EXCLUSIVE}
)
V2_ACQUISITION_PROJECTION_POLICY = (
    "latest_stimulus_frame_per_source_acquisition_index_v1"
)

_EVIDENCE_SEAL = object()
_TIMELINE_EVIDENCE_SEAL = object()

PROTOCOL_SEMANTIC_CHASER_WINDOWS_CAPABILITY_ID = (
    "protocol_semantic_chaser_windows"
)
STANDALONE_SOLID_BLACK_BASELINE_CAPABILITY_ID = (
    "standalone_solid_black_baseline"
)
PROTOCOL_SEMANTIC_SELECTION_PRODUCTION_CAPABILITY_ID = (
    "protocol_semantic_selection_production_eligibility"
)


class ProtocolSemanticChaserSelectionError(ComposableEpochSelectionAdapterError):
    """Raised when semantic chaser hierarchy or containment is not exact."""


@dataclass(frozen=True, slots=True, init=False)
class ProtocolSemanticTimelineEvidence:
    """Loader-minted current archive evidence for one semantic timeline."""

    evidence: TimelineAuthorityEvidence
    _verification_seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        evidence: TimelineAuthorityEvidence,
        _verification_seal: object | None = None,
    ) -> None:
        if _verification_seal is not _TIMELINE_EVIDENCE_SEAL:
            raise ProtocolSemanticChaserSelectionError(
                "Protocol semantic timeline evidence requires its strict loader."
            )
        if type(evidence) is not TimelineAuthorityEvidence:
            raise ProtocolSemanticChaserSelectionError(
                "Protocol semantic timeline evidence is malformed."
            )
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "_verification_seal", _TIMELINE_EVIDENCE_SEAL)

    def to_dict(self) -> dict[str, Any]:
        return self.evidence.to_dict()


def _exact_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ProtocolSemanticChaserSelectionError(
            f"{name} must be one exact nonempty string."
        )
    return value


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


@dataclass(frozen=True)
class ProtocolStepFrameBounds:
    """One producer interval projected onto a declared native frame axis.

    The historical attribute names remain for storage compatibility.  For v2
    their values are acquisition-array indices, as declared by ``frame_axis``.
    """

    step_index: int
    start_camera_frame: int
    end_camera_frame: int
    frame_axis: str = "legacy_camera_frame_id"
    source_start_stimulus_frame_inclusive: int | None = None
    source_end_stimulus_frame_exclusive: int | None = None
    mapped_stimulus_frame_count: int | None = None
    mapped_acquisition_frame_count: int | None = None
    reused_acquisition_frame_count: int = 0
    acquisition_gap_count: int = 0

    def __post_init__(self) -> None:
        if type(self.step_index) is not int or self.step_index < 0:
            raise ProtocolSemanticChaserSelectionError(
                "step_index must be one non-negative exact integer."
            )
        for name in ("start_camera_frame", "end_camera_frame"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ProtocolSemanticChaserSelectionError(
                    f"{name} must be one non-negative exact integer."
                )
        if self.end_camera_frame < self.start_camera_frame:
            raise ProtocolSemanticChaserSelectionError(
                "Raw protocol step end precedes its start."
            )
        if self.frame_axis not in {
            "legacy_camera_frame_id",
            "source_acquisition_frame_index",
        }:
            raise ProtocolSemanticChaserSelectionError(
                "Protocol step frame_axis is unsupported."
            )
        stimulus_values = (
            self.source_start_stimulus_frame_inclusive,
            self.source_end_stimulus_frame_exclusive,
            self.mapped_stimulus_frame_count,
            self.mapped_acquisition_frame_count,
        )
        if self.frame_axis == "legacy_camera_frame_id":
            if any(value is not None for value in stimulus_values):
                raise ProtocolSemanticChaserSelectionError(
                    "Legacy bounds cannot claim stimulus-frame mapping evidence."
                )
        else:
            start, end, stimulus_count, acquisition_count = stimulus_values
            if (
                type(start) is not int
                or type(end) is not int
                or type(stimulus_count) is not int
                or type(acquisition_count) is not int
                or start < 0
                or end <= start
                or stimulus_count != end - start
                or acquisition_count <= 0
                or acquisition_count > stimulus_count
            ):
                raise ProtocolSemanticChaserSelectionError(
                    "Acquisition bounds require one complete half-open stimulus interval."
                )
        if type(self.acquisition_gap_count) is not int or self.acquisition_gap_count < 0:
            raise ProtocolSemanticChaserSelectionError(
                "acquisition_gap_count must be one non-negative exact integer."
            )
        if (
            type(self.reused_acquisition_frame_count) is not int
            or self.reused_acquisition_frame_count < 0
        ):
            raise ProtocolSemanticChaserSelectionError(
                "reused_acquisition_frame_count must be one non-negative exact integer."
            )
        if (
            self.frame_axis == "legacy_camera_frame_id"
            and (
                self.acquisition_gap_count != 0
                or self.reused_acquisition_frame_count != 0
            )
        ):
            raise ProtocolSemanticChaserSelectionError(
                "Legacy camera bounds cannot claim acquisition-gap evidence."
            )

    def to_dict(self) -> dict[str, int]:
        return {
            "step_index": self.step_index,
            "start_camera_frame": self.start_camera_frame,
            "end_camera_frame": self.end_camera_frame,
            "frame_axis": self.frame_axis,
            "source_start_stimulus_frame_inclusive": (
                self.source_start_stimulus_frame_inclusive
            ),
            "source_end_stimulus_frame_exclusive": (
                self.source_end_stimulus_frame_exclusive
            ),
            "mapped_stimulus_frame_count": self.mapped_stimulus_frame_count,
            "mapped_acquisition_frame_count": (
                self.mapped_acquisition_frame_count
            ),
            "reused_acquisition_frame_count": (
                self.reused_acquisition_frame_count
            ),
            "acquisition_gap_count": self.acquisition_gap_count,
            "acquisition_projection_policy": (
                V2_ACQUISITION_PROJECTION_POLICY
                if self.frame_axis == "source_acquisition_frame_index"
                else None
            ),
        }


@dataclass(frozen=True)
class ProtocolPhaseFrameBounds:
    """One exact v2 CHASER phase projected to acquisition rows."""

    role: str
    bounds: ProtocolStepFrameBounds

    def __post_init__(self) -> None:
        if self.role not in CHASER_PHASE_NAMES:
            raise ProtocolSemanticChaserSelectionError(
                "Protocol phase role is unsupported."
            )
        if (
            type(self.bounds) is not ProtocolStepFrameBounds
            or self.bounds.frame_axis != "source_acquisition_frame_index"
        ):
            raise ProtocolSemanticChaserSelectionError(
                "Protocol phase requires exact acquisition-frame bounds."
            )

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "bounds": self.bounds.to_dict()}


@dataclass(frozen=True)
class ProtocolSemanticSelectionEvidence:
    """Exact semantic snapshot plus raw materialized step-bound evidence."""

    source_stimulus_run: str
    source_stimulus_path: str
    source_stimulus_fingerprint_algorithm: str
    source_stimulus_fingerprint: str
    source_direct_consolidated_metadata: Mapping[str, Any]
    snapshot: ProtocolSemanticSnapshot
    step_bounds: tuple[ProtocolStepFrameBounds, ...]
    chaser_phase_bounds: tuple[ProtocolPhaseFrameBounds, ...] = ()
    protocol_execution_hash: str | None = None
    frame_bound_acquisition_binding: Mapping[str, Any] | None = None
    step_end_interval_semantics: str = STEP_END_PENDING
    trial_index_integrity_status: str = TRIAL_INDEX_INTEGRITY_LOCAL
    _verification_seal: object | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self._verification_seal is not _EVIDENCE_SEAL:
            raise ProtocolSemanticChaserSelectionError(
                "Protocol semantic selection evidence requires its strict loader."
            )
        run_name = _exact_text(
            self.source_stimulus_run,
            name="source_stimulus_run",
        )
        if run_name in {"latest", "latest_complete", "authoritative_run"}:
            raise ProtocolSemanticChaserSelectionError(
                "source_stimulus_run must name one exact run, not a selector."
            )
        expected_path = f"analysis/stimulus_runs/{run_name}"
        if self.source_stimulus_path != expected_path:
            raise ProtocolSemanticChaserSelectionError(
                "source_stimulus_path is not bound to its exact run name."
            )
        _exact_text(
            self.source_stimulus_fingerprint_algorithm,
            name="source_stimulus_fingerprint_algorithm",
        )
        fingerprint = self.source_stimulus_fingerprint
        if (
            type(fingerprint) is not str
            or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
        ):
            raise ProtocolSemanticChaserSelectionError(
                "source_stimulus_fingerprint must be one lowercase SHA-256 digest."
            )
        if type(self.snapshot) is not ProtocolSemanticSnapshot:
            raise ProtocolSemanticChaserSelectionError(
                "A validated ProtocolSemanticSnapshot is required."
            )
        reloaded = validate_protocol_semantic_snapshot(
            semantic_hash=self.snapshot.semantic_hash,
            semantic_json=self.snapshot.semantic_json,
            trial_index_json=self.snapshot.trial_index_json,
            trial_index_hash=(
                self.snapshot.trial_index_sha256
                if self.snapshot.snapshot_schema_version == 2
                else None
            ),
            snapshot_schema_version=self.snapshot.snapshot_schema_version,
            snapshot_policy_id=self.snapshot.snapshot_policy_id,
        )
        if reloaded.recipe_record() != self.snapshot.recipe_record():
            raise ProtocolSemanticChaserSelectionError(
                "Protocol semantic snapshot did not revalidate exactly."
            )
        if self.step_end_interval_semantics not in STEP_END_POLICIES:
            raise ProtocolSemanticChaserSelectionError(
                "step_end_interval_semantics is unsupported."
            )
        if self.trial_index_integrity_status != (
            self.snapshot.trial_index_integrity_status
        ):
            raise ProtocolSemanticChaserSelectionError(
                "Trial-index integrity status differs from the exact snapshot."
            )
        if not isinstance(self.source_direct_consolidated_metadata, Mapping):
            raise ProtocolSemanticChaserSelectionError(
                "source_direct_consolidated_metadata must be one exact record."
            )
        metadata = _freeze_json(self.source_direct_consolidated_metadata)
        if not isinstance(metadata, Mapping):  # pragma: no cover - defensive
            raise ProtocolSemanticChaserSelectionError(
                "source_direct_consolidated_metadata must remain one record."
            )
        object.__setattr__(self, "source_direct_consolidated_metadata", metadata)
        bounds = tuple(self.step_bounds)
        if any(type(item) is not ProtocolStepFrameBounds for item in bounds):
            raise ProtocolSemanticChaserSelectionError(
                "step_bounds must contain exact ProtocolStepFrameBounds values."
            )
        expected_indices = set(range(self.snapshot.step_count))
        observed_indices = {item.step_index for item in bounds}
        if len(bounds) != len(observed_indices) or observed_indices != expected_indices:
            raise ProtocolSemanticChaserSelectionError(
                "step_bounds must bind every semantic recipe index exactly once."
            )
        ordered = tuple(sorted(bounds, key=lambda item: item.step_index))
        for previous, current in zip(ordered, ordered[1:]):
            overlaps = (
                current.start_camera_frame < previous.end_camera_frame
                if self.step_end_interval_semantics == STEP_END_EXCLUSIVE
                else current.start_camera_frame <= previous.end_camera_frame
            )
            if overlaps:
                raise ProtocolSemanticChaserSelectionError(
                    "Raw semantic step bounds overlap or are not strictly ordered."
                )
        object.__setattr__(self, "step_bounds", ordered)
        phase_bounds = tuple(self.chaser_phase_bounds)
        if any(type(item) is not ProtocolPhaseFrameBounds for item in phase_bounds):
            raise ProtocolSemanticChaserSelectionError(
                "chaser_phase_bounds must contain exact ProtocolPhaseFrameBounds."
            )
        if len({item.role for item in phase_bounds}) != len(phase_bounds):
            raise ProtocolSemanticChaserSelectionError(
                "chaser_phase_bounds contains duplicate semantic roles."
            )
        object.__setattr__(
            self,
            "chaser_phase_bounds",
            tuple(sorted(phase_bounds, key=lambda item: item.role)),
        )
        if self.snapshot.snapshot_schema_version == 2:
            if (
                self.step_end_interval_semantics != STEP_END_EXCLUSIVE
                or type(self.protocol_execution_hash) is not str
                or not self.protocol_execution_hash.startswith("sha256:")
                or len(self.protocol_execution_hash) != 71
                or not isinstance(self.frame_bound_acquisition_binding, Mapping)
                or any(
                    item.frame_axis != "source_acquisition_frame_index"
                    for item in ordered
                )
            ):
                raise ProtocolSemanticChaserSelectionError(
                    "Snapshot v2 requires exact execution and frame-bound acquisition evidence."
                )
            chaser_steps = [
                item
                for item in self.snapshot.steps
                if item.stimulus_mode == "CHASER"
            ]
            expected_phase_roles = set(CHASER_PHASE_NAMES) if chaser_steps else set()
            if {item.role for item in phase_bounds} != expected_phase_roles:
                raise ProtocolSemanticChaserSelectionError(
                    "Snapshot v2 CHASER evidence lacks its exact phase mappings."
                )
            binding = _freeze_json(self.frame_bound_acquisition_binding)
            if not isinstance(binding, Mapping):  # pragma: no cover - defensive
                raise ProtocolSemanticChaserSelectionError(
                    "Frame-bound acquisition binding is malformed."
                )
            object.__setattr__(self, "frame_bound_acquisition_binding", binding)
        elif (
            phase_bounds
            or self.protocol_execution_hash is not None
            or self.frame_bound_acquisition_binding is not None
            or any(item.frame_axis != "legacy_camera_frame_id" for item in ordered)
        ):
            raise ProtocolSemanticChaserSelectionError(
                "Legacy snapshot evidence cannot claim v2 acquisition mapping."
            )

    @property
    def bounds_by_step_index(self) -> Mapping[int, ProtocolStepFrameBounds]:
        return MappingProxyType({item.step_index: item for item in self.step_bounds})

    @property
    def phase_bounds_by_role(self) -> Mapping[str, ProtocolStepFrameBounds]:
        return MappingProxyType(
            {item.role: item.bounds for item in self.chaser_phase_bounds}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": f"{ADAPTER_SCHEMA_ID}.protocol_evidence",
            "schema_version": 2,
            "source_stimulus_run": self.source_stimulus_run,
            "source_stimulus_path": self.source_stimulus_path,
            "source_stimulus_fingerprint_algorithm": (
                self.source_stimulus_fingerprint_algorithm
            ),
            "source_stimulus_fingerprint": self.source_stimulus_fingerprint,
            "source_direct_consolidated_metadata": _thaw(
                self.source_direct_consolidated_metadata
            ),
            "protocol_semantic_hash": self.snapshot.semantic_hash,
            "protocol_trial_index_sha256": self.snapshot.trial_index_sha256,
            "palette_computed_trial_index_sha256": self.snapshot.trial_index_sha256,
            "trial_index_integrity_status": self.trial_index_integrity_status,
            "protocol_recipe": self.snapshot.recipe_record(),
            "protocol_execution_hash": self.protocol_execution_hash,
            "frame_bound_acquisition_binding": _thaw(
                self.frame_bound_acquisition_binding
            ),
            "step_end_interval_semantics": self.step_end_interval_semantics,
            "step_bounds": [item.to_dict() for item in self.step_bounds],
            "chaser_phase_bounds": [
                item.to_dict() for item in self.chaser_phase_bounds
            ],
        }


def _snapshots_match_exactly(
    left: ProtocolSemanticSnapshot,
    right: ProtocolSemanticSnapshot,
) -> bool:
    return (
        left.snapshot_schema_version == right.snapshot_schema_version
        and left.snapshot_policy_id == right.snapshot_policy_id
        and left.semantic_hash == right.semantic_hash
        and left.semantic_json == right.semantic_json
        and left.trial_index_json == right.trial_index_json
        and left.trial_index_sha256 == right.trial_index_sha256
        and left.recipe_record() == right.recipe_record()
    )


def _validate_frame_bound_mapping_axis(
    source: PairedFrameBoundChaserSource,
) -> tuple[np.ndarray, np.ndarray]:
    stimulus = np.asarray(source.companion.stimulus_frame_num, dtype=np.int64)
    acquisition = np.asarray(
        source.companion.frame_source_acquisition_frame_index,
        dtype=np.int64,
    )
    if (
        stimulus.ndim != 1
        or acquisition.ndim != 1
        or stimulus.shape != acquisition.shape
        or stimulus.size == 0
        or np.any(np.diff(stimulus) <= 0)
        or np.any(np.diff(acquisition) < 0)
        or np.any(acquisition < 0)
        or np.any(acquisition >= source.dimensions.total_frames)
    ):
        raise ProtocolSemanticChaserSelectionError(
            "Frame-bound stimulus-to-acquisition mapping must be one complete, "
            "monotone native-frame axis."
        )
    return stimulus, acquisition


def _project_execution_interval_to_acquisition_bounds(
    interval: ProtocolStimulusFrameInterval,
    *,
    step_index: int,
    stimulus_frame_num: np.ndarray,
    acquisition_frame_index: np.ndarray,
) -> ProtocolStepFrameBounds:
    start = interval.start_stimulus_frame_inclusive
    end = interval.end_stimulus_frame_exclusive
    if end <= start:
        raise ProtocolSemanticChaserSelectionError(
            "Protocol execution interval is empty and cannot define an analytic window."
        )
    selected = (stimulus_frame_num >= start) & (stimulus_frame_num < end)
    mapped_stimulus = stimulus_frame_num[selected]
    mapped_acquisition = acquisition_frame_index[selected]
    expected_count = end - start
    if (
        mapped_stimulus.size != expected_count
        or int(mapped_stimulus[0]) != start
        or int(mapped_stimulus[-1]) != end - 1
        or np.any(np.diff(mapped_stimulus) != 1)
    ):
        raise ProtocolSemanticChaserSelectionError(
            "Frame-bound acquisition evidence does not cover every producer-authored "
            "stimulus frame in one execution interval."
        )
    latest_by_acquisition = np.r_[
        acquisition_frame_index[1:] != acquisition_frame_index[:-1],
        True,
    ]
    selected_acquisition = acquisition_frame_index[
        latest_by_acquisition
        & (stimulus_frame_num >= start)
        & (stimulus_frame_num < end)
    ]
    if selected_acquisition.size == 0:
        raise ProtocolSemanticChaserSelectionError(
            "A producer interval has no acquisition row after exact latest-sample projection."
        )
    acquisition_start = int(selected_acquisition[0])
    acquisition_end = int(selected_acquisition[-1]) + 1
    mapped_acquisition_count = int(selected_acquisition.size)
    return ProtocolStepFrameBounds(
        step_index=step_index,
        start_camera_frame=acquisition_start,
        end_camera_frame=acquisition_end,
        frame_axis="source_acquisition_frame_index",
        source_start_stimulus_frame_inclusive=start,
        source_end_stimulus_frame_exclusive=end,
        mapped_stimulus_frame_count=expected_count,
        mapped_acquisition_frame_count=mapped_acquisition_count,
        reused_acquisition_frame_count=(
            expected_count - int(np.unique(mapped_acquisition).size)
        ),
        acquisition_gap_count=(
            acquisition_end - acquisition_start - mapped_acquisition_count
        ),
    )


def load_protocol_semantic_selection_evidence(
    analysis_zarr: str | Path,
    selection: ResolvedEpochSelection,
    *,
    step_end_interval_semantics: str = STEP_END_PENDING,
    use_consolidated: bool = True,
    frame_bound_source: PairedFrameBoundChaserSource | None = None,
    frame_bound_source_binding: Mapping[str, Any] | None = None,
) -> ProtocolSemanticSelectionEvidence:
    """Mint semantic evidence from one exact immutable stimulus authority."""

    try:
        resolved = _require_selection(selection)
    except ComposableEpochSelectionAdapterError as exc:
        raise ProtocolSemanticChaserSelectionError(str(exc)) from exc
    if type(use_consolidated) is not bool:
        raise ProtocolSemanticChaserSelectionError(
            "use_consolidated must be one exact boolean."
        )
    if frame_bound_source is not None and frame_bound_source_binding is not None:
        raise ProtocolSemanticChaserSelectionError(
            "Supply either a frame-bound source or its reload binding, not both."
        )
    archive = Path(analysis_zarr).expanduser().resolve()
    if not archive.is_dir():
        raise FileNotFoundError(f"Analysis Zarr archive does not exist: {archive}")
    timeline = resolved.source_timeline_identity
    source_run = _exact_text(
        timeline.get("source_stimulus_run"),
        name="resolved source stimulus run",
    )
    source_path = _exact_text(
        timeline.get("source_stimulus_path"),
        name="resolved source stimulus path",
    )
    fingerprint_algorithm = _exact_text(
        timeline.get("fingerprint_algorithm"),
        name="resolved source stimulus fingerprint algorithm",
    )
    if fingerprint_algorithm != STIMULUS_SOURCE_FINGERPRINT_ALGORITHM:
        raise ProtocolSemanticChaserSelectionError(
            "Resolved source stimulus fingerprint algorithm is unsupported."
        )
    expected_fingerprint = timeline.get("fingerprint")
    try:
        metadata = validate_direct_consolidated_subtree(
            archive,
            subtree_path=source_path,
        )
        root = open_zarr_root(
            archive,
            mode="r",
            use_consolidated=use_consolidated,
        )
        run_group = root[source_path]
        if not isinstance(run_group, zarr.Group):
            raise ProtocolSemanticChaserSelectionError(
                "Exact source stimulus path is not one Zarr group."
            )
        snapshot = read_materialized_protocol_semantic_snapshot(run_group)
        observed_fingerprint = stimulus_group_logical_fingerprint(run_group)
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
        ProtocolSemanticContractError,
    ) as exc:
        raise ProtocolSemanticChaserSelectionError(
            f"Unable to load exact materialized protocol semantic evidence: {exc}"
        ) from exc
    if observed_fingerprint != expected_fingerprint:
        raise ProtocolSemanticChaserSelectionError(
            "Materialized protocol semantic source fingerprint differs from the "
            "exact epoch-selection authority."
        )
    bounds: list[ProtocolStepFrameBounds] = []
    phase_bounds: list[ProtocolPhaseFrameBounds] = []
    execution_hash: str | None = None
    acquisition_binding: Mapping[str, Any] | None = None
    steps_group = run_group["steps"]
    if snapshot.snapshot_schema_version == 2:
        if step_end_interval_semantics != STEP_END_EXCLUSIVE:
            raise ProtocolSemanticChaserSelectionError(
                "Citrus snapshot v2 requires producer-declared half-open execution semantics."
            )
        if frame_bound_source_binding is not None:
            try:
                source = load_paired_frame_bound_chaser_source_from_binding(
                    frame_bound_source_binding
                )
            except StimulusCoordinateContractError as exc:
                raise ProtocolSemanticChaserSelectionError(
                    f"Unable to reload frame-bound acquisition evidence: {exc}"
                ) from exc
        else:
            if type(frame_bound_source) is not PairedFrameBoundChaserSource:
                raise ProtocolSemanticChaserSelectionError(
                    "Citrus snapshot v2 requires one strict paired frame-bound source."
                )
            try:
                source = frame_bound_source.assert_current()
            except StimulusCoordinateContractError as exc:
                raise ProtocolSemanticChaserSelectionError(
                    f"Unable to reverify frame-bound acquisition evidence: {exc}"
                ) from exc
        if (
            source.recording_id != timeline.get("recording_id")
            or source.dimensions.total_frames != resolved.native_frame_count
        ):
            raise ProtocolSemanticChaserSelectionError(
                "Frame-bound recording identity or native frame population differs "
                "from the exact epoch selection."
            )
        source_h5 = run_group.attrs.get("source_h5")
        if type(source_h5) is not str or Path(source_h5).expanduser().resolve() != source.raw_h5:
            raise ProtocolSemanticChaserSelectionError(
                "Materialized semantic stimulus run is not bound to the sealed raw H5."
            )
        try:
            materialized_execution = read_materialized_protocol_execution_index(
                run_group,
                snapshot=snapshot,
            )
            with h5py.File(source.raw_h5, "r") as raw_h5:
                raw_snapshot = read_protocol_semantic_snapshot(raw_h5)
                if raw_snapshot is None or not _snapshots_match_exactly(
                    snapshot,
                    raw_snapshot,
                ):
                    raise ProtocolSemanticChaserSelectionError(
                        "Materialized semantic snapshot differs from the sealed raw H5."
                    )
                raw_execution = read_protocol_execution_index(
                    raw_h5,
                    snapshot=raw_snapshot,
                )
        except (
            OSError,
            ProtocolExecutionContractError,
            ProtocolSemanticContractError,
        ) as exc:
            raise ProtocolSemanticChaserSelectionError(
                f"Unable to verify sealed v2 execution evidence: {exc}"
            ) from exc
        if (
            raw_execution.execution_json != materialized_execution.execution_json
            or raw_execution.execution_hash != materialized_execution.execution_hash
            or raw_execution.status != "complete"
        ):
            raise ProtocolSemanticChaserSelectionError(
                "Materialized protocol execution differs from the complete sealed raw H5."
            )
        stimulus_frames, acquisition_frames = _validate_frame_bound_mapping_axis(
            source
        )
        for realized in raw_execution.steps:
            bounds.append(
                _project_execution_interval_to_acquisition_bounds(
                    realized.interval,
                    step_index=realized.step_index,
                    stimulus_frame_num=stimulus_frames,
                    acquisition_frame_index=acquisition_frames,
                )
            )
            if realized.chaser_phases is not None:
                for role in CHASER_PHASE_NAMES:
                    phase_bounds.append(
                        ProtocolPhaseFrameBounds(
                            role=role,
                            bounds=_project_execution_interval_to_acquisition_bounds(
                                realized.chaser_phases[role],
                                step_index=realized.step_index,
                                stimulus_frame_num=stimulus_frames,
                                acquisition_frame_index=acquisition_frames,
                            ),
                        )
                    )
        execution_hash = raw_execution.execution_hash
        acquisition_binding = source.reload_binding()
        try:
            source.assert_verified()
        except StimulusCoordinateContractError as exc:
            raise ProtocolSemanticChaserSelectionError(
                f"Frame-bound acquisition evidence changed during semantic loading: {exc}"
            ) from exc
    else:
        if frame_bound_source is not None or frame_bound_source_binding is not None:
            raise ProtocolSemanticChaserSelectionError(
                "Legacy protocol snapshots cannot consume v2 frame-bound evidence."
            )
        for identity in snapshot.steps:
            attrs = steps_group[f"step_{identity.step_index}"].attrs
            start = attrs.get("start_camera_frame")
            end = attrs.get("end_camera_frame")
            if (
                type(start) is not int
                or type(end) is not int
                or start < 0
                or end < start
            ):
                raise ProtocolSemanticChaserSelectionError(
                    "Materialized semantic step lacks exact raw camera-frame bounds "
                    f"at step_index={identity.step_index}."
                )
            bounds.append(
                ProtocolStepFrameBounds(
                    step_index=identity.step_index,
                    start_camera_frame=start,
                    end_camera_frame=end,
                )
            )
    return ProtocolSemanticSelectionEvidence(
        source_stimulus_run=source_run,
        source_stimulus_path=source_path,
        source_stimulus_fingerprint_algorithm=fingerprint_algorithm,
        source_stimulus_fingerprint=observed_fingerprint,
        source_direct_consolidated_metadata=metadata.to_json(),
        snapshot=snapshot,
        step_bounds=tuple(bounds),
        chaser_phase_bounds=tuple(phase_bounds),
        protocol_execution_hash=execution_hash,
        frame_bound_acquisition_binding=acquisition_binding,
        step_end_interval_semantics=step_end_interval_semantics,
        trial_index_integrity_status=snapshot.trial_index_integrity_status,
        _verification_seal=_EVIDENCE_SEAL,
    )


def load_protocol_semantic_timeline_evidence(
    analysis_zarr: str | Path,
    selection: ResolvedEpochSelection,
) -> ProtocolSemanticTimelineEvidence:
    """Mint exact timeline evidence needed to compile a semantic hierarchy."""

    try:
        resolved = _require_selection(selection)
    except ComposableEpochSelectionAdapterError as exc:
        raise ProtocolSemanticChaserSelectionError(str(exc)) from exc
    archive = Path(analysis_zarr).expanduser().resolve()
    if not archive.is_dir():
        raise FileNotFoundError(f"Analysis Zarr archive does not exist: {archive}")
    try:
        direct = open_zarr_root(archive, mode="r", use_consolidated=False)
        consolidated = open_zarr_root(archive, mode="r", use_consolidated=True)
        direct_video = _thaw(direct.attrs.get("source_video_metadata"))
        consolidated_video = _thaw(
            consolidated.attrs.get("source_video_metadata")
        )
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
        raise ProtocolSemanticChaserSelectionError(
            f"Unable to load exact source-video metadata: {exc}"
        ) from exc
    if not isinstance(direct_video, Mapping) or direct_video != consolidated_video:
        raise ProtocolSemanticChaserSelectionError(
            "Direct and consolidated source-video metadata are absent or differ."
        )
    clock = _thaw(resolved.recording_timing_authority)
    if not isinstance(clock, Mapping):
        raise ProtocolSemanticChaserSelectionError(
            "Resolved epoch selection lacks its bound recording timing authority."
        )
    clock_record = clock.get("acquisition_frame_clock")
    if not isinstance(clock_record, Mapping):
        raise ProtocolSemanticChaserSelectionError(
            "Recording timing authority lacks its acquisition frame clock."
        )
    clock_ref = clock_record.get("run_path")
    if type(clock_ref) is not str or not clock_ref:
        raise ProtocolSemanticChaserSelectionError(
            "Recording timing authority lacks its exact clock run path."
        )
    clock_sha256 = resolved.recording_timing_authority_sha256
    if type(clock_sha256) is not str:
        raise ProtocolSemanticChaserSelectionError(
            "Resolved epoch selection lacks its recording timing digest."
        )
    timeline = resolved.source_timeline_identity
    recording_id = _exact_text(
        timeline.get("recording_id"),
        name="resolved source recording_id",
    )
    source_video = dict(direct_video)
    source_video_ref = "/@source_video_metadata"
    source_video_sha256 = canonical_sha256(source_video)
    source_metadata = {
        "recording_id": recording_id,
        "source_timeline_digest": resolved.source_timeline_digest,
        "source_epoch_run_path": resolved.run_path,
        "source_epoch_run_manifest_sha256": resolved.run_manifest_digest,
        "source_epoch_run_manifest_payload_sha256": (
            resolved.run_manifest_payload_digest
        ),
        "source_epoch_logical_content_sha256": (
            resolved.source_epoch_logical_content_digest
        ),
        "source_epoch_lineage_hash": resolved.source_epoch_lineage_hash,
        "source_epoch_lineage_payload_sha256": (
            resolved.source_epoch_lineage_payload_digest
        ),
        "timing_authority": _thaw(resolved.timing_authority),
        "source_video_metadata_ref": source_video_ref,
        "source_video_metadata_sha256": source_video_sha256,
        "acquisition_clock_authority_ref": clock_ref,
        "acquisition_clock_authority_sha256": clock_sha256,
        "acquisition_frame_domain": "camera_acquisition_frame",
        "frame_count": resolved.native_frame_count,
        "fps": resolved.fps,
    }
    return ProtocolSemanticTimelineEvidence(
        evidence=TimelineAuthorityEvidence(
            recording_id=recording_id,
            timeline_id=resolved.source_timeline_digest,
            stimulus_authority_id=resolved.run_path,
            acquisition_frame_domain="camera_acquisition_frame",
            source_video_metadata_ref=source_video_ref,
            source_video_metadata_sha256=source_video_sha256,
            source_video_metadata=source_video,
            acquisition_clock_authority_ref=clock_ref,
            acquisition_clock_authority_sha256=clock_sha256,
            acquisition_clock_authority=dict(clock),
            source_metadata_sha256=canonical_sha256(source_metadata),
            source_metadata=source_metadata,
        ),
        _verification_seal=_TIMELINE_EVIDENCE_SEAL,
    )


@dataclass(frozen=True)
class SemanticProtocolStepWindow:
    """One producer step plus its usable half-open containment interval."""

    identity: ProtocolStepIdentity
    raw_bounds: ProtocolStepFrameBounds
    usable_start_frame: int
    usable_end_frame: int
    containment_policy: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity.to_record(),
            "raw_bounds": self.raw_bounds.to_dict(),
            "usable_half_open_interval": [
                self.usable_start_frame,
                self.usable_end_frame,
            ],
            "containment_policy": self.containment_policy,
        }


def _step_window(
    evidence: ProtocolSemanticSelectionEvidence,
    identity: ProtocolStepIdentity,
) -> SemanticProtocolStepWindow:
    bounds = evidence.bounds_by_step_index[identity.step_index]
    if evidence.step_end_interval_semantics == STEP_END_INCLUSIVE:
        end_exclusive = bounds.end_camera_frame + 1
        policy = "producer_declared_inclusive_end_to_half_open_v1"
    elif evidence.step_end_interval_semantics == STEP_END_EXCLUSIVE:
        end_exclusive = bounds.end_camera_frame
        policy = "producer_declared_exclusive_end_v1"
    else:
        end_exclusive = bounds.end_camera_frame
        policy = "conservative_common_interior_pending_step_end_v1"
    if end_exclusive <= bounds.start_camera_frame:
        raise ProtocolSemanticChaserSelectionError(
            "Protocol step has no usable frame interval under its boundary policy."
        )
    return SemanticProtocolStepWindow(
        identity=identity,
        raw_bounds=bounds,
        usable_start_frame=bounds.start_camera_frame,
        usable_end_frame=end_exclusive,
        containment_policy=policy,
    )


def _semantic_steps(
    evidence: ProtocolSemanticSelectionEvidence,
) -> tuple[SemanticProtocolStepWindow | None, SemanticProtocolStepWindow]:
    chaser = [
        step
        for step in evidence.snapshot.steps
        if step.stimulus_mode == "CHASER"
        and step.stimulus_family == "chaser"
        and step.display_context == "chaser"
    ]
    if len(chaser) != 1:
        raise ProtocolSemanticChaserSelectionError(
            "Semantic chaser selection requires exactly one exact CHASER step."
        )
    black = [
        step
        for step in evidence.snapshot.steps
        if step.stimulus_mode == "SOLID_BLACK"
        and step.stimulus_family == "solid_color"
        and step.display_context == "solid_black"
    ]
    if len(black) > 1:
        raise ProtocolSemanticChaserSelectionError(
            "Semantic chaser selection supports at most one standalone SOLID_BLACK step."
        )
    return (
        _step_window(evidence, black[0]) if black else None,
        _step_window(evidence, chaser[0]),
    )


def _resolve_binding(
    selection: ResolvedEpochSelection,
    binding: EpochRoleBinding,
) -> Any:
    try:
        return _interval_by_binding(selection, binding)
    except ComposableEpochSelectionAdapterError as exc:
        raise ProtocolSemanticChaserSelectionError(str(exc)) from exc


def _contained_member(
    *,
    selection: ResolvedEpochSelection,
    authority: TimelineAuthority,
    role: str,
    binding: EpochRoleBinding,
    interval: Any,
    step: SemanticProtocolStepWindow,
    evidence: ProtocolSemanticSelectionEvidence,
) -> Any:
    source_start = int(interval.start_frame)
    source_end = int(interval.end_frame)
    exact_phase_bounds = evidence.phase_bounds_by_role.get(role)
    if exact_phase_bounds is not None and (
        source_start != exact_phase_bounds.start_camera_frame
        or source_end != exact_phase_bounds.end_camera_frame
    ):
        raise ProtocolSemanticChaserSelectionError(
            f"{role} does not equal its exact producer-authored v2 phase envelope."
        )
    if source_start < step.usable_start_frame:
        raise ProtocolSemanticChaserSelectionError(
            f"{role} starts before its exact producer step."
        )
    selected_end = source_end
    terminal_frame_excluded = False
    if source_end > step.usable_end_frame:
        if (
            evidence.step_end_interval_semantics == STEP_END_PENDING
            and source_end == step.usable_end_frame + 1
        ):
            selected_end = step.usable_end_frame
            terminal_frame_excluded = True
        else:
            raise ProtocolSemanticChaserSelectionError(
                f"{role} extends beyond its exact producer step."
            )
    if selected_end <= source_start:
        raise ProtocolSemanticChaserSelectionError(
            f"{role} is empty after conservative boundary handling."
        )
    occurrence_id = interval.occurrence_identity.get("occurrence_id")
    if type(occurrence_id) is not str or not occurrence_id:
        raise ProtocolSemanticChaserSelectionError(
            f"{role} source interval lacks an occurrence identity."
        )
    reference_payload = {
        "adapter_schema_id": ADAPTER_SCHEMA_ID,
        "role": role,
        "source_interval_digest": interval.source_interval_digest,
        "protocol_semantic_hash": evidence.snapshot.semantic_hash,
        "protocol_semantic_step_index": step.identity.step_index,
        "selected_half_open_interval": [source_start, selected_end],
        "containment_policy": step.containment_policy,
    }
    reference = interval_annotation_reference(
        reference_id=canonical_sha256(reference_payload),
        label=role,
        start_frame=source_start,
        end_frame=selected_end,
        authority=authority,
        occurrence_id=occurrence_id,
    )
    metadata = {
        **reference_payload,
        "role_binding": binding.to_dict(),
        "source_half_open_interval": [source_start, source_end],
        "terminal_frame_excluded_pending_step_end_contract": (
            terminal_frame_excluded
        ),
        "protocol_step": step.to_dict(),
        "protocol_phase_bounds": (
            exact_phase_bounds.to_dict()
            if exact_phase_bounds is not None
            else None
        ),
        "protocol_evidence_sha256": canonical_sha256(evidence.to_dict()),
        "resolved_epoch_selection": _selection_identity(selection),
        "source_metadata_identity": _thaw(interval.source_metadata_identity),
        "occurrence_identity": _thaw(interval.occurrence_identity),
    }
    return member(
        reference,
        role=RoleMetadata(role=role, label=role, metadata=metadata),
    )


def _compile_role(
    *,
    selection: ResolvedEpochSelection,
    authority: TimelineAuthority,
    role: str,
    binding: EpochRoleBinding,
    interval: Any,
    step: SemanticProtocolStepWindow,
    evidence: ProtocolSemanticSelectionEvidence,
) -> CompiledSelection:
    expression = _contained_member(
        selection=selection,
        authority=authority,
        role=role,
        binding=binding,
        interval=interval,
        step=step,
        evidence=evidence,
    )
    return compile_selection(
        SelectionSpec(
            selection_id=role,
            expression=expression,
            aggregation_policy="keep_occurrences",
            metadata={
                "adapter_schema_id": ADAPTER_SCHEMA_ID,
                "role": role,
                "protocol_semantic_hash": evidence.snapshot.semantic_hash,
                "protocol_semantic_step_index": step.identity.step_index,
                "protocol_semantic_step_ref": (
                    "protocol_semantic_snapshot@recipe.steps"
                    f"[{step.identity.step_index}]"
                ),
                "protocol_evidence_sha256": canonical_sha256(evidence.to_dict()),
                "resolved_epoch_selection": _selection_identity(selection),
            },
        ),
        expected_authority=authority,
    )


@dataclass(frozen=True)
class ProtocolSemanticChaserSelections:
    """Versioned semantic hierarchy for chaser analytics."""

    timeline_authority: TimelineAuthority
    timeline_evidence: ProtocolSemanticTimelineEvidence
    protocol_evidence: ProtocolSemanticSelectionEvidence
    chaser_step: SemanticProtocolStepWindow
    standalone_solid_black_step: SemanticProtocolStepWindow | None
    chaser_pre: CompiledSelection
    chaser_training: CompiledSelection
    chaser_post: CompiledSelection
    standalone_solid_black: CompiledSelection | None
    standalone_solid_black_status: str

    @property
    def named(self) -> Mapping[str, CompiledSelection]:
        result = {
            "chaser_pre": self.chaser_pre,
            "chaser_training": self.chaser_training,
            "chaser_post": self.chaser_post,
        }
        if self.standalone_solid_black is not None:
            result[STANDALONE_SOLID_BLACK_ROLE] = self.standalone_solid_black
        return MappingProxyType(result)

    def __getitem__(self, name: str) -> CompiledSelection:
        if name in LEGACY_ROLE_NAMES:
            raise KeyError(
                f"legacy role {name!r} is not a protocol-semantic selection name"
            )
        try:
            return self.named[name]
        except KeyError as exc:
            raise KeyError(f"unknown protocol-semantic selection name: {name!r}") from exc

    def identity_record(self) -> dict[str, Any]:
        """Return the bounded identity used by planners and downstream candidates."""

        named = self.named
        requested = _thaw(self.chaser_pre.requested)
        requested_metadata = requested.get("metadata", {})
        return {
            "schema_id": ADAPTER_SCHEMA_ID,
            "schema_version": 2,
            "selector_eligible": False,
            "scientific_use_class": (
                "exploratory_frame_bound_protocol_semantic_selection"
                if self.protocol_evidence.snapshot.snapshot_schema_version == 2
                else "exploratory_conservative_protocol_semantic_selection"
            ),
            "source_stimulus_run": self.protocol_evidence.source_stimulus_run,
            "source_stimulus_path": self.protocol_evidence.source_stimulus_path,
            "source_stimulus_fingerprint_algorithm": (
                self.protocol_evidence.source_stimulus_fingerprint_algorithm
            ),
            "source_stimulus_fingerprint": (
                self.protocol_evidence.source_stimulus_fingerprint
            ),
            "protocol_evidence_sha256": canonical_sha256(
                self.protocol_evidence.to_dict()
            ),
            "timeline_evidence_sha256": canonical_sha256(
                self.timeline_evidence.to_dict()
            ),
            "resolved_epoch_selection": requested_metadata.get(
                "resolved_epoch_selection"
            ),
            "protocol_semantic_hash": (
                self.protocol_evidence.snapshot.semantic_hash
            ),
            "protocol_trial_index_sha256": (
                self.protocol_evidence.snapshot.trial_index_sha256
            ),
            "protocol_snapshot_schema_version": (
                self.protocol_evidence.snapshot.snapshot_schema_version
            ),
            "protocol_execution_hash": (
                self.protocol_evidence.protocol_execution_hash
            ),
            "frame_bound_acquisition_mapping_verified": (
                self.protocol_evidence.frame_bound_acquisition_binding
                is not None
            ),
            "palette_computed_trial_index_sha256": (
                self.protocol_evidence.snapshot.trial_index_sha256
            ),
            "trial_index_integrity_status": (
                self.protocol_evidence.trial_index_integrity_status
            ),
            "step_end_interval_semantics": (
                self.protocol_evidence.step_end_interval_semantics
            ),
            "chaser_step": self.chaser_step.to_dict(),
            "standalone_solid_black_step": (
                self.standalone_solid_black_step.to_dict()
                if self.standalone_solid_black_step is not None
                else None
            ),
            "standalone_solid_black_status": (
                self.standalone_solid_black_status
            ),
            "selection_resolved_sha256_by_role": {
                role: named[role].resolved_digest for role in sorted(named)
            },
        }

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(self.identity_record())

    def capability_assessments(self) -> tuple[CapabilityAssessment, ...]:
        """Project semantic selection state into the existing strict planner.

        CHASER windows are available for selector-ineligible exploratory
        candidates.  Standalone-baseline work is independently applicable,
        missing, or inapplicable.  Production selection remains review-blocked
        for this historical v1 path while producer trial-index and raw
        ``STEP_END`` proof are absent.
        """

        identity = self.identity_record()
        common_evidence = {
            "adapter_schema_id": ADAPTER_SCHEMA_ID,
            "selection_identity_sha256": self.identity_sha256,
            "protocol_semantic_hash": identity["protocol_semantic_hash"],
            "source_stimulus_path": identity["source_stimulus_path"],
            "selector_eligible": False,
        }
        chaser = CapabilityAssessment(
            capability_id=PROTOCOL_SEMANTIC_CHASER_WINDOWS_CAPABILITY_ID,
            state=CapabilityState.READY,
            reason_code="semantic_chaser_windows_ready",
            evidence={
                **common_evidence,
                "scientific_use_class": identity["scientific_use_class"],
                "roles": list(CHASER_WINDOW_ROLES),
                "step_end_interval_semantics": (
                    identity["step_end_interval_semantics"]
                ),
            },
        )
        baseline_state: CapabilityState
        if self.standalone_solid_black_status.startswith("not_applicable_"):
            baseline_state = CapabilityState.NOT_APPLICABLE
            baseline_reason = "protocol_has_no_standalone_solid_black"
        elif self.standalone_solid_black_status == "present_not_selected":
            baseline_state = CapabilityState.MISSING
            baseline_reason = "standalone_solid_black_not_selected"
        else:
            baseline_state = CapabilityState.READY
            baseline_reason = "standalone_solid_black_selected"
        baseline = CapabilityAssessment(
            capability_id=STANDALONE_SOLID_BLACK_BASELINE_CAPABILITY_ID,
            state=baseline_state,
            reason_code=baseline_reason,
            evidence={
                **common_evidence,
                "standalone_solid_black_status": (
                    self.standalone_solid_black_status
                ),
            },
        )
        pending_contracts: list[str] = []
        if (
            self.protocol_evidence.trial_index_integrity_status
            == TRIAL_INDEX_INTEGRITY_LOCAL
        ):
            pending_contracts.append("producer_trial_index_checksum")
        if (
            self.protocol_evidence.step_end_interval_semantics
            == STEP_END_PENDING
        ):
            pending_contracts.append("producer_step_end_interval_semantics")
        frame_bound_v2 = (
            self.protocol_evidence.snapshot.snapshot_schema_version == 2
            and self.protocol_evidence.frame_bound_acquisition_binding is not None
        )
        production = CapabilityAssessment(
            capability_id=(
                PROTOCOL_SEMANTIC_SELECTION_PRODUCTION_CAPABILITY_ID
            ),
            state=CapabilityState.REVIEW_REQUIRED,
            reason_code=(
                "controlled_four_camera_hardware_validation_pending"
                if frame_bound_v2
                else "producer_contracts_pending"
            ),
            evidence={
                **common_evidence,
                "pending_contracts": pending_contracts,
                "trial_index_integrity_status": (
                    identity["trial_index_integrity_status"]
                ),
                "step_end_interval_semantics": (
                    identity["step_end_interval_semantics"]
                ),
                "frame_bound_acquisition_mapping_verified": frame_bound_v2,
                "promotion_blockers": (
                    ["controlled_four_camera_hardware_validation_pending"]
                    if frame_bound_v2
                    else []
                ),
            },
        )
        return tuple(
            sorted(
                (chaser, baseline, production),
                key=lambda assessment: assessment.capability_id,
            )
        )


def compile_protocol_semantic_chaser_selections(
    selection: ResolvedEpochSelection,
    *,
    timeline_evidence: ProtocolSemanticTimelineEvidence,
    protocol_evidence: ProtocolSemanticSelectionEvidence,
    role_bindings: Mapping[str, EpochRoleBinding],
) -> ProtocolSemanticChaserSelections:
    """Compile semantic chaser windows without reinterpreting legacy roles."""

    try:
        resolved = _require_selection(selection)
        if type(timeline_evidence) is not ProtocolSemanticTimelineEvidence:
            raise ProtocolSemanticChaserSelectionError(
                "Protocol semantic timeline evidence requires its strict loader."
            )
        authority = _validate_timeline_evidence(
            resolved,
            timeline_evidence.evidence,
        )
    except ComposableEpochSelectionAdapterError as exc:
        raise ProtocolSemanticChaserSelectionError(str(exc)) from exc
    if type(protocol_evidence) is not ProtocolSemanticSelectionEvidence:
        raise ProtocolSemanticChaserSelectionError(
            "ProtocolSemanticSelectionEvidence is required."
        )
    timeline = resolved.source_timeline_identity
    if (
        timeline.get("source_stimulus_run")
        != protocol_evidence.source_stimulus_run
        or timeline.get("source_stimulus_path")
        != protocol_evidence.source_stimulus_path
        or timeline.get("fingerprint_algorithm")
        != protocol_evidence.source_stimulus_fingerprint_algorithm
        or timeline.get("fingerprint")
        != protocol_evidence.source_stimulus_fingerprint
    ):
        raise ProtocolSemanticChaserSelectionError(
            "Protocol semantic evidence differs from the exact resolved source "
            "stimulus authority."
        )
    if not isinstance(role_bindings, Mapping):
        raise ProtocolSemanticChaserSelectionError(
            "role_bindings must be one explicit mapping."
        )
    supplied_roles = set(role_bindings)
    legacy = supplied_roles & LEGACY_ROLE_NAMES
    if legacy:
        raise ProtocolSemanticChaserSelectionError(
            "Legacy role names are forbidden in the semantic successor: "
            + ", ".join(sorted(legacy))
            + "."
        )
    allowed_roles = set(CHASER_WINDOW_ROLES) | {STANDALONE_SOLID_BLACK_ROLE}
    if not set(CHASER_WINDOW_ROLES).issubset(supplied_roles) or not supplied_roles.issubset(
        allowed_roles
    ):
        raise ProtocolSemanticChaserSelectionError(
            "role_bindings must contain chaser_pre, chaser_training, and "
            "chaser_post, with optional standalone_solid_black."
        )
    for role, binding in role_bindings.items():
        if type(binding) is not EpochRoleBinding:
            raise ProtocolSemanticChaserSelectionError(
                f"role binding for {role!r} must be an exact EpochRoleBinding."
            )
    standalone_step, chaser_step = _semantic_steps(protocol_evidence)
    has_standalone_binding = STANDALONE_SOLID_BLACK_ROLE in role_bindings
    if has_standalone_binding and standalone_step is None:
        raise ProtocolSemanticChaserSelectionError(
            "The exact protocol has no standalone SOLID_BLACK step."
        )
    if len(resolved.intervals) != len(role_bindings):
        raise ProtocolSemanticChaserSelectionError(
            "Every resolved epoch interval must have one explicit semantic role binding."
        )
    intervals = {
        role: _resolve_binding(resolved, binding)
        for role, binding in role_bindings.items()
    }
    if len({item.source_interval_digest for item in intervals.values()}) != len(
        intervals
    ):
        raise ProtocolSemanticChaserSelectionError(
            "Semantic role bindings must resolve to distinct source intervals."
        )
    if len(
        {
            item.occurrence_identity.get("occurrence_id")
            for item in intervals.values()
        }
    ) != len(intervals):
        raise ProtocolSemanticChaserSelectionError(
            "Semantic role bindings must preserve distinct occurrences."
        )
    for role in CHASER_WINDOW_ROLES:
        expected_label = CHASER_SOURCE_LABEL_BY_ROLE[role]
        if intervals[role].label != expected_label:
            raise ProtocolSemanticChaserSelectionError(
                f"{role} must bind the exact {expected_label!r} source window."
            )
    ordered_chaser = tuple(intervals[role] for role in CHASER_WINDOW_ROLES)
    for previous, current in zip(ordered_chaser, ordered_chaser[1:]):
        if int(previous.end_frame) > int(current.start_frame):
            raise ProtocolSemanticChaserSelectionError(
                "chaser_pre, chaser_training, and chaser_post must be "
                "non-overlapping and ordered."
            )

    compiled = {
        role: _compile_role(
            selection=resolved,
            authority=authority,
            role=role,
            binding=role_bindings[role],
            interval=intervals[role],
            step=chaser_step,
            evidence=protocol_evidence,
        )
        for role in CHASER_WINDOW_ROLES
    }
    standalone: CompiledSelection | None = None
    if has_standalone_binding:
        assert standalone_step is not None
        standalone = _compile_role(
            selection=resolved,
            authority=authority,
            role=STANDALONE_SOLID_BLACK_ROLE,
            binding=role_bindings[STANDALONE_SOLID_BLACK_ROLE],
            interval=intervals[STANDALONE_SOLID_BLACK_ROLE],
            step=standalone_step,
            evidence=protocol_evidence,
        )
    if standalone_step is None:
        standalone_status = "not_applicable_protocol_has_no_standalone_solid_black"
    elif standalone is None:
        standalone_status = "present_not_selected"
    else:
        standalone_status = "selected"

    return ProtocolSemanticChaserSelections(
        timeline_authority=authority,
        timeline_evidence=timeline_evidence,
        protocol_evidence=protocol_evidence,
        chaser_step=chaser_step,
        standalone_solid_black_step=standalone_step,
        chaser_pre=compiled["chaser_pre"],
        chaser_training=compiled["chaser_training"],
        chaser_post=compiled["chaser_post"],
        standalone_solid_black=standalone,
        standalone_solid_black_status=standalone_status,
    )


__all__ = [
    "ADAPTER_SCHEMA_ID",
    "CHASER_WINDOW_ROLES",
    "CHASER_SOURCE_LABEL_BY_ROLE",
    "LEGACY_ROLE_NAMES",
    "PROTOCOL_SEMANTIC_CHASER_WINDOWS_CAPABILITY_ID",
    "PROTOCOL_SEMANTIC_SELECTION_PRODUCTION_CAPABILITY_ID",
    "ProtocolSemanticChaserSelectionError",
    "ProtocolSemanticChaserSelections",
    "ProtocolSemanticSelectionEvidence",
    "ProtocolSemanticTimelineEvidence",
    "ProtocolStepFrameBounds",
    "STANDALONE_SOLID_BLACK_ROLE",
    "STANDALONE_SOLID_BLACK_BASELINE_CAPABILITY_ID",
    "STEP_END_EXCLUSIVE",
    "STEP_END_INCLUSIVE",
    "STEP_END_PENDING",
    "SemanticProtocolStepWindow",
    "compile_protocol_semantic_chaser_selections",
    "load_protocol_semantic_selection_evidence",
    "load_protocol_semantic_timeline_evidence",
]

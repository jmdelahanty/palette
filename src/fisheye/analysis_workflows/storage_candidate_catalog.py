"""Executable catalog of unpromoted derived-analysis storage candidates.

The production storage-contract catalog records the currently maintained
authority.  This module records physical candidates separately so creating and
benchmarking one cannot be mistaken for adopting it as a production profile.
Every entry is selector-ineligible and unpromoted by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from importlib import import_module
import re
from typing import Any

from .storage_contract_catalog import DERIVED_ANALYSIS_STORAGE_CONTRACT_BY_STAGE


_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")
_PROFILE_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_MODULE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
_CALLABLE_ATTR = re.compile(r"^_?[a-z][a-z0-9_]*$")


class StorageCandidatePublicationMode(str, Enum):
    """Closed publication boundaries implemented by candidate writers."""

    SHARED_ATOMIC = "shared_atomic_nonpromoting_v1"
    GUARDED_DIRECT = "guarded_direct_nonpromoting_v1"


@dataclass(frozen=True)
class DerivedAnalysisStorageCandidate:
    """One explicit, selector-ineligible physical-layout candidate."""

    stage_id: str
    profile_id: str
    owner_module: str
    entrypoint_attr: str
    publication_mode: StorageCandidatePublicationMode
    consolidates_before_return: bool
    repairs_failed_visibility: bool

    def __post_init__(self) -> None:
        for field, value, pattern in (
            ("stage_id", self.stage_id, _IDENTIFIER),
            ("profile_id", self.profile_id, _PROFILE_ID),
            ("owner_module", self.owner_module, _MODULE),
            ("entrypoint_attr", self.entrypoint_attr, _CALLABLE_ATTR),
        ):
            if type(value) is not str or not pattern.fullmatch(value):
                raise ValueError(f"{field} must be one canonical exact string")
        if self.stage_id not in DERIVED_ANALYSIS_STORAGE_CONTRACT_BY_STAGE:
            raise ValueError("candidate stage must have one central logical contract")
        if not isinstance(self.publication_mode, StorageCandidatePublicationMode):
            raise TypeError("publication_mode must use StorageCandidatePublicationMode")
        for field in ("consolidates_before_return", "repairs_failed_visibility"):
            if type(getattr(self, field)) is not bool:
                raise TypeError(f"{field} must be an exact bool")
        if self.repairs_failed_visibility and not self.consolidates_before_return:
            raise ValueError(
                "failure-visibility repair requires candidate consolidation"
            )
        if self.publication_mode is StorageCandidatePublicationMode.SHARED_ATOMIC:
            if not self.consolidates_before_return or not self.repairs_failed_visibility:
                raise ValueError(
                    "shared atomic candidates must consolidate and repair failed views"
                )
        elif self.consolidates_before_return or self.repairs_failed_visibility:
            raise ValueError(
                "guarded direct candidates do not own archive consolidation or repair"
            )

    def resolves_entrypoint(self) -> bool:
        module: Any = import_module(self.owner_module)
        return callable(getattr(module, self.entrypoint_attr, None))

    def uses_shared_atomic_publisher(self) -> bool:
        if self.publication_mode is not StorageCandidatePublicationMode.SHARED_ATOMIC:
            return False
        module: Any = import_module(self.owner_module)
        publisher = import_module(
            "fisheye.analysis_workflows.materializers.atomic_run_publisher"
        )
        return (
            getattr(module, "atomic_publish_run_group", None)
            is publisher.atomic_publish_run_group
        )

    def as_record(self) -> dict[str, object]:
        return {
            "stage_id": self.stage_id,
            "profile_id": self.profile_id,
            "owner_module": self.owner_module,
            "entrypoint": self.entrypoint_attr,
            "publication_mode": self.publication_mode.value,
            "consolidates_before_return": self.consolidates_before_return,
            "repairs_failed_visibility": self.repairs_failed_visibility,
            "selector_eligible": False,
            "profile_promoted": False,
        }


def _atomic_candidate(
    stage_id: str,
    *,
    profile_id: str,
    owner_module: str,
    entrypoint_attr: str,
) -> DerivedAnalysisStorageCandidate:
    return DerivedAnalysisStorageCandidate(
        stage_id=stage_id,
        profile_id=profile_id,
        owner_module=owner_module,
        entrypoint_attr=entrypoint_attr,
        publication_mode=StorageCandidatePublicationMode.SHARED_ATOMIC,
        consolidates_before_return=True,
        repairs_failed_visibility=True,
    )


def _direct_candidate(
    stage_id: str,
    *,
    profile_id: str,
    owner_module: str,
    entrypoint_attr: str,
) -> DerivedAnalysisStorageCandidate:
    return DerivedAnalysisStorageCandidate(
        stage_id=stage_id,
        profile_id=profile_id,
        owner_module=owner_module,
        entrypoint_attr=entrypoint_attr,
        publication_mode=StorageCandidatePublicationMode.GUARDED_DIRECT,
        consolidates_before_return=False,
        repairs_failed_visibility=False,
    )


DERIVED_ANALYSIS_STORAGE_CANDIDATES: tuple[
    DerivedAnalysisStorageCandidate, ...
] = (
    _atomic_candidate(
        "track_kinematics",
        profile_id="published_http_v1",
        owner_module=(
            "fisheye.analysis_workflows.materializers.track_kinematics_candidate"
        ),
        entrypoint_attr="materialize_track_kinematics_flat_candidate",
    ),
    _atomic_candidate(
        "swim_bouts",
        profile_id="published_http_v1",
        owner_module=(
            "fisheye.analysis_workflows.materializers.exact_tabular_candidate"
        ),
        entrypoint_attr="materialize_exact_tabular_candidate",
    ),
    _atomic_candidate(
        "bout_kinematics",
        profile_id="published_http_v1",
        owner_module=(
            "fisheye.analysis_workflows.materializers.exact_tabular_candidate"
        ),
        entrypoint_attr="materialize_exact_tabular_candidate",
    ),
    _atomic_candidate(
        "eye_angles",
        profile_id="eye_angle_access_aware_candidate_v1",
        owner_module="fisheye.analysis_workflows.materializers.eye_angles",
        entrypoint_attr="materialize_eye_angles",
    ),
    _atomic_candidate(
        "subject_shape",
        profile_id="subject_shape_access_aware_candidate_v1",
        owner_module="fisheye.analysis_workflows.materializers.subject_shape",
        entrypoint_attr="materialize_subject_shape",
    ),
    _atomic_candidate(
        "tail_kinematics",
        profile_id="published_http_v1",
        owner_module="fisheye.analysis_workflows.materializers.tail_kinematics",
        entrypoint_attr="materialize_tail_kinematics",
    ),
    _atomic_candidate(
        "stimulus_response",
        profile_id="published_http_v1",
        owner_module="fisheye.analysis_workflows.materializers.stimulus_response",
        entrypoint_attr="materialize_stimulus_response",
    ),
    _atomic_candidate(
        "stimulus_epochs",
        profile_id="published_http_v1",
        owner_module="fisheye.analysis_workflows.materializers.stimulus_epochs",
        entrypoint_attr="materialize_stimulus_epoch_candidate",
    ),
    _atomic_candidate(
        "detection_occupancy",
        profile_id="published_http_v1",
        owner_module=(
            "fisheye.analysis_workflows.materializers.exact_tabular_candidate"
        ),
        entrypoint_attr="materialize_exact_tabular_candidate",
    ),
    _atomic_candidate(
        "session_occupancy",
        profile_id="published_http_v1",
        owner_module=(
            "fisheye.analysis_workflows.materializers.exact_tabular_candidate"
        ),
        entrypoint_attr="materialize_exact_tabular_candidate",
    ),
    _direct_candidate(
        "tail_posture_view",
        profile_id="published_http_v1",
        owner_module="fisheye.analysis.tail_posture_view_runs",
        entrypoint_attr="write_tail_posture_view_run",
    ),
    _direct_candidate(
        "bout_classification",
        profile_id="published_http_v1",
        owner_module="fisheye.analysis.megabouts_classifier",
        entrypoint_attr="write_megabouts_classification_run",
    ),
)


DERIVED_ANALYSIS_STORAGE_CANDIDATE_BY_STAGE = {
    candidate.stage_id: candidate
    for candidate in DERIVED_ANALYSIS_STORAGE_CANDIDATES
}


def resolved_storage_candidates() -> tuple[dict[str, object], ...]:
    return tuple(candidate.as_record() for candidate in DERIVED_ANALYSIS_STORAGE_CANDIDATES)


__all__ = [
    "DERIVED_ANALYSIS_STORAGE_CANDIDATE_BY_STAGE",
    "DERIVED_ANALYSIS_STORAGE_CANDIDATES",
    "DerivedAnalysisStorageCandidate",
    "StorageCandidatePublicationMode",
    "resolved_storage_candidates",
]

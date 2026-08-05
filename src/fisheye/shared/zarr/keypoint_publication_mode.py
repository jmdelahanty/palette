"""Shared lifecycle attributes for keypoint-family immutable publications."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping


KEYPOINT_PUBLICATION_MODE_SHADOW = "benchmark_shadow"
KEYPOINT_PUBLICATION_MODE_PRODUCTION_CANDIDATE = "production_candidate"
KEYPOINT_PUBLICATION_MODES = (
    KEYPOINT_PUBLICATION_MODE_SHADOW,
    KEYPOINT_PUBLICATION_MODE_PRODUCTION_CANDIDATE,
)
ATOMIC_PUBLICATION_OWNER_ATTR = "atomic_publication_owner_uuid"
_UUID_HEX = re.compile(r"^[0-9a-f]{32}$")


@dataclass(frozen=True)
class KeypointPublicationDisposition:
    """Exact non-authoritative lifecycle state sealed into one local run."""

    mode: str = KEYPOINT_PUBLICATION_MODE_SHADOW
    publication_owner_uuid: str | None = None
    run_provenance: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.mode not in KEYPOINT_PUBLICATION_MODES:
            raise ValueError(
                f"Unsupported keypoint publication mode {self.mode!r}; "
                f"expected one of {KEYPOINT_PUBLICATION_MODES}."
            )
        owner = self.publication_owner_uuid
        if self.mode == KEYPOINT_PUBLICATION_MODE_PRODUCTION_CANDIDATE:
            if not isinstance(owner, str) or not _UUID_HEX.fullmatch(owner):
                raise ValueError(
                    "Production candidates require one lowercase 32-character "
                    "publication_owner_uuid."
                )
            if not isinstance(self.run_provenance, Mapping):
                raise ValueError("Production candidates require exact run_provenance.")
        elif owner is not None or self.run_provenance is not None:
            raise ValueError(
                "Benchmark shadows cannot carry production ownership or provenance."
            )

    @property
    def production_candidate(self) -> bool:
        return self.mode == KEYPOINT_PUBLICATION_MODE_PRODUCTION_CANDIDATE

    def root_attributes(self) -> dict[str, Any]:
        if not self.production_candidate:
            return {
                "benchmark_only": True,
                "canonical": False,
                "registry_registered": False,
                "selector_eligible": False,
            }
        return {
            "benchmark_only": False,
            "canonical": False,
            "registry_registered": False,
            "selector_eligible": False,
            "production_candidate": True,
            "production_selector_activation": "deferred_separate_reviewed_change",
        }

    def family_attributes(self) -> dict[str, Any]:
        if not self.production_candidate:
            return {
                "benchmark_only": True,
                "selector_eligible": False,
                "selection_contract": "none_shadow_direct_path_only",
            }
        return {
            "benchmark_only": False,
            "selector_eligible": False,
            "selection_contract": "none_production_candidate_direct_path_only",
            "production_candidate": True,
        }

    def run_attributes(self) -> dict[str, Any]:
        if not self.production_candidate:
            return {"stage_selector_eligible": False, "shadow_only": True}
        values = {
            "stage_selector_eligible": False,
            "shadow_only": False,
            "production_candidate": True,
            "production_selector_activation": "deferred_separate_reviewed_change",
        }
        if self.publication_owner_uuid is not None:
            values[ATOMIC_PUBLICATION_OWNER_ATTR] = self.publication_owner_uuid
        return values

    def array_attributes(self) -> dict[str, Any]:
        if not self.production_candidate:
            return {"benchmark_only": True, "selector_eligible": False}
        return {
            "benchmark_only": False,
            "selector_eligible": False,
            "production_candidate": True,
        }


@dataclass(frozen=True)
class KeypointChainPublicationDispositions:
    raw: KeypointPublicationDisposition = KeypointPublicationDisposition()
    quality: KeypointPublicationDisposition = KeypointPublicationDisposition()
    refined: KeypointPublicationDisposition = KeypointPublicationDisposition()
    body_frame: KeypointPublicationDisposition = KeypointPublicationDisposition()

    def __post_init__(self) -> None:
        values = (self.raw, self.quality, self.refined, self.body_frame)
        if not all(
            isinstance(value, KeypointPublicationDisposition) for value in values
        ):
            raise TypeError("Every keypoint-chain disposition must be exact.")
        modes = {value.mode for value in values}
        if len(modes) != 1:
            raise ValueError("A keypoint chain cannot mix publication modes.")

    @property
    def production_candidate(self) -> bool:
        return self.raw.production_candidate


__all__ = [
    "ATOMIC_PUBLICATION_OWNER_ATTR",
    "KEYPOINT_PUBLICATION_MODE_PRODUCTION_CANDIDATE",
    "KEYPOINT_PUBLICATION_MODE_SHADOW",
    "KEYPOINT_PUBLICATION_MODES",
    "KeypointChainPublicationDispositions",
    "KeypointPublicationDisposition",
]

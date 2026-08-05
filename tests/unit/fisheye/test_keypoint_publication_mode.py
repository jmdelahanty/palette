from __future__ import annotations

import pytest

from fisheye.shared.zarr.keypoint_publication_mode import (
    KEYPOINT_PUBLICATION_MODE_PRODUCTION_CANDIDATE,
    KeypointChainPublicationDispositions,
    KeypointPublicationDisposition,
)


def _candidate(owner: str) -> KeypointPublicationDisposition:
    return KeypointPublicationDisposition(
        mode=KEYPOINT_PUBLICATION_MODE_PRODUCTION_CANDIDATE,
        publication_owner_uuid=owner,
        run_provenance={"schema": "test"},
    )


def test_production_candidate_lifecycle_is_ineligible_and_owner_bound() -> None:
    value = _candidate("a" * 32)

    assert value.root_attributes()["selector_eligible"] is False
    assert value.family_attributes()["selection_contract"] == (
        "none_production_candidate_direct_path_only"
    )
    assert value.run_attributes() == {
        "stage_selector_eligible": False,
        "shadow_only": False,
        "production_candidate": True,
        "production_selector_activation": "deferred_separate_reviewed_change",
        "atomic_publication_owner_uuid": "a" * 32,
    }
    assert value.array_attributes()["production_candidate"] is True


def test_production_candidate_requires_owner_and_provenance() -> None:
    with pytest.raises(ValueError, match="publication_owner_uuid"):
        KeypointPublicationDisposition(
            mode=KEYPOINT_PUBLICATION_MODE_PRODUCTION_CANDIDATE,
            run_provenance={"schema": "test"},
        )
    with pytest.raises(ValueError, match="run_provenance"):
        KeypointPublicationDisposition(
            mode=KEYPOINT_PUBLICATION_MODE_PRODUCTION_CANDIDATE,
            publication_owner_uuid="a" * 32,
        )


def test_keypoint_chain_cannot_mix_shadow_and_production_modes() -> None:
    with pytest.raises(ValueError, match="cannot mix publication modes"):
        KeypointChainPublicationDispositions(raw=_candidate("a" * 32))

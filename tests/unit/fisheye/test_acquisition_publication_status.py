from __future__ import annotations

import copy

import pytest

from fisheye.shared.acquisition_publication_status import (
    ACQUISITION_AUTHORITY_NOT_PUBLISHED,
    ACQUISITION_AUTHORITY_PENDING,
    ACQUISITION_AUTHORITY_PUBLISHED,
    ACQUISITION_AUTHORITY_STATUS_ATTR,
    CLIPPED_EXTERNAL_ACQUISITION_AUTHORITY_MODE,
    CLIPPED_EXTERNAL_ACQUISITION_PUBLISHED_REASON,
    EXTERNAL_ACQUISITION_AUTHORITY_MODE,
    EXTERNAL_ACQUISITION_PENDING_REASON,
    MATERIALIZED_ACQUISITION_AUTHORITY_MODE,
    MATERIALIZED_ACQUISITION_PUBLISHED_REASON,
    AcquisitionPublicationStatusError,
    build_acquisition_authority_publication_status,
    load_acquisition_authority_publication_status,
    parse_acquisition_authority_publication_status,
    stamp_acquisition_authority_publication_status,
)


class _Group:
    def __init__(self) -> None:
        self.attrs: dict[str, object] = {}
        self.children: dict[str, _Group] = {}

    def get(self, name: str) -> _Group | None:
        return self.children.get(name)


def _root() -> tuple[_Group, _Group]:
    root = _Group()
    raw = _Group()
    root.children["raw_video"] = raw
    return root, raw


def test_status_builder_and_parser_are_mode_and_type_strict() -> None:
    materialized = build_acquisition_authority_publication_status(
        status=ACQUISITION_AUTHORITY_PUBLISHED,
        reason_code=MATERIALIZED_ACQUISITION_PUBLISHED_REASON,
        authority_mode=MATERIALIZED_ACQUISITION_AUTHORITY_MODE,
        authority_path="analysis/acquisition_camera_frames/camera-1",
    )
    assert (
        parse_acquisition_authority_publication_status(materialized.to_dict())
        == materialized
    )
    clipped = build_acquisition_authority_publication_status(
        status=ACQUISITION_AUTHORITY_PUBLISHED,
        reason_code=CLIPPED_EXTERNAL_ACQUISITION_PUBLISHED_REASON,
        authority_mode=CLIPPED_EXTERNAL_ACQUISITION_AUTHORITY_MODE,
        authority_path="analysis/acquisition_camera_frames/camera-1",
    )
    assert parse_acquisition_authority_publication_status(clipped.to_dict()) == clipped

    wrong_type = materialized.to_dict()
    wrong_type["schema_version"] = float(wrong_type["schema_version"])
    with pytest.raises(AcquisitionPublicationStatusError, match="schema_version"):
        parse_acquisition_authority_publication_status(wrong_type)

    with pytest.raises(AcquisitionPublicationStatusError, match="reason"):
        build_acquisition_authority_publication_status(
            status=ACQUISITION_AUTHORITY_PENDING,
            reason_code=EXTERNAL_ACQUISITION_PENDING_REASON,
            authority_mode=MATERIALIZED_ACQUISITION_AUTHORITY_MODE,
            authority_path="analysis/acquisition_camera_frames/camera-1",
        )


def test_mirrored_status_loader_returns_typed_mode_and_fails_on_conflict() -> None:
    root, raw = _root()
    stamped = stamp_acquisition_authority_publication_status(
        root,
        raw,
        status=ACQUISITION_AUTHORITY_PENDING,
        reason_code=EXTERNAL_ACQUISITION_PENDING_REASON,
        authority_mode=EXTERNAL_ACQUISITION_AUTHORITY_MODE,
        authority_path="analysis/acquisition_camera_frames/camera-1",
    )
    loaded = load_acquisition_authority_publication_status(root)
    assert loaded == stamped
    assert loaded.authority_mode == EXTERNAL_ACQUISITION_AUTHORITY_MODE

    raw.attrs[ACQUISITION_AUTHORITY_STATUS_ATTR] = copy.deepcopy(
        raw.attrs[ACQUISITION_AUTHORITY_STATUS_ATTR]
    )
    raw.attrs[ACQUISITION_AUTHORITY_STATUS_ATTR]["reason_code"] = "conflict"  # type: ignore[index]
    with pytest.raises(AcquisitionPublicationStatusError, match="conflict"):
        load_acquisition_authority_publication_status(root)


def test_noncanonical_status_cannot_claim_an_authority_mode_or_path() -> None:
    record = build_acquisition_authority_publication_status(
        status=ACQUISITION_AUTHORITY_NOT_PUBLISHED,
        reason_code="sampled_or_training_import",
    )
    assert record.authority_mode is None
    assert record.authority_path is None

    with pytest.raises(AcquisitionPublicationStatusError, match="cannot claim"):
        build_acquisition_authority_publication_status(
            status=ACQUISITION_AUTHORITY_NOT_PUBLISHED,
            reason_code="sampled_or_training_import",
            authority_mode=EXTERNAL_ACQUISITION_AUTHORITY_MODE,
        )

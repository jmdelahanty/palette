"""The registry projects canonical subject fields, whatever the source wrote."""

from __future__ import annotations

import copy
import json

import pytest
import zarr

from fisheye.registry.db import Registry, _extract_provenance, _extract_snapshot
from fisheye.shared import zebrobot_subject_reference as zsr
from fisheye.shared.subject_metadata import publish_subject_metadata
from tests.unit.fisheye.test_subject_fields import (
    CITRUS_V3,
    LEGACY_H5,
    ORANGE_REFERENCE,
    SERVED,
    SUBJECT,
)

PARENTS = [{"identifier": "8130_M13_A2", "sex": "F"}, {"identifier": "6728_M19_E5", "sex": "M"}]


def _orange_metadata():
    def fetch(url):
        return (200, copy.deepcopy(SERVED)) if url.endswith("/citrus-snapshot") else (200, {"items": []})

    return zsr.resolve_subject_reference(ORANGE_REFERENCE, camera="CAM-1", fetch=fetch).metadata


def _project(tmp_path, name, metadata):
    root = zarr.open_group(str(tmp_path / f"{name}.zarr"), mode="w")
    publish_subject_metadata(root, metadata, source_artifact={"kind": name})
    subject, source, raw = _extract_snapshot(root)
    registry = Registry(tmp_path / f"{name}.sqlite")
    registry.upsert_recording(recording_id=f"rec-{name}")
    registry.upsert_dataset(
        f"dataset-{name}", session_uuid=f"session-{name}", zarr_path=tmp_path / f"{name}.zarr",
        recording_id=f"rec-{name}", artifact_kind="source_recording", zarr_use="training",
    )
    registry.upsert_subject_snapshot_entities(
        f"dataset-{name}", recording_id=f"rec-{name}", snapshot=subject, snapshot_source=source
    )
    rows = registry.conn.execute(
        "SELECT subject_id, dpf_at_acquisition, sex, genotype FROM recording_subjects"
    ).fetchall()
    parents = registry.conn.execute("SELECT parents_json FROM crosses").fetchall()
    return subject, _extract_provenance(subject, raw), [tuple(r) for r in rows], parents


@pytest.mark.parametrize("name, metadata", [("legacy", LEGACY_H5), ("citrus", CITRUS_V3)])
def test_h5_sources_project_identity_age_and_parents(tmp_path, name, metadata):
    _subject, provenance, rows, parents = _project(tmp_path, name, metadata)
    # fish_id (legacy) and subject_id (Citrus v3) both land as the subject.
    assert rows == [(SUBJECT, 8, "unknown", "AB [AB IC] SEPT25")]
    assert provenance["fish_id"] == SUBJECT
    # "ID [sex]" is parsed: the sex is no longer stuck inside the identifier.
    assert json.loads(parents[0][0]) == PARENTS
    assert provenance["dpf_at_acquisition"] == 8


def test_orange_reference_reaches_the_registry_with_its_age(tmp_path):
    subject, provenance, rows, parents = _project(tmp_path, "orange", _orange_metadata())
    assert rows == []  # Orange names no Citrus subject, so no subject row is invented
    assert provenance["dpf_at_acquisition"] == 8  # previously lost (dpf_at_recording unread)
    assert provenance["dish_id"] == "18867_10" and provenance["parents"] == PARENTS
    assert provenance["fish_id"] is None


def test_placeholder_identities_are_not_projected(tmp_path):
    _subject, provenance, rows, _parents = _project(
        tmp_path, "placeholder",
        {"subject_ids": ["rec-local-1"], "subject_count": 1, "species": "Danio rerio",
         "identity_scope": "recording_local_placeholder",
         "source": "manual_subject_context_backfill", "status": "manual_backfill"},
    )
    assert rows == [] and provenance["fish_id"] is None

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path

import pytest

from fisheye.shared.coordinate_record import (
    stamp_and_bind_persisted_coordinate_record,
)
from fisheye.shared.zarr.coordinate_successor_authority import (
    COORDINATE_SUCCESSOR_AUTHORITY_ATTR,
    CoordinateSuccessorAuthorityError,
    KEYPOINT_COORDINATE_SUCCESSOR_KIND,
    build_coordinate_successor_authority,
    load_coordinate_successor_authority,
    stamp_coordinate_successor_authority,
    validate_coordinate_successor_authority,
)
from fisheye.shared.zarr.coordinate_successor_files import (
    copy_metadata_and_link_payload,
    metadata_tree_sha256,
)


class _Node:
    _archive = object()

    def __init__(self, *, path: str, children=None) -> None:
        self.path = path
        self._coordinate_archive_token = self._archive
        self.attrs: dict[str, object] = {}
        self.children = dict(children or {})

    def __getitem__(self, key: str):
        return self.children[key]


def _source_manifest() -> dict[str, object]:
    return {
        "schema_id": "palette.keypoint.run_manifest",
        "schema_version": 1,
        "payload_digest": "1" * 64,
        "payload": {"logical_content": {"digest": "2" * 64}},
    }


def _authority(run: _Node) -> dict[str, object]:
    coordinate = stamp_and_bind_persisted_coordinate_record(
        run,
        {"schema_id": "coordinate-test", "schema_version": 1},
        attr_name="coordinate_context",
    )
    return build_coordinate_successor_authority(
        kind=KEYPOINT_COORDINATE_SUCCESSOR_KIND,
        source_family="keypoints_runs",
        source_run_path="keypoints_runs/source",
        source_manifest=_source_manifest(),
        source_authority_kind="sealed_keypoint_bundle",
        source_authority={"schema_id": "source-authority", "schema_version": 1},
        successor_family="keypoints_runs",
        successor_run_path=run.path,
        payload_equivalence={"mode": "hardlink_exact_payload"},
        coordinate_records={
            "context": {
                "record_ref": coordinate.record_ref,
                "record_sha256": coordinate.record_sha256,
            }
        },
    )


def test_coordinate_successor_authority_revalidates_persisted_record() -> None:
    run = _Node(path="keypoints_runs/successor")
    authority = _authority(run)

    assert validate_coordinate_successor_authority(authority) == ()
    stamp_coordinate_successor_authority(run, authority)
    assert load_coordinate_successor_authority(
        run,
        expected_kind=KEYPOINT_COORDINATE_SUCCESSOR_KIND,
        expected_successor_run_path=run.path,
    ) == authority

    run.attrs["coordinate_context"] = {
        "schema_id": "coordinate-test",
        "schema_version": 2,
    }
    with pytest.raises(
        CoordinateSuccessorAuthorityError,
        match="missing, malformed, or stale",
    ):
        load_coordinate_successor_authority(
            run,
            expected_kind=KEYPOINT_COORDINATE_SUCCESSOR_KIND,
            expected_successor_run_path=run.path,
        )


def test_coordinate_successor_authority_tampering_fails_closed() -> None:
    run = _Node(path="keypoints_runs/successor")
    authority = _authority(run)
    changed = deepcopy(authority)
    changed["payload"]["successor"]["stage_selector_eligible"] = True

    errors = validate_coordinate_successor_authority(changed)

    assert "coordinate successor authority payload digest differs" in errors
    assert "coordinate successor target lifecycle is invalid" in errors

    stamp_coordinate_successor_authority(run, authority)
    persisted = deepcopy(run.attrs[COORDINATE_SUCCESSOR_AUTHORITY_ATTR])
    persisted["payload"]["production_state_changes"] = ["latest"]
    run.attrs[COORDINATE_SUCCESSOR_AUTHORITY_ATTR] = persisted
    with pytest.raises(CoordinateSuccessorAuthorityError, match="payload digest"):
        load_coordinate_successor_authority(
            run,
            expected_kind=KEYPOINT_COORDINATE_SUCCESSOR_KIND,
            expected_successor_run_path=run.path,
        )


def test_successor_copy_separates_metadata_and_hardlinks_payload(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    (source / "keypoints" / "c").mkdir(parents=True)
    (source / "zarr.json").write_text('{"node_type":"group"}\n')
    (source / "keypoints" / "zarr.json").write_text(
        '{"node_type":"array"}\n'
    )
    (source / "keypoints" / "c" / "0").write_bytes(b"payload")
    source_metadata = metadata_tree_sha256(source)

    receipt = copy_metadata_and_link_payload(source, target)

    assert receipt == {
        "metadata_files_copied": 2,
        "payload_files_hardlinked": 1,
    }
    assert metadata_tree_sha256(source) == source_metadata
    assert os.stat(source / "zarr.json").st_ino != os.stat(
        target / "zarr.json"
    ).st_ino
    assert os.stat(source / "keypoints" / "c" / "0").st_ino == os.stat(
        target / "keypoints" / "c" / "0"
    ).st_ino

    (target / "zarr.json").write_text('{"node_type":"group","attributes":{}}\n')
    assert metadata_tree_sha256(source) == source_metadata
    assert (source / "keypoints" / "c" / "0").read_bytes() == b"payload"

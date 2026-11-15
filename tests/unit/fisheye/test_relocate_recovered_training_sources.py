"""Route recovered training archives to the exact manifest recording names."""

import hashlib
import json
from pathlib import Path

import pytest

from fisheye.training import relocate_recovered_training_sources as relocate_module
from fisheye.training.relocate_recovered_training_sources import (
    build_relocation_plan,
)


def _inputs() -> tuple[dict, dict]:
    collection = {
        "schema_id": "palette.training.merged_recovery_collection.v1",
        "stage_selector_eligible": False,
        "source_detect_set_id": "detect_set",
        "pose_rows": 2,
        "detect_rows": 3,
        "detect_only_rows": 1,
        "recordings_with_pose": 1,
        "recordings": [
            {
                "recording_id": "rec_a",
                "path": "/tmp/recovered/rec_a_recovered_training.zarr",
                "pose_rows": 2,
                "detect_rows": 3,
                "detect_only_rows": 1,
                "array_digest_index_sha256": "a" * 64,
            }
        ],
    }
    detect_manifest = {
        "set_id": "detect_set",
        "merged_export": {
            "source_datasets": [{"dataset_id": "rec_a", "name": "rec_a_DefaultScreen"}]
        },
    }
    return collection, detect_manifest


def test_plan_uses_manifest_recording_name_and_preserves_old_path() -> None:
    collection, detect_manifest = _inputs()
    plan = build_relocation_plan(
        collection=collection,
        collection_dir=Path("/tmp/recovered"),
        detect_manifest=detect_manifest,
        recordings_root=Path("/groups/recordings"),
    )
    assert plan[0]["source_path"] == "/tmp/recovered/rec_a_recovered_training.zarr"
    assert plan[0]["destination_path"] == (
        "/groups/recordings/rec_a_DefaultScreen/zarr/"
        "rec_a_DefaultScreen_recovered_training.zarr"
    )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("path", "/tmp/other/rec_a_recovered_training.zarr", "path disagrees"),
        ("detect_rows", 4, "detect_rows total disagrees"),
    ],
)
def test_plan_refuses_conflicting_collection(
    field: str, replacement: object, message: str
) -> None:
    collection, detect_manifest = _inputs()
    collection["recordings"][0][field] = replacement
    with pytest.raises(ValueError, match=message):
        build_relocation_plan(
            collection=collection,
            collection_dir=Path("/tmp/recovered"),
            detect_manifest=detect_manifest,
            recordings_root=Path("/groups/recordings"),
        )


def test_plan_refuses_unsafe_manifest_name() -> None:
    collection, detect_manifest = _inputs()
    detect_manifest["merged_export"]["source_datasets"][0]["name"] = "rec_a/evil"
    with pytest.raises(ValueError, match="Unsafe"):
        build_relocation_plan(
            collection=collection,
            collection_dir=Path("/tmp/recovered"),
            detect_manifest=detect_manifest,
            recordings_root=Path("/groups/recordings"),
        )


def test_relocation_keeps_archive_inode_and_records_both_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection, detect_manifest = _inputs()
    old = tmp_path / "recovered" / "rec_a_recovered_training.zarr"
    old.mkdir(parents=True)
    (old / "payload").write_bytes(b"unchanged")
    original_inode = old.stat().st_ino
    attrs = {
        "pose_source_row_count": 2,
        "detect_source_row_count": 3,
        "detect_only_row_count": 1,
        "array_sha256": {"payload": hashlib.sha256(b"unchanged").hexdigest()},
    }
    digest = hashlib.sha256(
        json.dumps(
            attrs["array_sha256"], sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    collection["recordings"][0]["path"] = str(old)
    collection["recordings"][0]["array_digest_index_sha256"] = digest
    collection["recordings_with_pose"] = 1
    manifest_path = tmp_path / "detect_manifest.json"
    manifest_path.write_text(json.dumps(detect_manifest))
    collection["source_detect_manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    collection_path = old.parent / "recovery_collection.json"
    collection_path.write_text(json.dumps(collection))
    monkeypatch.setattr(
        relocate_module,
        "validate_recovered_recording",
        lambda path, **kwargs: attrs,
    )
    recordings_root = tmp_path / "recordings"
    receipt = relocate_module.relocate_full_recovery(
        collection_path=collection_path,
        detect_manifest_path=manifest_path,
        recordings_root=recordings_root,
    )
    destination = Path(receipt["recordings"][0]["destination_path"])
    assert not old.exists()
    assert destination.stat().st_ino == original_inode
    assert (destination / "payload").read_bytes() == b"unchanged"
    assert receipt["recordings"][0]["source_path"] == str(old)
    assert (recordings_root / "_index" / relocate_module.RECEIPT_NAME).exists()

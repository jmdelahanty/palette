"""Derived indexing preserves source rows; it is not codec/import acceptance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pyarrow.parquet as pq
import pytest

from fisheye.shared import recording_transfer_snapshot as transfer
from fisheye.utils import build_transfer_parent_frame_index as indexer
from fisheye.utils.build_recording_frame_index import TABLE_SCHEMA

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures/recording_transfer_v2"


def _copy(tmp_path: Path, name: str = "rolling") -> Path:
    return Path(shutil.copytree(FIXTURES / name, tmp_path / name))


def _bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _resign(root: Path) -> None:
    snapshot = transfer.build_snapshot(root, transfer.MARKER_NAME, destination=True)
    data = transfer.canonical_bytes(snapshot)
    (root / transfer.SNAPSHOT_PATH).write_bytes(data)
    marker_path = root / transfer.MARKER_NAME
    marker = json.loads(marker_path.read_bytes())
    digest = hashlib.sha256(data).hexdigest()
    marker["snapshot_id"] = "sha256:" + digest
    marker["snapshot"].update(size_bytes=len(data), sha256=digest)
    marker_path.write_bytes(transfer.canonical_bytes(marker))


@pytest.mark.parametrize("name", ["whole", "rolling"])
@pytest.mark.parametrize("batch_rows", [1, 2, 65536])
def test_parent_index_preserves_rows_and_source_bytes(
    tmp_path: Path, name: str, batch_rows: int
) -> None:
    root = _copy(tmp_path, name)
    before = _bytes(root)
    destination = tmp_path / "index"
    result = indexer.build_transfer_parent_frame_index(
        root, camera_id="02010093", output_dir=destination, batch_rows=batch_rows
    )
    assert result["status"] == "ok"
    assert result["import_complete"] is False
    assert result["source_cleanup_authorized"] is False
    assert _bytes(root) == before
    table = pq.read_table(destination / "recording_frame_index.parquet")
    assert table.schema == TABLE_SCHEMA
    rows = table.to_pylist()
    count = 2 if name == "whole" else 3
    assert [row["recording_frame_id"] for row in rows] == list(range(1, count + 1))
    assert [row["parent_frame_index"] for row in rows] == list(range(count))
    assert [row["clip_local_frame_index"] for row in rows] == (
        [0, 1] if name == "whole" else [0, 1, 0]
    )
    assert {row["camera_serial"] for row in rows} == {"02010093"}
    assert all("crop" not in row["video_path"] for row in rows)
    assert all(Path(row["video_path"]).is_file() for row in rows)
    snapshot = transfer.verify_transfer_snapshot(root)
    parent = transfer.plan_parent_recordings(snapshot)[0]
    assert {row["recording_id"] for row in rows} == {parent.recording_id}
    assert {row["session_id"] for row in rows} == {parent.session_uuid}
    assert [row["clip_id"] for row in rows] == [
        clip.clip_id
        for clip in parent.clips
        for output in clip.outputs
        if output.output_kind == "full"
        for _ in range(output.frame_count)
    ]
    manifest = json.loads(
        (destination / "recording_frame_index_manifest.json").read_bytes()
    )
    assert manifest["source_transfer"]["snapshot_id"] == snapshot.snapshot_id
    assert manifest["source_transfer"]["camera_id"] == "02010093"
    assert manifest["row_count"] == count
    assert manifest["execution"]["max_buffered_rows"] <= batch_rows
    assert manifest["clock_validity"] == "not_evaluated_by_index"
    assert (
        manifest["parquet_sha256"]
        == hashlib.sha256(
            (destination / "recording_frame_index.parquet").read_bytes()
        ).hexdigest()
    )
    clip_index = json.loads((destination / "recording_clip_index.json").read_bytes())
    assert len(clip_index["clips"]) == len(parent.clips)
    assert clip_index["source_transfer"] == manifest["source_transfer"]


def test_dry_run_does_not_create_outputs(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    destination = tmp_path / "not-created" / "index"
    result = indexer.build_transfer_parent_frame_index(
        root, camera_id="02010093", output_dir=destination, dry_run=True, batch_rows=1
    )
    assert result["status"] == "ok"
    assert result["wrote_parquet"] is False
    assert not destination.parent.exists()


def test_batch_storage_changes_preserve_logical_rows(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    before = _bytes(root)
    outputs = [tmp_path / "one-row-batches", tmp_path / "large-batches"]
    for output, batch in zip(outputs, [1, 65536]):
        indexer.build_transfer_parent_frame_index(
            root, camera_id="02010093", output_dir=output, batch_rows=batch
        )
    first, second = [
        pq.read_table(output / "recording_frame_index.parquet") for output in outputs
    ]
    assert first.equals(second)
    assert _bytes(root) == before
    # Different physical batching may change Parquet/digest bytes. No claim of
    # physical-digest stability is made by equality of logical rows.


@pytest.mark.parametrize("batch_rows", [0, -1, True, 1.5, 1048577])
def test_invalid_batch_size_refuses_without_writes(tmp_path: Path, batch_rows) -> None:
    root = _copy(tmp_path)
    with pytest.raises(ValueError, match="batch_rows"):
        indexer.build_transfer_parent_frame_index(
            root,
            camera_id="02010093",
            output_dir=tmp_path / "out",
            batch_rows=batch_rows,
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("camera_id", ["2010093", "absent", 2010093, None])
def test_camera_identity_is_exact(tmp_path: Path, camera_id) -> None:
    root = _copy(tmp_path)
    with pytest.raises(ValueError, match="camera"):
        indexer.build_transfer_parent_frame_index(
            root, camera_id=camera_id, output_dir=tmp_path / "out"
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("nested", [False, True])
def test_source_tree_is_never_an_output_destination(
    tmp_path: Path, nested: bool
) -> None:
    root = _copy(tmp_path)
    before = _bytes(root)
    with pytest.raises(ValueError, match="source"):
        indexer.build_transfer_parent_frame_index(
            root, camera_id="02010093", output_dir=root / "derived" if nested else root
        )
    assert _bytes(root) == before


@pytest.mark.parametrize("dry_run", [False, True])
def test_uint64_timestamp_cannot_be_silently_cast_to_int64(
    tmp_path: Path, dry_run: bool
) -> None:
    root = _copy(tmp_path, "whole")
    metadata = root / "Cam02010093_full.csv"
    lines = metadata.read_text().splitlines()
    header = lines[0].split(",")
    row = lines[1].split(",")
    row[header.index("timestamp")] = str(2**63)
    lines[1] = ",".join(row)
    metadata.write_text("\n".join(lines) + "\n")
    _resign(root)
    transfer.verify_transfer_snapshot(root)  # Transport's uint64 is valid.
    with pytest.raises(ValueError, match="signed-int64"):
        indexer.build_transfer_parent_frame_index(
            root, camera_id="02010093", output_dir=tmp_path / "out", dry_run=dry_run
        )
    assert not (tmp_path / "out" / "recording_frame_index_manifest.json").exists()


def test_existing_destination_never_overwritten(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    destination = tmp_path / "out"
    destination.mkdir()
    (destination / "owned-by-other-worker").write_text("keep")
    before = _bytes(destination)
    with pytest.raises(FileExistsError):
        indexer.build_transfer_parent_frame_index(
            root, camera_id="02010093", output_dir=destination
        )
    assert _bytes(destination) == before


def test_failed_optional_proof_is_refused_before_index_creation(tmp_path: Path) -> None:
    root = _copy(tmp_path, "failed_optional_proof")
    with pytest.raises(ValueError, match="frame_identity_proof.*failed"):
        indexer.build_transfer_parent_frame_index(
            root, camera_id="02010093", output_dir=tmp_path / "out"
        )
    assert not (tmp_path / "out").exists()


def test_changed_source_during_write_leaves_no_success_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    root = _copy(tmp_path)
    real_write = indexer.pq.ParquetWriter.write_table
    changed = False

    def mutate(self, table, *args, **kwargs):
        nonlocal changed
        result = real_write(self, table, *args, **kwargs)
        if not changed:
            changed = True
            (root / "unexpected.txt").write_text("concurrent source mutation")
        return result

    monkeypatch.setattr(indexer.pq.ParquetWriter, "write_table", mutate)
    with pytest.raises(ValueError):
        indexer.build_transfer_parent_frame_index(
            root, camera_id="02010093", output_dir=tmp_path / "out"
        )
    assert not (tmp_path / "out" / "recording_frame_index_manifest.json").exists()


def test_write_interruption_leaves_no_success_and_no_source_mutation(
    tmp_path: Path, monkeypatch
) -> None:
    root = _copy(tmp_path)
    before = _bytes(root)

    def fail(*args, **kwargs):
        raise OSError("injected storage failure")

    monkeypatch.setattr(indexer.pq.ParquetWriter, "write_table", fail)
    with pytest.raises(OSError, match="injected"):
        indexer.build_transfer_parent_frame_index(
            root, camera_id="02010093", output_dir=tmp_path / "out"
        )
    assert not (tmp_path / "out" / "recording_frame_index_manifest.json").exists()
    assert _bytes(root) == before


def test_output_ownership_loss_cannot_publish_into_replacement(
    tmp_path: Path, monkeypatch
) -> None:
    root = _copy(tmp_path)
    destination = tmp_path / "out"
    retained = tmp_path / "displaced-output"
    real_write = indexer.pq.ParquetWriter.write_table
    changed = False

    def replace_directory(self, table, *args, **kwargs):
        nonlocal changed
        result = real_write(self, table, *args, **kwargs)
        if not changed:
            changed = True
            destination.rename(retained)
            destination.mkdir()
            (destination / "other-owner").write_text("keep")
        return result

    monkeypatch.setattr(indexer.pq.ParquetWriter, "write_table", replace_directory)
    with pytest.raises(ValueError, match="ownership"):
        indexer.build_transfer_parent_frame_index(
            root, camera_id="02010093", output_dir=destination
        )
    assert _bytes(destination) == {"other-owner": b"keep"}
    assert not (retained / "recording_frame_index_manifest.json").exists()


def test_failed_manifest_publication_needs_fresh_attempt(
    tmp_path: Path, monkeypatch
) -> None:
    root = _copy(tmp_path)
    first_output = tmp_path / "failed-attempt"
    real_write = indexer.write_json_atomic

    def fail_manifest(path, *args, **kwargs):
        if path.name == "recording_frame_index_manifest.json":
            raise OSError("injected final publication failure")
        return real_write(path, *args, **kwargs)

    monkeypatch.setattr(indexer, "write_json_atomic", fail_manifest)
    with pytest.raises(OSError, match="final publication"):
        indexer.build_transfer_parent_frame_index(
            root, camera_id="02010093", output_dir=first_output
        )
    assert not (first_output / "recording_frame_index_manifest.json").exists()
    monkeypatch.setattr(indexer, "write_json_atomic", real_write)
    with pytest.raises(FileExistsError):
        indexer.build_transfer_parent_frame_index(
            root, camera_id="02010093", output_dir=first_output
        )
    result = indexer.build_transfer_parent_frame_index(
        root, camera_id="02010093", output_dir=tmp_path / "fresh-attempt"
    )
    assert result["status"] == "ok"
    assert first_output.is_dir()  # Failed evidence is preserved, not deleted.

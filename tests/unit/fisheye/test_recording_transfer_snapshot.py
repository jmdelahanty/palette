"""Transport is not admission; pinned fake media tests make no codec claim."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from fisheye.shared.recording_transfer_snapshot import (
    MARKER_NAME,
    SNAPSHOT_PATH,
    TransferSnapshotError,
    build_snapshot,
    canonical_bytes,
    normalized_path,
    plan_parent_recordings,
    verify_transfer_snapshot,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures/recording_transfer_v2"
GOLDENS = {
    "whole": "8dfeac35339f39b031cb2b33b4358cba0435bcad2e6967322518755d24d5ae21",
    "rolling": "5f25f35c56561e6d8ac5c00948948373d18f1d9eafedbfedc019f0c6a8e5a97f",
    "failed_optional_proof": "51ab32e4d2f4be1ebf58703181a1c8bb1a6927e0b1db3f3a0d5546f06679795b",
}
REVIEW_CASES = json.loads((FIXTURES / "review_cases.json").read_bytes())


@pytest.mark.parametrize("case", REVIEW_CASES["paths"])
def test_shared_artifact_path_corpus(case: dict) -> None:
    if case["artifact"]:
        assert normalized_path(case["value"]) == case["value"]
    else:
        with pytest.raises(TransferSnapshotError):
            normalized_path(case["value"])


@pytest.mark.parametrize(
    "case", REVIEW_CASES["clip_contradictions"], ids=lambda case: case["name"]
)
def test_shared_clip_closure_contradictions(tmp_path: Path, case: dict) -> None:
    root = copy_bundle(tmp_path, "whole")
    manifest = json.loads((root / "recording_session.json").read_bytes())
    manifest["clips"][0].update(case["set"])
    write_json(root / "recording_session.json", manifest)
    # The semantic reconstruction must refuse even before a new snapshot can
    # be sealed. This is not merely the old inventory noticing a changed hash.
    with pytest.raises(TransferSnapshotError, match="clip"):
        build_snapshot(root, MARKER_NAME, destination=True)


@pytest.mark.parametrize(
    "case", REVIEW_CASES["finalization_contradictions"], ids=lambda case: case["name"]
)
def test_shared_container_contradictions(tmp_path: Path, case: dict) -> None:
    root = copy_bundle(tmp_path, "whole")
    final_path = next(root.rglob("*.finalization.json"))
    final = json.loads(final_path.read_bytes())
    final[case["section"]].update(case["set"])
    write_json(final_path, final)
    with pytest.raises(TransferSnapshotError):
        build_snapshot(root, MARKER_NAME, destination=True)


def copy_bundle(tmp_path: Path, name: str = "rolling") -> Path:
    return Path(shutil.copytree(FIXTURES / name, tmp_path / name))


def write_json(path: Path, value: dict) -> None:
    path.write_bytes(canonical_bytes(value))


def resign_snapshot(root: Path, snapshot: dict) -> None:
    data = canonical_bytes(snapshot)
    (root / SNAPSHOT_PATH).write_bytes(data)
    marker = json.loads((root / MARKER_NAME).read_bytes())
    digest = hashlib.sha256(data).hexdigest()
    marker["snapshot_id"] = "sha256:" + digest
    marker["snapshot"].update(size_bytes=len(data), sha256=digest)
    write_json(root / MARKER_NAME, marker)


@pytest.mark.parametrize("name", GOLDENS)
def test_exact_shared_golden_transport(name: str) -> None:
    verified = verify_transfer_snapshot(FIXTURES / name)
    assert verified.snapshot_id == "sha256:" + GOLDENS[name]


def test_two_cameras_are_two_parents_not_four_clip_recordings() -> None:
    verified = verify_transfer_snapshot(FIXTURES / "rolling")
    parents = plan_parent_recordings(verified)
    assert [parent.camera_id for parent in parents] == ["02010093", "02010094"]
    assert len({parent.recording_id for parent in parents}) == 2
    assert all(parent.session_uuid == "fixture-session" for parent in parents)
    assert all(len(parent.clips) == 2 for parent in parents)
    assert all(parent.total_frames == 3 for parent in parents)
    assert all(parent.recording_layout == "rolling_clips" for parent in parents)
    assert all(
        {output.output_kind for output in parent.clips[0].outputs} == {"full", "crop"}
        for parent in parents
    )


def test_native_single_clip_omissions_and_null_success_are_preserved() -> None:
    verified = verify_transfer_snapshot(FIXTURES / "whole")
    (parent,) = plan_parent_recordings(verified)
    assert parent.recording_layout == "single_video"
    assert verified.recording_payload_kind == "citrus_h5"
    assert parent.total_frames == 2
    assert parent.clips[0].directory == "."


def test_failed_optional_proof_is_transport_valid_but_not_parent_plan() -> None:
    verified = verify_transfer_snapshot(FIXTURES / "failed_optional_proof")
    with pytest.raises(TransferSnapshotError, match="frame_identity_proof.*failed"):
        plan_parent_recordings(verified)


def test_proof_failure_not_a_fake_codec_failure(tmp_path: Path) -> None:
    root = copy_bundle(tmp_path, "failed_optional_proof")
    with pytest.raises(TransferSnapshotError, match="frame_identity_proof.*failed"):
        plan_parent_recordings(verify_transfer_snapshot(root))
    # Keep the same fake media and every other input. Only remove the optional
    # proof, then regenerate transport evidence. This demonstrates which gate
    # refused the invalid sibling without claiming encoded-media admission.
    write_json(root / "Cam02010093_full.summary.json", {})
    resign_snapshot(root, build_snapshot(root, MARKER_NAME, destination=True))
    assert len(plan_parent_recordings(verify_transfer_snapshot(root))) == 1


def change_output_frames(
    root: Path,
    manifest: dict,
    clip_index: int,
    kind: str,
    frames: list[int],
    *,
    crop_offset: int = 0,
) -> None:
    output = manifest["clips"][clip_index]["recording_outputs"]["02010093"][kind]
    metadata = root / output["metadata"]
    header, *rows = metadata.read_text().splitlines()
    columns = header.split(",")
    rewritten = []
    for index, frame in enumerate(frames):
        fields = dict(zip(columns, rows[min(index, len(rows) - 1)].split(",")))
        fields.update(frame_id=str(frame), recording_frame_id=str(frame))
        if kind == "crop":
            fields.update(
                crop_video_frame_index=str(index),
                session_crop_video_frame_index=str(crop_offset + index),
            )
        rewritten.append(",".join(fields[column] for column in columns))
    metadata.write_text(header + "\n" + "\n".join(rewritten) + "\n")
    output.update(
        frame_count=len(frames),
        first_recording_frame_id=frames[0],
        last_recording_frame_id=frames[-1],
        recording_frame_id_gaps=frames[-1] - frames[0] + 1 - len(frames),
    )
    final_path = root / (output["video"] + ".finalization.json")
    final = json.loads(final_path.read_bytes())
    final["packet_writes"].update(
        submissions_accepted=len(frames),
        write_attempts=len(frames),
        packets_written=len(frames),
    )
    write_json(final_path, final)


def test_sparse_crop_subset_preserved_without_dense_renumbering(tmp_path: Path) -> None:
    root = copy_bundle(tmp_path)
    manifest = json.loads((root / "recording_session.json").read_bytes())
    change_output_frames(root, manifest, 0, "crop", [1])
    change_output_frames(root, manifest, 1, "crop", [3], crop_offset=1)
    write_json(root / "recording_session.json", manifest)
    resign_snapshot(root, build_snapshot(root, MARKER_NAME, destination=True))
    verified = verify_transfer_snapshot(root)
    first, _ = plan_parent_recordings(verified)
    crop = next(
        output for output in first.clips[1].outputs if output.output_kind == "crop"
    )
    assert crop.first_recording_frame_id == 3
    assert first.total_frames == 3


def test_transport_valid_full_gaps_refused_by_dense_parent_profile(
    tmp_path: Path,
) -> None:
    root = copy_bundle(tmp_path)
    manifest = json.loads((root / "recording_session.json").read_bytes())
    change_output_frames(root, manifest, 0, "full", [1, 3])
    change_output_frames(root, manifest, 1, "full", [4])
    write_json(root / "recording_session.json", manifest)
    resign_snapshot(root, build_snapshot(root, MARKER_NAME, destination=True))
    verified = verify_transfer_snapshot(root)
    with pytest.raises(TransferSnapshotError, match="dense.*frame-id gaps"):
        plan_parent_recordings(verified)


@pytest.mark.parametrize("error_value", [True, "0", 0.5, -1])
def test_success_booleans_do_not_override_present_finalization_error(
    tmp_path: Path, error_value: object
) -> None:
    root = copy_bundle(tmp_path)
    path = next(root.rglob("*.finalization.json"))
    final = json.loads(path.read_bytes())
    final["container"]["trailer_error_code"] = error_value
    write_json(path, final)
    with pytest.raises(TransferSnapshotError, match="trailer_error_code"):
        verify_transfer_snapshot(root)


def test_old_verified_object_cannot_plan_changed_source(tmp_path: Path) -> None:
    root = copy_bundle(tmp_path)
    old = verify_transfer_snapshot(root)
    (root / "fixture_notice.txt").write_text("another immutable generation\n")
    resign_snapshot(root, build_snapshot(root, MARKER_NAME, destination=True))
    assert verify_transfer_snapshot(root).snapshot_id != old.snapshot_id
    with pytest.raises(TransferSnapshotError, match="generation changed"):
        plan_parent_recordings(old)


def test_location_and_delivery_attempt_are_not_parent_or_snapshot_identity(
    tmp_path: Path,
) -> None:
    root = copy_bundle(tmp_path)
    original = verify_transfer_snapshot(root)
    marker = json.loads((root / MARKER_NAME).read_bytes())
    marker["delivery"].update(attempt_id="f" * 32, destination_dir="/another/delivery")
    write_json(root / MARKER_NAME, marker)
    replay = verify_transfer_snapshot(root)
    assert replay.snapshot_id == original.snapshot_id
    assert replay.attempt_id != original.attempt_id
    assert plan_parent_recordings(replay) == plan_parent_recordings(original)


@pytest.mark.parametrize("mutation", ["missing", "extra", "changed", "symlink"])
def test_closed_inventory_refuses_mutations(tmp_path: Path, mutation: str) -> None:
    root = copy_bundle(tmp_path)
    video = next(root.rglob("*.mp4"))
    if mutation == "missing":
        video.unlink()
    elif mutation == "extra":
        (root / "extra.txt").write_text("undeclared")
    elif mutation == "changed":
        video.write_bytes(b"changed media")
    else:
        video.unlink()
        video.symlink_to(FIXTURES / "whole/Cam02010093_full.mp4")
    with pytest.raises(TransferSnapshotError):
        verify_transfer_snapshot(root)


@pytest.mark.parametrize("target", [MARKER_NAME, SNAPSHOT_PATH])
def test_duplicate_json_keys_refused(tmp_path: Path, target: str) -> None:
    root = copy_bundle(tmp_path)
    path = root / target
    path.write_bytes(path.read_bytes().replace(b"{", b'{"schema_id":"wrong",', 1))
    with pytest.raises(TransferSnapshotError, match="duplicate"):
        verify_transfer_snapshot(root)


def test_snapshot_exact_bytes_not_reserialized_digest(tmp_path: Path) -> None:
    root = copy_bundle(tmp_path)
    path = root / SNAPSHOT_PATH
    path.write_text(json.dumps(json.loads(path.read_bytes()), indent=2))
    with pytest.raises(TransferSnapshotError):
        verify_transfer_snapshot(root)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("recording_layout", "single_video"),
        ("recording_payload_kind", "external_ipc_video_only"),
        ("parent_recording_count", 4),
        ("required_consumer_profile", "legacy"),
        ("new_unrecognized_field", 1),
    ],
)
def test_marker_binding_is_closed_and_exact(
    tmp_path: Path, field: str, value: object
) -> None:
    root = copy_bundle(tmp_path)
    marker = json.loads((root / MARKER_NAME).read_bytes())
    marker[field] = value
    write_json(root / MARKER_NAME, marker)
    with pytest.raises(TransferSnapshotError):
        verify_transfer_snapshot(root)


@pytest.mark.parametrize(
    "mutation", ["clip_order", "camera", "frame_map", "unknown_field"]
)
def test_resigned_snapshot_cannot_override_source_semantics(
    tmp_path: Path, mutation: str
) -> None:
    root = copy_bundle(tmp_path)
    snapshot = json.loads((root / SNAPSHOT_PATH).read_bytes())
    parent = snapshot["parents"][0]
    if mutation == "clip_order":
        parent["clips"].reverse()
    elif mutation == "camera":
        parent["parent_key"]["camera_serial"] = "2010093"
    elif mutation == "frame_map":
        parent["clips"][0]["outputs"][0]["frame_map"]["row_correspondence_sha256"] = (
            "0" * 64
        )
    else:
        parent["unexpected"] = True
    resign_snapshot(root, snapshot)
    with pytest.raises(TransferSnapshotError):
        verify_transfer_snapshot(root)


def test_unmarked_partial_delivery_is_not_ingestible(tmp_path: Path) -> None:
    root = copy_bundle(tmp_path)
    (root / MARKER_NAME).unlink()
    with pytest.raises(TransferSnapshotError):
        verify_transfer_snapshot(root)


def test_planning_does_not_modify_retained_source(tmp_path: Path) -> None:
    root = copy_bundle(tmp_path)
    before = {
        p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }
    plan_parent_recordings(verify_transfer_snapshot(root))
    after = {
        p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }
    assert after == before


def test_inspection_cli_reports_plan_not_import(capsys: pytest.CaptureFixture) -> None:
    from fisheye.utils.inspect_recording_transfer import main

    assert main([str(FIXTURES / "rolling")]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "parent_plan_only"
    assert payload["parent_recording_count"] == 2
    assert payload["import_complete"] is False
    assert payload["registry_admitted"] is False
    assert payload["source_cleanup_authorized"] is False


def test_inspection_cli_does_not_downgrade_failed_proof(
    capsys: pytest.CaptureFixture,
) -> None:
    from fisheye.utils.inspect_recording_transfer import main

    assert main([str(FIXTURES / "failed_optional_proof")]) == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert "frame_identity_proof" in json.loads(captured.err)["error"]

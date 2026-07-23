from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from fisheye.utils.audit_coordinate_contracts import (
    AUDIT_RULESET_ID,
    AUDIT_RULESET_VERSION,
    NORMALIZED_ARTIFACT_FILENAMES,
    _ruleset_content_sha256,
)
from fisheye.utils.summarize_coordinate_audit import (
    _fingerprint,
    build_coordinate_audit_aggregate,
    main,
    verify_coordinate_audit_aggregate,
    write_coordinate_audit_aggregate,
)


def _write_run(
    zarr_path: Path,
    run_path: str,
    attributes: dict[str, object],
) -> None:
    path = zarr_path / run_path / "zarr.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": attributes,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _surface(
    *,
    zarr_path: Path,
    surface_path: str,
    family: str | None,
    run_path: str | None,
    recording_id: str | None,
    status: str,
    issue_codes: list[str],
) -> dict[str, object]:
    return {
        "audit_schema_id": "palette.coordinate_contract_inventory",
        "audit_schema_version": 12,
        "record_type": "coordinate_surface",
        "dataset_key": f"dataset-{recording_id}",
        "recording_id": recording_id,
        "zarr_path": str(zarr_path),
        "surface_path": surface_path,
        "surface_type": "test_geometry",
        "status": status,
        "issue_codes": issue_codes,
        "run_context": {"family": family, "run_path": run_path},
    }


def _write_inventory(path: Path, records: list[dict[str, object]]) -> None:
    dataset_records: dict[str, dict[str, object]] = {}
    normalized_records: list[dict[str, object]] = []
    for source_record in records:
        record = dict(source_record)
        dataset_key = str(record["dataset_key"])
        normalized_records.append(record)
        dataset_records.setdefault(
            dataset_key,
            {
                "audit_schema_id": "palette.coordinate_contract_inventory",
                "audit_schema_version": 12,
                "record_type": "coordinate_dataset",
                "dataset_key": dataset_key,
                "recording_id": record.get("recording_id"),
                "zarr_path": record.get("zarr_path"),
            },
        )
    ordered_records = [
        *(dataset_records[key] for key in sorted(dataset_records)),
        *normalized_records,
    ]
    for dataset_key in dataset_records:
        bundle_records = [
            dict(record)
            for record in ordered_records
            if record["dataset_key"] == dataset_key
        ]
        bundle_sha256 = _fingerprint(bundle_records)
        for record in ordered_records:
            if record["dataset_key"] == dataset_key:
                record["record_bundle_sha256"] = bundle_sha256
    path.write_text(
        "".join(
            json.dumps(record, sort_keys=True) + "\n"
            for record in ordered_records
        ),
        encoding="utf-8",
    )


def _write_source_artifact_manifest(
    artifact_dir: Path,
    inventory: Path,
    *,
    complete: bool = True,
) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    file_records: dict[str, dict[str, object]] = {}
    for name in NORMALIZED_ARTIFACT_FILENAMES:
        if name == "artifact_manifest.json":
            continue
        path = artifact_dir / name
        content = b"{}\n" if path.suffix == ".json" else b""
        path.write_bytes(content)
        file_records[f"artifact:{name}"] = {
            "path": name,
            "path_kind": "relative_to_manifest",
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
        }
    inventory = inventory.resolve(strict=True)
    inventory_content = inventory.read_bytes()
    file_records["external:inventory_jsonl"] = {
        "path": str(inventory),
        "path_kind": "absolute",
        "sha256": hashlib.sha256(inventory_content).hexdigest(),
        "size_bytes": len(inventory_content),
    }
    manifest_payload: dict[str, object] = {
        "schema_id": "palette.coordinate_contract_audit.artifact_generation",
        "schema_version": 12,
        "audit_ruleset_id": AUDIT_RULESET_ID,
        "audit_ruleset_version": AUDIT_RULESET_VERSION,
        "ruleset_content_sha256": _ruleset_content_sha256(),
        "complete": complete,
        "integrity_manifest_complete": True,
        "declared_output_files": sorted(
            [*NORMALIZED_ARTIFACT_FILENAMES, str(inventory)]
        ),
        "manifest_file": "artifact_manifest.json",
        "manifest_self_digest_policy": (
            "canonical_json_payload_excluding_generation_sha256_v1"
        ),
        "files": file_records,
        "inventory_records_sha256": "0" * 64,
        "registry_snapshot_sha256": file_records[
            "artifact:registry_snapshot.json"
        ]["sha256"],
    }
    manifest_payload["generation_sha256"] = _fingerprint(manifest_payload)
    manifest = artifact_dir / "artifact_manifest.json"
    manifest.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def test_aggregate_binds_exact_run_producer_and_deduplicates_run_reads(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "recording.zarr"
    run_path = "analysis/track_kinematics_runs/offline/run-a"
    _write_run(
        zarr_path,
        run_path,
        {
            "method": "track_kinematics_offline",
            "method_version": 2,
            "git_commit": "a" * 40,
            "run_provenance": {
                "git_sha": "a" * 40,
                "fisheye_version": "0.1.0",
            },
            "lineage_payload_json": json.dumps(
                {
                    "method": "track_kinematics_offline",
                    "method_version": 2,
                    "code": {"git_commit": "a" * 40},
                }
            ),
            "analysis_schema_id": "palette.track_motion",
            "analysis_schema_version": 2,
        },
    )
    inventory = tmp_path / "inventory.jsonl"
    _write_inventory(
        inventory,
        [
            _surface(
                zarr_path=zarr_path,
                surface_path=f"{run_path}/tracks/id_0/positions_px",
                family="track_kinematics_runs",
                run_path=run_path,
                recording_id="rec-a",
                status="ambiguous_fail_closed",
                issue_codes=["ARRAY_COORDINATE_DESCRIPTOR_MISSING"],
            ),
            _surface(
                zarr_path=zarr_path,
                surface_path=f"{run_path}/tracks/id_0/positions_mm",
                family="track_kinematics_runs",
                run_path=run_path,
                recording_id="rec-a",
                status="numerical_validation_required",
                issue_codes=["CALIBRATION_LINEAGE_MISSING"],
            ),
        ],
    )

    manifest = _write_source_artifact_manifest(tmp_path / "artifacts", inventory)
    payload = build_coordinate_audit_aggregate(
        inventory,
        artifact_manifest=manifest,
    )

    assert verify_coordinate_audit_aggregate(payload)
    assert payload["surface_count"] == 2
    assert payload["run_count"] == 1
    assert payload["recording_count"] == 1
    run = payload["run_records"][0]
    assert run["producer_status"] == "resolved"
    assert run["method"] == "track_kinematics_offline"
    assert run["method_version"] == 2
    assert run["git_commit"] == "a" * 40
    assert run["software_version"] == "0.1.0"
    assert run["run_schema_evidence"] == {
        "analysis_schema_id": "palette.track_motion",
        "analysis_schema_version": 2,
    }
    family = payload["by_run_family"]["track_kinematics_runs"]
    assert family["surface_count"] == 2
    assert family["run_count"] == 1
    assert family["affected_recording_ids"] == ["rec-a"]
    assert family["issue_counts"] == {
        "ARRAY_COORDINATE_DESCRIPTOR_MISSING": 1,
        "CALIBRATION_LINEAGE_MISSING": 1,
    }
    assert payload["by_recording"]["rec-a"]["surface_count"] == 2


def test_aggregate_preserves_conflicts_missing_metadata_and_no_run_context(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "recording.zarr"
    conflicting_run = "keypoints_runs/run-a"
    _write_run(
        zarr_path,
        conflicting_run,
        {
            "method": "yolo_pose",
            "git_commit": "a" * 40,
            "provenance": {"git": {"commit": "b" * 40}},
        },
    )
    missing_run = "crop_runs/missing"
    inventory = tmp_path / "inventory.jsonl"
    _write_inventory(
        inventory,
        [
            _surface(
                zarr_path=zarr_path,
                surface_path=f"{conflicting_run}/keypoints_img",
                family="keypoints_runs",
                run_path=conflicting_run,
                recording_id="rec-a",
                status="ambiguous_fail_closed",
                issue_codes=["COORDINATE_SPACE_MISSING"],
            ),
            _surface(
                zarr_path=zarr_path,
                surface_path=f"{missing_run}/roi_coordinates_full",
                family="crop_runs",
                run_path=missing_run,
                recording_id="rec-b",
                status="missing_or_unreadable",
                issue_codes=["ROW_IDENTITY_MISSING"],
            ),
            _surface(
                zarr_path=zarr_path,
                surface_path="analysis/calibration/homography",
                family=None,
                run_path=None,
                recording_id="rec-b",
                status="ambiguous_fail_closed",
                issue_codes=["HOMOGRAPHY_DIRECTION_MISSING"],
            ),
        ],
    )

    manifest = _write_source_artifact_manifest(tmp_path / "artifacts", inventory)
    payload = build_coordinate_audit_aggregate(
        inventory,
        artifact_manifest=manifest,
    )

    assert verify_coordinate_audit_aggregate(payload)
    assert payload["run_count"] == 3
    assert payload["recording_count"] == 2
    assert payload["recording_bucket_count"] == 2
    assert payload["portable_or_unbound_surface_count"] == 0
    statuses = {record["metadata_status"] for record in payload["run_records"]}
    assert statuses == {"readable", "missing", "run_context_missing"}
    conflict = next(
        record
        for record in payload["run_records"]
        if record["run_path"] == conflicting_run
    )
    assert conflict["producer_status"] == "conflicting_or_invalid"
    assert conflict["candidate_resolution"]["git_commit"] == "conflicting"
    assert "__no_run_family__" in payload["by_run_family"]
    issue = next(
        row
        for row in payload["issue_by_run_family"]
        if row["issue_code"] == "ROW_IDENTITY_MISSING"
    )
    assert issue["affected_recording_ids"] == ["rec-b"]


def test_aggregate_uses_and_conflict_checks_legacy_git_commit_hash(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "recording.zarr"
    legacy_run = "crop_runs/legacy"
    conflicting_run = "crop_runs/conflicting"
    _write_run(zarr_path, legacy_run, {"git_commit_hash": "a" * 40})
    _write_run(
        zarr_path,
        conflicting_run,
        {
            "git_commit": "a" * 40,
            "git_commit_hash": "b" * 40,
        },
    )
    inventory = tmp_path / "inventory.jsonl"
    _write_inventory(
        inventory,
        [
            _surface(
                zarr_path=zarr_path,
                surface_path=f"{legacy_run}/roi_coordinates_full",
                family="crop_runs",
                run_path=legacy_run,
                recording_id="rec-a",
                status="ambiguous_fail_closed",
                issue_codes=["COORDINATE_SPACE_MISSING"],
            ),
            _surface(
                zarr_path=zarr_path,
                surface_path=f"{conflicting_run}/roi_coordinates_full",
                family="crop_runs",
                run_path=conflicting_run,
                recording_id="rec-a",
                status="ambiguous_fail_closed",
                issue_codes=["COORDINATE_SPACE_MISSING"],
            ),
        ],
    )
    manifest = _write_source_artifact_manifest(tmp_path / "artifacts", inventory)

    payload = build_coordinate_audit_aggregate(
        inventory,
        artifact_manifest=manifest,
    )

    assert verify_coordinate_audit_aggregate(payload)
    legacy = next(
        record for record in payload["run_records"] if record["run_path"] == legacy_run
    )
    assert legacy["producer_status"] == "resolved"
    assert legacy["git_commit"] == "a" * 40
    assert legacy["git_commit_candidates"] == [
        {
            "attribute_path": "git_commit_hash",
            "valid_scalar": True,
            "value": "a" * 40,
        }
    ]
    conflict = next(
        record
        for record in payload["run_records"]
        if record["run_path"] == conflicting_run
    )
    assert conflict["producer_status"] == "conflicting_or_invalid"
    assert conflict["candidate_resolution"]["git_commit"] == "conflicting"


def test_cli_writes_a_verifiable_deterministic_payload(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording.zarr"
    run_path = "detect_runs/run-a"
    _write_run(zarr_path, run_path, {"git_commit": "c" * 40})
    inventory = tmp_path / "inventory.jsonl"
    _write_inventory(
        inventory,
        [
            _surface(
                zarr_path=zarr_path,
                surface_path=f"{run_path}/bbox_img_xyxy",
                family="detect_runs",
                run_path=run_path,
                recording_id="rec-a",
                status="ambiguous_fail_closed",
                issue_codes=["ARRAY_COORDINATE_DESCRIPTOR_MISSING"],
            )
        ],
    )
    output_a = tmp_path / "aggregate-a.json"
    output_b = tmp_path / "aggregate-b.json"
    manifest = _write_source_artifact_manifest(tmp_path / "artifacts", inventory)

    assert (
        main(
            [
                "--inventory-jsonl",
                str(inventory),
                "--artifact-manifest",
                str(manifest),
                "--output-json",
                str(output_a),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "--inventory-jsonl",
                str(inventory),
                "--artifact-manifest",
                str(manifest),
                "--output-json",
                str(output_b),
            ]
        )
        == 0
    )
    assert output_a.read_bytes() == output_b.read_bytes()
    payload = json.loads(output_a.read_text(encoding="utf-8"))
    assert verify_coordinate_audit_aggregate(payload)


def test_aggregate_excludes_portable_bucket_from_recording_counts(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "portable.zarr"
    run_path = "keypoints_runs/portable"
    _write_run(zarr_path, run_path, {"method": "merged_export"})
    inventory = tmp_path / "inventory.jsonl"
    _write_inventory(
        inventory,
        [
            _surface(
                zarr_path=zarr_path,
                surface_path=f"{run_path}/keypoints_img",
                family="keypoints_runs",
                run_path=run_path,
                recording_id=None,
                status="ambiguous_fail_closed",
                issue_codes=["COORDINATE_SPACE_MISSING"],
            )
        ],
    )
    manifest = _write_source_artifact_manifest(tmp_path / "artifacts", inventory)

    payload = build_coordinate_audit_aggregate(
        inventory,
        artifact_manifest=manifest,
    )

    assert verify_coordinate_audit_aggregate(payload)
    assert payload["recording_count"] == 0
    assert payload["recording_bucket_count"] == 1
    family = payload["by_run_family"]["keypoints_runs"]
    assert family["affected_recording_count"] == 0
    assert family["affected_recording_ids"] == []
    assert family["portable_or_unbound_surface_count"] == 1
    issue = payload["issue_by_run_family"][0]
    assert issue["affected_recording_count"] == 0
    assert issue["affected_recording_ids"] == []
    assert issue["portable_or_unbound_occurrence_count"] == 1


def test_aggregate_requires_complete_manifest_bound_to_exact_inventory(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "recording.zarr"
    run_path = "detect_runs/run-a"
    _write_run(zarr_path, run_path, {"method": "detect"})
    inventory = tmp_path / "inventory.jsonl"
    surface = _surface(
        zarr_path=zarr_path,
        surface_path=f"{run_path}/bbox_img_xyxy",
        family="detect_runs",
        run_path=run_path,
        recording_id="rec-a",
        status="ambiguous_fail_closed",
        issue_codes=["ARRAY_COORDINATE_DESCRIPTOR_MISSING"],
    )
    _write_inventory(inventory, [surface])
    with pytest.raises(ValueError, match="artifact_manifest is required"):
        build_coordinate_audit_aggregate(inventory)
    incomplete_manifest = _write_source_artifact_manifest(
        tmp_path / "incomplete-artifacts",
        inventory,
        complete=False,
    )
    with pytest.raises(ValueError, match="not marked complete"):
        build_coordinate_audit_aggregate(
            inventory,
            artifact_manifest=incomplete_manifest,
        )

    other_inventory = tmp_path / "other-inventory.jsonl"
    _write_inventory(other_inventory, [surface])
    manifest = _write_source_artifact_manifest(tmp_path / "artifacts", inventory)
    with pytest.raises(ValueError, match="different inventory_jsonl"):
        build_coordinate_audit_aggregate(
            other_inventory,
            artifact_manifest=manifest,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda record: record.__setitem__("issue_codes", "NOT_A_LIST"), "issue_codes"),
        (
            lambda record: record["run_context"].__setitem__(
                "run_path", "../escape"
            ),
            "run family/path context",
        ),
        (
            lambda record: record.__setitem__("audit_schema_version", 11),
            "schema_version",
        ),
    ],
)
def test_aggregate_rejects_malformed_surface_records(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    zarr_path = tmp_path / "recording.zarr"
    run_path = "detect_runs/run-a"
    _write_run(zarr_path, run_path, {"method": "detect"})
    record = _surface(
        zarr_path=zarr_path,
        surface_path=f"{run_path}/bbox_img_xyxy",
        family="detect_runs",
        run_path=run_path,
        recording_id="rec-a",
        status="ambiguous_fail_closed",
        issue_codes=["ARRAY_COORDINATE_DESCRIPTOR_MISSING"],
    )
    assert callable(mutation)
    mutation(record)
    inventory = tmp_path / "inventory.jsonl"
    _write_inventory(inventory, [record])
    manifest = _write_source_artifact_manifest(tmp_path / "artifacts", inventory)

    with pytest.raises(ValueError, match=message):
        build_coordinate_audit_aggregate(
            inventory,
            artifact_manifest=manifest,
        )


def test_aggregate_rejects_duplicate_coordinate_surfaces(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording.zarr"
    run_path = "detect_runs/run-a"
    _write_run(zarr_path, run_path, {"method": "detect"})
    record = _surface(
        zarr_path=zarr_path,
        surface_path=f"{run_path}/bbox_img_xyxy",
        family="detect_runs",
        run_path=run_path,
        recording_id="rec-a",
        status="ambiguous_fail_closed",
        issue_codes=["ARRAY_COORDINATE_DESCRIPTOR_MISSING"],
    )
    inventory = tmp_path / "inventory.jsonl"
    _write_inventory(inventory, [record, dict(record)])
    manifest = _write_source_artifact_manifest(tmp_path / "artifacts", inventory)

    with pytest.raises(ValueError, match="duplicates a coordinate surface"):
        build_coordinate_audit_aggregate(
            inventory,
            artifact_manifest=manifest,
        )


def test_aggregate_preserves_malformed_lineage_and_dual_metadata_as_invalid(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "recording.zarr"
    malformed_run = "keypoints_runs/malformed"
    _write_run(
        zarr_path,
        malformed_run,
        {
            "method": "yolo_pose",
            "git_commit": "a" * 40,
            "lineage_payload_json": "{not-json",
        },
    )
    dual_run = "detect_runs/dual"
    _write_run(zarr_path, dual_run, {"method": "detect"})
    (zarr_path / dual_run / ".zattrs").write_text("{}\n", encoding="utf-8")
    inventory = tmp_path / "inventory.jsonl"
    _write_inventory(
        inventory,
        [
            _surface(
                zarr_path=zarr_path,
                surface_path=f"{malformed_run}/keypoints_img",
                family="keypoints_runs",
                run_path=malformed_run,
                recording_id="rec-a",
                status="ambiguous_fail_closed",
                issue_codes=["COORDINATE_SPACE_MISSING"],
            ),
            _surface(
                zarr_path=zarr_path,
                surface_path=f"{dual_run}/bbox_img_xyxy",
                family="detect_runs",
                run_path=dual_run,
                recording_id="rec-a",
                status="ambiguous_fail_closed",
                issue_codes=["COORDINATE_SPACE_MISSING"],
            ),
        ],
    )
    manifest = _write_source_artifact_manifest(tmp_path / "artifacts", inventory)

    payload = build_coordinate_audit_aggregate(
        inventory,
        artifact_manifest=manifest,
    )

    malformed = next(
        record for record in payload["run_records"] if record["run_path"] == malformed_run
    )
    assert malformed["producer_status"] == "conflicting_or_invalid"
    assert malformed["candidate_resolution"]["method"] == "invalid"
    dual = next(
        record for record in payload["run_records"] if record["run_path"] == dual_run
    )
    assert dual["metadata_format"] == "conflicting_formats"
    assert dual["metadata_status"] == "invalid_location"
    assert dual["producer_key"] == "unavailable:run_metadata_conflicting_formats"


def test_aggregate_rejects_output_aliases_and_archive_locations(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "recording.zarr"
    run_path = "detect_runs/run-a"
    _write_run(zarr_path, run_path, {"method": "detect"})
    inventory = tmp_path / "inventory.jsonl"
    _write_inventory(
        inventory,
        [
            _surface(
                zarr_path=zarr_path,
                surface_path=f"{run_path}/bbox_img_xyxy",
                family="detect_runs",
                run_path=run_path,
                recording_id="rec-a",
                status="ambiguous_fail_closed",
                issue_codes=["COORDINATE_SPACE_MISSING"],
            )
        ],
    )
    manifest = _write_source_artifact_manifest(tmp_path / "artifacts", inventory)
    payload = build_coordinate_audit_aggregate(
        inventory,
        artifact_manifest=manifest,
    )

    with pytest.raises(ValueError, match="aliases a source audit artifact"):
        write_coordinate_audit_aggregate(manifest, payload)
    with pytest.raises(ValueError, match="outside source Zarr roots"):
        write_coordinate_audit_aggregate(zarr_path / "aggregate.json", payload)


def test_aggregate_protects_zero_surface_dataset_archive_locations(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "zero-surfaces.zarr"
    zarr_path.mkdir()
    dataset_record: dict[str, object] = {
        "audit_schema_id": "palette.coordinate_contract_inventory",
        "audit_schema_version": 12,
        "record_type": "coordinate_dataset",
        "dataset_key": "dataset-zero",
        "recording_id": "rec-zero",
        "zarr_path": str(zarr_path),
    }
    dataset_record["record_bundle_sha256"] = _fingerprint([dataset_record])
    inventory = tmp_path / "inventory.jsonl"
    inventory.write_text(
        json.dumps(dataset_record, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = _write_source_artifact_manifest(tmp_path / "artifacts", inventory)

    payload = build_coordinate_audit_aggregate(
        inventory,
        artifact_manifest=manifest,
    )

    assert verify_coordinate_audit_aggregate(payload)
    assert payload["surface_count"] == 0
    assert payload["source_zarr_roots"] == [str(zarr_path)]
    with pytest.raises(ValueError, match="outside source Zarr roots"):
        write_coordinate_audit_aggregate(zarr_path / "aggregate.json", payload)


def test_aggregate_does_not_follow_run_metadata_symlink_outside_archive(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "recording.zarr"
    run_path = "detect_runs/escape"
    _write_run(tmp_path / "outside-root", "outside-run", {"method": "outside"})
    outside_run = tmp_path / "outside-root" / "outside-run"
    link = zarr_path / run_path
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside_run, target_is_directory=True)
    inventory = tmp_path / "inventory.jsonl"
    _write_inventory(
        inventory,
        [
            _surface(
                zarr_path=zarr_path,
                surface_path=f"{run_path}/bbox_img_xyxy",
                family="detect_runs",
                run_path=run_path,
                recording_id="rec-a",
                status="ambiguous_fail_closed",
                issue_codes=["COORDINATE_SPACE_MISSING"],
            )
        ],
    )
    manifest = _write_source_artifact_manifest(tmp_path / "artifacts", inventory)

    payload = build_coordinate_audit_aggregate(
        inventory,
        artifact_manifest=manifest,
    )

    assert verify_coordinate_audit_aggregate(payload)
    run = payload["run_records"][0]
    assert run["metadata_format"] == "unsafe_path"
    assert run["metadata_status"] == "invalid_location"
    assert run["producer_key"] == "unavailable:run_metadata_unsafe_path"


def test_verifier_rejects_rehashed_but_inconsistent_counts(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording.zarr"
    run_path = "detect_runs/run-a"
    _write_run(zarr_path, run_path, {"method": "detect"})
    inventory = tmp_path / "inventory.jsonl"
    _write_inventory(
        inventory,
        [
            _surface(
                zarr_path=zarr_path,
                surface_path=f"{run_path}/bbox_img_xyxy",
                family="detect_runs",
                run_path=run_path,
                recording_id="rec-a",
                status="ambiguous_fail_closed",
                issue_codes=["COORDINATE_SPACE_MISSING"],
            )
        ],
    )
    manifest = _write_source_artifact_manifest(tmp_path / "artifacts", inventory)
    payload = build_coordinate_audit_aggregate(
        inventory,
        artifact_manifest=manifest,
    )
    payload["surface_count"] += 1
    payload.pop("aggregate_payload_sha256")
    payload["aggregate_payload_sha256"] = _fingerprint(payload)

    assert not verify_coordinate_audit_aggregate(payload)


def test_verifier_rejects_rehashed_membership_and_invalid_producer_types(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "recording.zarr"
    run_path = "detect_runs/run-a"
    _write_run(zarr_path, run_path, {"method": "detect"})
    inventory = tmp_path / "inventory.jsonl"
    _write_inventory(
        inventory,
        [
            _surface(
                zarr_path=zarr_path,
                surface_path=f"{run_path}/bbox_img_xyxy",
                family="detect_runs",
                run_path=run_path,
                recording_id="rec-a",
                status="ambiguous_fail_closed",
                issue_codes=["COORDINATE_SPACE_MISSING"],
            )
        ],
    )
    manifest = _write_source_artifact_manifest(tmp_path / "artifacts", inventory)
    payload = build_coordinate_audit_aggregate(
        inventory,
        artifact_manifest=manifest,
    )

    wrong_run_count = copy.deepcopy(payload)
    wrong_run_count["by_run_family"]["detect_runs"]["run_count"] = 0
    wrong_run_count.pop("aggregate_payload_sha256")
    wrong_run_count["aggregate_payload_sha256"] = _fingerprint(wrong_run_count)
    assert not verify_coordinate_audit_aggregate(wrong_run_count)

    wrong_recording = copy.deepcopy(payload)
    wrong_recording["by_run_family"]["detect_runs"][
        "affected_recording_ids"
    ] = ["fabricated-recording"]
    wrong_recording.pop("aggregate_payload_sha256")
    wrong_recording["aggregate_payload_sha256"] = _fingerprint(wrong_recording)
    assert not verify_coordinate_audit_aggregate(wrong_recording)

    wrong_issue_recording = copy.deepcopy(payload)
    wrong_issue_recording["issue_by_run_family"][0][
        "affected_recording_ids"
    ] = ["fabricated-recording"]
    wrong_issue_recording.pop("aggregate_payload_sha256")
    wrong_issue_recording["aggregate_payload_sha256"] = _fingerprint(
        wrong_issue_recording
    )
    assert not verify_coordinate_audit_aggregate(wrong_issue_recording)

    invalid_producer = copy.deepcopy(payload)
    invalid_producer["run_records"][0]["producer_key"] = []
    invalid_producer.pop("aggregate_payload_sha256")
    invalid_producer["aggregate_payload_sha256"] = _fingerprint(invalid_producer)
    assert not verify_coordinate_audit_aggregate(invalid_producer)


def test_verifier_rejects_source_inventory_changed_after_generation(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "recording.zarr"
    run_path = "detect_runs/run-a"
    _write_run(zarr_path, run_path, {"method": "detect"})
    inventory = tmp_path / "inventory.jsonl"
    _write_inventory(
        inventory,
        [
            _surface(
                zarr_path=zarr_path,
                surface_path=f"{run_path}/bbox_img_xyxy",
                family="detect_runs",
                run_path=run_path,
                recording_id="rec-a",
                status="ambiguous_fail_closed",
                issue_codes=["COORDINATE_SPACE_MISSING"],
            )
        ],
    )
    manifest = _write_source_artifact_manifest(tmp_path / "artifacts", inventory)
    payload = build_coordinate_audit_aggregate(
        inventory,
        artifact_manifest=manifest,
    )
    inventory.write_text(
        inventory.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    assert not verify_coordinate_audit_aggregate(payload)

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from fisheye.shared.zarr_payload_receipt import (
    build_payload_integrity_receipt,
    build_payload_validation_receipt,
    canonical_json_sha256,
    canonical_decoded_content_inventory,
    canonical_payload_integrity_receipt,
    decoded_content_inventory_from_copy_report,
    decoded_payload_receipt_from_copy_report,
    verify_payload_integrity_receipt,
    verify_payload_validation_receipt,
)

RUN_REF = "/analysis/track_kinematics_runs/offline/canary"


def _copy_report() -> dict:
    return {
        "schema_id": "palette.zarr_sharded_run_copy.v1",
        "status": "complete",
        "exact_decoded_validation": True,
        "exact_full_decoded_content_hashes": True,
        "arrays": [
            {
                "path": "track_ids",
                "dtype": "int32",
                "shape": [2],
                "decoded_content_bytes": 8,
                "decoded_content_sha256": "a" * 64,
            },
            {
                "path": "tracks/id_0/speed_raw_px",
                "dtype": "float32",
                "shape": [5],
                "decoded_content_bytes": 20,
                "decoded_content_sha256": "b" * 64,
            },
            {
                "path": "tracks/id_0/source_instance_key",
                "dtype": "[('valid', '?'), ('instance_key', '<u8')]",
                "shape": [5],
                "decoded_content_bytes": 45,
                "decoded_content_sha256": "c" * 64,
            },
        ],
        "shards": [
            {
                "path": "track_ids",
                "start_row": 0,
                "stop_row": 2,
                "decoded_bytes": 8,
                "decoded_sha256": "1" * 64,
            },
            {
                "path": "tracks/id_0/speed_raw_px",
                "start_row": 0,
                "stop_row": 3,
                "decoded_bytes": 12,
                "decoded_sha256": "2" * 64,
            },
            {
                "path": "tracks/id_0/speed_raw_px",
                "start_row": 3,
                "stop_row": 5,
                "decoded_bytes": 8,
                "decoded_sha256": "3" * 64,
            },
        ],
        "static_arrays": [
            {
                "path": "tracks/id_0/source_instance_key",
                "decoded_bytes": 45,
                "decoded_sha256": "4" * 64,
            }
        ],
    }


def _run_tree(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    (run / "tracks" / "id_0" / "speed_raw_px" / "c").mkdir(parents=True)
    (run / "tracks" / "id_0" / "speed_raw_px" / "c" / "0").write_bytes(
        b"compressed-speed-shard"
    )
    (run / "tracks" / "id_0" / "speed_raw_px" / "zarr.json").write_text(
        '{"node_type": "array", "attributes": {"status": "running"}}',
        encoding="utf-8",
    )
    (run / "zarr.json").write_text(
        '{"node_type": "group", "attributes": '
        '{"stage_selector_eligible": false}}',
        encoding="utf-8",
    )
    return run


def test_receipt_survives_metadata_changes_and_rejects_payload_changes(
    tmp_path: Path,
) -> None:
    run = _run_tree(tmp_path)
    receipt = build_payload_integrity_receipt(
        run,
        run_ref=RUN_REF,
        decoded_copy_report=_copy_report(),
        hash_workers=2,
    )

    canonical = verify_payload_integrity_receipt(
        run,
        receipt,
        expected_run_ref=RUN_REF,
        hash_workers=2,
    )
    assert canonical["decoded_payload"]["array_count"] == 3
    assert canonical["physical_payload"]["file_count"] == 1
    assert canonical["immutable_metadata"]["file_count"] == 2

    # Coordinate/completion binding is metadata-only and must not invalidate
    # the immutable payload root.
    (run / "zarr.json").write_text(
        '{"node_type": "group", "attributes": '
        '{"stage_selector_eligible": true}}',
        encoding="utf-8",
    )
    verify_payload_integrity_receipt(
        run,
        receipt,
        expected_run_ref=RUN_REF,
        hash_workers=2,
    )
    # Array interpretation metadata is immutable even though attributes are
    # allowed to advance during publication.
    (run / "tracks" / "id_0" / "speed_raw_px" / "zarr.json").write_text(
        '{"attributes": {}, "node_type": "array", "shape": [999]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="immutable Zarr metadata differs"):
        verify_payload_integrity_receipt(
            run,
            receipt,
            expected_run_ref=RUN_REF,
            hash_workers=2,
        )

    (run / "tracks" / "id_0" / "speed_raw_px" / "zarr.json").write_text(
        '{"node_type": "array", "attributes": {"status": "running"}}',
        encoding="utf-8",
    )

    payload = run / "tracks" / "id_0" / "speed_raw_px" / "c" / "0"
    payload.write_bytes(b"mutated-compressed-speed-shard")
    with pytest.raises(ValueError, match="physical payload differs"):
        verify_payload_integrity_receipt(
            run,
            receipt,
            expected_run_ref=RUN_REF,
            hash_workers=2,
        )


def test_decoded_receipt_can_bind_local_science_before_physical_publication(
    tmp_path: Path,
) -> None:
    decoded = decoded_payload_receipt_from_copy_report(_copy_report())
    integrity = build_payload_integrity_receipt(
        _run_tree(tmp_path),
        run_ref=RUN_REF,
        decoded_copy_report=_copy_report(),
    )

    assert decoded == integrity["decoded_payload"]


def test_whole_array_content_inventory_is_closed_and_bound_to_decoded_root() -> None:
    report = _copy_report()
    decoded = decoded_payload_receipt_from_copy_report(report)
    inventory = decoded_content_inventory_from_copy_report(report)

    assert inventory["decoded_payload_root_sha256"] == decoded["root_sha256"]
    assert inventory["array_count"] == 3
    assert inventory["decoded_bytes"] == 73
    assert canonical_decoded_content_inventory(inventory) == inventory

    tampered = deepcopy(inventory)
    tampered["arrays"][0]["content_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="digest is stale"):
        canonical_decoded_content_inventory(tampered)


def test_receipt_rejects_decoded_gaps_and_tampered_records(tmp_path: Path) -> None:
    run = _run_tree(tmp_path)
    report = _copy_report()
    report["shards"][2]["start_row"] = 4
    with pytest.raises(ValueError, match="gap or overlap"):
        build_payload_integrity_receipt(
            run,
            run_ref=RUN_REF,
            decoded_copy_report=report,
        )

    receipt = build_payload_integrity_receipt(
        run,
        run_ref=RUN_REF,
        decoded_copy_report=_copy_report(),
    )
    tampered = deepcopy(receipt)
    tampered["decoded_payload"]["arrays"][0]["dtype"] = "float64"
    with pytest.raises(ValueError, match="unsupported or stale"):
        canonical_payload_integrity_receipt(tampered)

    # Recomputing only the outer record digest cannot conceal a stale child
    # root or broken closed inventory.
    tampered["record_sha256"] = canonical_json_sha256(
        {key: value for key, value in tampered.items() if key != "record_sha256"}
    )
    with pytest.raises(ValueError, match="array root|payload root"):
        canonical_payload_integrity_receipt(tampered)


def test_validation_receipt_binds_integrity_manifest_validator_and_policy(
    tmp_path: Path,
) -> None:
    run = _run_tree(tmp_path)
    integrity = build_payload_integrity_receipt(
        run,
        run_ref=RUN_REF,
        decoded_copy_report=_copy_report(),
    )
    validation = build_payload_validation_receipt(
        integrity,
        scientific_manifest_schema_id="palette.track_motion_publication_manifest",
        scientific_manifest_schema_version=2,
        scientific_manifest_sha256="a" * 64,
        validator_schema_id="palette.track_motion_full_validator",
        validator_schema_version=1,
        numerical_policy={
            "schema_id": "palette.track_motion_numerical_policy",
            "schema_version": 1,
            "floating_comparison": "persisted_dtype_exact_plus_versioned_tolerance",
        },
    )

    verified = verify_payload_validation_receipt(
        validation,
        integrity_receipt=integrity,
        expected_scientific_manifest_sha256="a" * 64,
        expected_validator_schema_id="palette.track_motion_full_validator",
        expected_validator_schema_version=1,
    )
    assert verified["result"] == "valid"

    altered_integrity = deepcopy(integrity)
    altered_integrity["physical_payload"]["files"][0]["size_bytes"] += 1
    with pytest.raises(ValueError):
        verify_payload_validation_receipt(
            validation,
            integrity_receipt=altered_integrity,
            expected_scientific_manifest_sha256="a" * 64,
            expected_validator_schema_id="palette.track_motion_full_validator",
            expected_validator_schema_version=1,
        )

    with pytest.raises(ValueError, match="validator or manifest binding"):
        verify_payload_validation_receipt(
            validation,
            integrity_receipt=integrity,
            expected_scientific_manifest_sha256="b" * 64,
            expected_validator_schema_id="palette.track_motion_full_validator",
            expected_validator_schema_version=1,
        )

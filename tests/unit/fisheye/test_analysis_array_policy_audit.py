from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from fisheye.analysis_workflows.analysis_array_policy_audit import (
    ANALYSIS_ARRAY_POLICY_AUDIT_SCHEMA_ID,
    build_analysis_array_policy_audit,
    require_analysis_array_policy_audit,
)
from fisheye.shared.zarr.analysis_array_contracts import (
    AnalysisArrayDeclaration,
    AnalysisAuthorityRole,
)
from fisheye.shared.zarr.analysis_benchmark_suite import (
    AnalysisBenchmarkScale,
    build_analysis_benchmark_suite,
)
from fisheye.shared.zarr.analysis_storage_planning import (
    AnalysisArrayStorageFacts,
    plan_analysis_storage,
)
from fisheye.shared.zarr.array_contracts import FLOAT32, ArrayContract
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.storage_intent import AccessPattern, WriteMode
from fisheye.shared.zarr.storage_profiles import PUBLISHED_HTTP_V1


def _suite(
    *,
    byte_planner_adopted: bool = True,
    coordinate_space: str | None = "arena_physical_mm",
):
    declaration = AnalysisArrayDeclaration(
        path="tracks/position_mm",
        contract=ArrayContract(
            schema_id="palette.test.array_policy.position_mm",
            schema_version=1,
            dtype=FLOAT32,
            shape_template=("n_rows", 2),
            axis_names=("row", "xy"),
            description="Synthetic physical position used by the policy audit.",
            units="millimetres",
            coordinate_space=coordinate_space,
        ),
        required=True,
        access_pattern=AccessPattern.WINDOWED,
        write_mode=WriteMode.IMMUTABLE,
        authority_role=AnalysisAuthorityRole.SCIENTIFIC_AUTHORITY,
        fill_semantics="nan_means_invalid_position",
        null_semantics="validity_is_declared_by_a_peer_array",
        physical_policy_owner="test_array_policy",
        byte_planner_adopted=byte_planner_adopted,
    )
    receipt = plan_analysis_storage(
        (declaration,),
        {
            declaration.path: AnalysisArrayStorageFacts(
                path=declaration.path,
                shape=(1_000_000, 2),
                dtype=np.dtype("float32"),
                access_unit_semantics="one complete xy position row",
            )
        },
        profile=PUBLISHED_HTTP_V1,
        dimensions={"n_rows": 1_000_000},
    )
    return build_analysis_benchmark_suite(
        family_id="track_kinematics",
        scale=AnalysisBenchmarkScale(
            scale_id="million_rows",
            dimensions=receipt.dimensions,
            description="Synthetic million-row policy audit.",
        ),
        storage_receipt=receipt,
        repetitions=5,
    )


def test_policy_audit_binds_logical_plan_codec_and_complete_scan() -> None:
    suite = _suite()
    audit = build_analysis_array_policy_audit("track_kinematics", suite)

    assert audit["schema_id"] == ANALYSIS_ARRAY_POLICY_AUDIT_SCHEMA_ID
    payload = audit["payload"]
    assert payload["codec_profile"]["zarr_format"] == 3
    assert [item["name"] for item in payload["codec_profile"]["codec_chain"]] == [
        "bytes",
        "zstd",
    ]
    assert payload["codec_profile"]["codec_chain"] == [
        {"name": "bytes", "configuration": {"endian": "little"}},
        {
            "name": "zstd",
            "configuration": {"level": 0, "checksum": False},
        },
    ]
    assert payload["codec_profile"]["sharding_index"]["codec_chain"][-1][
        "name"
    ] == "crc32c"
    assert payload["summary"]["array_count"] == 1
    assert payload["summary"]["complete_scan_case_count"] == 1
    assert payload["summary"]["sharded_array_count"] == 1
    array = payload["arrays"][0]
    assert array["effective_chunk_shape"] == [131_072, 2]
    assert array["effective_shard_shape"] is not None
    assert array["independently_readable_inner_chunks"] is True
    assert array["byte_planner_adopted"] is True


def test_policy_audit_rejects_non_adopted_array() -> None:
    with pytest.raises(ValueError, match="shared byte planner is not adopted"):
        build_analysis_array_policy_audit(
            "track_kinematics",
            _suite(byte_planner_adopted=False),
        )


def test_policy_audit_rejects_noncanonical_semantic_label() -> None:
    with pytest.raises(ValueError, match="coordinate_space"):
        build_analysis_array_policy_audit(
            "track_kinematics",
            _suite(coordinate_space=" arena_physical_mm"),
        )


def test_policy_audit_rejects_recomputed_digest_tampering() -> None:
    suite = _suite()
    audit = build_analysis_array_policy_audit("track_kinematics", suite)
    tampered = deepcopy(audit)
    tampered["payload"]["arrays"][0]["payload_object_estimate"] += 1
    tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])

    with pytest.raises(ValueError, match="differs from executable evidence"):
        require_analysis_array_policy_audit(
            tampered,
            stage_id="track_kinematics",
            benchmark_suite=suite,
        )


def test_policy_audit_requires_one_complete_scan_case_per_array() -> None:
    suite = _suite()
    broken = deepcopy(suite)
    broken["payload"]["array_cases"] = [
        record
        for record in broken["payload"]["array_cases"]
        if record["case"]["workload"]["workload_id"]
        != "palette.storage_workload.full_scan_read.v1"
    ]
    broken["payload_digest"] = canonical_json_sha256(broken["payload"])

    with pytest.raises(ValueError, match="does not have the exact workloads"):
        build_analysis_array_policy_audit("track_kinematics", broken)

from __future__ import annotations

import json

import numpy as np
import pytest

from fisheye.shared.zarr.analysis_array_contracts import (
    AnalysisArrayDeclaration,
    AnalysisAuthorityRole,
)
from fisheye.shared.zarr.analysis_storage_planning import (
    ANALYSIS_STORAGE_PLAN_SCHEMA_ID,
    AnalysisArrayStorageFacts,
    analysis_array_declaration_from_manifest,
    analysis_storage_plan_receipt_from_manifest,
    plan_analysis_array_storage,
    plan_analysis_storage,
)
from fisheye.shared.zarr.array_contracts import (
    BOOL,
    FLOAT32,
    INT64,
    UINT8,
    ArrayContract,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.storage_intent import AccessPattern, WriteMode
from fisheye.shared.zarr.storage_profiles import PUBLISHED_HTTP_V1


MIB = 1024 * 1024


def _declaration(
    path: str,
    *,
    dtype=FLOAT32,
    shape: tuple[str | int, ...] = ("n_rows",),
    axes: tuple[str, ...] = ("row",),
    access: AccessPattern = AccessPattern.WINDOWED,
    required: bool = True,
) -> AnalysisArrayDeclaration:
    return AnalysisArrayDeclaration(
        path=path,
        contract=ArrayContract(
            schema_id="palette.test.analysis." + path.replace("/", "."),
            schema_version=1,
            dtype=dtype,
            shape_template=shape,
            axis_names=axes,
            description=f"Exact test contract for {path}.",
        ),
        required=required,
        access_pattern=access,
        write_mode=WriteMode.IMMUTABLE,
        authority_role=AnalysisAuthorityRole.SCIENTIFIC_AUTHORITY,
        fill_semantics="every stored value follows the exact test contract",
        null_semantics="none",
        physical_policy_owner="test_unadopted_writer",
        byte_planner_adopted=False,
    )


def _facts(
    path: str,
    shape: tuple[int, ...],
    dtype: object,
    *,
    semantics: str = "one complete logical row",
) -> AnalysisArrayStorageFacts:
    return AnalysisArrayStorageFacts(
        path=path,
        shape=shape,
        dtype=dtype,
        access_unit_semantics=semantics,
    )


def test_narrow_bool_and_int64_columns_derive_rows_from_actual_bytes() -> None:
    bool_receipt = plan_analysis_array_storage(
        _declaration("valid", dtype=BOOL),
        _facts("valid", (10_000_000,), np.bool_),
        profile=PUBLISHED_HTTP_V1,
    )
    int_receipt = plan_analysis_array_storage(
        _declaration("instance_key", dtype=INT64),
        _facts("instance_key", (10_000_000,), np.int64),
        profile=PUBLISHED_HTTP_V1,
    )

    assert bool_receipt.plan.access_unit_nbytes == 1
    assert bool_receipt.plan.chunk_shape == (1_048_576,)
    assert bool_receipt.plan.chunk_nbytes == MIB
    assert int_receipt.plan.access_unit_nbytes == 8
    assert int_receipt.plan.chunk_shape == (131_072,)
    assert int_receipt.plan.chunk_nbytes == MIB
    assert bool_receipt.lifecycle_classification == "immutable_snapshot_array"
    assert bool_receipt.plan.is_sharded
    assert int_receipt.plan.is_sharded


def test_float32_keypoint_rows_keep_the_complete_trailing_record() -> None:
    receipt = plan_analysis_array_storage(
        _declaration(
            "keypoints_img",
            shape=("n_rows", 5, 2),
            axes=("row", "keypoint", "xy"),
        ),
        _facts(
            "keypoints_img",
            (1_000_000, 5, 2),
            np.float32,
            semantics="all five source-camera xy keypoints for one observation",
        ),
        profile=PUBLISHED_HTTP_V1,
    )

    assert receipt.facts.access_unit_shape == (1, 5, 2)
    assert receipt.plan.access_unit_shape == (1, 5, 2)
    assert receipt.plan.access_unit_nbytes == 40
    assert receipt.plan.chunk_shape == (32_768, 5, 2)
    assert receipt.plan.chunk_nbytes == 1_310_720
    assert receipt.plan.shard_axes == (0,)
    assert receipt.plan.shard_shape is not None
    assert receipt.plan.shard_shape[1:] == (5, 2)


def test_small_eager_semantic_table_remains_one_regular_object() -> None:
    receipt = plan_analysis_array_storage(
        _declaration(
            "channel_name_bytes",
            dtype=UINT8,
            shape=("n_channels", 64),
            axes=("channel", "utf8_byte"),
            access=AccessPattern.EAGER,
        ),
        _facts(
            "channel_name_bytes",
            (12, 64),
            np.uint8,
            semantics="one complete fixed-width channel-name record",
        ),
        profile=PUBLISHED_HTTP_V1,
    )

    assert receipt.plan.chunk_shape == (12, 64)
    assert receipt.plan.chunk_nbytes == 768
    assert receipt.plan.shard_shape is None
    assert receipt.plan.estimated_payload_objects == 1


def test_indexed_array_below_eager_cap_still_uses_access_budgeted_inner_chunks() -> None:
    receipt = plan_analysis_array_storage(
        _declaration(
            "probability",
            access=AccessPattern.INDEXED,
        ),
        _facts(
            "probability",
            (1_000_000,),
            np.float32,
            semantics="one independently indexed probability row",
        ),
        profile=PUBLISHED_HTTP_V1,
    )

    # The logical payload is only 4 MiB, below the profile's 8 MiB eager cap.
    # That cap is deliberately irrelevant to INDEXED access: decode granularity
    # remains governed by the access-specific uncompressed byte budget.
    assert receipt.plan.logical_nbytes == 4_000_000
    assert receipt.plan.chunk_shape == (262_144,)
    assert receipt.plan.chunk_nbytes == MIB
    assert receipt.plan.estimated_chunk_count == 4
    assert receipt.plan.shard_shape == (1_048_576,)
    assert receipt.plan.estimated_payload_objects == 1


def test_indexed_flat_rows_use_complete_xy_records_and_json_receipt() -> None:
    declaration = _declaration(
        "contour_points_xy",
        shape=("n_points", 2),
        axes=("point", "xy"),
        access=AccessPattern.INDEXED,
    )
    facts = _facts(
        "contour_points_xy",
        (110_685_000, 2),
        np.float32,
        semantics="one complete indexed xy point",
    )
    receipt = plan_analysis_storage(
        [declaration.as_manifest()],
        {facts.path: facts.as_manifest()},
        profile=PUBLISHED_HTTP_V1,
    )
    manifest = receipt.as_manifest()

    assert receipt.entries[0].plan.access_pattern == "indexed"
    assert receipt.entries[0].plan.access_unit_nbytes == 8
    assert receipt.entries[0].plan.chunk_shape == (131_072, 2)
    assert receipt.entries[0].plan.chunk_nbytes == MIB
    assert manifest["schema_id"] == ANALYSIS_STORAGE_PLAN_SCHEMA_ID
    assert manifest["payload_digest"] == canonical_json_sha256(manifest["payload"])
    assert json.loads(json.dumps(manifest)) == manifest
    assert manifest["payload"]["object_estimate"]["payload_objects"] == 27


def test_zero_row_array_has_a_complete_record_contract_and_no_payloads() -> None:
    receipt = plan_analysis_storage(
        [
            _declaration(
                "vectors",
                shape=("n_rows", 2),
                axes=("row", "xy"),
            )
        ],
        {"vectors": _facts("vectors", (0, 2), np.float32)},
        profile=PUBLISHED_HTTP_V1,
    )
    entry = receipt.entries[0]
    estimate = receipt.as_manifest()["payload"]["object_estimate"]

    assert entry.facts.access_unit_shape == (1, 2)
    assert entry.plan.logical_nbytes == 0
    assert entry.plan.chunk_grid_shape == (0, 1)
    assert entry.plan.estimated_payload_objects == 0
    assert estimate["empty_arrays"] == 1
    assert estimate["array_metadata_objects"] == 1
    assert estimate["array_objects_excluding_group_metadata"] == 1


def test_schema_dtype_shape_and_path_mismatches_fail_before_planning() -> None:
    declaration = _declaration(
        "keypoints",
        shape=("n_rows", 5, 2),
        axes=("row", "keypoint", "xy"),
    )
    with pytest.raises(ValueError, match="dtype mismatch"):
        plan_analysis_array_storage(
            declaration,
            _facts("keypoints", (10, 5, 2), np.float64),
            profile=PUBLISHED_HTTP_V1,
        )
    with pytest.raises(ValueError, match="shape contract failed"):
        plan_analysis_array_storage(
            declaration,
            _facts("keypoints", (10, 4, 2), np.float32),
            profile=PUBLISHED_HTTP_V1,
        )
    with pytest.raises(ValueError, match="does not match facts path"):
        plan_analysis_array_storage(
            declaration,
            _facts("other", (10, 5, 2), np.float32),
            profile=PUBLISHED_HTTP_V1,
        )
    with pytest.raises(ValueError, match="expected symbolic dimension n_rows=11"):
        plan_analysis_array_storage(
            declaration,
            _facts("keypoints", (10, 5, 2), np.float32),
            profile=PUBLISHED_HTTP_V1,
            dimensions={"n_rows": 11},
        )


def test_shared_symbolic_dimension_mismatch_fails_closed() -> None:
    with pytest.raises(ValueError, match="expected symbolic dimension n_rows=10"):
        plan_analysis_storage(
            [_declaration("scores"), _declaration("valid", dtype=BOOL)],
            {
                "scores": _facts("scores", (10,), np.float32),
                "valid": _facts("valid", (11,), np.bool_),
            },
            profile=PUBLISHED_HTTP_V1,
        )


def test_manifest_parsers_reject_rehashed_or_noncanonical_schema_facts() -> None:
    declaration = _declaration("scores")
    parsed = analysis_array_declaration_from_manifest(declaration.as_manifest())
    assert parsed == declaration

    tampered_declaration = declaration.as_manifest()
    tampered_declaration["logical_contract"]["dtype"]["itemsize_bytes"] = 8
    with pytest.raises(ValueError, match="dtype contract is not canonical"):
        analysis_array_declaration_from_manifest(tampered_declaration)

    facts = _facts("scores", (100,), np.float32)
    tampered_facts = facts.as_manifest()
    tampered_facts["access_unit_shape"] = [2]
    with pytest.raises(ValueError, match="not canonical"):
        AnalysisArrayStorageFacts.from_manifest(tampered_facts)


def test_missing_required_unexpected_and_optional_absence_are_exact() -> None:
    required = _declaration("scores")
    optional = _declaration("quality", required=False)
    receipt = plan_analysis_storage(
        [required, optional],
        {"scores": _facts("scores", (4,), np.float32)},
        profile=PUBLISHED_HTTP_V1,
    )
    assert tuple(entry.declaration.path for entry in receipt.entries) == ("scores",)

    with pytest.raises(ValueError, match="Missing required"):
        plan_analysis_storage([required], {}, profile=PUBLISHED_HTTP_V1)
    with pytest.raises(ValueError, match="Unexpected analysis array facts"):
        plan_analysis_storage(
            [required],
            {
                "scores": _facts("scores", (4,), np.float32),
                "other": _facts("other", (4,), np.float32),
            },
            profile=PUBLISHED_HTTP_V1,
        )


def test_receipt_is_deterministic_across_input_mapping_order() -> None:
    declarations = [_declaration("scores"), _declaration("valid", dtype=BOOL)]
    facts = {
        "scores": _facts("scores", (10_000,), np.float32),
        "valid": _facts("valid", (10_000,), np.bool_),
    }
    forward = plan_analysis_storage(
        declarations,
        facts,
        profile=PUBLISHED_HTTP_V1,
    )
    reverse = plan_analysis_storage(
        reversed(declarations),
        dict(reversed(tuple(facts.items()))),
        profile=PUBLISHED_HTTP_V1,
    )

    assert forward == reverse
    assert forward.as_manifest() == reverse.as_manifest()


def test_receipt_parser_replans_and_rejects_rehashed_physical_tampering() -> None:
    receipt = plan_analysis_storage(
        [_declaration("scores"), _declaration("valid", dtype=BOOL)],
        {
            "scores": _facts("scores", (200_000,), np.float32),
            "valid": _facts("valid", (200_000,), np.bool_),
        },
        profile=PUBLISHED_HTTP_V1,
    )
    manifest = receipt.as_manifest()

    assert analysis_storage_plan_receipt_from_manifest(manifest) == receipt

    tampered = json.loads(json.dumps(manifest))
    tampered["payload"]["arrays"][0]["plan"]["chunk_shape"][0] = 1
    tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])
    with pytest.raises(ValueError, match="differs from executable byte planning"):
        analysis_storage_plan_receipt_from_manifest(tampered)

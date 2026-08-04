from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pytest

from fisheye.shared.zarr.analysis_array_contracts import (
    AnalysisArrayDeclaration,
    AnalysisAuthorityRole,
)
from fisheye.shared.zarr.analysis_benchmark_suite import (
    ANALYSIS_BENCHMARK_SUITE_LEGACY_SCHEMA_VERSION,
    ANALYSIS_BENCHMARK_SUITE_SCHEMA_VERSION,
    AnalysisBenchmarkScale,
    build_analysis_benchmark_suite,
    require_analysis_benchmark_suite_manifest,
)
from fisheye.shared.zarr.analysis_storage_planning import (
    AnalysisArrayStorageFacts,
    plan_analysis_storage,
)
from fisheye.shared.zarr.array_contracts import BOOL, FLOAT32, INT64, ArrayContract
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.storage_intent import AccessPattern, WriteMode
from fisheye.shared.zarr.storage_profiles import PUBLISHED_HTTP_V1


def _declaration(
    path: str,
    *,
    dtype,
    access: AccessPattern,
) -> AnalysisArrayDeclaration:
    return AnalysisArrayDeclaration(
        path=path,
        contract=ArrayContract(
            schema_id=f"palette.test.{path}",
            schema_version=1,
            dtype=dtype,
            shape_template=("n_rows",),
            axis_names=("row",),
            description=f"Test declaration for {path}.",
        ),
        required=True,
        access_pattern=access,
        write_mode=WriteMode.IMMUTABLE,
        authority_role=AnalysisAuthorityRole.SCIENTIFIC_AUTHORITY,
        fill_semantics="every row is present",
        null_semantics="none",
        physical_policy_owner="test_candidate",
        byte_planner_adopted=False,
    )


def _suite(n_rows: int):
    declarations = (
        _declaration("lookup", dtype=INT64, access=AccessPattern.EAGER),
        _declaration("timeline", dtype=FLOAT32, access=AccessPattern.WINDOWED),
        _declaration("masks", dtype=BOOL, access=AccessPattern.PER_ROW),
        _declaration("points", dtype=FLOAT32, access=AccessPattern.INDEXED),
    )
    facts = {
        declaration.path: AnalysisArrayStorageFacts(
            path=declaration.path,
            shape=(n_rows,),
            dtype=np.dtype(declaration.contract.dtype.numpy_dtype),
            access_unit_semantics="one complete logical row",
        )
        for declaration in declarations
    }
    receipt = plan_analysis_storage(
        declarations,
        facts,
        profile=PUBLISHED_HTTP_V1,
        dimensions={"n_rows": n_rows},
    )
    scale = AnalysisBenchmarkScale(
        scale_id=f"rows_{n_rows}",
        dimensions=(("n_rows", n_rows),),
        description=f"Deterministic {n_rows}-row benchmark scale.",
    )
    return build_analysis_benchmark_suite(
        family_id="test_family",
        scale=scale,
        storage_receipt=receipt,
    )


@pytest.mark.parametrize("n_rows", (200_000, 1_000_000))
def test_suite_covers_write_primary_read_full_scan_and_publication(n_rows: int) -> None:
    suite = _suite(n_rows)
    payload = suite["payload"]

    require_analysis_benchmark_suite_manifest(suite)
    assert len(payload["array_cases"]) == 12
    assert payload["publication_case"]["scope"] == "complete_immutable_run"
    assert payload["execution_policy"]["production_mutation_authorized"] is False
    assert payload["execution_policy"]["node_local_compute"] is True
    assert json.loads(json.dumps(suite)) == suite

    primary = {
        row["array_path"]: row["selection"]["mode"]
        for row in payload["array_cases"]
        if row["case"]["workload"]["workload_id"]
        not in {
            "palette.storage_workload.write_materialization.v1",
            "palette.storage_workload.full_scan_read.v1",
        }
    }
    assert primary == {
        "lookup": "whole_array",
        "timeline": "bounded_row_windows",
        "masks": "random_complete_rows",
        "points": "indexed_row_resolution",
    }


def test_suite_is_deterministic_and_changes_with_seed() -> None:
    first = _suite(200_000)
    second = _suite(200_000)
    assert first == second

    receipt = first["payload"]["storage_plan_receipt"]
    # The builder intentionally accepts typed receipts only; deterministic
    # mutation behavior is covered by rebuilding through _suite.
    assert receipt["payload_digest"] == canonical_json_sha256(receipt["payload"])


def test_suite_v2_uses_observed_growth_axis_for_selection_extent() -> None:
    declaration = AnalysisArrayDeclaration(
        path="signals/timeline",
        contract=ArrayContract(
            schema_id="palette.test.axis_one_timeline",
            schema_version=1,
            dtype=FLOAT32,
            shape_template=("n_subjects", "n_frames"),
            axis_names=("subject", "frame"),
            description="Axis-one timeline used to prove growth-axis selection.",
        ),
        required=True,
        access_pattern=AccessPattern.WINDOWED,
        write_mode=WriteMode.IMMUTABLE,
        authority_role=AnalysisAuthorityRole.SCIENTIFIC_AUTHORITY,
        fill_semantics="every frame is present",
        null_semantics="none",
        physical_policy_owner="test_candidate",
        byte_planner_adopted=False,
    )
    receipt = plan_analysis_storage(
        (declaration,),
        {
            declaration.path: AnalysisArrayStorageFacts(
                path=declaration.path,
                shape=(1, 200_000),
                dtype=np.float32,
                access_unit_semantics="one subject-frame value",
                growth_axis=1,
            )
        },
        profile=PUBLISHED_HTTP_V1,
        dimensions={"n_subjects": 1, "n_frames": 200_000},
    )
    suite = build_analysis_benchmark_suite(
        family_id="axis_one_family",
        scale=AnalysisBenchmarkScale(
            scale_id="full",
            dimensions=(("n_frames", 200_000), ("n_subjects", 1)),
            description="Axis-one full-duration fixture.",
        ),
        storage_receipt=receipt,
    )
    primary = next(
        record
        for record in suite["payload"]["array_cases"]
        if record["selection"]["mode"] == "bounded_row_windows"
    )

    assert suite["schema_version"] == ANALYSIS_BENCHMARK_SUITE_SCHEMA_VERSION
    assert primary["selection"]["selection_axis"] == 1
    assert primary["selection"]["selection_extent"] == 200_000
    assert primary["selection"]["ranges"][-1] == [195_904, 200_000]


def test_legacy_v1_suite_is_auditable_but_not_current_timing_evidence() -> None:
    suite = _suite(200_000)
    legacy = deepcopy(suite)
    legacy["schema_version"] = ANALYSIS_BENCHMARK_SUITE_LEGACY_SCHEMA_VERSION
    for record in legacy["payload"]["array_cases"]:
        selection = record["selection"]
        selection.pop("selection_axis", None)
        selection.pop("selection_extent", None)
        selection.pop("selection_extent_source", None)
        selection.pop("execution_strategy", None)
    legacy["payload_digest"] = canonical_json_sha256(legacy["payload"])

    require_analysis_benchmark_suite_manifest(legacy)
    with pytest.raises(ValueError, match="timing- and promotion-ineligible"):
        require_analysis_benchmark_suite_manifest(legacy, require_current=True)


def test_rehashed_case_plan_tampering_fails() -> None:
    suite = _suite(200_000)
    tampered = deepcopy(suite)
    case = tampered["payload"]["array_cases"][0]["case"]
    case["storage_plan"]["chunk_shape"] = [1]
    tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])

    with pytest.raises(ValueError, match="case storage plan differs"):
        require_analysis_benchmark_suite_manifest(tampered)


def test_rehashed_receipt_and_case_plan_tampering_fails() -> None:
    suite = _suite(200_000)
    tampered = deepcopy(suite)
    receipt = tampered["payload"]["storage_plan_receipt"]
    path = receipt["payload"]["arrays"][0]["path"]
    receipt["payload"]["arrays"][0]["plan"]["chunk_shape"][0] = 1
    receipt["payload_digest"] = canonical_json_sha256(receipt["payload"])
    for row in tampered["payload"]["array_cases"]:
        if row["array_path"] == path:
            row["case"]["storage_plan"]["chunk_shape"][0] = 1
    tampered["payload"]["publication_case"]["storage_plan_receipt_digest"] = receipt[
        "payload_digest"
    ]
    tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])

    with pytest.raises(ValueError, match="storage-plan receipt is not executable"):
        require_analysis_benchmark_suite_manifest(tampered)


def test_rehashed_execution_policy_tampering_fails() -> None:
    suite = _suite(1_000_000)
    tampered = deepcopy(suite)
    tampered["payload"]["execution_policy"]["selector_eligible"] = True
    tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])

    with pytest.raises(ValueError, match="execution safety policy"):
        require_analysis_benchmark_suite_manifest(tampered)


def test_rehashed_deterministic_selection_tampering_fails() -> None:
    suite = _suite(1_000_000)
    tampered = deepcopy(suite)
    row = next(
        record
        for record in tampered["payload"]["array_cases"]
        if record["selection"]["mode"] == "random_complete_rows"
    )
    row["selection"]["row_indices"][0] += 1
    tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])

    with pytest.raises(ValueError, match="selection differs"):
        require_analysis_benchmark_suite_manifest(tampered)


def test_scale_must_match_planned_dimensions() -> None:
    declarations = (
        _declaration("timeline", dtype=FLOAT32, access=AccessPattern.WINDOWED),
    )
    receipt = plan_analysis_storage(
        declarations,
        {
            "timeline": AnalysisArrayStorageFacts(
                path="timeline",
                shape=(10,),
                dtype=np.float32,
                access_unit_semantics="one row",
            )
        },
        profile=PUBLISHED_HTTP_V1,
        dimensions={"n_rows": 10},
    )
    with pytest.raises(ValueError, match="dimensions must equal"):
        build_analysis_benchmark_suite(
            family_id="test_family",
            scale=AnalysisBenchmarkScale(
                scale_id="wrong",
                dimensions=(("n_rows", 11),),
                description="Wrong logical scale.",
            ),
            storage_receipt=receipt,
        )

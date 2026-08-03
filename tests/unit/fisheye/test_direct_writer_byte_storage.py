from __future__ import annotations

import numpy as np
import zarr

from fisheye.analysis.bout_classification_runs import (
    validate_staged_bout_classification_run,
)
from fisheye.analysis.bout_classification_schema import (
    BOUT_CLASSIFICATION_ACCESS_UNIT_SEMANTICS,
    BOUT_CLASSIFICATION_CANDIDATE_ARRAY_DECLARATIONS,
)
from fisheye.analysis.direct_writer_storage import (
    ANALYSIS_STORAGE_PLAN_DIGEST_ATTR,
    ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR,
    ANALYSIS_STORAGE_PROFILE_ROLE,
    ANALYSIS_STORAGE_PROFILE_ROLE_ATTR,
)
from fisheye.analysis.megabouts_classifier import (
    write_megabouts_classification_run,
)
from fisheye.analysis.tail_posture_view_runs import (
    write_tail_posture_view_run_group,
)
from fisheye.analysis.tail_posture_view_schema import (
    TAIL_POSTURE_VIEW_ACCESS_UNIT_SEMANTICS,
    TAIL_POSTURE_VIEW_ARRAY_SCHEMA_ATTR,
    TAIL_POSTURE_VIEW_CANDIDATE_ARRAY_DECLARATIONS,
    TailPostureViewDimensions,
    validate_tail_posture_view_arrays,
)
from fisheye.shared.zarr.analysis_storage_planning import (
    AnalysisArrayStorageFacts,
    plan_analysis_storage,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.storage_profiles import PUBLISHED_HTTP_V1
from tests.unit.fisheye.test_megabouts_publication_lifecycle import (
    _pack,
    _result,
)
from tests.unit.fisheye.test_tail_posture_view_runs import (
    _build_shape_root,
    _patch_provenance,
)


def _facts_for_declarations(
    declarations,
    *,
    dimensions: dict[str, int],
    semantics: dict[str, str],
) -> dict[str, AnalysisArrayStorageFacts]:
    facts: dict[str, AnalysisArrayStorageFacts] = {}
    for declaration in declarations:
        shape = tuple(
            dimensions[value] if isinstance(value, str) else value
            for value in declaration.contract.shape_template
        )
        facts[declaration.path] = AnalysisArrayStorageFacts(
            path=declaration.path,
            shape=shape,
            dtype=declaration.contract.dtype.numpy_dtype,
            access_unit_semantics=semantics[declaration.path],
        )
    return facts


def _entry(receipt, path: str):
    return next(item for item in receipt.entries if item.declaration.path == path)


def _copy_group(source, destination) -> None:
    destination.attrs.update(dict(source.attrs))
    for name in source.array_keys():
        source_array = source[name]
        destination_array = destination.create_array(
            name,
            data=np.asarray(source_array[:]),
        )
        destination_array.attrs.update(dict(source_array.attrs))
    for name in source.group_keys():
        _copy_group(source[name], destination.create_group(name))


def test_tail_and_bout_candidate_plans_are_byte_derived_at_empty_short_and_full_scale() -> (
    None
):
    tail_estimates: dict[int, dict[str, object]] = {}
    bout_estimates: dict[int, dict[str, object]] = {}
    for row_count in (0, 200_000, 1_000_000):
        tail_dimensions = {
            "n_rows": row_count,
            "n_keypoints": 11,
            "n_angles": 10,
        }
        tail = plan_analysis_storage(
            TAIL_POSTURE_VIEW_CANDIDATE_ARRAY_DECLARATIONS,
            _facts_for_declarations(
                TAIL_POSTURE_VIEW_CANDIDATE_ARRAY_DECLARATIONS,
                dimensions=tail_dimensions,
                semantics=TAIL_POSTURE_VIEW_ACCESS_UNIT_SEMANTICS,
            ),
            profile=PUBLISHED_HTTP_V1,
            dimensions=tail_dimensions,
        )
        tail_estimates[row_count] = tail.as_manifest()["payload"]["object_estimate"]

        bout_dimensions = {"n_bouts": row_count}
        bout = plan_analysis_storage(
            BOUT_CLASSIFICATION_CANDIDATE_ARRAY_DECLARATIONS,
            _facts_for_declarations(
                BOUT_CLASSIFICATION_CANDIDATE_ARRAY_DECLARATIONS,
                dimensions=bout_dimensions,
                semantics=BOUT_CLASSIFICATION_ACCESS_UNIT_SEMANTICS,
            ),
            profile=PUBLISHED_HTTP_V1,
            dimensions=bout_dimensions,
        )
        bout_estimates[row_count] = bout.as_manifest()["payload"]["object_estimate"]

        if row_count:
            assert _entry(tail, "instance_key").plan.chunk_shape[0] == 131_072
            assert _entry(tail, "failure_reason_bytes").plan.chunk_shape == (
                16_384,
                64,
            )
            assert _entry(tail, "tail_keypoints_xy").plan.chunk_shape == (
                8_192,
                11,
                2,
            )
            assert _entry(bout, "per_bout/category_label_bytes").plan.chunk_shape == (
                16_384,
                64,
            )
            assert _entry(bout, "per_bout/failure_reason_bytes").plan.chunk_shape == (
                8_192,
                128,
            )

    assert tail_estimates[0]["payload_objects"] == 0
    assert tail_estimates[0]["array_metadata_objects"] == 10
    assert bout_estimates[0]["payload_objects"] == 0
    assert bout_estimates[0]["array_metadata_objects"] == 20
    assert tail_estimates[200_000]["payload_objects"] == 10
    assert tail_estimates[1_000_000]["payload_objects"] == 15
    assert bout_estimates[200_000]["payload_objects"] == 20
    assert bout_estimates[1_000_000]["payload_objects"] == 24


def test_tail_candidate_is_complete_ineligible_and_rejects_rehashed_plan_tampering(
    monkeypatch,
) -> None:
    _patch_provenance(monkeypatch)
    root = _build_shape_root()

    summary = write_tail_posture_view_run_group(
        root,
        subject_shape_run="shape_001",
        run_name="posture_byte_candidate",
        storage_profile=PUBLISHED_HTTP_V1,
    )

    parent = root["analysis/tail_posture_view_runs"]
    run = parent["posture_byte_candidate"]
    assert summary["status"] == "candidate_complete"
    assert run.attrs["palette_run_completion_status"] == "complete"
    assert run.attrs["stage_selector_eligible"] is False
    assert parent.attrs.get("latest") != "posture_byte_candidate"
    assert parent.attrs.get("latest_complete") != "posture_byte_candidate"
    assert run.attrs[ANALYSIS_STORAGE_PROFILE_ROLE_ATTR] == (
        ANALYSIS_STORAGE_PROFILE_ROLE
    )
    assert (
        run.attrs[TAIL_POSTURE_VIEW_ARRAY_SCHEMA_ATTR]["byte_planner_adopted"] is True
    )
    assert not validate_tail_posture_view_arrays(
        run,
        dimensions=TailPostureViewDimensions(
            n_rows=2,
            n_keypoints=11,
            n_angles=10,
        ),
    )

    receipt = run.attrs[ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR]
    receipt["payload"]["arrays"][0]["plan"]["chunk_shape"][0] += 1
    receipt["payload_digest"] = canonical_json_sha256(receipt["payload"])
    run.attrs[ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR] = receipt
    run.attrs[ANALYSIS_STORAGE_PLAN_DIGEST_ATTR] = receipt["payload_digest"]

    issues = validate_tail_posture_view_arrays(
        run,
        dimensions=TailPostureViewDimensions(
            n_rows=2,
            n_keypoints=11,
            n_angles=10,
        ),
    )
    assert any(issue.code == "storage_plan_mismatch" for issue in issues)
    assert any("executable planning" in issue.message for issue in issues)


def test_bout_candidate_roundtrips_and_direct_matches_consolidated_metadata(
    tmp_path,
) -> None:
    store_path = tmp_path / "bout_candidate.zarr"
    root = zarr.open_group(store_path, mode="w", zarr_format=3)
    run_name = write_megabouts_classification_run(
        root,
        run_name="bout_byte_candidate",
        pack=_pack(),
        result=_result(),
        storage_profile=PUBLISHED_HTTP_V1,
    )
    assert run_name == "bout_byte_candidate"
    parent = root["analysis/bout_classification_runs"]
    run = parent[run_name]
    assert run.attrs["palette_run_completion_status"] == "complete"
    assert run.attrs["stage_selector_eligible"] is False
    assert parent.attrs.get("latest") != run_name
    assert parent.attrs.get("latest_complete") != run_name
    assert (
        validate_staged_bout_classification_run(
            root,
            run_name,
            strict=True,
        )["ok"]
        is True
    )

    zarr.consolidate_metadata(store_path)
    direct = zarr.open_group(store_path, mode="r", use_consolidated=False)
    consolidated = zarr.open_group(store_path, mode="r", use_consolidated=True)
    direct_run = direct[f"analysis/bout_classification_runs/{run_name}"]
    consolidated_run = consolidated[f"analysis/bout_classification_runs/{run_name}"]
    assert dict(direct_run.attrs) == dict(consolidated_run.attrs)
    for declaration in BOUT_CLASSIFICATION_CANDIDATE_ARRAY_DECLARATIONS:
        direct_array = direct_run[declaration.path]
        consolidated_array = consolidated_run[declaration.path]
        assert direct_array.metadata.to_dict() == consolidated_array.metadata.to_dict()
        np.testing.assert_array_equal(direct_array[:], consolidated_array[:])


def test_tail_candidate_direct_matches_consolidated_metadata(
    monkeypatch,
    tmp_path,
) -> None:
    _patch_provenance(monkeypatch)
    store_path = tmp_path / "tail_candidate.zarr"
    root = zarr.open_group(store_path, mode="w", zarr_format=3)
    _copy_group(_build_shape_root(), root)
    write_tail_posture_view_run_group(
        root,
        subject_shape_run="shape_001",
        run_name="tail_byte_candidate",
        storage_profile=PUBLISHED_HTTP_V1,
    )

    zarr.consolidate_metadata(store_path)
    direct = zarr.open_group(store_path, mode="r", use_consolidated=False)
    consolidated = zarr.open_group(store_path, mode="r", use_consolidated=True)
    run_path = "analysis/tail_posture_view_runs/tail_byte_candidate"
    direct_run = direct[run_path]
    consolidated_run = consolidated[run_path]
    assert dict(direct_run.attrs) == dict(consolidated_run.attrs)
    for declaration in TAIL_POSTURE_VIEW_CANDIDATE_ARRAY_DECLARATIONS:
        direct_array = direct_run[declaration.path]
        consolidated_array = consolidated_run[declaration.path]
        assert direct_array.metadata.to_dict() == consolidated_array.metadata.to_dict()
        np.testing.assert_array_equal(direct_array[:], consolidated_array[:])

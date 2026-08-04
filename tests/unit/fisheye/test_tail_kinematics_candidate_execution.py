from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from fisheye.analysis.tail_kinematics_schema import (
    TailKinematicsDimensions,
    stamp_tail_kinematics_array_schema,
)
from fisheye.analysis.tail_kinematics_storage import (
    build_tail_kinematics_storage_receipt,
    create_tail_kinematics_arrays_from_receipt,
    persist_tail_kinematics_storage_receipt,
)
from fisheye.analysis_workflows.tail_kinematics_candidate_execution import (
    TAIL_KINEMATICS_CORE_ARRAY_COUNT,
    TAIL_KINEMATICS_EXECUTION_FAMILY_ID,
    TAIL_KINEMATICS_REVISION_BUNDLE_ARRAY_COUNT,
    build_tail_kinematics_coordinate_evidence,
    build_tail_kinematics_execution_suite,
    build_tail_kinematics_invocation,
    compute_tail_kinematics_logical_hashes,
    require_tail_kinematics_execution_suite,
    require_tail_kinematics_invocation_manifest,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.storage_profiles import PUBLISHED_HTTP_V1


def _run(*, include_revision_bundle: bool) -> zarr.Group:
    dimensions = TailKinematicsDimensions(
        n_rows=7,
        n_tail_samples=10,
        n_components=3 if include_revision_bundle else None,
    )
    receipt = build_tail_kinematics_storage_receipt(
        dimensions,
        profile=PUBLISHED_HTTP_V1,
    )
    root = zarr.group()
    run = root.require_group("analysis/tail_kinematics_runs/source")
    create_tail_kinematics_arrays_from_receipt(
        run,
        receipt=receipt,
        dimensions=dimensions,
    )
    run.attrs["byte_planner_adopted"] = True
    stamp_tail_kinematics_array_schema(
        run,
        dimensions,
        byte_planner_adopted=True,
    )
    persist_tail_kinematics_storage_receipt(run, receipt)
    run["instance_key"][:] = np.arange(10, 17, dtype=np.uint64)
    run["source_crop_row_ids"][:] = np.arange(20, 27, dtype=np.int64)
    run["source_acquisition_frame_index"][:] = np.arange(30, 37, dtype=np.int64)
    run["tail_angle_sample_s"][:] = np.linspace(0, 1, 10, dtype=np.float32)
    if include_revision_bundle:
        run["source_refined_subject_masks/row_revision"][:] = np.arange(
            21, dtype=np.int64
        ).reshape(7, 3)
        run["source_refined_subject_masks/row_revision_available"][:] = True
    return run


@pytest.mark.parametrize("include_revision_bundle", [False, True])
def test_suite_reconstructs_exact_core_and_atomic_revision_bundle(
    include_revision_bundle: bool,
) -> None:
    source = _run(include_revision_bundle=include_revision_bundle)
    suite = build_tail_kinematics_execution_suite(
        source,
        scale_id="unit",
        description="exact tail execution fixture",
        repetitions=1,
    )

    require_tail_kinematics_execution_suite(
        TAIL_KINEMATICS_EXECUTION_FAMILY_ID,
        suite,
    )
    expected_count = TAIL_KINEMATICS_CORE_ARRAY_COUNT + (
        TAIL_KINEMATICS_REVISION_BUNDLE_ARRAY_COUNT if include_revision_bundle else 0
    )
    assert len(suite["payload"]["storage_plan_receipt"]["payload"]["arrays"]) == (
        expected_count
    )
    assert len(compute_tail_kinematics_logical_hashes(source)["arrays"]) == (
        expected_count
    )


def test_suite_rejects_rehashed_storage_plan_tampering() -> None:
    suite = build_tail_kinematics_execution_suite(
        _run(include_revision_bundle=True),
        scale_id="unit",
        description="exact tail execution fixture",
        repetitions=1,
    )
    tampered = deepcopy(suite)
    tampered["payload"]["storage_plan_receipt"]["payload"]["arrays"][0]["plan"][
        "chunk_shape"
    ][0] += 1
    receipt = tampered["payload"]["storage_plan_receipt"]
    receipt["payload_digest"] = canonical_json_sha256(receipt["payload"])
    tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])

    with pytest.raises(ValueError, match="storage.plan"):
        require_tail_kinematics_execution_suite(
            TAIL_KINEMATICS_EXECUTION_FAMILY_ID,
            tampered,
        )


def test_family_invocation_is_closed_and_digest_bound() -> None:
    invocation = build_tail_kinematics_invocation(
        source_subject_shape_run="shape_1",
        source_tail_coordinate_manifest_sha256="a" * 64,
        source_subject_shape_manifest_sha256="b" * 64,
        tail_angle_sample_count=10,
        block_rows=8_192,
        output_shard_rows=131_072,
        storage_profile_id="published_http_v1",
        copy_backend="python",
        keep_scratch=False,
        check_capacity=True,
    )

    require_tail_kinematics_invocation_manifest(invocation)
    assert invocation["payload"]["contract_id"] == "tail_kinematics_v1"

    tampered = deepcopy(invocation)
    tampered["payload"]["parameters"]["num_workers"] = 2
    tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])
    with pytest.raises(ValueError, match="num_workers"):
        require_tail_kinematics_invocation_manifest(tampered)


def _publication(*, run: str, tail_digest: str, shape_digest: str):
    return SimpleNamespace(
        manifest=SimpleNamespace(
            record_sha256=tail_digest,
            record_ref=f"/analysis/tail_kinematics_runs/{run}@manifest",
        ),
        source=SimpleNamespace(
            manifest=SimpleNamespace(
                record_sha256=shape_digest,
                record_ref="/analysis/subject_shape_runs/shape_1@manifest",
            )
        ),
    )


def test_coordinate_evidence_binds_source_and_published_authorities() -> None:
    evidence = build_tail_kinematics_coordinate_evidence(
        source_publication=_publication(
            run="source",
            tail_digest="a" * 64,
            shape_digest="b" * 64,
        ),
        candidate_publication=_publication(
            run="candidate",
            tail_digest="c" * 64,
            shape_digest="b" * 64,
        ),
    )

    assert evidence["status"] == "verified_canonical_publication"
    assert evidence["coordinate_gate_passed"] is True
    assert evidence["published_authority_sha256"] == "c" * 64
    assert [item["role"] for item in evidence["source_authority_digests"]] == [
        "canonical_subject_shape",
        "source_tail_kinematics",
    ]


def test_coordinate_evidence_rejects_authority_swap() -> None:
    with pytest.raises(ValueError, match="different subject-shape"):
        build_tail_kinematics_coordinate_evidence(
            source_publication=_publication(
                run="source",
                tail_digest="a" * 64,
                shape_digest="b" * 64,
            ),
            candidate_publication=_publication(
                run="candidate",
                tail_digest="c" * 64,
                shape_digest="d" * 64,
            ),
        )

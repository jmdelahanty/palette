from __future__ import annotations

import json

import pytest

from fisheye.shared.zarr.array_contracts import (
    FRAME_COUNTS_V1,
    KEYPOINTS_IMG_V1,
)
from fisheye.shared.zarr.benchmark_contracts import (
    PER_ROW_RANDOM_READ_V1,
    WINDOWED_ROWS_READ_V1,
    BenchmarkPhase,
    StorageBenchmarkCase,
    benchmark_result_envelope,
)
from fisheye.shared.zarr.storage_intent import AccessPattern, WriteMode
from fisheye.shared.zarr.storage_planner import plan_storage
from fisheye.shared.zarr.storage_profiles import PUBLISHED_HTTP_V1


def _keypoint_plan():
    intent = KEYPOINTS_IMG_V1.storage_intent(
        name="keypoints_img",
        shape=(1_000_000, 5, 2),
        access=AccessPattern.WINDOWED,
        write_mode=WriteMode.IMMUTABLE,
    )
    return plan_storage(intent, PUBLISHED_HTTP_V1)


def test_benchmark_case_locks_schema_plan_phase_and_workload() -> None:
    case = StorageBenchmarkCase(
        case_id="keypoints_img__published_http_v1__windowed",
        phase=BenchmarkPhase.READ,
        array_contract=KEYPOINTS_IMG_V1,
        storage_plan=_keypoint_plan(),
        workload=WINDOWED_ROWS_READ_V1,
    )

    manifest = case.as_manifest()
    assert manifest["logical_schema"] == {
        "id": "palette.array.keypoints_img",
        "version": 1,
    }
    assert manifest["storage_plan"]["logical_dtype"] == "float64"
    assert json.loads(json.dumps(manifest)) == manifest


def test_benchmark_case_rejects_wrong_logical_contract() -> None:
    with pytest.raises(ValueError, match="logical schema identity"):
        StorageBenchmarkCase(
            case_id="wrong_contract",
            phase=BenchmarkPhase.READ,
            array_contract=FRAME_COUNTS_V1,
            storage_plan=_keypoint_plan(),
            workload=WINDOWED_ROWS_READ_V1,
        )


def test_benchmark_case_rejects_incompatible_access_workload() -> None:
    with pytest.raises(ValueError, match="does not support access pattern"):
        StorageBenchmarkCase(
            case_id="wrong_workload",
            phase=BenchmarkPhase.READ,
            array_contract=KEYPOINTS_IMG_V1,
            storage_plan=_keypoint_plan(),
            workload=PER_ROW_RANDOM_READ_V1,
        )


def test_common_result_envelope_is_json_safe_and_schema_linked() -> None:
    case = StorageBenchmarkCase(
        case_id="keypoints_img__published_http_v1__windowed",
        phase=BenchmarkPhase.READ,
        array_contract=KEYPOINTS_IMG_V1,
        storage_plan=_keypoint_plan(),
        workload=WINDOWED_ROWS_READ_V1,
    )
    result = benchmark_result_envelope(
        case,
        source_identity={"digest": "abc"},
        environment={"storage_tier": "local"},
        trials=[{"read_seconds": 1.0}],
        summary={"median_seconds": 1.0},
        validation={"exact_dtype": True, "decoded_match": True},
    )

    assert result["schema_id"] == "palette.storage_benchmark"
    assert result["schema_version"] == 1
    assert result["storage_plan"]["logical_schema_id"] == (
        "palette.array.keypoints_img"
    )
    assert json.loads(json.dumps(result)) == result

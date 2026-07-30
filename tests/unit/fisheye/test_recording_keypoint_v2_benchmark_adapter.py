from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fisheye.utils import finalize_recording_keypoint_v2_benchmark_adapter as mod


def test_row_lookup_reorders_by_stable_instance_key() -> None:
    available = np.asarray([90, 10, 40], dtype=np.uint64)
    requested = np.asarray([10, 40, 90], dtype=np.uint64)

    rows = mod._row_lookup(requested=requested, available=available)

    np.testing.assert_array_equal(rows, np.asarray([1, 2, 0], dtype=np.int64))


@pytest.mark.parametrize(
    "requested,available,message",
    [
        ([10, 10], [10, 20], "Requested crop-v2 instance keys are not unique"),
        ([10, 20], [10, 10], "Historical keypoint instance keys are not unique"),
        ([10, 30], [10, 20], "do not cover"),
        ([10], [10, 20], "contain rows absent"),
    ],
)
def test_row_lookup_fails_closed_on_nonbijective_keys(
    requested: list[int],
    available: list[int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        mod._row_lookup(
            requested=np.asarray(requested, dtype=np.uint64),
            available=np.asarray(available, dtype=np.uint64),
        )


def test_node_local_scratch_rejects_shared_storage(tmp_path: Path) -> None:
    assert mod._require_node_local_scratch(tmp_path) == tmp_path.resolve()
    with pytest.raises(ValueError, match="node-local"):
        mod._require_node_local_scratch(Path("/groups/example"))


def test_rebound_receipt_changes_only_output_locations(tmp_path: Path) -> None:
    payload = {
        "status": "complete",
        "selector_eligible": False,
        "registry_registered": False,
        "crop": {},
        "clip_inputs": [],
        "preparation": {},
        "outputs": {
            name: {
                "path": f"/tmp/local/{name}.zarr",
                "selector_eligible": False,
            }
            for name in (
                "raw_keypoints",
                "keypoint_quality",
                "refined_keypoints",
                "body_frame",
            )
        },
        "selector_activation": "none_direct_path_only",
        "production_state_changes": [],
    }
    receipt = {
        "schema_id": "palette.keypoint.clipped_recording_finalization",
        "schema_version": 1,
        "digest_algorithm": "sha256_canonical_json_v1",
        "payload_digest": mod.canonical_json_sha256(payload),
        "payload": payload,
    }

    rebound = mod._rebind_finalization_receipt(
        receipt,
        destination=tmp_path / "published",
    )

    assert rebound["payload_digest"] != receipt["payload_digest"]
    for binding in rebound["payload"]["outputs"].values():
        assert Path(binding["path"]).parent == tmp_path / "published"

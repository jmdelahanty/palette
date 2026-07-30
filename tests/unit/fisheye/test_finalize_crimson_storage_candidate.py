from __future__ import annotations

import json
from pathlib import Path

import pytest

from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.utils import finalize_crimson_storage_candidate as mod


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _result(
    tmp_path: Path,
    *,
    name: str,
    receipt_path: Path | None = None,
) -> Path:
    payload: dict[str, object] = {
        "status": "complete",
        "selector_eligible": False,
        "registry_registered": False,
        "production_state_changes": [],
    }
    if receipt_path is not None:
        payload["finalization_receipt_path"] = str(receipt_path)
    if name == "refined":
        payload.update(
            {
                "output_archive": str(tmp_path / "refined.zarr"),
                "output_run_id": "refined_run",
            }
        )
    elif name == "crop":
        payload.update(
            {
                "output_archive": str(tmp_path / "crop.zarr"),
                "output_run_id": "crop_run",
            }
        )
    return _write(tmp_path / f"{name}.json", payload)


def test_final_handoff_freezes_scale_and_all_seven_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refined_receipt = _write(
        tmp_path / "refined_receipt.json",
        {"payload_digest": "1" * 64, "payload": {}},
    )
    keypoint_outputs = {
        stage: {
            "path": str(tmp_path / f"{stage}.zarr"),
            "run_id": f"{stage}_run",
        }
        for stage in (
            "raw_keypoints",
            "keypoint_quality",
            "refined_keypoints",
            "body_frame",
        )
    }
    keypoint_receipt = _write(
        tmp_path / "keypoint_receipt.json",
        {
            "payload_digest": "2" * 64,
            "payload": {"outputs": keypoint_outputs},
        },
    )
    refined_result = _result(
        tmp_path, name="refined", receipt_path=refined_receipt
    )
    crop_result = _result(tmp_path, name="crop")
    keypoint_result = _result(
        tmp_path, name="keypoint", receipt_path=keypoint_receipt
    )
    monkeypatch.setattr(
        mod, "validate_clipped_refined_detection_finalization_receipt", lambda _: ()
    )
    monkeypatch.setattr(
        mod, "validate_clipped_keypoint_finalization_receipt", lambda _: ()
    )
    monkeypatch.setattr(
        mod,
        "_artifact",
        lambda *, stage, archive, run_id, **_: {
            "stage": stage,
            "server_path": str(archive),
            "run_id": run_id,
        },
    )
    monkeypatch.setattr(
        mod,
        "_git_state",
        lambda: {"commit": "3" * 40, "branch": "agent/test", "worktree_clean": True},
    )
    output = (
        tmp_path
        / ".palette_benchmarks"
        / "candidate"
        / "handoff_manifest.json"
    )

    handoff = mod.finalize_crimson_storage_candidate(
        candidate_id="sleepyfish_full_v1",
        classification="full_duration_fixture",
        expected_n_frames=1_188_000,
        expected_n_instances=1_169_010,
        canonical_archive=tmp_path / "canonical.zarr",
        canonical_run="canonical_run",
        refined_result_path=refined_result,
        crop_result_path=crop_result,
        keypoint_result_path=keypoint_result,
        crimson_contract_commit="a" * 40,
        crimson_contract_sha256="b" * 64,
        expected_palette_commit="3" * 40,
        output=output,
    )

    assert output.is_file()
    assert handoff["payload_digest"] == canonical_json_sha256(handoff["payload"])
    payload = handoff["payload"]
    assert payload["classification"] == "full_duration_fixture"
    assert payload["promotion_semantics"] == (
        "full_duration_candidate_requires_crimson_gate"
    )
    assert payload["analysis_crop_pixels_included"] is False
    assert set(payload["artifacts"]) == {
        "canonical_detection",
        "refined_detection",
        "crop_geometry",
        "raw_keypoints",
        "keypoint_quality",
        "refined_keypoints",
        "body_frame",
    }


def test_handoff_refuses_non_benchmark_destination(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=".palette_benchmarks"):
        mod.finalize_crimson_storage_candidate(
            candidate_id="bad",
            classification="integration_fixture",
            expected_n_frames=1,
            expected_n_instances=1,
            canonical_archive=tmp_path / "canonical.zarr",
            canonical_run="canonical",
            refined_result_path=tmp_path / "refined.json",
            crop_result_path=tmp_path / "crop.json",
            keypoint_result_path=tmp_path / "keypoint.json",
            crimson_contract_commit="a" * 40,
            crimson_contract_sha256="b" * 64,
            expected_palette_commit="3" * 40,
            output=tmp_path / "handoff_manifest.json",
        )

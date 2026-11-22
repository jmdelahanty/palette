"""palette.orange_crop_cqp_gop1.v1: Orange's constant-QP all-intra crops.

Shapes are those of the 2026-11-20 chaser pilots (Orange edc5450 recorder,
app config recording.crop.external_ipc.qp 29).
"""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path

import pytest

from fisheye.diagnostics.video import container
from fisheye.shared.orange_crop_compression import (
    CropCompressionRefusal,
    cqp_gop1_compression,
    require_start_snapshot_qp,
)
from fisheye.shared.recording_transfer_snapshot import TransferSnapshotError
from fisheye.utils.organize_transfer_recordings import _require_sealed_crop_qp

from tests.unit.fisheye.test_video_diagnostics_orange_sync import (
    _write_evidence,
    _write_mp4,
)


def _cqp_summary(qp: int = 29) -> dict:
    return {
        "tuning": "hq",
        "rate_control": {
            "const_qp": qp,
            "const_qp_active": True,
            "nvenc_rate_control_mode": "constqp",
            "requested_mode": "cqp",
            "resolved_strategy": "cqp",
            "max_bitrate_bps": 0,
        },
        "video_metadata": {
            "encoder": {
                "codec": "hevc",
                "profile_name": "external_crop_hevc_cqp_gop1",
                "tuning": "hq",
                "rate_control_strategy": "cqp",
                "qp": qp,
                "quality_value": qp,
                "resolved_gop_length": 1,
                "requested_gop_length": 25,
            }
        },
    }


def _check(tmp_path: Path, monkeypatch, summary: dict) -> dict:
    video = tmp_path / "crop.mp4"
    _write_mp4(video, has_stss=False)
    evidence = dataclasses.replace(
        _write_evidence(tmp_path, extra_summary=summary),
        declared_tuning=summary.get("tuning", "hq"),
    )
    monkeypatch.setattr(container, "_probe_codec_name", lambda _: "hevc")
    return container.check_hevc_keyframe_flags(video, orange_crop_evidence=evidence)


def test_qp29_all_intra_crop_is_admitted_and_records_its_qp(tmp_path, monkeypatch):
    result = _check(tmp_path, monkeypatch, _cqp_summary())

    assert result["sync_sample_proof"] == "orange_idr_sidecar_verified"
    assert result["needs_fix"] is False
    assert result["orange_evidence"]["crop_compression"] == {
        "rule": "palette.orange_crop_cqp_gop1.v1",
        "profile_name": "external_crop_hevc_cqp_gop1",
        "tuning": "hq",
        "qp": 29,
    }


def test_lossless_crop_assessment_is_unchanged(tmp_path, monkeypatch):
    video = tmp_path / "crop.mp4"
    _write_mp4(video, has_stss=False)
    monkeypatch.setattr(container, "_probe_codec_name", lambda _: "hevc")

    result = container.check_hevc_keyframe_flags(
        video, orange_crop_evidence=_write_evidence(tmp_path)
    )

    assert result["sync_sample_proof"] == "orange_idr_sidecar_verified"
    assert "crop_compression" not in result["orange_evidence"]


def _mutated(path: tuple, value):
    summary = _cqp_summary()
    target = summary
    for key in path[:-1]:
        target = target[key]
    if value is KeyError:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return summary


ENCODER = ("video_metadata", "encoder")


@pytest.mark.parametrize(
    "summary",
    [
        pytest.param(_mutated(ENCODER + ("profile_name",), "external_crop_hevc_vbr"), id="profile"),
        pytest.param(_mutated(ENCODER + ("tuning",), "ll"), id="encoder-tuning"),
        pytest.param(_mutated(ENCODER + ("codec",), "h264"), id="codec"),
        pytest.param(_mutated(ENCODER + ("rate_control_strategy",), "vbr"), id="strategy"),
        pytest.param(_mutated(ENCODER + ("qp",), 26), id="encoder-qp-disagrees"),
        pytest.param(_mutated(ENCODER + ("resolved_gop_length",), 25), id="encoder-gop"),
        pytest.param(_mutated(("rate_control", "const_qp_active"), False), id="qp-inactive"),
        pytest.param(_mutated(("rate_control", "resolved_strategy"), "vbr"), id="rc-strategy"),
        pytest.param(_mutated(("rate_control", "const_qp"), 0), id="qp-zero"),
        pytest.param(_mutated(("rate_control", "const_qp"), 52), id="qp-too-high"),
        pytest.param(_mutated(("rate_control", "const_qp"), True), id="qp-bool"),
        pytest.param(_mutated(("rate_control", "const_qp"), 29.0), id="qp-float"),
        pytest.param(_mutated(("rate_control",), KeyError), id="no-rate-control"),
        pytest.param(_mutated(("video_metadata",), KeyError), id="no-encoder"),
    ],
)
def test_any_other_lossy_crop_is_still_refused(tmp_path, monkeypatch, summary):
    result = _check(tmp_path, monkeypatch, summary)

    assert result["sync_sample_proof"] == "orange_idr_sidecar_contradiction"
    assert result["needs_fix"] is True
    assert "not lossless: 'hq'" in result["orange_evidence"]["error"]
    assert "crop_compression" not in result["orange_evidence"]


def test_other_lossy_tuning_is_refused(tmp_path, monkeypatch):
    summary = _cqp_summary()
    summary["tuning"] = "ll"

    result = _check(tmp_path, monkeypatch, summary)

    assert result["sync_sample_proof"] == "orange_idr_sidecar_contradiction"


def _snapshot(value) -> dict:
    environment = {} if value is None else {
        "ORANGE_CROP_EXTERNAL_QP": {"value": value, "source": "app_config"}
    }
    return {"session": {"effective_configuration": {"environment": environment}}}


def test_start_snapshot_qp_must_match():
    require_start_snapshot_qp(_snapshot("29"), 29)
    for value in ("26", None, 29, "", "2 9", "29.0", "٢٩"):
        with pytest.raises(CropCompressionRefusal):
            require_start_snapshot_qp(_snapshot(value), 29)


def _assessments(qp: int | None) -> dict:
    evidence = {"status": "orange_idr_sidecar_verified"}
    if qp is not None:
        evidence["crop_compression"] = cqp_gop1_compression(_cqp_summary(qp))
    return {"cams/x/Cam1_crop_external.mp4": {"orange_evidence": evidence}}


def test_organizer_refuses_a_qp_the_start_snapshot_did_not_seal(tmp_path):
    plan = {"source_dir": str(tmp_path)}
    snapshot = tmp_path / "recording_snapshot_start.json"

    _require_sealed_crop_qp(plan, _assessments(None))  # lossless: no snapshot read
    with pytest.raises(TransferSnapshotError, match="missing"):
        _require_sealed_crop_qp(plan, _assessments(29))

    snapshot.write_text(json.dumps(_snapshot("29")))
    _require_sealed_crop_qp(plan, _assessments(29))
    with pytest.raises(TransferSnapshotError, match="disagrees"):
        _require_sealed_crop_qp(plan, _assessments(26))

    snapshot.write_text(json.dumps(_snapshot(None)))
    with pytest.raises(TransferSnapshotError, match="decimal"):
        _require_sealed_crop_qp(plan, _assessments(29))


def test_summary_input_is_not_mutated():
    summary = _cqp_summary()
    before = copy.deepcopy(summary)
    cqp_gop1_compression(summary)
    assert summary == before

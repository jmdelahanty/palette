"""Training sources: lossless or palette.orange_crop_cqp_gop1.v1, always tagged."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fisheye.shared.orange_crop_compression import (
    CropCompressionRefusal,
    classify_crop_summary,
)
from fisheye.training.sampled_training_acquisition_crop_materialization import (
    resolve_acquisition_crop_source,
)
from fisheye.utils import export_acquisition_crop_pose_training_zarr as export

from tests.unit.fisheye.test_orange_crop_cqp_gop1 import _cqp_summary
from tests.unit.fisheye.test_regenerate_training_crops_pynvvc import (
    _make_lossless_acquisition_crop_source,
)


def _summary_path(recording: Path) -> Path:
    return next(
        (recording / "derived" / "external_crop_recorder").glob("*_crop_external_summary.json")
    )


def _make_cqp_source(tmp_path: Path, qp: int = 29) -> Path:
    recording = _make_lossless_acquisition_crop_source(tmp_path)
    path = _summary_path(recording)
    document = json.loads(path.read_text(encoding="utf-8"))
    cqp = _cqp_summary(qp)
    document["tuning"] = cqp["tuning"]
    document["rate_control"] = cqp["rate_control"]
    encoder = document["video_metadata"]["encoder"]
    encoder.update(cqp["video_metadata"]["encoder"])
    encoder.pop("quality_value")
    path.write_text(json.dumps(document), encoding="utf-8")
    return recording


def test_lossless_source_contract_is_unchanged(tmp_path):
    source = resolve_acquisition_crop_source(_make_lossless_acquisition_crop_source(tmp_path))

    assert "crop_compression" not in source.encoder_contract
    assert source.encoder_contract["encoder"]["profile_name"] == "external_crop_hevc_lossless_gop1"


def test_qp29_source_is_admitted_with_its_qp(tmp_path):
    source = resolve_acquisition_crop_source(_make_cqp_source(tmp_path))

    assert source.encoder_contract["crop_compression"] == {
        "rule": "palette.orange_crop_cqp_gop1.v1",
        "profile_name": "external_crop_hevc_cqp_gop1",
        "tuning": "hq",
        "qp": 29,
    }
    assert source.encoder_contract["encoder"]["qp"] == 29
    assert source.encoder_contract["encoder"]["tuning"] == "hq"


@pytest.mark.parametrize(
    "edit",
    [
        pytest.param(lambda d: d["video_metadata"]["encoder"].update(qp=26), id="encoder-qp"),
        pytest.param(lambda d: d["video_metadata"]["encoder"].update(resize_enabled=True), id="resize"),
        pytest.param(lambda d: d["video_metadata"]["encoder"].update(output_width=192), id="size"),
        pytest.param(lambda d: d["rate_control"].update(const_qp_active=False), id="qp-inactive"),
        pytest.param(lambda d: d.update(tuning="ll"), id="top-tuning"),
        pytest.param(lambda d: d.update(encode_dropped=3), id="dropped"),
        pytest.param(lambda d: d["video_metadata"]["source_pixel_contract"].update(dtype="uint16"), id="pixels"),
    ],
)
def test_materializer_refuses_any_other_lossy_source(tmp_path, edit):
    recording = _make_cqp_source(tmp_path)
    path = _summary_path(recording)
    document = json.loads(path.read_text(encoding="utf-8"))
    edit(document)
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="palette.orange_crop_cqp_gop1.v1"):
        resolve_acquisition_crop_source(recording)


def test_classifier():
    assert classify_crop_summary({"tuning": "lossless"})["lossless"] is True
    assert classify_crop_summary(_cqp_summary())["qp"] == 29
    assert classify_crop_summary(_cqp_summary())["lossless"] is False
    contradiction = {"tuning": "lossless", "video_metadata": {"encoder": {"tuning": "hq"}}}
    for summary in (contradiction, {}, {"tuning": "ll"}, {"tuning": None}):
        with pytest.raises(CropCompressionRefusal):
            classify_crop_summary(summary)


def _export_inputs(tmp_path: Path, summary: dict | None) -> tuple[Path, Path]:
    video = tmp_path / "derived" / "external_crop_recorder" / "Cam1_crop_external.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"x")
    if summary is not None:
        video.with_name("Cam1_crop_external_summary.json").write_text(json.dumps(summary))
    return tmp_path, video


def test_export_tags_the_source_compression(tmp_path):
    recording, video = _export_inputs(tmp_path, _cqp_summary())

    compression = export.crop_source_compression(recording, video)

    assert compression["rule"] == "palette.orange_crop_cqp_gop1.v1"
    assert compression["qp"] == 29
    assert compression["summary_path"].endswith("Cam1_crop_external_summary.json")
    assert len(compression["summary_sha256"]) == 64


@pytest.mark.parametrize("summary", [None, {"tuning": "ll"}, {"tuning": "hq"}])
def test_export_refuses_unverifiable_or_other_lossy_crops(tmp_path, summary):
    recording, video = _export_inputs(tmp_path, summary)

    with pytest.raises(ValueError):
        export.inspect_crop_video_stream(recording, tmp_path / "meta.csv", video)


def test_writers_refuse_a_stream_without_verified_compression():
    info = export.CropVideoStreamInfo(
        crop_video_path="v.mp4", crop_meta_path="m.csv", width=1, height=1,
        codec_name="hevc", pix_fmt="gray", color_range="pc", color_space=None,
        nb_frames="1", frame_format_confirmation_status="pending_orange_confirmation",
    )
    with pytest.raises(ValueError, match="no verified compression"):
        export.require_source_compression(info)

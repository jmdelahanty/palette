"""Helper tests for ONNX->TensorRT export logging helpers."""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.training.onnx_to_tensorrt import (
    _infer_onnx_manifest_path,
    _load_output_contract_from_manifest,
    _format_output_contract,
)


def test_infer_onnx_manifest_path_appends_manifest_suffix() -> None:
    onnx_path = Path("/tmp/model.onnx")
    assert _infer_onnx_manifest_path(onnx_path) == Path("/tmp/model.onnx.manifest.json")


def test_load_output_contract_from_manifest_reads_outputs(tmp_path: Path) -> None:
    manifest = tmp_path / "model.onnx.manifest.json"
    payload = {
        "onnx": {
            "outputs": [
                {"name": "num_dets", "dtype": "INT32", "shape": [1, 1]},
                {"name": "bboxes", "dtype": "FLOAT", "shape": [1, 1, 4]},
            ]
        }
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    out = _load_output_contract_from_manifest(manifest)
    assert out is not None
    assert out[0]["name"] == "num_dets"
    assert out[1]["dtype"] == "FLOAT"


def test_format_output_contract_renders_expected_string() -> None:
    text = _format_output_contract(
        [
            {"name": "scores", "dtype": "FLOAT", "shape": [1, 1]},
            {"name": "labels", "dtype": "INT32", "shape": [1, 1]},
        ]
    )
    assert text is not None
    assert "scores[FLOAT,(1,1)]" in text
    assert "labels[INT32,(1,1)]" in text


"""Path/helper tests for export_detection parameter resolution."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.training.export_detection import _resolve_training_params


class _Args:
    def __init__(self, imgsz=None, device=None):
        self.imgsz = imgsz
        self.device = device


def test_resolve_training_params_prefers_effective_params_from_report() -> None:
    report = {
        "training_params": {"imgsz": 2304, "device": "0"},
        "training_history": {
            "effective_training_params": {"imgsz": [640, 640], "device": "1"},
        },
    }
    params = _resolve_training_params(report, _Args())
    assert params["imgsz"] == [640, 640]
    assert params["device"] == "1"


def test_resolve_training_params_imgsz_override_wins() -> None:
    report = {
        "training_history": {
            "effective_training_params": {"imgsz": [640, 640], "device": "0"},
        },
    }
    params = _resolve_training_params(report, _Args(imgsz=[1024], device=None))
    assert params["imgsz"] == 1024

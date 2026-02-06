"""Unit tests for training registry status rendering helpers."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.utils.check_training_registry import (
    _metrics_summary_from_json,
    _status_with_details,
)


def test_metrics_summary_from_json_prefers_common_fields() -> None:
    payload = (
        '{"mAP50": 0.95231, "mAP50_95": 0.72111, '
        '"precision": 0.91001, "recall": 0.83002}'
    )
    assert (
        _metrics_summary_from_json(payload)
        == "mAP50=0.952, mAP50-95=0.721, P=0.910, R=0.830"
    )


def test_metrics_summary_from_json_supports_alt_keys() -> None:
    payload = '{"map50": 0.9, "map50-95": 0.6}'
    assert _metrics_summary_from_json(payload) == "mAP50=0.900, mAP50-95=0.600"


def test_status_with_details_appends_suffix_only_for_ok() -> None:
    assert _status_with_details(True, "mAP50=0.900", rich=False) == "OK (mAP50=0.900)"
    assert _status_with_details(False, "mAP50=0.900", rich=False) == "MISS"
    assert _status_with_details(None, "mAP50=0.900", rich=False) == "—"

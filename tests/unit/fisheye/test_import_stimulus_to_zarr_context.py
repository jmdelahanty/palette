"""Tests for session context extraction during stimulus import."""

from pathlib import Path
import sys

import h5py

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.analysis.import_stimulus_to_zarr import _read_h5_session_context


def test_read_h5_session_context_does_not_promote_protocol_name_to_canvas_name(tmp_path: Path) -> None:
    h5_path = tmp_path / "session.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.attrs["session_uuid"] = "session_1"
        h5.attrs["protocol_name_from_definition"] = "DefaultScreen"
        h5.attrs["loaded_protocol_filepath"] = "group_screen.json"

    with h5py.File(h5_path, "r") as h5:
        context = _read_h5_session_context(h5)

    assert context is not None
    assert context["protocol_name_from_definition"] == "DefaultScreen"
    assert "canvas_name" not in context
    assert "canvas_name_source" not in context


def test_read_h5_session_context_preserves_explicit_canvas_name(tmp_path: Path) -> None:
    h5_path = tmp_path / "session_with_canvas.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.attrs["session_uuid"] = "session_2"
        h5.attrs["canvas_name"] = "shadow"
        h5.attrs["protocol_name_from_definition"] = "DefaultScreen"

    with h5py.File(h5_path, "r") as h5:
        context = _read_h5_session_context(h5)

    assert context is not None
    assert context["canvas_name"] == "shadow"
    assert context["protocol_name_from_definition"] == "DefaultScreen"

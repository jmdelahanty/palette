from __future__ import annotations

import json
import sys
from pathlib import Path

from fisheye.diagnostics import benchmark_kvikio_gds as mod


def test_status_from_returncode_formats_success_failure_and_signal() -> None:
    assert mod._status_from_returncode(0) == "success"
    assert mod._status_from_returncode(2) == "failed(2)"
    assert mod._status_from_returncode(-11) == "signal 11"


def test_json_payload_from_stdout_uses_last_json_line() -> None:
    payload = mod._json_payload_from_stdout(
        "starting\n"
        + json.dumps({"first": True})
        + "\n"
        + "noise\n"
        + json.dumps({"second": True, "bytes_written": 4096})
        + "\n"
    )

    assert payload == {"second": True, "bytes_written": 4096}


def test_child_command_uses_current_python_and_module(tmp_path: Path) -> None:
    cmd = mod._child_command(
        "cufile-gpu",
        scratch_dir=tmp_path,
        size_mib=8,
        os_exit=True,
    )

    assert cmd[:4] == [sys.executable, "-m", "fisheye.diagnostics.benchmark_kvikio_gds", "_child"]
    assert "cufile-gpu" in cmd
    assert "--scratch-dir" in cmd
    assert str(tmp_path) in cmd
    assert "--size-mib" in cmd
    assert "8" in cmd
    assert "--os-exit" in cmd


def test_aligned_size_rounds_to_page_multiple() -> None:
    assert mod._aligned_size(1) == 1024 * 1024
    assert mod._aligned_size(0) == 1024 * 1024

from __future__ import annotations

from pathlib import Path

import pytest

from fisheye.shared import runtime_config as mod


def test_runtime_config_dirs_include_checkout_and_wheel_prefix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(mod.sys, "prefix", str(tmp_path))

    candidates = mod.runtime_config_dirs("pose_schemas")

    assert candidates[0].as_posix().endswith("/configs/fisheye/pose_schemas")
    assert candidates[-1] == (
        tmp_path / "share/palette/configs/fisheye/pose_schemas"
    )


@pytest.mark.parametrize("value", ["", "/absolute", "../escape", "a/../../escape"])
def test_runtime_config_dirs_reject_unsafe_paths(value: str) -> None:
    with pytest.raises(ValueError):
        mod.runtime_config_dirs(value)

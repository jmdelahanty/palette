"""Import-safety tests for optional native video dependencies."""

from __future__ import annotations

import builtins
import importlib
import sys
from pathlib import Path

import pytest


def test_import_video_is_import_safe_without_native_decode_deps(monkeypatch):
    """Collection must not require decord/cupy in CPU-only CI environments."""

    module_name = "fisheye.capture.import_video"
    capture_pkg = importlib.import_module("fisheye.capture")
    sentinel = object()
    previous_module = sys.modules.pop(module_name, None)
    previous_attr = getattr(capture_pkg, "import_video", sentinel)
    if previous_attr is not sentinel:
        delattr(capture_pkg, "import_video")

    real_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name.split(".")[0] in {"cupy", "decord"}:
            raise ModuleNotFoundError(f"No module named {name!r}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    try:
        module = importlib.import_module(module_name)
        assert module._HAVE_CUPY is False
        assert module._HAVE_DECORD is False
        with pytest.raises(RuntimeError, match="decord"):
            module._setup_video_reader(Path("missing.mp4"), use_gpu=False, force_cpu=True, console=None)
    finally:
        sys.modules.pop(module_name, None)
        if previous_module is not None:
            sys.modules[module_name] = previous_module
            setattr(capture_pkg, "import_video", previous_module)
        elif previous_attr is not sentinel:
            setattr(capture_pkg, "import_video", previous_attr)

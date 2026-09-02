"""Shared pytest fixtures."""

import socket
import tempfile
import shutil
from pathlib import Path

import pytest


pytest_plugins = ["tests.unit.fisheye.chaser_test_fixtures"]

_PALETTE_TEST_FORBIDDEN_HOSTS = frozenset({"login1", "login2"})


def _palette_test_short_hostname() -> str:
    return socket.gethostname().strip().lower().split(".", 1)[0]


def _require_palette_test_host() -> None:
    hostname = _palette_test_short_hostname()
    if hostname in _PALETTE_TEST_FORBIDDEN_HOSTS:
        raise pytest.UsageError(
            "Palette tests must run from a workstation checkout, never on "
            f"the campus login node {hostname!r}."
        )


def pytest_sessionstart(session: pytest.Session) -> None:
    """Fail before collection when pytest is invoked on a login node."""

    del session
    _require_palette_test_host()

@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir)

@pytest.fixture
def sample_video_metadata():
    """Standard video metadata for tests."""
    return {
        'fps': 30.0,
        'width': 640,
        'height': 480,
        'total_frames': 100,
        'source_video': 'test.mp4'
    }

@pytest.fixture
def sample_config():
    """Standard pipeline config for tests."""
    return {
        'import': {
            'downsample_size': [240, 320],
            'chunk_size': 10,
            'batch_size': 5
        }
    }

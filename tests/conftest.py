"""Shared pytest fixtures."""

import pytest
import tempfile
import shutil
from pathlib import Path

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
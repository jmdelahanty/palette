"""Unit tests for Zarr schema."""

import pytest
import tempfile
import shutil
from pathlib import Path
import sys
import zarr
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.shared.zarr.schema import (
    create_palette_zarr,
    get_run_group,
    ZARR_SCHEMA_VERSION
)

class TestZarrSchema:
    """Test Zarr schema creation and structure."""
    
    @pytest.fixture
    def temp_zarr_path(self):
        """Create temporary directory for zarr store."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir)
    
    def test_create_palette_zarr(self, temp_zarr_path):
        """Test basic zarr creation."""
        video_metadata = {
            'fps': 30.0,
            'width': 640,
            'height': 480,
            'total_frames': 100,
            'source_video': 'test.mp4'
        }
        
        config = {
            'import': {
                'downsample_size': [240, 320],
                'chunk_size': 10,
                'batch_size': 5
            }
        }
        
        root = create_palette_zarr(
            temp_zarr_path,
            video_metadata,
            config
        )
        
        # Check root attributes
        assert root.attrs['schema_version'] == ZARR_SCHEMA_VERSION
        assert root.attrs['fps'] == 30.0
        assert root.attrs['width'] == 640
        
        # Check group structure
        assert 'raw_video' in root
        assert 'processing' in root
        assert 'analysis' in root
        assert 'metadata' in root
        
        # Check subgroups
        assert 'background' in root['processing']
        assert 'detection' in root['processing']
        assert 'tracking' in root['processing']
        
        # Check arrays
        assert 'images_full' in root['raw_video']
        assert 'images_ds' in root['raw_video']
        assert 'timestamps' in root['raw_video']
        
        # Check array shapes
        images_full = root['raw_video']['images_full']
        assert images_full.shape == (100, 480, 640)
        
        images_ds = root['raw_video']['images_ds']
        assert images_ds.shape == (100, 240, 320)
    
    def test_get_run_group(self, temp_zarr_path):
        """Test run group creation."""
        video_metadata = {'fps': 30, 'width': 640, 'height': 480, 'total_frames': 10}
        config = {'import': {'downsample_size': [240, 320], 'chunk_size': 10}}
        
        root = create_palette_zarr(temp_zarr_path, video_metadata, config)
        
        # Create a new run group
        run_group = get_run_group(root, 'test_run_001')
        assert 'test_run_001' in root['processing']
        
        # Getting same group should return existing
        run_group2 = get_run_group(root, 'test_run_001')
        assert run_group == run_group2
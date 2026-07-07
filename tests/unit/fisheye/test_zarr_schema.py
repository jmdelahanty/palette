"""Unit tests for Zarr schema."""

import pytest
import asyncio
import threading
import tempfile
import shutil
from pathlib import Path
import sys
import zarr
import numpy as np
import zarr.core.sync as zarr_sync
import zarr.api.synchronous as zarr_sync_api
from zarr.storage import MemoryStore

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.shared.zarr.schema import (
    create_palette_zarr,
    get_run_group,
    ZARR_SCHEMA,
    ZARR_SCHEMA_VERSION
)
from fisheye.shared.run_provenance import build_writer_run_provenance

class TestZarrSchema:
    """Test Zarr schema creation and structure."""

    @pytest.fixture(autouse=True)
    def _patch_zarr_sync(self, monkeypatch):
        """Patch zarr's sync helper to avoid hangs in this environment."""

        def _sync_via_asyncio_run(coro, loop=None, timeout=None):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(coro)

            result = {}
            error = {}

            def _runner():
                try:
                    result["value"] = asyncio.run(coro)
                except Exception as exc:  # pragma: no cover - defensive
                    error["exc"] = exc

            thread = threading.Thread(target=_runner, daemon=True)
            thread.start()
            thread.join()

            if "exc" in error:
                raise error["exc"]
            return result.get("value")

        monkeypatch.setattr(zarr_sync, "sync", _sync_via_asyncio_run)
        monkeypatch.setattr(zarr_sync_api, "sync", _sync_via_asyncio_run)
        # Use an in-memory store to avoid LocalStore async deadlocks in this environment.
        monkeypatch.setattr(
            "fisheye.shared.zarr.schema.LocalStore",
            lambda *_args, **_kwargs: MemoryStore(),
        )
        # Avoid slow or GPU-dependent environment introspection in tests.
        monkeypatch.setattr(
            "fisheye.shared.zarr.schema.get_environment_summary",
            lambda: {
                "environment_type": "test",
                "environment_name": "pytest",
                "python_version": "3.x",
                "total_packages": 0,
                "key_packages": {},
            },
        )
        monkeypatch.setattr(
            "fisheye.shared.zarr.schema.get_platform_info",
            lambda **_kwargs: {"hostname": "test-host", "system": "test", "cpu_cores": 1},
        )
        monkeypatch.setattr(
            "fisheye.shared.zarr.schema.get_git_info",
            lambda: {"commit_hash": "test", "short_hash": "test", "branch": "test", "is_dirty": False},
        )
        monkeypatch.setattr(
            "fisheye.shared.zarr.schema.get_gpu_info",
            lambda: {"devices": [{"name": "test-gpu"}]},
        )
    
    @pytest.fixture
    def temp_zarr_path(self):
        """Create temporary directory for zarr store."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir)
    
    def test_create_palette_zarr(self, temp_zarr_path, monkeypatch):
        """Test basic zarr creation."""
        video_metadata = {
            'fps': 30.0,
            'width': 64,
            'height': 48,
            'total_frames': 10,
            'source_video': 'test.mp4'
        }
        
        config = {
            'import': {
                'downsample_size': [24, 32],
                'chunk_size': 4,
                'batch_size': 2
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
        assert root.attrs['width'] == 64
        assert root.attrs['height'] == 48
        assert root.attrs['total_frames'] == 10
        
        # Check group structure
        assert 'raw_video' in root
        assert 'pipeline_params' in root
        assert 'background_runs' in root
        assert 'detect_runs' in root
        assert 'refined_detect_runs' in root
        assert 'crop_runs' in root
        assert 'keypoints_runs' in root
        assert 'analysis_metadata' in root
        
        # Check arrays
        assert 'images_full' in root['raw_video']
        assert 'images_ds' in root['raw_video']
        assert 'timestamps' in root['raw_video']
        
        # Check array shapes
        images_full = root['raw_video']['images_full']
        assert images_full.shape == (10, 48, 64)
        
        images_ds = root['raw_video']['images_ds']
        assert images_ds.shape == (10, 24, 32)
    
    def test_get_run_group(self, temp_zarr_path):
        """Test run group creation."""
        video_metadata = {'fps': 30, 'width': 640, 'height': 480, 'total_frames': 10}
        config = {'import': {'downsample_size': [240, 320], 'chunk_size': 10}}
        
        root = create_palette_zarr(temp_zarr_path, video_metadata, config)
        
        # Create a new run group
        run_group, run_name = get_run_group(root, 'detect')
        assert 'detect_runs' in root
        assert run_name in root['detect_runs']
        assert root['detect_runs'].attrs.get('latest') is None
        assert root['detect_runs'].attrs['latest_pending'] == run_name
        assert run_group == root['detect_runs'][run_name]

        from fisheye.shared.zarr_run_completion import mark_run_complete

        mark_run_complete(
            run_group,
            parent_group=root["detect_runs"],
            run_name=run_name,
            run_provenance=build_writer_run_provenance(
                command="test_zarr_schema",
                params={"stage": "detect"},
            ),
        )

        # Getting latest run should return existing complete run when create_new=False
        run_group2, run_name2 = get_run_group(root, 'detect', create_new=False)
        assert run_name2 == run_name
        assert run_group2 == run_group

    def test_schema_refined_detect_run_attributes(self):
        """Test refined detect run attribute schema keys."""
        attrs = ZARR_SCHEMA["groups"]["refined_detect_runs"]["run_attributes"]
        expected = {
            "source_detect_run",
            "source_quality_run",
            "refinement_timestamp",
            "operations",
            "parameters",
            "coverage_comparison",
            "coverage_frames_total",
            "coverage_frame_source",
            "coverage_frames_full",
            "manual_review_latest",
            "detect_review_status",
            "retune_params",
        }
        assert expected.issubset(set(attrs.keys()))

        parent_attrs = ZARR_SCHEMA["groups"]["refined_detect_runs"]["parent_attributes"]
        assert "detect_review_status_latest" in parent_attrs
        legacy_description = parent_attrs["detect_review_status_latest"]
        assert "Historical" in legacy_description
        assert "no reader consults it" in legacy_description

    def test_schema_crop_run_attributes(self):
        """Test crop run attribute schema keys."""
        attrs = ZARR_SCHEMA["groups"]["crop_runs"]["run_attributes"]
        expected = {
            "detection_source_type",
            "detection_source_path",
            "detection_selection_policy",
            "detect_review_status",
            "detect_review_status_ref",
            "source_detect_run",
            "source_refined_run",
            "source_refined_row_ids_available",
            "source_refined_row_id_policy",
            "source_detect_row_index_available",
        }
        assert expected.issubset(set(attrs.keys()))

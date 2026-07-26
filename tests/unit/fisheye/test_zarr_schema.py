"""Unit tests for Zarr schema."""

from pathlib import Path
import sys
import zarr
from zarr.storage import MemoryStore

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.shared.zarr.schema import (
    get_run_group,
    ZARR_SCHEMA,
)
from fisheye.shared.run_provenance import build_writer_run_provenance

class TestZarrSchema:
    """Test Zarr schema creation and structure."""

    def test_get_run_group(self):
        """Test run group creation."""
        root = zarr.open_group(store=MemoryStore(), mode="w", zarr_format=3)
        
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

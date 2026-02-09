# Update tests/unit/fisheye/test_video_import.py
"""Unit tests for video import functionality."""

import pytest
import tempfile
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

# Only test if dependencies are available
cv2 = pytest.importorskip("cv2", reason="OpenCV not installed")
try:
    decord = pytest.importorskip("decord", reason="Decord not installed")
except Exception as exc:  # pragma: no cover - environment-specific shared lib issues
    pytest.skip(f"Decord import failed at runtime: {exc}", allow_module_level=True)
torch = pytest.importorskip("torch", reason="PyTorch not installed")

from fisheye.capture.import_video import (
    _process_video_cpu,
    _process_video_gpu
)

class TestVideoProcessing:
    """Test video processing functions."""
    
    def test_imports_work(self):
        """Test that imports are successful."""
        assert _process_video_cpu is not None
        assert _process_video_gpu is not None
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), 
                        reason="GPU not available")
    def test_gpu_available(self):
        """Test GPU detection."""
        assert torch.cuda.device_count() > 0

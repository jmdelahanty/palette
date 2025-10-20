"""Eye segmentation utilities."""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from skimage import measure


def _extract_contour(mask: np.ndarray, min_points: int = 5) -> Optional[np.ndarray]:
    """Extract contour from mask."""
    # Placeholder implementation
    return None


def _feret_mask_from_region(
    mask: np.ndarray, threshold: float
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """Extract Feret mask from region."""
    # Placeholder implementation
    return None, None

# src/fisheye/refinement/utils.py
"""
Shared utilities for refinement analysis across pipeline stages.
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class Gap:
    """Represents a gap in any binary coverage/presence data."""
    start_frame: int
    end_frame: int
    size: int
    
    @property
    def frames(self) -> List[int]:
        return list(range(self.start_frame, self.end_frame + 1))


def identify_gaps(presence_mask: np.ndarray, 
                 max_gap: Optional[int] = None) -> List[Gap]:
    """
    Identify gaps in any binary presence data.
    
    Args:
        presence_mask: Boolean or binary array where True/1 = present, False/0 = gap
        max_gap: Optional maximum gap size to return (for filtering)
        
    Returns:
        List of Gap objects
    """
    gaps = []
    
    # Find frames with presence
    present_frames = np.where(presence_mask)[0]
    
    if len(present_frames) < 2:
        return gaps
    
    # Find gaps between consecutive presences
    for i in range(1, len(present_frames)):
        gap_start = present_frames[i-1] + 1
        gap_end = present_frames[i] - 1
        gap_size = gap_end - gap_start + 1
        
        if gap_size > 0:
            if max_gap is None or gap_size <= max_gap:
                gaps.append(Gap(
                    start_frame=int(gap_start),
                    end_frame=int(gap_end),
                    size=int(gap_size)
                ))
    
    return gaps


def categorize_gaps(gaps: List[Gap]) -> Dict[str, int]:
    """
    Categorize gaps by size for reporting.
    
    Args:
        gaps: List of Gap objects
        
    Returns:
        Dictionary with gap counts by category
    """
    categories = {
        'very_small (1-5)': 0,
        'small (6-10)': 0,
        'medium (11-50)': 0,
        'large (51-200)': 0,
        'very_large (>200)': 0
    }
    
    for gap in gaps:
        if gap.size <= 5:
            categories['very_small (1-5)'] += 1
        elif gap.size <= 10:
            categories['small (6-10)'] += 1
        elif gap.size <= 50:
            categories['medium (11-50)'] += 1
        elif gap.size <= 200:
            categories['large (51-200)'] += 1
        else:
            categories['very_large (>200)'] += 1
    
    return categories


def calculate_coverage_stats(presence_mask: np.ndarray) -> Dict:
    """
    Calculate basic coverage statistics for any binary presence data.
    
    Args:
        presence_mask: Boolean or binary array where True/1 = present
        
    Returns:
        Dictionary with coverage statistics
    """
    total_frames = len(presence_mask)
    present_frames = np.sum(presence_mask)
    absent_frames = total_frames - present_frames
    coverage_percent = (present_frames / total_frames * 100) if total_frames > 0 else 0
    
    return {
        'total_frames': total_frames,
        'present_frames': int(present_frames),
        'absent_frames': int(absent_frames),
        'coverage_percent': float(coverage_percent)
    }
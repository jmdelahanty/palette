#!/usr/bin/env python3
"""
Zarr Frame Interpolator for YOLO Detection Files

This module adds interpolated detection data to existing YOLO zarr files,
maintaining all data in a single file with versioned interpolation runs.

The interpolated data is stored in a new group within the same zarr file:
/interpolation_runs/{timestamp}/
    ├── bboxes              # Interpolated bounding boxes
    ├── scores              # Interpolated scores
    ├── class_ids           # Class IDs
    ├── n_detections        # Number of detections per frame
    ├── interpolation_mask  # Boolean mask of interpolated frames
    └── gap_info/           # Gap statistics

/analysis/
    ├── distance_traveled   # Cumulative distance based on interpolated data
    └── velocity            # Frame-to-frame velocity
"""

import zarr
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import logging
from enum import Enum


class InterpolationMethod(Enum):
    """Supported interpolation methods."""
    LINEAR = "linear"
    NEAREST = "nearest"
    CUBIC = "cubic"
    NONE = "none"


@dataclass
class InterpolationMetadata:
    """Metadata for interpolation process."""
    created_at: str
    run_name: str
    interpolation_method: str
    total_frames: int
    original_detected_frames: int
    interpolated_frames: int
    gap_statistics: Dict[str, Any]
    parameters: Dict[str, Any]
    version: str = "1.0.0"


@dataclass
class FrameGap:
    """Information about a gap in the detection sequence."""
    start_frame: int
    end_frame: int
    gap_size: int
    prev_valid_frame: Optional[int]
    next_valid_frame: Optional[int]


class YOLOZarrFrameInterpolator:
    """
    Adds interpolated detections to existing YOLO zarr files.
    
    This class:
    - Detects gaps in YOLO detection data
    - Interpolates missing detections
    - Stores interpolated data in the same zarr file
    - Calculates derived metrics (distance, velocity)
    - Maintains version history of interpolation runs
    """
    
    def __init__(self, 
                 zarr_path: str,
                 method: InterpolationMethod = InterpolationMethod.LINEAR,
                 confidence_decay: float = 0.9,
                 run_name: Optional[str] = None,
                 verbose: bool = True):
        """
        Initialize the interpolator.
        
        Args:
            zarr_path: Path to YOLO detection zarr file
            method: Interpolation method to use
            confidence_decay: Factor to reduce confidence for interpolated frames (0-1)
            run_name: Optional name for this interpolation run
            verbose: Enable verbose logging
        """
        self.zarr_path = Path(zarr_path)
        self.method = method
        self.confidence_decay = confidence_decay
        self.verbose = verbose
        
        # Generate run name if not provided
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_name = f"interp_{self.method.value}_{timestamp}"
        else:
            self.run_name = run_name
        
        # Setup logging
        self._setup_logging()
        
        # Data containers
        self.root = None
        self.bboxes = None
        self.scores = None
        self.class_ids = None
        self.n_detections = None
        self.gaps: List[FrameGap] = []
        self.interpolation_mask = None
        self.metadata = None
        
    def _setup_logging(self):
        """Configure logging based on verbosity."""
        level = logging.INFO if self.verbose else logging.WARNING
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_data(self) -> bool:
        """
        Load data from the zarr file.
        
        Returns:
            True if data loaded successfully, False otherwise
        """
        try:
            self.logger.info(f"Loading zarr file: {self.zarr_path}")
            self.root = zarr.open(str(self.zarr_path), mode='r+')
            
            # Check for required datasets
            required_datasets = ['bboxes', 'scores', 'class_ids', 'n_detections']
            for dataset_name in required_datasets:
                if dataset_name not in self.root:
                    raise ValueError(f"Required dataset '{dataset_name}' not found in zarr file")
            
            # Load original detection data
            self.bboxes = self.root['bboxes'][:]
            self.scores = self.root['scores'][:]
            self.class_ids = self.root['class_ids'][:]
            self.n_detections = self.root['n_detections'][:]
            
            self.logger.info(f"Loaded detection data:")
            self.logger.info(f"  - Bboxes shape: {self.bboxes.shape}")
            self.logger.info(f"  - Total frames: {len(self.n_detections)}")
            self.logger.info(f"  - Max detections per frame: {self.bboxes.shape[1]}")
            
            # Get metadata
            self.fps = self.root.attrs.get('fps', 30.0)
            self.video_width = self.root.attrs.get('width', 1920)
            self.video_height = self.root.attrs.get('height', 1080)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            return False
    
    def detect_gaps(self) -> List[FrameGap]:
        """
        Detect gaps in the YOLO detection data.
        
        Returns:
            List of detected gaps
        """
        self.logger.info("Detecting gaps in detection data...")
        
        # Create validity mask (frame has valid detection if n_detections > 0)
        validity_mask = self.n_detections > 0
        
        # Find gaps (sequences of frames without detections)
        gaps = []
        in_gap = False
        gap_start = 0
        
        for i, is_valid in enumerate(validity_mask):
            if not is_valid and not in_gap:
                # Start of a gap
                in_gap = True
                gap_start = i
            elif is_valid and in_gap:
                # End of a gap
                in_gap = False
                gap_end = i - 1
                
                # Find previous and next valid frames
                prev_valid = gap_start - 1 if gap_start > 0 else None
                next_valid = i
                
                # Only add if we have valid neighbors for interpolation
                if prev_valid is not None and validity_mask[prev_valid]:
                    gaps.append(FrameGap(
                        start_frame=gap_start,
                        end_frame=gap_end,
                        gap_size=gap_end - gap_start + 1,
                        prev_valid_frame=prev_valid,
                        next_valid_frame=next_valid
                    ))
        
        # Handle gap that extends to the end
        if in_gap:
            if gap_start > 0 and validity_mask[gap_start - 1]:
                gaps.append(FrameGap(
                    start_frame=gap_start,
                    end_frame=len(validity_mask) - 1,
                    gap_size=len(validity_mask) - gap_start,
                    prev_valid_frame=gap_start - 1,
                    next_valid_frame=None
                ))
        
        self.gaps = gaps
        
        # Log gap statistics
        if gaps:
            total_gap_frames = sum(g.gap_size for g in gaps)
            self.logger.info(f"Found {len(gaps)} gaps totaling {total_gap_frames} frames")
            self.logger.info(f"Largest gap: {max(g.gap_size for g in gaps)} frames")
            self.logger.info(f"Average gap size: {total_gap_frames/len(gaps):.1f} frames")
        else:
            self.logger.info("No gaps detected - detection data is complete!")
        
        return gaps
    
    def interpolate_gaps(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Interpolate detection data for gaps.
        
        Returns:
            Tuple of (interpolated_bboxes, interpolated_scores, interpolated_class_ids, interpolated_n_detections)
        """
        if not self.gaps:
            self.logger.info("No gaps to interpolate")
            return self.bboxes.copy(), self.scores.copy(), self.class_ids.copy(), self.n_detections.copy()
        
        self.logger.info(f"Interpolating {len(self.gaps)} gaps using {self.method.value} method")
        
        # Create copies for interpolation
        interp_bboxes = self.bboxes.copy()
        interp_scores = self.scores.copy()
        interp_class_ids = self.class_ids.copy()
        interp_n_detections = self.n_detections.copy()
        
        # Create interpolation mask
        self.interpolation_mask = np.zeros(len(self.n_detections), dtype=bool)
        
        for gap in self.gaps:
            # Skip gaps without valid neighbors
            if gap.prev_valid_frame is None:
                self.logger.warning(f"Skipping gap at frames {gap.start_frame}-{gap.end_frame} (no previous frame)")
                continue
            
            # Mark frames as interpolated
            self.interpolation_mask[gap.start_frame:gap.end_frame + 1] = True
            
            # Get detections from boundary frames
            prev_n_dets = self.n_detections[gap.prev_valid_frame]
            
            if gap.next_valid_frame is not None:
                next_n_dets = self.n_detections[gap.next_valid_frame]
                # Use minimum number of detections for stability
                n_dets_to_interp = min(prev_n_dets, next_n_dets)
            else:
                # No next frame, just use previous
                n_dets_to_interp = prev_n_dets
            
            if n_dets_to_interp == 0:
                continue  # No detections to interpolate
            
            # Get boundary detection data
            prev_boxes = self.bboxes[gap.prev_valid_frame, :n_dets_to_interp]
            prev_scores = self.scores[gap.prev_valid_frame, :n_dets_to_interp]
            prev_classes = self.class_ids[gap.prev_valid_frame, :n_dets_to_interp]
            
            if gap.next_valid_frame is not None:
                next_boxes = self.bboxes[gap.next_valid_frame, :n_dets_to_interp]
                next_scores = self.scores[gap.next_valid_frame, :n_dets_to_interp]
                next_classes = self.class_ids[gap.next_valid_frame, :n_dets_to_interp]
            else:
                # No next frame, use previous as endpoint
                next_boxes = prev_boxes
                next_scores = prev_scores * (self.confidence_decay ** gap.gap_size)
                next_classes = prev_classes
            
            # Interpolate based on method
            if self.method == InterpolationMethod.LINEAR:
                # Linear interpolation for each frame in the gap
                for frame_offset in range(gap.gap_size):
                    frame_idx = gap.start_frame + frame_offset
                    
                    # Calculate interpolation weight
                    if gap.next_valid_frame is not None:
                        alpha = (frame_offset + 1) / (gap.gap_size + 1)
                    else:
                        alpha = 0  # Use previous frame values
                    
                    # Interpolate bounding boxes
                    interp_bboxes[frame_idx, :n_dets_to_interp] = \
                        (1 - alpha) * prev_boxes + alpha * next_boxes
                    
                    # Interpolate scores with decay
                    base_score = (1 - alpha) * prev_scores + alpha * next_scores
                    decay_factor = self.confidence_decay ** (min(frame_offset + 1, 
                                                                  gap.gap_size - frame_offset))
                    interp_scores[frame_idx, :n_dets_to_interp] = base_score * decay_factor
                    
                    # Use majority class (nearest neighbor for class IDs)
                    if alpha < 0.5:
                        interp_class_ids[frame_idx, :n_dets_to_interp] = prev_classes
                    else:
                        interp_class_ids[frame_idx, :n_dets_to_interp] = next_classes
                    
                    # Set number of detections
                    interp_n_detections[frame_idx] = n_dets_to_interp
            
            elif self.method == InterpolationMethod.NEAREST:
                # Use nearest valid frame
                for frame_offset in range(gap.gap_size):
                    frame_idx = gap.start_frame + frame_offset
                    
                    # Determine which frame is closer
                    if gap.next_valid_frame is None:
                        use_prev = True
                    else:
                        dist_to_prev = frame_offset + 1
                        dist_to_next = gap.gap_size - frame_offset
                        use_prev = dist_to_prev <= dist_to_next
                    
                    if use_prev:
                        interp_bboxes[frame_idx, :n_dets_to_interp] = prev_boxes
                        interp_scores[frame_idx, :n_dets_to_interp] = \
                            prev_scores * (self.confidence_decay ** (frame_offset + 1))
                        interp_class_ids[frame_idx, :n_dets_to_interp] = prev_classes
                    else:
                        interp_bboxes[frame_idx, :n_dets_to_interp] = next_boxes
                        interp_scores[frame_idx, :n_dets_to_interp] = \
                            next_scores * (self.confidence_decay ** (gap.gap_size - frame_offset))
                        interp_class_ids[frame_idx, :n_dets_to_interp] = next_classes
                    
                    interp_n_detections[frame_idx] = n_dets_to_interp
        
        self.logger.info(f"Interpolated {np.sum(self.interpolation_mask)} frames")
        return interp_bboxes, interp_scores, interp_class_ids, interp_n_detections
    
    def calculate_derived_metrics(self, bboxes: np.ndarray, n_detections: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate derived metrics from interpolated detection data.
        
        Args:
            bboxes: Interpolated bounding boxes
            n_detections: Number of detections per frame
            
        Returns:
            Dictionary of derived metrics
        """
        self.logger.info("Calculating derived metrics...")
        
        metrics = {}
        
        # Calculate center points for primary detection
        centers = np.full((len(bboxes), 2), np.nan)
        for i in range(len(bboxes)):
            if n_detections[i] > 0:
                bbox = bboxes[i, 0]  # First detection
                centers[i, 0] = (bbox[0] + bbox[2]) / 2  # center_x
                centers[i, 1] = (bbox[1] + bbox[3]) / 2  # center_y
        
        # Store center positions
        metrics['center_x'] = centers[:, 0]
        metrics['center_y'] = centers[:, 1]
        
        # Calculate frame-to-frame distances
        distances = np.full(len(centers) - 1, np.nan)
        valid_pairs = ~np.isnan(centers[:-1]).any(axis=1) & ~np.isnan(centers[1:]).any(axis=1)
        distances[valid_pairs] = np.sqrt(np.sum((centers[1:] - centers[:-1])[valid_pairs]**2, axis=1))
        
        # Pad to match frame count
        distances = np.concatenate([[0], distances])
        metrics['frame_distance'] = distances
        
        # Calculate cumulative distance
        metrics['cumulative_distance'] = np.nancumsum(distances)
        
        # Calculate velocity (pixels per second)
        velocity = distances * self.fps
        metrics['velocity'] = velocity
        
        # Calculate smoothed velocity (rolling average)
        window_size = int(self.fps / 2)  # Half-second window
        smoothed_velocity = np.convolve(np.nan_to_num(velocity), 
                                       np.ones(window_size)/window_size, 
                                       mode='same')
        metrics['velocity_smoothed'] = smoothed_velocity
        
        # Calculate acceleration
        acceleration = np.gradient(np.nan_to_num(velocity)) * self.fps
        metrics['acceleration'] = acceleration
        
        # Calculate detection confidence (mean confidence per frame)
        mean_confidence = np.full(len(n_detections), np.nan)
        for i in range(len(n_detections)):
            if n_detections[i] > 0:
                mean_confidence[i] = np.mean(self.scores[i, :n_detections[i]])
        metrics['mean_confidence'] = mean_confidence
        
        # Log summary statistics
        total_distance = np.nanmax(metrics['cumulative_distance'])
        mean_velocity = np.nanmean(velocity)
        max_velocity = np.nanmax(velocity)
        
        self.logger.info(f"  Total distance: {total_distance:.2f} pixels")
        self.logger.info(f"  Mean velocity: {mean_velocity:.2f} pixels/sec")
        self.logger.info(f"  Max velocity: {max_velocity:.2f} pixels/sec")
        
        return metrics
    
    def save_to_zarr(self, interp_bboxes: np.ndarray, interp_scores: np.ndarray,
                     interp_class_ids: np.ndarray, interp_n_detections: np.ndarray,
                     metrics: Dict[str, np.ndarray]):
        """
        Save interpolated data and metrics to the zarr file.
        """
        self.logger.info(f"Saving interpolation run: {self.run_name}")
        
        # Create interpolation_runs group if it doesn't exist
        if 'interpolation_runs' not in self.root:
            interp_runs_group = self.root.create_group('interpolation_runs')
            interp_runs_group.attrs['created_at'] = datetime.now().isoformat()
            interp_runs_group.attrs['description'] = "Interpolated detection data from gap filling"
        else:
            interp_runs_group = self.root['interpolation_runs']
        
        # Create group for this interpolation run
        run_group = interp_runs_group.create_group(self.run_name)
        run_group.attrs['created_at'] = datetime.now().isoformat()
        run_group.attrs['method'] = self.method.value
        run_group.attrs['confidence_decay'] = self.confidence_decay
        
        # Store interpolated detection data
        run_group.create_dataset(
            'bboxes',
            data=interp_bboxes,
            chunks=self.root['bboxes'].chunks,
            dtype='float32'
        )
        
        run_group.create_dataset(
            'scores',
            data=interp_scores,
            chunks=self.root['scores'].chunks,
            dtype='float32'
        )
        
        run_group.create_dataset(
            'class_ids',
            data=interp_class_ids,
            chunks=self.root['class_ids'].chunks,
            dtype='int32'
        )
        
        run_group.create_dataset(
            'n_detections',
            data=interp_n_detections,
            chunks=self.root['n_detections'].chunks,
            dtype='int32'
        )
        
        # Store interpolation mask
        mask_dataset = run_group.create_dataset(
            'interpolation_mask',
            data=self.interpolation_mask,
            chunks=(min(10000, len(self.interpolation_mask)),),
            dtype=bool
        )
        mask_dataset.attrs['description'] = "True for interpolated frames, False for original detections"
        mask_dataset.attrs['total_interpolated'] = int(np.sum(self.interpolation_mask))
        
        # Store gap information
        gap_info_group = run_group.create_group('gap_info')
        gap_info_group.attrs['total_gaps'] = len(self.gaps)
        gap_info_group.attrs['gaps'] = [asdict(gap) for gap in self.gaps]
        
        if self.gaps:
            gap_starts = np.array([g.start_frame for g in self.gaps], dtype='int32')
            gap_ends = np.array([g.end_frame for g in self.gaps], dtype='int32')
            gap_sizes = np.array([g.gap_size for g in self.gaps], dtype='int32')
            
            gap_info_group.create_dataset('gap_starts', data=gap_starts)
            gap_info_group.create_dataset('gap_ends', data=gap_ends)
            gap_info_group.create_dataset('gap_sizes', data=gap_sizes)
        
        # Create analysis group if it doesn't exist
        if 'analysis' not in self.root:
            analysis_group = self.root.create_group('analysis')
            analysis_group.attrs['created_at'] = datetime.now().isoformat()
        else:
            analysis_group = self.root['analysis']
        
        # Create subgroup for this run's analysis
        run_analysis = analysis_group.create_group(self.run_name)
        run_analysis.attrs['created_at'] = datetime.now().isoformat()
        run_analysis.attrs['source_interpolation'] = self.run_name
        
        # Store derived metrics
        for metric_name, metric_data in metrics.items():
            dataset = run_analysis.create_dataset(
                metric_name,
                data=metric_data,
                chunks=(min(10000, len(metric_data)),),
                dtype='float32'
            )
            
            # Add descriptive attributes
            if metric_name == 'cumulative_distance':
                dataset.attrs['units'] = 'pixels'
                dataset.attrs['description'] = 'Cumulative distance traveled'
            elif metric_name == 'velocity':
                dataset.attrs['units'] = 'pixels/second'
                dataset.attrs['description'] = 'Frame-to-frame velocity'
            elif metric_name == 'acceleration':
                dataset.attrs['units'] = 'pixels/second^2'
                dataset.attrs['description'] = 'Frame-to-frame acceleration'
        
        # Update latest interpolation run
        interp_runs_group.attrs['latest'] = self.run_name
        
        # Store metadata
        self._create_metadata(interp_n_detections)
        run_group.attrs['metadata'] = json.dumps(asdict(self.metadata))
        
        self.logger.info(f"✅ Saved interpolation run and analysis to: {self.zarr_path}")
    
    def _create_metadata(self, interp_n_detections: np.ndarray):
        """
        Create comprehensive metadata for the interpolation process.
        """
        # Calculate statistics
        gap_statistics = {
            'total_gaps': len(self.gaps),
            'total_gap_frames': sum(g.gap_size for g in self.gaps) if self.gaps else 0,
            'largest_gap': max(g.gap_size for g in self.gaps) if self.gaps else 0,
            'smallest_gap': min(g.gap_size for g in self.gaps) if self.gaps else 0,
            'average_gap_size': sum(g.gap_size for g in self.gaps) / len(self.gaps) if self.gaps else 0
        }
        
        # Count frames with detections
        original_detected = np.sum(self.n_detections > 0)
        interpolated_detected = np.sum(interp_n_detections > 0)
        
        self.metadata = InterpolationMetadata(
            created_at=datetime.now().isoformat(),
            run_name=self.run_name,
            interpolation_method=self.method.value,
            total_frames=len(interp_n_detections),
            original_detected_frames=int(original_detected),
            interpolated_frames=int(np.sum(self.interpolation_mask)),
            gap_statistics=gap_statistics,
            parameters={
                'method': self.method.value,
                'confidence_decay': self.confidence_decay,
                'fps': self.fps,
                'video_width': self.video_width,
                'video_height': self.video_height
            }
        )
    
    def list_existing_runs(self) -> List[str]:
        """
        List existing interpolation runs in the zarr file.
        
        Returns:
            List of run names
        """
        if 'interpolation_runs' not in self.root:
            return []
        
        runs = list(self.root['interpolation_runs'].keys())
        return runs
    
    def generate_report(self) -> str:
        """
        Generate a detailed report of the interpolation process.
        
        Returns:
            Report text
        """
        report = []
        report.append("=" * 80)
        report.append("YOLO ZARR FRAME INTERPOLATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Run name: {self.run_name}")
        report.append("")
        
        # File Information
        report.append("FILE INFORMATION")
        report.append("-" * 40)
        report.append(f"Zarr file: {self.zarr_path}")
        
        # Check for existing runs
        existing_runs = self.list_existing_runs()
        if existing_runs:
            report.append(f"Existing interpolation runs: {len(existing_runs)}")
            for run in existing_runs[-5:]:  # Show last 5 runs
                report.append(f"  - {run}")
            if len(existing_runs) > 5:
                report.append(f"  ... and {len(existing_runs) - 5} more")
        report.append("")
        
        # Data Information
        report.append("DATA INFORMATION")
        report.append("-" * 40)
        if self.n_detections is not None:
            report.append(f"Total frames: {len(self.n_detections)}")
            report.append(f"Max detections per frame: {self.bboxes.shape[1]}")
            report.append(f"Video dimensions: {self.video_width}x{self.video_height}")
            report.append(f"FPS: {self.fps}")
            report.append(f"Interpolation method: {self.method.value}")
            report.append(f"Confidence decay: {self.confidence_decay}")
            report.append("")
        
        # Detection Statistics
        report.append("DETECTION STATISTICS")
        report.append("-" * 40)
        if self.n_detections is not None:
            frames_with_dets = np.sum(self.n_detections > 0)
            total_dets = np.sum(self.n_detections)
            avg_dets = total_dets / len(self.n_detections) if len(self.n_detections) > 0 else 0
            
            report.append(f"Original frames with detections: {frames_with_dets}/{len(self.n_detections)} "
                         f"({frames_with_dets/len(self.n_detections)*100:.1f}%)")
            report.append(f"Total detections: {total_dets}")
            report.append(f"Average detections per frame: {avg_dets:.2f}")
            report.append("")
        
        # Gap Analysis
        report.append("GAP ANALYSIS")
        report.append("-" * 40)
        if self.gaps:
            report.append(f"Total gaps found: {len(self.gaps)}")
            report.append(f"Total gap frames: {sum(g.gap_size for g in self.gaps)}")
            report.append(f"Largest gap: {max(g.gap_size for g in self.gaps)} frames")
            report.append(f"Smallest gap: {min(g.gap_size for g in self.gaps)} frames")
            report.append(f"Average gap size: {sum(g.gap_size for g in self.gaps)/len(self.gaps):.1f} frames")
            report.append("")
            
            # Show first few gaps
            report.append("First 10 gaps:")
            for i, gap in enumerate(self.gaps[:10]):
                report.append(f"  Gap {i+1}: Frames {gap.start_frame}-{gap.end_frame} ({gap.gap_size} frames)")
            if len(self.gaps) > 10:
                report.append(f"  ... and {len(self.gaps) - 10} more gaps")
        else:
            report.append("No gaps detected - detection data is complete!")
        report.append("")
        
        # Interpolation Results
        if self.metadata:
            report.append("INTERPOLATION RESULTS")
            report.append("-" * 40)
            report.append(f"Original frames with detections: {self.metadata.original_detected_frames}")
            report.append(f"Frames interpolated: {self.metadata.interpolated_frames}")
            
            if self.metadata.total_frames > 0:
                orig_coverage = self.metadata.original_detected_frames / self.metadata.total_frames * 100
                new_coverage = (self.metadata.original_detected_frames + self.metadata.interpolated_frames) / \
                              self.metadata.total_frames * 100
                report.append(f"Detection coverage: {orig_coverage:.1f}% → {new_coverage:.1f}%")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def run(self) -> bool:
        """
        Execute the complete interpolation pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        self.logger.info("🚀 Starting YOLO zarr frame interpolation pipeline...")
        
        try:
            # Step 1: Load data
            if not self.load_data():
                return False
            
            # Step 2: Detect gaps
            gaps = self.detect_gaps()
            
            if not gaps:
                self.logger.info("✨ No gaps detected - detection data is complete!")
                return True
            
            # Step 3: Interpolate gaps
            interp_bboxes, interp_scores, interp_class_ids, interp_n_detections = self.interpolate_gaps()
            
            # Step 4: Calculate derived metrics
            metrics = self.calculate_derived_metrics(interp_bboxes, interp_n_detections)
            
            # Step 5: Save to zarr
            self.save_to_zarr(interp_bboxes, interp_scores, interp_class_ids, 
                            interp_n_detections, metrics)
            
            # Step 6: Generate report
            print("\n" + self.generate_report())
            
            self.logger.info("✨ Interpolation pipeline completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Add interpolated detections and analysis to YOLO zarr files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s detections.zarr
  %(prog)s detections.zarr --method linear --confidence-decay 0.8
  %(prog)s detections.zarr --run-name my_interpolation_v2
  %(prog)s detections.zarr --list-runs
  %(prog)s detections.zarr --dry-run
        """
    )
    
    parser.add_argument(
        'zarr_path',
        help='Path to YOLO detection zarr file'
    )
    parser.add_argument(
        '-m', '--method',
        choices=['linear', 'nearest', 'cubic', 'none'],
        default='linear',
        help='Interpolation method (default: linear)'
    )
    parser.add_argument(
        '--confidence-decay',
        type=float,
        default=0.9,
        help='Confidence decay factor for interpolated frames (0-1, default: 0.9)'
    )
    parser.add_argument(
        '--run-name',
        help='Name for this interpolation run (default: auto-generated)'
    )
    parser.add_argument(
        '--list-runs',
        action='store_true',
        help='List existing interpolation runs and exit'
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Analyze gaps without saving'
    )
    
    args = parser.parse_args()
    
    # Validate confidence decay
    if not 0 < args.confidence_decay <= 1:
        parser.error("Confidence decay must be between 0 and 1")
    
    # Create interpolator
    method = InterpolationMethod(args.method)
    interpolator = YOLOZarrFrameInterpolator(
        args.zarr_path,
        method=method,
        confidence_decay=args.confidence_decay,
        run_name=args.run_name,
        verbose=not args.quiet
    )
    
    # List runs if requested
    if args.list_runs:
        interpolator.load_data()
        runs = interpolator.list_existing_runs()
        if runs:
            print(f"\nExisting interpolation runs in {args.zarr_path}:")
            for run in runs:
                run_group = interpolator.root['interpolation_runs'][run]
                created = run_group.attrs.get('created_at', 'Unknown')
                method = run_group.attrs.get('method', 'Unknown')
                print(f"  - {run}")
                print(f"      Created: {created}")
                print(f"      Method: {method}")
        else:
            print(f"\nNo interpolation runs found in {args.zarr_path}")
        return 0
    
    # Run interpolation
    if args.dry_run:
        # Just analyze gaps
        interpolator.load_data()
        interpolator.detect_gaps()
        print("\n" + interpolator.generate_report())
        success = True
    else:
        success = interpolator.run()
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
#!/usr/bin/env python3
"""
Interpolation BBox Saver

Creates and saves actual bounding boxes for interpolated detections.
Handles partial ROI interpolation and merges with existing detections.
"""

import zarr
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json
from rich.console import Console
from rich.progress import track

console = Console()


class InterpolationBBoxSaver:
    """Save interpolated bounding boxes to zarr in a way that can be visualized."""
    
    def __init__(self, zarr_path: str, verbose: bool = True):
        self.zarr_path = Path(zarr_path)
        self.verbose = verbose
        self.root = zarr.open_group(self.zarr_path, mode='r+')
        
    def load_interpolation_data(self) -> Tuple[Optional[Dict], Optional[np.ndarray]]:
        """Load existing interpolation runs and masks."""
        if 'interpolation_runs' not in self.root:
            console.print("[yellow]No interpolation runs found[/yellow]")
            return None, None
            
        interp_group = self.root['interpolation_runs']
        if 'latest' not in interp_group.attrs:
            console.print("[yellow]No latest interpolation run[/yellow]")
            return None, None
            
        latest_run = interp_group.attrs['latest']
        run_group = interp_group[latest_run]
        
        # Load masks
        masks = run_group['interpolation_masks'][:]
        
        # Load statistics
        roi_stats = json.loads(run_group.attrs.get('roi_statistics', '{}'))
        processed_rois = run_group.attrs.get('processed_rois', [])
        
        if self.verbose:
            console.print(f"[green]Loaded interpolation from:[/green] {latest_run}")
            console.print(f"  Processed ROIs: {processed_rois}")
            
        return {'run_name': latest_run, 'roi_stats': roi_stats, 'processed_rois': processed_rois}, masks
    
    def create_interpolated_bboxes(self, roi_id: int, interpolation_mask: np.ndarray) -> Dict:
        """
        Create interpolated bounding boxes for a specific ROI.
        
        Returns:
            Dict with frame_idx -> bbox mapping
        """
        # Load detection data
        detect_group = self.root['detect_runs']
        latest_detect = detect_group.attrs['latest']
        n_detections = detect_group[latest_detect]['n_detections'][:]
        bbox_coords = detect_group[latest_detect]['bbox_norm_coords'][:]
        
        # Load ID assignments
        id_key = 'id_assignments_runs' if 'id_assignments_runs' in self.root else 'id_assignments'
        id_group = self.root[id_key]
        latest_id = id_group.attrs['latest']
        detection_ids = id_group[latest_id]['detection_ids'][:]
        n_detections_per_roi = id_group[latest_id]['n_detections_per_roi'][:]
        
        # Extract existing positions for this ROI
        existing_positions = {}
        cumulative_idx = 0
        
        for frame_idx in range(len(n_detections)):
            frame_det_count = int(n_detections[frame_idx])
            
            if frame_det_count > 0 and n_detections_per_roi[frame_idx, roi_id] > 0:
                frame_detection_ids = detection_ids[cumulative_idx:cumulative_idx + frame_det_count]
                roi_mask = frame_detection_ids == roi_id
                
                if np.any(roi_mask):
                    roi_idx = np.where(roi_mask)[0][0]
                    bbox = bbox_coords[cumulative_idx + roi_idx]
                    existing_positions[frame_idx] = bbox
            
            cumulative_idx += frame_det_count
        
        if self.verbose:
            console.print(f"ROI {roi_id}: Found {len(existing_positions)} existing detections")
        
        # Create interpolated bboxes
        interpolated_bboxes = {}
        frames_to_interpolate = np.where(interpolation_mask)[0]
        
        for frame_idx in frames_to_interpolate:
            # Find nearest real detections before and after
            before_frame = None
            after_frame = None
            
            # Search backwards for before frame
            for f in range(frame_idx - 1, -1, -1):
                if f in existing_positions:
                    before_frame = f
                    break
            
            # Search forwards for after frame
            for f in range(frame_idx + 1, len(interpolation_mask)):
                if f in existing_positions:
                    after_frame = f
                    break
            
            # Interpolate if we have both bounds
            if before_frame is not None and after_frame is not None:
                bbox_before = existing_positions[before_frame]
                bbox_after = existing_positions[after_frame]
                
                # Linear interpolation
                t = (frame_idx - before_frame) / (after_frame - before_frame)
                interpolated_bbox = bbox_before * (1 - t) + bbox_after * t
                
                interpolated_bboxes[frame_idx] = interpolated_bbox
        
        if self.verbose:
            console.print(f"ROI {roi_id}: Created {len(interpolated_bboxes)} interpolated bboxes")
        
        return interpolated_bboxes
    
    def save_interpolated_detections(self):
        """Save interpolated bounding boxes to zarr."""
        # Load interpolation data
        interp_info, masks = self.load_interpolation_data()
        if interp_info is None:
            console.print("[red]No interpolation data to process[/red]")
            return
        
        # Create storage for interpolated detections
        if 'interpolated_detections' not in self.root:
            interp_det_group = self.root.create_group('interpolated_detections')
        else:
            interp_det_group = self.root['interpolated_detections']
        
        # Generate timestamp for this save
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        save_name = f'interpolated_{timestamp}'
        
        # Create save group
        save_group = interp_det_group.create_group(save_name)
        
        # Process each ROI that was interpolated
        all_interpolated_bboxes = {}
        for roi_id in interp_info['processed_rois']:
            roi_mask = masks[roi_id]
            
            # Create interpolated bboxes for this ROI
            interpolated_bboxes = self.create_interpolated_bboxes(roi_id, roi_mask)
            
            # Store for saving
            for frame_idx, bbox in interpolated_bboxes.items():
                if frame_idx not in all_interpolated_bboxes:
                    all_interpolated_bboxes[frame_idx] = {}
                all_interpolated_bboxes[frame_idx][roi_id] = bbox
        
        # Convert to arrays for storage
        # We'll create a sparse representation
        total_interpolated = sum(len(rois) for rois in all_interpolated_bboxes.values())
        
        # Create arrays
        interp_frames = []
        interp_roi_ids = []
        interp_bboxes = []
        
        for frame_idx in sorted(all_interpolated_bboxes.keys()):
            for roi_id, bbox in all_interpolated_bboxes[frame_idx].items():
                interp_frames.append(frame_idx)
                interp_roi_ids.append(roi_id)
                interp_bboxes.append(bbox)
        
        # Save as zarr arrays
        save_group.create_dataset('frame_indices', data=np.array(interp_frames, dtype=np.int32))
        save_group.create_dataset('roi_ids', data=np.array(interp_roi_ids, dtype=np.int32))
        save_group.create_dataset('bboxes', data=np.array(interp_bboxes, dtype=np.float32))
        
        # Save metadata
        save_group.attrs.update({
            'created_at': datetime.now().isoformat(),
            'source_interpolation_run': interp_info['run_name'],
            'total_interpolated_detections': total_interpolated,
            'processed_rois': interp_info['processed_rois'],
            'roi_statistics': json.dumps(interp_info['roi_stats'])
        })
        
        # Update latest pointer
        interp_det_group.attrs['latest'] = save_name
        
        if self.verbose:
            console.print(f"\n[green]✓ Saved interpolated detections:[/green] {save_name}")
            console.print(f"  Total interpolated detections: {total_interpolated}")
            console.print(f"  Frames with interpolations: {len(all_interpolated_bboxes)}")
            console.print(f"  ROIs processed: {interp_info['processed_rois']}")
        
        return save_name
    
    def create_merged_detection_view(self):
        """
        Create a merged view combining original and interpolated detections.
        This creates new arrays that include both real and interpolated data.
        """
        console.print("\n[bold cyan]Creating Merged Detection View[/bold cyan]")
        
        # Load original detections
        detect_group = self.root['detect_runs']
        latest_detect = detect_group.attrs['latest']
        orig_n_detections = detect_group[latest_detect]['n_detections'][:]
        orig_bbox_coords = detect_group[latest_detect]['bbox_norm_coords'][:]
        
        # Load ID assignments
        id_key = 'id_assignments_runs' if 'id_assignments_runs' in self.root else 'id_assignments'
        id_group = self.root[id_key]
        latest_id = id_group.attrs['latest']
        orig_detection_ids = id_group[latest_id]['detection_ids'][:]
        
        # Load interpolated detections
        if 'interpolated_detections' not in self.root:
            console.print("[yellow]No interpolated detections found[/yellow]")
            return
        
        interp_det_group = self.root['interpolated_detections']
        latest_interp = interp_det_group.attrs['latest']
        interp_group = interp_det_group[latest_interp]
        
        interp_frames = interp_group['frame_indices'][:]
        interp_roi_ids = interp_group['roi_ids'][:]
        interp_bboxes = interp_group['bboxes'][:]
        
        # Build frame -> interpolations mapping
        frame_interpolations = {}
        for i, frame_idx in enumerate(interp_frames):
            if frame_idx not in frame_interpolations:
                frame_interpolations[frame_idx] = []
            frame_interpolations[frame_idx].append({
                'roi_id': interp_roi_ids[i],
                'bbox': interp_bboxes[i]
            })
        
        # Create merged arrays
        merged_bboxes = []
        merged_ids = []
        merged_is_interpolated = []
        merged_n_detections = []
        
        cumulative_idx = 0
        for frame_idx in range(len(orig_n_detections)):
            frame_det_count = int(orig_n_detections[frame_idx])
            frame_bboxes = []
            frame_ids = []
            frame_is_interp = []
            
            # Add original detections
            if frame_det_count > 0:
                for i in range(frame_det_count):
                    frame_bboxes.append(orig_bbox_coords[cumulative_idx + i])
                    frame_ids.append(orig_detection_ids[cumulative_idx + i])
                    frame_is_interp.append(False)
                cumulative_idx += frame_det_count
            
            # Add interpolated detections for this frame
            if frame_idx in frame_interpolations:
                for interp in frame_interpolations[frame_idx]:
                    # Check if this ROI already has a real detection
                    if interp['roi_id'] not in frame_ids:
                        frame_bboxes.append(interp['bbox'])
                        frame_ids.append(interp['roi_id'])
                        frame_is_interp.append(True)
            
            # Store frame data
            if frame_bboxes:
                merged_bboxes.extend(frame_bboxes)
                merged_ids.extend(frame_ids)
                merged_is_interpolated.extend(frame_is_interp)
            merged_n_detections.append(len(frame_bboxes))
        
        # Save merged view
        if 'merged_detections' not in self.root:
            merged_group = self.root.create_group('merged_detections')
        else:
            merged_group = self.root['merged_detections']
        
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        run_name = f'merged_{timestamp}'
        run_group = merged_group.create_group(run_name)
        
        # Save arrays
        run_group.create_dataset('bbox_coords', data=np.array(merged_bboxes, dtype=np.float32))
        run_group.create_dataset('detection_ids', data=np.array(merged_ids, dtype=np.int32))
        run_group.create_dataset('is_interpolated', data=np.array(merged_is_interpolated, dtype=bool))
        run_group.create_dataset('n_detections', data=np.array(merged_n_detections, dtype=np.int32))
        
        # Save metadata
        run_group.attrs.update({
            'created_at': datetime.now().isoformat(),
            'source_detection_run': f"{id_key}/{latest_id}",
            'source_interpolation_run': f"interpolated_detections/{latest_interp}",
            'total_detections': len(merged_bboxes),
            'total_interpolated': sum(merged_is_interpolated),
            'total_original': sum(~np.array(merged_is_interpolated))
        })
        
        # Update latest
        merged_group.attrs['latest'] = run_name
        
        if self.verbose:
            console.print(f"\n[green]✓ Created merged detection view:[/green] {run_name}")
            console.print(f"  Total detections: {len(merged_bboxes)}")
            console.print(f"  Original: {sum(~np.array(merged_is_interpolated))}")
            console.print(f"  Interpolated: {sum(merged_is_interpolated)}")
        
        return run_name


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Save interpolated bounding boxes for visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('zarr_path', help='Path to zarr file')
    parser.add_argument('--merge', action='store_true',
                       help='Also create merged detection view')
    
    args = parser.parse_args()
    
    # Create saver
    saver = InterpolationBBoxSaver(args.zarr_path)
    
    # Save interpolated detections
    save_name = saver.save_interpolated_detections()
    
    # Create merged view if requested
    if args.merge and save_name:
        saver.create_merged_detection_view()


if __name__ == "__main__":
    main()
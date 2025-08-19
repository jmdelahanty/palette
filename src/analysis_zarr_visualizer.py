#!/usr/bin/env python3
"""
Visualizer for analysis H5 files and detection Zarr outputs.
Combines fast video decoding with Decord and GPU-accelerated rendering with Vispy.
Supports both H5 analysis files and Zarr detection formats.
"""

import argparse
from pathlib import Path
import numpy as np
import zarr
import h5py
from decord import VideoReader, cpu, gpu
from vispy import app, scene
from vispy.scene import visuals
import time
import os
import colorsys

# Set Decord EOF retry limit for difficult videos
os.environ['DECORD_EOF_RETRY_MAX'] = '20480'

class AnalysisDetectionVisualizer:
    def __init__(self, video_path=None, h5_path=None, zarr_path=None, 
                 device='cpu', target_fps=30):
        """
        Initialize visualizer for analysis H5 files and/or Zarr detections.
        
        Args:
            video_path: Path to video file (optional if H5 contains video)
            h5_path: Path to analysis H5 file
            zarr_path: Path to detection Zarr file
            device: 'cpu' or 'gpu' for video decoding
            target_fps: Target playback FPS
        """
        self.h5_path = h5_path
        self.zarr_path = zarr_path
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        
        # Initialize data containers
        self.h5_data = {}
        self.zarr_data = {}
        self.video_data = {}
        
        # Load H5 data if provided
        if h5_path:
            self._load_h5_data()
        
        # Load Zarr data if provided
        if zarr_path:
            self._load_zarr_data()
        
        # Load video (from path or H5)
        self._load_video(video_path, device)
        
        # Generate colors for different classes/sources
        self._generate_colors()
        
        # Playback state
        self.current_frame = 0
        self.is_playing = False
        self.last_frame_time = time.time()
        self.show_h5 = True
        self.show_zarr = True
        self.show_overlay = True
        
        # Setup visualization
        self._setup_canvas()
    
    def _load_h5_data(self):
        """Load data from analysis H5 file."""
        print(f"Loading H5 analysis file: {self.h5_path}")
        
        with h5py.File(self.h5_path, 'r') as hf:
            # Check for different possible data locations
            if '/tracking_data/bounding_boxes' in hf:
                # Standard tracking format
                bbox_dataset = hf['/tracking_data/bounding_boxes']
                frame_ids = bbox_dataset['payload_frame_id'][:]
                x_min = bbox_dataset['x_min'][:]
                y_min = bbox_dataset['y_min'][:]
                width = bbox_dataset['width'][:]
                height = bbox_dataset['height'][:]
                
                # Organize by frame
                bbox_map = {}
                for i, frame_id in enumerate(frame_ids):
                    if frame_id not in bbox_map:
                        bbox_map[frame_id] = []
                    x1, y1 = x_min[i], y_min[i]
                    x2, y2 = x1 + width[i], y1 + height[i]
                    bbox_map[frame_id].append([x1, y1, x2, y2])
                
                self.h5_data['bboxes'] = bbox_map
                print(f"  Loaded {len(frame_ids)} H5 detections")
            
            # Check for video data in H5
            if '/video_data/frames' in hf:
                self.h5_data['has_video'] = True
                self.h5_data['video_shape'] = hf['/video_data/frames'].shape
                print(f"  Found video data: {self.h5_data['video_shape']}")
            
            # Load metadata if available
            if '/video_metadata' in hf:
                metadata = hf['/video_metadata']
                if 'fps' in metadata.attrs:
                    self.h5_data['fps'] = metadata.attrs['fps']
                if 'width' in metadata.attrs:
                    self.h5_data['width'] = metadata.attrs['width']
                if 'height' in metadata.attrs:
                    self.h5_data['height'] = metadata.attrs['height']
    
    def _load_zarr_data(self):
        """Load data from Zarr detection file."""
        print(f"Loading Zarr detection file: {self.zarr_path}")
        
        zarr_root = zarr.open(self.zarr_path, mode='r')
        
        # Check for standard detection format
        if 'bboxes' in zarr_root:
            self.zarr_data['bboxes'] = zarr_root['bboxes'][:]
            self.zarr_data['scores'] = zarr_root['scores'][:]
            self.zarr_data['n_detections'] = zarr_root['n_detections'][:]
            
            # Load class IDs if available
            if 'class_ids' in zarr_root:
                self.zarr_data['class_ids'] = zarr_root['class_ids'][:]
            
            print(f"  Loaded {self.zarr_data['bboxes'].shape[0]} frames of Zarr detections")
        
        # Check for alternative formats (detect_runs structure)
        elif 'detect_runs' in zarr_root:
            latest_run = zarr_root['detect_runs'].attrs.get('latest')
            if latest_run:
                detect_group = zarr_root[f'detect_runs/{latest_run}']
                self.zarr_data['bboxes'] = detect_group['bbox_norm_coords'][:]
                self.zarr_data['n_detections'] = detect_group['n_detections'][:]
                print(f"  Loaded detect_runs/{latest_run} format")
        
        # Load metadata
        if zarr_root.attrs:
            self.zarr_data['metadata'] = dict(zarr_root.attrs)
    
    def _load_video(self, video_path, device):
        """Load video from file or H5."""
        if video_path:
            # Load from video file
            ctx = cpu() if device == 'cpu' else gpu()
            print(f"Loading video with Decord on {device.upper()}: {video_path}")
            self.vr = VideoReader(str(video_path), ctx=ctx)
            self.num_frames = len(self.vr)
            self.video_fps = self.vr.get_avg_fps()
            initial_frame = self.vr[0].asnumpy()
            self.video_height, self.video_width = initial_frame.shape[:2]
            self.video_source = 'file'
            
        elif self.h5_data.get('has_video'):
            # Load from H5 file
            print("Loading video from H5 file")
            with h5py.File(self.h5_path, 'r') as hf:
                video_data = hf['/video_data/frames']
                self.num_frames = video_data.shape[0]
                self.video_height, self.video_width = video_data.shape[1:3]
                self.video_fps = self.h5_data.get('fps', 30.0)
                # We'll load frames on demand
                self.video_source = 'h5'
        else:
            raise ValueError("No video source provided (neither video_path nor H5 contains video)")
        
        print(f"Video: {self.num_frames} frames, {self.video_width}x{self.video_height} @ {self.video_fps:.2f} FPS")
    
    def _generate_colors(self):
        """Generate distinct colors for visualization."""
        # Generate colors for different detection sources
        self.colors = {
            'h5': (0, 1, 0, 1),      # Green for H5
            'zarr': (0, 1, 1, 1),    # Cyan for Zarr
            'overlay': (1, 1, 0, 1)  # Yellow for overlay
        }
        
        # Generate class colors if we have class IDs
        if 'class_ids' in self.zarr_data:
            unique_classes = np.unique(self.zarr_data['class_ids'])
            n_classes = len(unique_classes)
            self.class_colors = {}
            for i, class_id in enumerate(unique_classes):
                hue = i / n_classes
                rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
                self.class_colors[class_id] = (*rgb, 1.0)
    
    def _setup_canvas(self):
        """Setup Vispy canvas with GPU-accelerated rendering."""
        self.canvas = scene.SceneCanvas(
            title="Analysis H5 & Zarr Detection Visualizer",
            size=(self.video_width * 1.5, self.video_height * 1.2),
            keys='interactive',
            bgcolor='black'
        )
        self.canvas.show()
        
        # Create viewbox
        self.view = self.canvas.central_widget.add_view()
        
        # Add video display
        self.image = visuals.Image(
            np.zeros((self.video_height, self.video_width, 3), dtype=np.uint8),
            interpolation='nearest',
            parent=self.view.scene
        )
        
        # Add line visuals for bounding boxes
        self.h5_lines = visuals.Line(
            pos=np.zeros((0, 2)),
            color=self.colors['h5'],
            width=2,
            connect='segments',
            parent=self.view.scene
        )
        
        self.zarr_lines = visuals.Line(
            pos=np.zeros((0, 2)),
            color=self.colors['zarr'],
            width=2,
            connect='segments',
            parent=self.view.scene
        )
        
        # Add text for labels
        self.text = visuals.Text(
            '',
            pos=(10, 30),
            color='white',
            font_size=12,
            parent=self.view.scene
        )
        
        # Setup camera
        self.view.camera = scene.PanZoomCamera(
            aspect=1.0,
            rect=(0, 0, self.video_width, self.video_height),
            flip=(False, True)
        )
        
        # Connect events
        self.canvas.events.key_press.connect(self._on_key_press)
        self.timer = app.Timer(interval=0.016, connect=self._update, start=True)
        
        # Show instructions
        self._print_instructions()
    
    def _print_instructions(self):
        """Print usage instructions."""
        print("\n" + "="*60)
        print("CONTROLS:")
        print("  SPACE     : Play/Pause")
        print("  LEFT/RIGHT: Previous/Next frame")
        print("  UP/DOWN   : Jump 10 frames")
        print("  HOME/END  : First/Last frame")
        print("  H         : Toggle H5 detections")
        print("  Z         : Toggle Zarr detections")
        print("  O         : Toggle overlay mode")
        print("  R         : Reset to first frame")
        print("  Q/ESC     : Quit")
        print("="*60 + "\n")
    
    def _get_current_frame(self):
        """Get the current video frame."""
        if self.video_source == 'file':
            return self.vr[self.current_frame].asnumpy()
        else:  # H5 source
            with h5py.File(self.h5_path, 'r') as hf:
                return hf['/video_data/frames'][self.current_frame]
    
    def _update(self, event):
        """Update the visualization."""
        # Handle playback timing
        if self.is_playing:
            current_time = time.time()
            if current_time - self.last_frame_time >= self.frame_interval:
                self.current_frame = (self.current_frame + 1) % self.num_frames
                self.last_frame_time = current_time
        
        # Get and display video frame
        frame = self._get_current_frame()
        if frame.ndim == 2:  # Grayscale
            frame = np.stack([frame] * 3, axis=-1)
        self.image.set_data(frame)
        
        # Update H5 detections
        if self.show_h5 and self.h5_data.get('bboxes'):
            h5_boxes = self.h5_data['bboxes'].get(self.current_frame, [])
            self._draw_boxes(self.h5_lines, h5_boxes, self.colors['h5'])
        else:
            self.h5_lines.set_data(pos=np.zeros((0, 2)))
        
        # Update Zarr detections
        if self.show_zarr and 'bboxes' in self.zarr_data:
            n_dets = self.zarr_data['n_detections'][self.current_frame]
            if n_dets > 0:
                zarr_boxes = self.zarr_data['bboxes'][self.current_frame, :n_dets]
                
                # Use class colors if available
                if 'class_ids' in self.zarr_data and hasattr(self, 'class_colors'):
                    class_ids = self.zarr_data['class_ids'][self.current_frame, :n_dets]
                    # Draw boxes by class
                    for class_id in np.unique(class_ids):
                        class_mask = class_ids == class_id
                        class_boxes = zarr_boxes[class_mask]
                        color = self.class_colors.get(class_id, self.colors['zarr'])
                        self._draw_boxes(self.zarr_lines, class_boxes, color)
                else:
                    self._draw_boxes(self.zarr_lines, zarr_boxes, self.colors['zarr'])
            else:
                self.zarr_lines.set_data(pos=np.zeros((0, 2)))
        else:
            self.zarr_lines.set_data(pos=np.zeros((0, 2)))
        
        # Update status text
        status = "▶" if self.is_playing else "⏸"
        sources = []
        if self.show_h5: sources.append("H5")
        if self.show_zarr: sources.append("Zarr")
        source_text = " | ".join(sources) if sources else "None"
        
        info = f"{status} Frame: {self.current_frame:06d}/{self.num_frames-1} | Showing: {source_text}"
        self.text.text = info
        
        self.canvas.update()
    
    def _draw_boxes(self, line_visual, boxes, color):
        """Draw bounding boxes using line segments."""
        if len(boxes) == 0:
            line_visual.set_data(pos=np.zeros((0, 2)))
            return
        
        line_pos = []
        for box in boxes:
            x1, y1, x2, y2 = box[:4]
            # Create line segments for rectangle
            line_pos.extend([
                [x1, y1], [x2, y1],  # Top
                [x2, y1], [x2, y2],  # Right
                [x2, y2], [x1, y2],  # Bottom
                [x1, y2], [x1, y1]   # Left
            ])
        
        if line_pos:
            line_visual.set_data(
                pos=np.array(line_pos, dtype=np.float32),
                color=color
            )
    
    def _on_key_press(self, event):
        """Handle keyboard input."""
        if event.key == 'Space':
            self.is_playing = not self.is_playing
        elif event.key == 'Left':
            self.current_frame = max(0, self.current_frame - 1)
            self.is_playing = False
        elif event.key == 'Right':
            self.current_frame = min(self.num_frames - 1, self.current_frame + 1)
            self.is_playing = False
        elif event.key == 'Up':
            self.current_frame = min(self.num_frames - 1, self.current_frame + 10)
        elif event.key == 'Down':
            self.current_frame = max(0, self.current_frame - 10)
        elif event.key == 'Home':
            self.current_frame = 0
            self.is_playing = False
        elif event.key == 'End':
            self.current_frame = self.num_frames - 1
            self.is_playing = False
        elif event.key == 'h':
            self.show_h5 = not self.show_h5
            print(f"H5 detections: {'ON' if self.show_h5 else 'OFF'}")
        elif event.key == 'z':
            self.show_zarr = not self.show_zarr
            print(f"Zarr detections: {'ON' if self.show_zarr else 'OFF'}")
        elif event.key == 'o':
            self.show_overlay = not self.show_overlay
            print(f"Overlay mode: {'ON' if self.show_overlay else 'OFF'}")
        elif event.key == 'r':
            self.current_frame = 0
            self.is_playing = False
        elif event.key in ['q', 'Escape']:
            self.canvas.close()
            app.quit()
    
    def run(self):
        """Start the visualizer."""
        app.run()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize detections from analysis H5 files and Zarr outputs"
    )
    parser.add_argument(
        '--video', 
        type=str, 
        help='Path to video file (optional if H5 contains video)'
    )
    parser.add_argument(
        '--h5', 
        type=str, 
        help='Path to analysis H5 file'
    )
    parser.add_argument(
        '--zarr', 
        type=str, 
        help='Path to Zarr detection file'
    )
    parser.add_argument(
        '--device', 
        choices=['cpu', 'gpu'], 
        default='cpu',
        help='Device for video decoding (default: cpu)'
    )
    parser.add_argument(
        '--fps', 
        type=int, 
        default=30,
        help='Target playback FPS (default: 30)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.h5 and not args.zarr:
        parser.error("At least one of --h5 or --zarr must be provided")
    
    if not args.video and not args.h5:
        parser.error("Either --video or --h5 (with embedded video) must be provided")
    
    # Create and run visualizer
    visualizer = AnalysisDetectionVisualizer(
        video_path=args.video,
        h5_path=args.h5,
        zarr_path=args.zarr,
        device=args.device,
        target_fps=args.fps
    )
    visualizer.run()


if __name__ == "__main__":
    main()
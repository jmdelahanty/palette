#!/usr/bin/env python3
"""
High-performance YOLO detection visualizer using Vispy and Decord.
Achieves smooth playback with GPU-accelerated rendering.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import zarr
from decord import VideoReader, cpu, gpu
import colorsys
from vispy import app, scene
from vispy.scene import visuals, SceneCanvas
import time
import os

# Set Decord EOF retry limit to handle difficult videos
os.environ['DECORD_EOF_RETRY_MAX'] = '20480'


class HighPerfDetectionVisualizer:
    def __init__(self, video_path, zarr_path, device='cpu', target_fps=30):
        """
        Initialize high-performance detection visualizer.
        
        Args:
            video_path: Path to video file
            zarr_path: Path to zarr file with detections
            device: 'cpu' or 'gpu' for decord
            target_fps: Target playback FPS
        """
        self.video_path = video_path
        self.zarr_path = zarr_path
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        
        # Load video with decord
        ctx = cpu() if device == 'cpu' else gpu()
        try:
            # Try with sequential reading first
            self.vr = VideoReader(str(video_path), ctx=ctx)
            self.num_frames = len(self.vr)
            self.video_fps = self.vr.get_avg_fps()
            
            # Get video dimensions - use sequential read
            self.vr.seek(0)
            test_frame = self.vr.next().asnumpy()
            self.video_height, self.video_width = test_frame.shape[:2]
            self.vr.seek(0)  # Reset to beginning
        except Exception as e:
            print(f"Error loading video with Decord: {e}")
            print("Trying alternative loading method...")
            # Try with different settings
            self.vr = VideoReader(str(video_path), ctx=ctx, num_threads=1)
            self.num_frames = len(self.vr)
            self.video_fps = self.vr.get_avg_fps()
            
            # Use shape property if available
            if hasattr(self.vr, 'shape'):
                self.video_height, self.video_width = self.vr.shape[1:3]
            else:
                # Fallback: assume HD resolution
                self.video_height, self.video_width = 1080, 1920
                print(f"Warning: Could not determine video dimensions, using {self.video_width}x{self.video_height}")
        
        # Load zarr data
        self.zarr_data = zarr.open(zarr_path, 'r')
        self.bboxes = self.zarr_data['bboxes']
        self.scores = self.zarr_data['scores']
        self.class_ids = self.zarr_data['class_ids']
        self.n_detections = self.zarr_data['n_detections']
        
        # Preload all detection data for faster access
        print("Loading detection data...")
        self.n_detections_array = self.n_detections[:]
        self.max_dets_per_frame = self.bboxes.shape[1]
        
        # Get class names if available
        self.class_names = {}
        if 'class_names' in self.zarr_data.attrs:
            self.class_names = json.loads(self.zarr_data.attrs['class_names'])
            if isinstance(self.class_names, list):
                self.class_names = {i: name for i, name in enumerate(self.class_names)}
        
        # Get unique classes for color mapping
        class_ids_array = self.class_ids[:]
        valid_mask = class_ids_array >= 0
        all_classes = class_ids_array[valid_mask].flatten()
        self.unique_classes = np.unique(all_classes).astype(int)
        self.class_colors = self._generate_colors(len(self.unique_classes))
        self.class_to_color = {cls: color for cls, color in zip(self.unique_classes, self.class_colors)}
        
        # Playback state
        self.current_frame = 0
        self.is_playing = False
        self.show_labels = True
        self.show_boxes = True  # Boxes on by default
        self.conf_threshold = 0.25
        self.selected_classes = set(self.unique_classes)
        self.last_frame_time = time.time()
        
        # Check if bounding boxes need scaling
        # YOLO models typically use 640x640, but boxes might already be in original coordinates
        # We'll check the max values to determine if scaling is needed
        sample_boxes = self.bboxes[0:min(100, self.num_frames)]  # Sample first 100 frames
        max_x = np.max(sample_boxes[:, :, [0, 2]])
        max_y = np.max(sample_boxes[:, :, [1, 3]])
        
        # If max values are close to 640, we need to scale to video dimensions
        self.needs_scaling = (max_x < 1000 and max_y < 1000)
        if self.needs_scaling:
            self.scale_x = self.video_width / 640.0
            self.scale_y = self.video_height / 640.0
            print(f"Detected YOLO coordinates (640x640), scaling by {self.scale_x:.2f}x{self.scale_y:.2f}")
        else:
            self.scale_x = 1.0
            self.scale_y = 1.0
            print("Bounding boxes appear to be in original video coordinates")
        
        # Setup Vispy canvas
        self._setup_canvas()
        
    def _generate_colors(self, n):
        """Generate visually distinct colors for each class."""
        colors = []
        # Start with red as the primary color
        if n > 0:
            colors.append((1.0, 0.0, 0.0, 1.0))  # Red for first class
        for i in range(1, n):
            hue = (i / n) * 0.8 + 0.1  # Avoid red hue (0.0)
            saturation = 0.9
            value = 0.9
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(rgb + (1.0,))  # Add alpha
        return colors
    
    def _setup_canvas(self):
        """Setup Vispy canvas and visual elements."""
        # Create canvas
        self.canvas = SceneCanvas(
            title=f'YOLO Detections - {Path(self.video_path).name}',
            size=(1280, 720),
            bgcolor='black',
            keys='interactive'
        )
        self.canvas.show()
        
        # Create view
        self.view = self.canvas.central_widget.add_view()
        
        # Create image visual for video frames
        # Use the test frame we already loaded
        initial_frame = test_frame if 'test_frame' in locals() else np.zeros((self.video_height, self.video_width, 3), dtype=np.uint8)
        self.image = visuals.Image(
            initial_frame,
            interpolation='nearest',
            parent=self.view.scene,
            method='subdivide'
        )
        
        # CRITICAL: Set up camera to match image dimensions
        # Use PanZoomCamera for interactive zooming and panning
        self.view.camera = scene.PanZoomCamera(aspect=1)
        # Set the camera range to match the full image dimensions
        self.view.camera.rect = (0, 0, self.video_width, self.video_height)
        
        # Create line visual for bounding boxes (preallocate for performance)
        # Maximum possible lines needed (5 lines per box: 4 sides + 1 gap)
        max_boxes = 100  # Reasonable maximum boxes to display
        self.max_lines = max_boxes * 5
        self.line_pos = np.zeros((self.max_lines * 2, 2), dtype=np.float32)
        self.line_colors = np.zeros((self.max_lines * 2, 4), dtype=np.float32)
        self.line_visual = visuals.Line(
            pos=self.line_pos,
            color=self.line_colors,
            connect='segments',
            parent=self.view.scene,
            width=3,  # Thicker lines for better visibility
            antialias=True
        )
        
        # FIX RENDERING ORDER FOR LINES - Draw on top of image
        self.line_visual.set_gl_state('translucent', depth_test=False, cull_face=False)
        self.line_visual.order = 1  # Draw after image (image is 0 by default)
        
        # Create text visual for labels with larger font
        self.text_visual = visuals.Text(
            text='',
            color='white',
            parent=self.view.scene,
            anchor_x='left',
            anchor_y='bottom',
            font_size=14,  # Larger font size
            bold=True  # Bold text for better visibility
        )
        
        # FIX RENDERING ORDER FOR TEXT - Draw on top of lines
        self.text_visual.set_gl_state('translucent', depth_test=False, cull_face=False)
        self.text_visual.order = 2  # Draw after lines
        
        # Info text overlay
        self.info_text = visuals.Text(
            text='',
            color='white',
            pos=(10, 30),
            parent=self.canvas.scene,
            anchor_x='left',
            font_size=12
        )
        
        # Control instructions
        controls = [
            "Controls:",
            "Space: Play/Pause",
            "←/→: Previous/Next frame",
            "↑/↓: Adjust confidence ±0.05",
            "L: Toggle labels",
            "B: Toggle boxes",
            "R: Reset view",
            "Mouse: Pan (drag) & Zoom (scroll)",
            "Q: Quit"
        ]
        self.control_text = visuals.Text(
            text='\n'.join(controls),
            color='white',
            pos=(10, 100),
            parent=self.canvas.scene,
            anchor_x='left',
            font_size=10
        )
        
        # Connect events
        self.canvas.events.key_press.connect(self._on_key_press)
        self.timer = app.Timer(interval=self.frame_interval, connect=self._on_timer, start=False)
        
        # Initial display
        self._update_frame()
        
    def _on_timer(self, event):
        """Timer callback for video playback."""
        if self.is_playing and self.current_frame < self.num_frames - 1:
            current_time = time.time()
            if current_time - self.last_frame_time >= self.frame_interval:
                self.current_frame += 1
                self._update_frame()
                self.last_frame_time = current_time
        elif self.current_frame >= self.num_frames - 1:
            self.is_playing = False
            self.timer.stop()
    
    def _on_key_press(self, event):
        """Handle keyboard input."""
        if event.key == 'Space':
            self._toggle_play()
        elif event.key == 'Left':
            self._prev_frame()
        elif event.key == 'Right':
            self._next_frame()
        elif event.key == 'Up':
            self.conf_threshold = min(1.0, self.conf_threshold + 0.05)
            self._update_frame()
        elif event.key == 'Down':
            self.conf_threshold = max(0.0, self.conf_threshold - 0.05)
            self._update_frame()
        elif event.key == 'L':
            self.show_labels = not self.show_labels
            self._update_frame()
        elif event.key == 'B':
            self.show_boxes = not self.show_boxes
            self._update_frame()
        elif event.key == 'R':
            # Reset view to show full image
            self.view.camera.rect = (0, 0, self.video_width, self.video_height)
        elif event.key == 'Q':
            self.canvas.close()
            app.quit()
    
    def _toggle_play(self):
        """Toggle play/pause."""
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.last_frame_time = time.time()
            self.timer.start()
        else:
            self.timer.stop()
    
    def _prev_frame(self):
        """Go to previous frame."""
        if self.current_frame > 0:
            self.current_frame -= 1
            self._update_frame()
    
    def _next_frame(self):
        """Go to next frame."""
        if self.current_frame < self.num_frames - 1:
            self.current_frame += 1
            self._update_frame()
    
    def _update_frame(self):
        """Update displayed frame with detections."""
        # Get and display video frame
        try:
            # Try random access first
            frame = self.vr[self.current_frame].asnumpy()
        except:
            # Fallback to sequential access
            self.vr.seek(self.current_frame)
            frame = self.vr.next().asnumpy()
        
        self.image.set_data(frame)
        
        # Get detections for this frame
        n_dets = self.n_detections_array[self.current_frame]
        
        # Update info text
        info = f"Frame: {self.current_frame}/{self.num_frames-1} | "
        info += f"FPS: {self.video_fps:.1f} | "
        info += f"Detections: {n_dets} | "
        info += f"Conf: {self.conf_threshold:.2f}"
        if self.is_playing:
            info += " | ▶ Playing"
        else:
            info += " | ⏸ Paused"
        self.info_text.text = info
        
        # Clear previous boxes
        self.line_pos.fill(0)
        self.line_colors.fill(0)
        line_idx = 0
        
        # Process detections if boxes are enabled
        if self.show_boxes and n_dets > 0:
            # Load detection data for this frame
            boxes = self.bboxes[self.current_frame, :n_dets]
            scores = self.scores[self.current_frame, :n_dets]
            classes = self.class_ids[self.current_frame, :n_dets]
            
            # Debug: Print first detection to check coordinates
            if n_dets > 0 and self.current_frame % 100 == 0:  # Print every 100 frames
                print(f"Frame {self.current_frame} - First box: {boxes[0]}, Score: {scores[0]:.3f}, Class: {classes[0]}")
                if self.needs_scaling:
                    print(f"  Scaled to: {boxes[0] * [self.scale_x, self.scale_y, self.scale_x, self.scale_y]}")
            
            # Collect label positions and texts
            label_texts = []
            label_positions = []
            
            for i in range(n_dets):
                # Check confidence threshold
                if scores[i] < self.conf_threshold:
                    continue
                
                # Check class filter
                class_id = int(classes[i])
                if class_id not in self.selected_classes:
                    continue
                
                # Get box coordinates and scale if needed
                x1, y1, x2, y2 = boxes[i]
                
                # Scale coordinates if they're in YOLO space (640x640)
                x1 *= self.scale_x
                y1 *= self.scale_y
                x2 *= self.scale_x
                y2 *= self.scale_y
                
                # Ensure coordinates are valid
                if x2 <= x1 or y2 <= y1:
                    print(f"Warning: Invalid box at frame {self.current_frame}: [{x1}, {y1}, {x2}, {y2}]")
                    continue
                
                # Get color for this class (red for first class, then other colors)
                color = self.class_to_color.get(class_id, (1, 0, 0, 1))
                
                # Create box lines (4 segments)
                if line_idx < self.max_lines - 4:
                    # Top line
                    self.line_pos[line_idx*2] = [x1, y1]
                    self.line_pos[line_idx*2 + 1] = [x2, y1]
                    self.line_colors[line_idx*2] = color
                    self.line_colors[line_idx*2 + 1] = color
                    line_idx += 1
                    
                    # Right line
                    self.line_pos[line_idx*2] = [x2, y1]
                    self.line_pos[line_idx*2 + 1] = [x2, y2]
                    self.line_colors[line_idx*2] = color
                    self.line_colors[line_idx*2 + 1] = color
                    line_idx += 1
                    
                    # Bottom line
                    self.line_pos[line_idx*2] = [x2, y2]
                    self.line_pos[line_idx*2 + 1] = [x1, y2]
                    self.line_colors[line_idx*2] = color
                    self.line_colors[line_idx*2 + 1] = color
                    line_idx += 1
                    
                    # Left line
                    self.line_pos[line_idx*2] = [x1, y2]
                    self.line_pos[line_idx*2 + 1] = [x1, y1]
                    self.line_colors[line_idx*2] = color
                    self.line_colors[line_idx*2 + 1] = color
                    line_idx += 1
                
                # Add label if enabled
                if self.show_labels:
                    label_text = ""
                    if str(class_id) in self.class_names:
                        label_text = self.class_names[str(class_id)]
                    else:
                        label_text = f"cls{class_id}"
                    label_text += f" {scores[i]:.2f}"
                    label_texts.append(label_text)
                    label_positions.append([x1, y1 - 5])
            
            # Update label text
            if self.show_labels and label_texts:
                self.text_visual.text = label_texts
                self.text_visual.pos = np.array(label_positions)
            else:
                self.text_visual.text = ''
            
            # Debug: Report how many lines we're drawing
            if line_idx > 0:
                print(f"Drawing {line_idx} line segments for boxes")
        else:
            self.text_visual.text = ''
        
        # Update line visual with the actual data
        if line_idx > 0:
            # Only pass the valid portion of the arrays
            valid_pos = self.line_pos[:line_idx*2]
            valid_colors = self.line_colors[:line_idx*2]
            self.line_visual.set_data(pos=valid_pos, color=valid_colors)
        else:
            # No boxes to draw - hide the lines
            self.line_visual.set_data(pos=np.array([[0, 0], [0, 0]]), color=(0, 0, 0, 0))
        
        # Update canvas
        self.canvas.update()
    
    def run(self):
        """Run the visualizer."""
        app.run()

def main():
    parser = argparse.ArgumentParser(description='High-performance YOLO detection visualizer')
    parser.add_argument('--video', type=str, required=True,
                       help='Path to video file')
    parser.add_argument('--zarr', type=str, required=True,
                       help='Path to zarr file with detections')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'gpu'],
                       help='Device for decord video reading (default: cpu)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Target playback FPS (default: 30)')
    
    args = parser.parse_args()
    
    # Create and run visualizer
    viz = HighPerfDetectionVisualizer(
        video_path=args.video,
        zarr_path=args.zarr,
        device=args.device,
        target_fps=args.fps
    )
    viz.run()


if __name__ == '__main__':
    main()
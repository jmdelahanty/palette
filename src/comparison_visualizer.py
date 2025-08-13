#!/usr/bin/env python3
"""
High-performance comparison visualizer for TensorRT (H5) and Offline (Zarr) models
using Vispy for GPU-accelerated rendering and Decord for fast video decoding.
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

# Set Decord EOF retry limit to handle difficult videos
os.environ['DECORD_EOF_RETRY_MAX'] = '20480'

class HighPerfComparisonVisualizer:
    def __init__(self, video_path, h5_path, zarr_path, device='cpu', target_fps=30):
        self.video_path = video_path
        self.h5_path = h5_path
        self.zarr_path = zarr_path
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps

        # --- Load Video with Decord ---
        ctx = cpu() if device == 'cpu' else gpu()
        print(f"Loading video with Decord on {device.upper()}...")
        self.vr = VideoReader(str(video_path), ctx=ctx)
        self.num_frames = len(self.vr)
        self.video_fps = self.vr.get_avg_fps()
        self.initial_frame = self.vr[0].asnumpy() # Load the first frame for initialization
        self.video_height, self.video_width = self.initial_frame.shape[:2]
        print(f"Video loaded: {self.num_frames} frames, {self.video_width}x{self.video_height} @ {self.video_fps:.2f} FPS")

        # --- Load H5 (TensorRT) Detections ---
        print("Loading H5 (TensorRT) detections...")
        with h5py.File(self.h5_path, 'r') as hf:
            bbox_dataset = hf['/tracking_data/bounding_boxes']
            h5_frame_ids = bbox_dataset['payload_frame_id'][:]
            x_min = bbox_dataset['x_min'][:]
            y_min = bbox_dataset['y_min'][:]
            width = bbox_dataset['width'][:]
            height = bbox_dataset['height'][:]

            self.h5_bbox_map = {}
            for i, frame_id in enumerate(h5_frame_ids):
                if frame_id not in self.h5_bbox_map:
                    self.h5_bbox_map[frame_id] = []
                x1, y1 = x_min[i], y_min[i]
                x2, y2 = x1 + width[i], y1 + height[i]
                self.h5_bbox_map[frame_id].append([x1, y1, x2, y2])
        print(f"Loaded {len(h5_frame_ids)} detections from H5 file.")

        # --- Load Zarr (Offline) Detections ---
        print("Loading Zarr (Offline) detections...")
        zarr_root = zarr.open(self.zarr_path, mode='r')
        self.zarr_bboxes = zarr_root['bboxes'][:]
        self.zarr_scores = zarr_root['scores'][:]
        self.zarr_n_detections = zarr_root['n_detections'][:]
        print(f"Loaded {self.zarr_bboxes.shape[0]} frames of detection data from Zarr file.")

        # --- Playback State ---
        self.current_frame = 0
        self.is_playing = False
        self.last_frame_time = time.time()
        
        # --- Canvas Setup ---
        self._setup_canvas()

    def _setup_canvas(self):
        """Setup Vispy canvas with two side-by-side views."""
        self.canvas = scene.SceneCanvas(
            title="TensorRT (H5) vs. Offline (Zarr) Comparison",
            size=(1920, 720),
            bgcolor='black',
            keys='interactive'
        )

        # Use a grid to layout the two views
        grid = self.canvas.central_widget.add_grid()

        # --- View 1: H5 (TensorRT) ---
        self.view1 = grid.add_view(row=0, col=0, border_color='lime')
        self.image1 = visuals.Image(self.initial_frame, parent=self.view1.scene, method='subdivide')
        self.lines1 = visuals.Line(parent=self.view1.scene, connect='segments', width=2, antialias=True)
        self.text1 = visuals.Text(parent=self.view1.scene, color='lime', font_size=14, bold=True, anchor_x='left', anchor_y='bottom')
        self.view1_title = visuals.Text("TensorRT (H5)", parent=self.view1.scene, color='lime', pos=(10, 30), font_size=16, bold=True)
        self.lines1.set_gl_state('translucent', depth_test=False, cull_face=False)
        self.text1.set_gl_state('translucent', depth_test=False, cull_face=False)


        # --- View 2: Zarr (Offline) ---
        self.view2 = grid.add_view(row=0, col=1, border_color='cyan')
        self.image2 = visuals.Image(self.initial_frame, parent=self.view2.scene, method='subdivide')
        self.lines2 = visuals.Line(parent=self.view2.scene, connect='segments', width=2, antialias=True)
        self.text2 = visuals.Text(parent=self.view2.scene, color='cyan', font_size=14, bold=True, anchor_x='left', anchor_y='bottom')
        self.view2_title = visuals.Text("Offline (Zarr)", parent=self.view2.scene, color='cyan', pos=(10, 30), font_size=16, bold=True)
        self.lines2.set_gl_state('translucent', depth_test=False, cull_face=False)
        self.text2.set_gl_state('translucent', depth_test=False, cull_face=False)
        
        # --- Link Cameras ---
        self.view1.camera = scene.PanZoomCamera(aspect=1)
        self.view2.camera = self.view1.camera  # Link cameras for synchronized pan/zoom
        self.view1.camera.rect = (0, 0, self.video_width, self.video_height)
        
        # --- Info Text ---
        self.info_text = visuals.Text("", parent=self.canvas.scene, color='white', pos=(10, 50), font_size=12)

        # --- Event Handlers ---
        self.canvas.events.key_press.connect(self._on_key_press)
        self.timer = app.Timer(interval=self.frame_interval, connect=self._on_timer, start=False)
        
        # --- Initial Display ---
        self._update_frame()
        self.canvas.show()

    def _update_frame(self):
        """Update both views with the current frame and detections."""
        # Get video frame
        frame = self.vr[self.current_frame].asnumpy()
        self.image1.set_data(frame)
        self.image2.set_data(frame)

        # --- Update H5 (TensorRT) View ---
        h5_boxes = np.array(self.h5_bbox_map.get(self.current_frame, []))
        if h5_boxes.shape[0] > 0:
            h5_scores = np.ones(len(h5_boxes)) # Dummy scores
            self._update_view_visuals(self.lines1, self.text1, h5_boxes, h5_scores, "TRT", (0, 1, 0, 1))
        else:
            self.lines1.set_data(pos=np.zeros((0,2)))
            self.text1.text = ''

        # --- Update Zarr (Offline) View ---
        num_zarr_dets = self.zarr_n_detections[self.current_frame]
        if num_zarr_dets > 0:
            zarr_boxes = self.zarr_bboxes[self.current_frame, :num_zarr_dets]
            zarr_scores = self.zarr_scores[self.current_frame, :num_zarr_dets]
            self._update_view_visuals(self.lines2, self.text2, zarr_boxes, zarr_scores, "Offline", (0, 1, 1, 1))
        else:
            self.lines2.set_data(pos=np.zeros((0,2)))
            self.text2.text = ''

        # Update info text
        status = "▶ Playing" if self.is_playing else "⏸ Paused"
        self.info_text.text = f"Frame: {self.current_frame}/{self.num_frames-1} | {status}"

        # Request a canvas update
        self.canvas.update()

    def _update_view_visuals(self, line_visual, text_visual, boxes, scores, prefix, color):
        """Helper to update the lines and text for a single view."""
        line_pos = []
        texts, text_positions = [], []

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            # Create 4 line segments for each box
            line_pos.extend([[x1, y1], [x2, y1], [x2, y1], [x2, y2], 
                             [x2, y2], [x1, y2], [x1, y2], [x1, y1]])
            
            texts.append(f"{prefix}: {scores[i]:.2f}")
            text_positions.append([x1, y1 - 5])

        if line_pos:
            line_visual.set_data(pos=np.array(line_pos, dtype=np.float32), color=color)
            text_visual.text = texts
            text_visual.pos = np.array(text_positions, dtype=np.float32)
        else:
            line_visual.set_data(pos=np.zeros((0,2)))
            text_visual.text = ''
            
    def _on_key_press(self, event):
        """Handle keyboard input."""
        if event.key == 'Space':
            self.is_playing = not self.is_playing
            if self.is_playing: self.timer.start()
            else: self.timer.stop()
        elif event.key == 'Left':
            if self.current_frame > 0:
                self.current_frame -= 1
                self._update_frame()
        elif event.key == 'Right':
            if self.current_frame < self.num_frames - 1:
                self.current_frame += 1
                self._update_frame()
        elif event.key == 'R':
            self.view1.camera.rect = (0, 0, self.video_width, self.video_height)
        elif event.key == 'Q' or event.key == 'Escape':
            self.canvas.close()
            app.quit()
    
    def _on_timer(self, event):
        """Timer callback for video playback."""
        if self.is_playing and self.current_frame < self.num_frames - 1:
            self.current_frame += 1
            self._update_frame()
        elif self.current_frame >= self.num_frames - 1:
            self.is_playing = False
            self.timer.stop()

    def run(self):
        """Run the application."""
        app.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="High-performance comparison of YOLO detections.")
    parser.add_argument("--video", required=True, type=Path, help="Path to the source video file.")
    parser.add_argument("--h5", required=True, type=Path, help="Path to the H5 file with TensorRT detections.")
    parser.add_argument("--zarr", required=True, type=Path, help="Path to the Zarr file with offline detections.")
    parser.add_argument("--device", default='cpu', choices=['cpu', 'gpu'], help="Device for decord video decoding.")
    parser.add_argument("--fps", type=int, default=30, help="Target playback FPS.")
    
    args = parser.parse_args()
    
    visualizer = HighPerfComparisonVisualizer(
        video_path=args.video,
        h5_path=args.h5,
        zarr_path=args.zarr,
        device=args.device,
        target_fps=args.fps
    )
    visualizer.run()
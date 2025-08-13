#!/usr/bin/env python3
"""
Advanced YOLO Detection Visualizer using PyQtGraph
Multi-window interface with timeline, statistics, and interactive controls
"""

import sys
import argparse
import json
from pathlib import Path
import numpy as np
import zarr
from decord import VideoReader, cpu, gpu
import colorsys
import time
import os

from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
import pyqtgraph.dockarea as dock

# Set Decord EOF retry limit
os.environ['DECORD_EOF_RETRY_MAX'] = '20480'

# Configure PyQtGraph
pg.setConfigOptions(antialias=True, imageAxisOrder='row-major')


class YOLODetectionVisualizer(QtWidgets.QMainWindow):
    def __init__(self, video_path, zarr_path, device='cpu'):
        super().__init__()
        self.video_path = video_path
        self.zarr_path = zarr_path
        
        # Load video with decord
        ctx = cpu() if device == 'cpu' else gpu()
        self.vr = VideoReader(str(video_path), ctx=ctx)
        self.num_frames = len(self.vr)
        self.video_fps = self.vr.get_avg_fps()
        
        # Get video dimensions
        self.vr.seek(0)
        test_frame = self.vr.next().asnumpy()
        self.video_height, self.video_width = test_frame.shape[:2]
        self.vr.seek(0)
        
        # Load zarr data
        print("Loading detection data...")
        self.zarr_data = zarr.open(zarr_path, 'r')
        self.bboxes = self.zarr_data['bboxes'][:]
        self.scores = self.zarr_data['scores'][:]
        self.class_ids = self.zarr_data['class_ids'][:]
        self.n_detections = self.zarr_data['n_detections'][:]
        
        # Get class names
        self.class_names = {}
        if 'class_names' in self.zarr_data.attrs:
            self.class_names = json.loads(self.zarr_data.attrs['class_names'])
            if isinstance(self.class_names, list):
                self.class_names = {i: name for i, name in enumerate(self.class_names)}
        
        # Generate colors for classes
        valid_mask = self.class_ids >= 0
        all_classes = self.class_ids[valid_mask].flatten()
        self.unique_classes = np.unique(all_classes).astype(int)
        self.class_colors = self._generate_colors(len(self.unique_classes))
        self.class_to_color = {cls: color for cls, color in zip(self.unique_classes, self.class_colors)}
        
        # Check if scaling is needed
        self._detect_coordinate_system()
        
        # Playback state
        self.current_frame = 0
        self.is_playing = False
        self.playback_speed = 1.0
        self.conf_threshold = 0.25
        self.show_boxes = True
        self.show_labels = True
        self.selected_classes = set(self.unique_classes)
        
        # Setup UI
        self.setup_ui()
        
        # Setup timer for playback
        self.play_timer = QtCore.QTimer()
        self.play_timer.timeout.connect(self.next_frame)
        self.play_timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)  # Use precise timer
        self.play_timer.setInterval(int(1000 / self.video_fps))
        
        # Frame time tracking for smooth playback
        self.last_frame_time = time.time()
        self.target_frame_interval = 1.0 / self.video_fps
        
        # Initial update
        self.update_frame()
        
    def _generate_colors(self, n):
        """Generate visually distinct colors."""
        colors = []
        for i in range(n):
            hue = i / max(1, n-1) * 0.8
            rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
            colors.append([int(c * 255) for c in rgb] + [255])
        return colors
    
    def _detect_coordinate_system(self):
        """Detect if coordinates need scaling."""
        sample_boxes = self.bboxes[:min(100, self.num_frames)]
        valid_mask = sample_boxes[:, :, 0] >= 0
        valid_boxes = sample_boxes[valid_mask]
        
        if len(valid_boxes) > 0:
            max_x = np.max(valid_boxes[:, [0, 2]])
            max_y = np.max(valid_boxes[:, [1, 3]])
            
            if max_x <= 1.01 and max_y <= 1.01:
                self.scale_x = self.video_width
                self.scale_y = self.video_height
                print("Detected normalized coordinates")
            elif 600 <= max_x <= 700 and 600 <= max_y <= 700:
                self.scale_x = self.video_width / 640.0
                self.scale_y = self.video_height / 640.0
                print(f"Detected YOLO 640x640 coordinates, scaling by {self.scale_x:.2f}x{self.scale_y:.2f}")
            else:
                self.scale_x = 1.0
                self.scale_y = 1.0
                print("Coordinates appear to be in video space")
        else:
            self.scale_x = 1.0
            self.scale_y = 1.0
    
    def setup_ui(self):
        """Setup the PyQtGraph UI with dockable areas."""
        self.setWindowTitle(f'YOLO Detection Visualizer - {Path(self.video_path).name}')
        self.resize(1400, 900)
        
        # Create dock area
        self.dock_area = dock.DockArea()
        self.setCentralWidget(self.dock_area)
        
        # Create docks
        self.video_dock = dock.Dock("Video", size=(800, 600))
        self.timeline_dock = dock.Dock("Detection Timeline", size=(800, 150))
        self.controls_dock = dock.Dock("Controls", size=(300, 600))
        self.stats_dock = dock.Dock("Statistics", size=(300, 300))
        self.class_dock = dock.Dock("Classes", size=(300, 300))
        
        # Arrange docks
        self.dock_area.addDock(self.video_dock, 'left')
        self.dock_area.addDock(self.timeline_dock, 'bottom', self.video_dock)
        self.dock_area.addDock(self.controls_dock, 'right')
        self.dock_area.addDock(self.stats_dock, 'bottom', self.controls_dock)
        self.dock_area.addDock(self.class_dock, 'bottom', self.stats_dock)
        
        # Setup video display
        self.setup_video_display()
        
        # Setup timeline
        self.setup_timeline()
        
        # Setup controls
        self.setup_controls()
        
        # Setup statistics
        self.setup_statistics()
        
        # Setup class filter
        self.setup_class_filter()
        
    def setup_video_display(self):
        """Setup the video display widget."""
        # Create graphics layout
        self.video_layout = pg.GraphicsLayoutWidget()
        self.video_dock.addWidget(self.video_layout)
        
        # Create viewbox for video
        self.video_view = self.video_layout.addViewBox()
        self.video_view.setAspectLocked(True)
        self.video_view.invertY()  # Flip Y to match image coordinates
        
        # Create image item for video frames
        self.video_image = pg.ImageItem()
        self.video_view.addItem(self.video_image)
        
        # Single item for ALL boxes using connected line segments
        self.all_boxes_item = pg.PlotDataItem(
            [], [], 
            pen=pg.mkPen('r', width=2),
            connect='finite'  # This allows disconnected segments
        )
        self.video_view.addItem(self.all_boxes_item)
        
        # Simplified text overlay for FPS/stats (single text item)
        self.overlay_text = pg.TextItem('', color='w', anchor=(0, 0))
        self.overlay_text.setPos(10, 10)
        self.video_view.addItem(self.overlay_text)
    
    def setup_timeline(self):
        """Setup the detection timeline plot."""
        self.timeline_widget = pg.PlotWidget()
        self.timeline_dock.addWidget(self.timeline_widget)
        
        # Configure timeline
        self.timeline_widget.setLabel('left', 'Detections')
        self.timeline_widget.setLabel('bottom', 'Frame')
        self.timeline_widget.showGrid(x=True, y=True, alpha=0.3)
        
        # Plot detection counts
        self.timeline_plot = self.timeline_widget.plot(
            self.n_detections, 
            pen=pg.mkPen('c', width=2),
            name='Detections'
        )
        
        # Add confidence threshold line
        self.conf_line = pg.InfiniteLine(
            pos=0, 
            angle=0, 
            pen=pg.mkPen('y', width=1, style=QtCore.Qt.PenStyle.DashLine)
        )
        self.timeline_widget.addItem(self.conf_line)
        
        # Add current frame indicator
        self.frame_line = pg.InfiniteLine(
            pos=0, 
            angle=90, 
            pen=pg.mkPen('r', width=2)
        )
        self.timeline_widget.addItem(self.frame_line)
        
        # Make timeline clickable
        self.timeline_widget.scene().sigMouseClicked.connect(self.timeline_clicked)
    
    def setup_controls(self):
        """Setup control panel."""
        controls_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        controls_widget.setLayout(layout)
        self.controls_dock.addWidget(controls_widget)
        
        # Frame control
        frame_group = QtWidgets.QGroupBox("Frame Control")
        frame_layout = QtWidgets.QVBoxLayout()
        
        # Frame slider
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, self.num_frames - 1)
        self.frame_slider.valueChanged.connect(self.slider_changed)
        frame_layout.addWidget(self.frame_slider)
        
        # Frame spinbox
        frame_spin_layout = QtWidgets.QHBoxLayout()
        frame_spin_layout.addWidget(QtWidgets.QLabel("Frame:"))
        self.frame_spinbox = QtWidgets.QSpinBox()
        self.frame_spinbox.setRange(0, self.num_frames - 1)
        self.frame_spinbox.valueChanged.connect(self.spinbox_changed)
        frame_spin_layout.addWidget(self.frame_spinbox)
        frame_spin_layout.addWidget(QtWidgets.QLabel(f"/ {self.num_frames - 1}"))
        frame_layout.addLayout(frame_spin_layout)
        
        frame_group.setLayout(frame_layout)
        layout.addWidget(frame_group)
        
        # Playback controls
        playback_group = QtWidgets.QGroupBox("Playback")
        playback_layout = QtWidgets.QVBoxLayout()
        
        # Play/Pause button
        self.play_button = QtWidgets.QPushButton("▶ Play")
        self.play_button.clicked.connect(self.toggle_play)
        playback_layout.addWidget(self.play_button)
        
        # Speed control
        speed_layout = QtWidgets.QHBoxLayout()
        speed_layout.addWidget(QtWidgets.QLabel("Speed:"))
        self.speed_combo = QtWidgets.QComboBox()
        self.speed_combo.addItems(['0.25x', '0.5x', '1x', '2x', '4x'])
        self.speed_combo.setCurrentText('1x')
        self.speed_combo.currentTextChanged.connect(self.speed_changed)
        speed_layout.addWidget(self.speed_combo)
        playback_layout.addLayout(speed_layout)
        
        playback_group.setLayout(playback_layout)
        layout.addWidget(playback_group)
        
        # Detection controls
        detection_group = QtWidgets.QGroupBox("Detection Settings")
        detection_layout = QtWidgets.QVBoxLayout()
        
        # Confidence threshold
        conf_layout = QtWidgets.QHBoxLayout()
        conf_layout.addWidget(QtWidgets.QLabel("Confidence:"))
        self.conf_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(int(self.conf_threshold * 100))
        self.conf_slider.valueChanged.connect(self.conf_changed)
        conf_layout.addWidget(self.conf_slider)
        self.conf_label = QtWidgets.QLabel(f"{self.conf_threshold:.2f}")
        conf_layout.addWidget(self.conf_label)
        detection_layout.addLayout(conf_layout)
        
        # Show boxes checkbox
        self.show_boxes_check = QtWidgets.QCheckBox("Show Boxes")
        self.show_boxes_check.setChecked(True)
        self.show_boxes_check.stateChanged.connect(self.toggle_boxes)
        detection_layout.addWidget(self.show_boxes_check)
        
        # Show labels checkbox
        self.show_labels_check = QtWidgets.QCheckBox("Show Labels")
        self.show_labels_check.setChecked(True)
        self.show_labels_check.stateChanged.connect(self.toggle_labels)
        detection_layout.addWidget(self.show_labels_check)
        
        detection_group.setLayout(detection_layout)
        layout.addWidget(detection_group)
        
        # Export button
        self.export_button = QtWidgets.QPushButton("📸 Export Frame")
        self.export_button.clicked.connect(self.export_frame)
        layout.addWidget(self.export_button)
        
        layout.addStretch()
    
    def setup_statistics(self):
        """Setup statistics display."""
        stats_widget = QtWidgets.QTextEdit()
        stats_widget.setReadOnly(True)
        self.stats_dock.addWidget(stats_widget)
        self.stats_display = stats_widget
    
    def setup_class_filter(self):
        """Setup class filter checkboxes."""
        class_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        class_widget.setLayout(layout)
        self.class_dock.addWidget(class_widget)
        
        layout.addWidget(QtWidgets.QLabel("Filter Classes:"))
        
        # Create checkbox for each class
        self.class_checkboxes = {}
        for class_id in self.unique_classes:
            class_name = self.class_names.get(class_id, f"Class {class_id}")
            checkbox = QtWidgets.QCheckBox(class_name)
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(lambda state, cid=class_id: self.toggle_class(cid, state))
            layout.addWidget(checkbox)
            self.class_checkboxes[class_id] = checkbox
        
        # Select/Deselect all buttons
        button_layout = QtWidgets.QHBoxLayout()
        select_all = QtWidgets.QPushButton("All")
        select_all.clicked.connect(self.select_all_classes)
        deselect_all = QtWidgets.QPushButton("None")
        deselect_all.clicked.connect(self.deselect_all_classes)
        button_layout.addWidget(select_all)
        button_layout.addWidget(deselect_all)
        layout.addLayout(button_layout)
        
        layout.addStretch()
    
    def update_frame(self, update_ui=True):
        """Update the displayed frame and detections."""
        # Track FPS for debugging
        if hasattr(self, '_fps_counter'):
            fps = 1.0 / (time.time() - self._fps_counter)
            self.current_fps = fps
        else:
            self.current_fps = 0
        self._fps_counter = time.time()
        
        # Get video frame
        try:
            frame = self.vr[self.current_frame].asnumpy()
        except:
            self.vr.seek(self.current_frame)
            frame = self.vr.next().asnumpy()
        
        # Convert BGR to RGB for display
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = frame[:, :, ::-1]  # BGR to RGB
        
        # Display frame
        self.video_image.setImage(frame)
        
        # Get detections for this frame
        n_dets = self.n_detections[self.current_frame]
        
        # Build all boxes in one go using numpy
        if self.show_boxes and n_dets > 0:
            boxes = self.bboxes[self.current_frame, :n_dets]
            scores = self.scores[self.current_frame, :n_dets]
            classes = self.class_ids[self.current_frame, :n_dets]
            
            # Filter by confidence and class
            valid_mask = scores >= self.conf_threshold
            valid_boxes = []
            
            for i in range(n_dets):
                if valid_mask[i] and int(classes[i]) in self.selected_classes:
                    x1, y1, x2, y2 = boxes[i]
                    x1 *= self.scale_x
                    y1 *= self.scale_y
                    x2 *= self.scale_x
                    y2 *= self.scale_y
                    
                    # Add box corners with NaN separator for disconnected segments
                    valid_boxes.extend([
                        [x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1],
                        [np.nan, np.nan]  # Separator
                    ])
            
            if valid_boxes:
                box_array = np.array(valid_boxes)
                self.all_boxes_item.setData(box_array[:, 0], box_array[:, 1])
            else:
                self.all_boxes_item.setData([], [])
        else:
            self.all_boxes_item.setData([], [])
        
        # Update overlay text with FPS and stats
        overlay_info = f"Frame: {self.current_frame}/{self.num_frames-1} | FPS: {self.current_fps:.1f} | Dets: {n_dets}"
        self.overlay_text.setText(overlay_info)
        
        # Only update UI elements if requested (skip during fast playback)
        if update_ui:
            # Update frame indicator on timeline
            self.frame_line.setPos(self.current_frame)
            
            # Update sliders/spinboxes
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(self.current_frame)
            self.frame_slider.blockSignals(False)
            
            self.frame_spinbox.blockSignals(True)
            self.frame_spinbox.setValue(self.current_frame)
            self.frame_spinbox.blockSignals(False)
            
            # Update statistics less frequently
            if self.current_frame % 5 == 0:  # Every 5th frame
                self.update_statistics(n_dets)
    
    def update_statistics(self, n_dets):
        """Update statistics display."""
        stats = []
        stats.append(f"Frame: {self.current_frame}/{self.num_frames-1}")
        stats.append(f"FPS: {self.video_fps:.1f}")
        stats.append(f"Detections: {n_dets}")
        
        if n_dets > 0:
            # Count by class
            frame_classes = self.class_ids[self.current_frame, :n_dets]
            frame_scores = self.scores[self.current_frame, :n_dets]
            
            # Apply threshold
            valid_mask = frame_scores >= self.conf_threshold
            if np.any(valid_mask):
                valid_classes = frame_classes[valid_mask]
                unique, counts = np.unique(valid_classes, return_counts=True)
                
                stats.append("\nBy Class:")
                for cls_id, count in zip(unique, counts):
                    if cls_id in self.selected_classes:
                        class_name = self.class_names.get(int(cls_id), f"Class {cls_id}")
                        stats.append(f"  {class_name}: {count}")
                
                stats.append(f"\nMax Confidence: {frame_scores[valid_mask].max():.3f}")
                stats.append(f"Avg Confidence: {frame_scores[valid_mask].mean():.3f}")
        
        self.stats_display.setText('\n'.join(stats))
    
    def slider_changed(self, value):
        """Handle frame slider change."""
        self.current_frame = value
        self.update_frame()
    
    def spinbox_changed(self, value):
        """Handle frame spinbox change."""
        self.current_frame = value
        self.update_frame()
    
    def conf_changed(self, value):
        """Handle confidence threshold change."""
        self.conf_threshold = value / 100.0
        self.conf_label.setText(f"{self.conf_threshold:.2f}")
        self.update_frame()
    
    def toggle_boxes(self, state):
        """Toggle box display."""
        self.show_boxes = state == QtCore.Qt.CheckState.Checked
        self.update_frame()
    
    def toggle_labels(self, state):
        """Toggle label display."""
        self.show_labels = state == QtCore.Qt.CheckState.Checked
        self.update_frame()
    
    def toggle_class(self, class_id, state):
        """Toggle class visibility."""
        if state == QtCore.Qt.CheckState.Checked:
            self.selected_classes.add(class_id)
        else:
            self.selected_classes.discard(class_id)
        self.update_frame()
    
    def select_all_classes(self):
        """Select all classes."""
        for class_id, checkbox in self.class_checkboxes.items():
            checkbox.setChecked(True)
            self.selected_classes.add(class_id)
        self.update_frame()
    
    def deselect_all_classes(self):
        """Deselect all classes."""
        for class_id, checkbox in self.class_checkboxes.items():
            checkbox.setChecked(False)
        self.selected_classes.clear()
        self.update_frame()
    
    def timeline_clicked(self, event):
        """Handle timeline click to jump to frame."""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            pos = self.timeline_widget.plotItem.vb.mapSceneToView(event.scenePos())
            frame = int(np.clip(pos.x(), 0, self.num_frames - 1))
            self.current_frame = frame
            self.update_frame()
    
    def toggle_play(self):
        """Toggle playback."""
        if self.is_playing:
            self.is_playing = False
            self.play_timer.stop()
            self.play_button.setText("▶ Play")
        else:
            self.is_playing = True
            self.last_frame_time = time.time()  # Reset frame timer
            # Use faster timer interval for 60 FPS
            self.play_timer.setInterval(8)  # 8ms for ~120Hz polling, we'll control actual rate with frame skipping
            self.play_timer.start()
            self.play_button.setText("⏸ Pause")
    
    def next_frame(self):
        """Advance to next frame with frame skipping for smooth playback."""
        if not self.is_playing:
            return
            
        current_time = time.time()
        elapsed = current_time - self.last_frame_time
        
        # Calculate how many frames we should advance based on elapsed time
        frames_to_advance = int(elapsed / self.target_frame_interval * self.playback_speed)
        
        if frames_to_advance >= 1:
            # Advance frames (possibly skipping if we're behind)
            new_frame = min(self.current_frame + frames_to_advance, self.num_frames - 1)
            
            if new_frame >= self.num_frames - 1:
                self.current_frame = self.num_frames - 1
                self.update_frame(update_ui=True)
                self.toggle_play()  # Stop at end
            else:
                self.current_frame = new_frame
                # Only update UI elements every few frames during playback for speed
                update_ui = (self.current_frame % 10 == 0)
                self.update_frame(update_ui=update_ui)
                self.last_frame_time = current_time
    
    def speed_changed(self, text):
        """Handle playback speed change."""
        speed_map = {'0.25x': 0.25, '0.5x': 0.5, '1x': 1.0, '2x': 2.0, '4x': 4.0}
        self.playback_speed = speed_map.get(text, 1.0)
        # Keep timer fast, control speed through frame skipping
        self.play_timer.setInterval(8)
    
    def export_frame(self):
        """Export current frame with annotations."""
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Frame", f"frame_{self.current_frame:06d}.png", "Images (*.png *.jpg)"
        )
        if filename:
            # Create export image
            exporter = pg.exporters.ImageExporter(self.video_view)
            exporter.export(filename)
            print(f"Exported frame to: {filename}")
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        if event.key() == QtCore.Qt.Key.Key_Space:
            self.toggle_play()
        elif event.key() == QtCore.Qt.Key.Key_Left:
            self.current_frame = max(0, self.current_frame - 1)
            self.update_frame()
        elif event.key() == QtCore.Qt.Key.Key_Right:
            self.current_frame = min(self.num_frames - 1, self.current_frame + 1)
            self.update_frame()
        elif event.key() == QtCore.Qt.Key.Key_Up:
            self.conf_threshold = min(1.0, self.conf_threshold + 0.05)
            self.conf_slider.setValue(int(self.conf_threshold * 100))
        elif event.key() == QtCore.Qt.Key.Key_Down:
            self.conf_threshold = max(0.0, self.conf_threshold - 0.05)
            self.conf_slider.setValue(int(self.conf_threshold * 100))


def main():
    parser = argparse.ArgumentParser(description='Advanced YOLO detection visualizer with PyQtGraph')
    parser.add_argument('--video', type=str, required=True,
                       help='Path to video file')
    parser.add_argument('--zarr', type=str, required=True,
                       help='Path to zarr file with detections')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'gpu'],
                       help='Device for decord video reading (default: cpu)')
    
    args = parser.parse_args()
    
    # Create Qt application
    app = QtWidgets.QApplication(sys.argv)
    
    # Create and show visualizer
    viz = YOLODetectionVisualizer(
        video_path=args.video,
        zarr_path=args.zarr,
        device=args.device
    )
    viz.show()
    
    # Run application
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
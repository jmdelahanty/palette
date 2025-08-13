#!/usr/bin/env python3
"""
High-performance comparison visualizer using a Vispy canvas embedded in a PyQt UI.
This version uses a QThread to offload data loading and processing, ensuring a responsive UI.
"""

import argparse
from pathlib import Path
import numpy as np
import zarr
import h5py
from decord import VideoReader, cpu, gpu
from vispy import scene
from pyqtgraph.Qt import QtWidgets, QtCore
import sys
import os

# Set Decord EOF retry limit
os.environ['DECORD_EOF_RETRY_MAX'] = '20480'

class DataLoaderThread(QtCore.QThread):
    """A worker thread for loading video frames and detection data."""
    data_loaded = QtCore.pyqtSignal(object)

    def __init__(self, video_path, h5_path, zarr_path, device='cpu'):
        super().__init__()
        self.video_path = video_path
        self.h5_path = h5_path
        self.zarr_path = zarr_path
        self.device = device
        self.current_frame = 0
        self.running = True
        self.vr = None
        self.h5_bbox_map = {}
        self.zarr_bboxes = None
        self.zarr_scores = None
        self.zarr_n_detections = None
        self._load_initial_data()

    def _load_initial_data(self):
        """Loads metadata and detection files once."""
        ctx = cpu() if self.device == 'cpu' else gpu()
        self.vr = VideoReader(str(self.video_path), ctx=ctx)
        
        with h5py.File(self.h5_path, 'r') as hf:
            bbox_dataset = hf['/tracking_data/bounding_boxes']
            h5_frame_ids = bbox_dataset['payload_frame_id'][:]
            x_min, y_min = bbox_dataset['x_min'][:], bbox_dataset['y_min'][:]
            width, height = bbox_dataset['width'][:], bbox_dataset['height'][:]
            
            for i, frame_id in enumerate(h5_frame_ids):
                if frame_id not in self.h5_bbox_map: self.h5_bbox_map[frame_id] = []
                x1, y1 = x_min[i], y_min[i]
                self.h5_bbox_map[frame_id].append([x1, y1, x1 + width[i], y1 + height[i]])

        zarr_root = zarr.open(str(self.zarr_path), mode='r')
        self.zarr_bboxes = zarr_root['bboxes']
        self.zarr_scores = zarr_root['scores']
        self.zarr_n_detections = zarr_root['n_detections']

    def run(self):
        """The entry point for the thread."""
        while self.running:
            frame_data = self.load_frame_data(self.current_frame)
            if frame_data:
                self.data_loaded.emit(frame_data)
            # This sleep is important to prevent this thread from hogging the CPU
            self.msleep(10) 

    def load_frame_data(self, frame_idx):
        """Loads all data for a specific frame."""
        if not (0 <= frame_idx < len(self.vr)):
            return None

        frame = self.vr[frame_idx].asnumpy()
        h5_boxes = np.array(self.h5_bbox_map.get(frame_idx, []))
        
        num_zarr_dets = self.zarr_n_detections[frame_idx]
        zarr_boxes = self.zarr_bboxes[frame_idx, :num_zarr_dets] if num_zarr_dets > 0 else np.zeros((0,4))
        zarr_scores = self.zarr_scores[frame_idx, :num_zarr_dets] if num_zarr_dets > 0 else np.zeros((0,))

        return {
            'frame_idx': frame_idx,
            'frame': frame,
            'h5_boxes': h5_boxes,
            'zarr_boxes': zarr_boxes,
            'zarr_scores': zarr_scores
        }

    def set_frame(self, frame_idx):
        """Sets the current frame to be loaded by the thread."""
        self.current_frame = frame_idx

    def stop(self):
        """Stops the thread."""
        self.running = False
        self.wait()

class VispyPyQtVisualizer(QtWidgets.QMainWindow):
    def __init__(self, video_path, h5_path, zarr_path, device='cpu', target_fps=30):
        super().__init__()
        # --- Data Loading Thread ---
        self.data_loader_thread = DataLoaderThread(video_path, h5_path, zarr_path, device)
        self.data_loader_thread.data_loaded.connect(self.on_data_loaded)
        self.num_frames = len(self.data_loader_thread.vr)
        self.video_width = self.data_loader_thread.vr[0].shape[1]
        self.video_height = self.data_loader_thread.vr[0].shape[0]

        # --- Playback State ---
        self.current_frame = 0
        self.is_playing = False
        self.target_fps = target_fps

        # --- UI and Vispy Canvas Setup ---
        self._setup_ui_and_canvas()

        # --- Timer for Playback ---
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.next_frame)

        # --- Start the data loader thread ---
        self.data_loader_thread.start()
        
    def _setup_ui_and_canvas(self):
        """Initializes the PyQt UI and embeds the Vispy canvas."""
        self.setWindowTitle("Vispy + PyQt Comparison Visualizer")
        self.resize(1800, 800)
        
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)

        # --- Vispy Canvas ---
        self.canvas = scene.SceneCanvas(keys='interactive', show=True)
        main_layout.addWidget(self.canvas.native)

        # Grid for side-by-side views
        grid = self.canvas.central_widget.add_grid()
        self.view1 = grid.add_view(row=0, col=0, border_color='lime')
        self.view2 = grid.add_view(row=0, col=1, border_color='cyan')

        # Link cameras
        self.view1.camera = scene.PanZoomCamera(aspect=1)
        self.view2.camera = self.view1.camera
        self.view1.camera.rect = (0, 0, self.video_width, self.video_height)

        # Visuals for View 1 (H5)
        self.image1 = scene.visuals.Image(parent=self.view1.scene, method='subdivide')
        self.lines1 = scene.visuals.Line(parent=self.view1.scene, connect='segments', width=2)
        self.text1 = scene.visuals.Text(parent=self.view1.scene, color='lime', font_size=12, bold=True, anchor_x='left', anchor_y='bottom')

        # Visuals for View 2 (Zarr)
        self.image2 = scene.visuals.Image(parent=self.view2.scene, method='subdivide')
        self.lines2 = scene.visuals.Line(parent=self.view2.scene, connect='segments', width=2)
        self.text2 = scene.visuals.Text(parent=self.view2.scene, color='cyan', font_size=12, bold=True, anchor_x='left', anchor_y='bottom')
        
        # --- FIX: Set GL state and rendering order ---
        for vis in [self.lines1, self.lines2, self.text1, self.text2]:
            vis.set_gl_state('translucent', depth_test=False, cull_face=False)
        
        self.lines1.order = 1
        self.lines2.order = 1
        self.text1.order = 2
        self.text2.order = 2

        # --- Controls Layout ---
        controls_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(controls_layout)

        self.play_button = QtWidgets.QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_play)
        controls_layout.addWidget(self.play_button)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, self.num_frames - 1)
        self.slider.valueChanged.connect(self.slider_moved)
        controls_layout.addWidget(self.slider)

        self.frame_label = QtWidgets.QLabel(f"Frame: {self.current_frame} / {self.num_frames - 1}")
        controls_layout.addWidget(self.frame_label)

    @QtCore.pyqtSlot(object)
    def on_data_loaded(self, data):
        """This slot is called when the worker thread has loaded data."""
        self.image1.set_data(data['frame'])
        self.image2.set_data(data['frame'])

        # Update H5 BBoxes
        h5_scores = np.ones(len(data['h5_boxes'])) # Dummy scores
        self._update_bboxes_vispy(self.lines1, self.text1, data['h5_boxes'], h5_scores, "TRT", (0, 1, 0, 1))

        # Update Zarr BBoxes
        self._update_bboxes_vispy(self.lines2, self.text2, data['zarr_boxes'], data['zarr_scores'], "Offline", (0, 1, 1, 1))

        # Update UI elements
        self.slider.blockSignals(True)
        self.slider.setValue(data['frame_idx'])
        self.slider.blockSignals(False)
        self.frame_label.setText(f"Frame: {data['frame_idx']} / {self.num_frames - 1}")

    def _update_bboxes_vispy(self, line_visual, text_visual, boxes, scores, prefix, color):
        """Helper to draw bounding boxes and text on a Vispy visual."""
        if len(boxes) == 0:
            line_visual.set_data(pos=np.zeros((0, 2)))
            text_visual.text = ''
            return

        line_pos = []
        texts, text_positions = [], []
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            line_pos.extend([[x1, y1], [x2, y1], [x2, y1], [x2, y2], 
                             [x2, y2], [x1, y2], [x1, y2], [x1, y1]])
            texts.append(f"{prefix}: {scores[i]:.2f}")
            text_positions.append([x1, y1 - 5])
        
        line_visual.set_data(pos=np.array(line_pos, dtype=np.float32), color=color)
        text_visual.text = texts
        text_visual.pos = np.array(text_positions, dtype=np.float32)

    def slider_moved(self, value):
        self.current_frame = value
        self.data_loader_thread.set_frame(value)

    def toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_button.setText("Pause")
            self.timer.start(int(1000 / self.target_fps))
        else:
            self.play_button.setText("Play")
            self.timer.stop()

    def next_frame(self):
        if self.current_frame < self.num_frames - 1:
            self.current_frame += 1
            self.data_loader_thread.set_frame(self.current_frame)
        else:
            self.toggle_play()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_Space: self.toggle_play()
        elif event.key() == QtCore.Qt.Key.Key_Right:
            if self.current_frame < self.num_frames - 1:
                self.current_frame += 1; self.data_loader_thread.set_frame(self.current_frame)
        elif event.key() == QtCore.Qt.Key.Key_Left:
            if self.current_frame > 0:
                self.current_frame -= 1; self.data_loader_thread.set_frame(self.current_frame)
        elif event.key() == QtCore.Qt.Key.Key_Q or event.key() == QtCore.Qt.Key.Key_Escape:
            self.close()

    def closeEvent(self, event):
        """Ensure the worker thread is stopped when the window closes."""
        self.data_loader_thread.stop()
        event.accept()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vispy/PyQt YOLO Detections Comparison.")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--h5", required=True, type=Path)
    parser.add_argument("--zarr", required=True, type=Path)
    parser.add_argument("--device", default='cpu', choices=['cpu', 'gpu'])
    parser.add_argument("--fps", type=int, default=30)
    
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    visualizer = VispyPyQtVisualizer(
        video_path=args.video,
        h5_path=args.h5,
        zarr_path=args.zarr,
        device=args.device,
        target_fps=args.fps
    )
    visualizer.show()
    sys.exit(app.exec())
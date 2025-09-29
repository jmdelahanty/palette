#!/usr/bin/env python3
"""
Simple Zarr viewer with minimal dependencies.
"""

import os
# Fix the MKL/OpenMP issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend to avoid potential issues
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from pathlib import Path

def view_zarr_manual(zarr_path: str):
    """Manually load and view Zarr chunks."""
    
    zarr_path = Path(zarr_path)
    array_path = zarr_path / "raw_video" / "images_full"
    
    # Array parameters (from your import)
    n_frames = 45627
    height = 4512
    width = 4512
    frames_per_chunk = 64
    
    print(f"Array: {n_frames} frames, {height}x{width}, chunks of {frames_per_chunk}")
    
    def load_frame(frame_idx):
        """Load a specific frame from chunk files."""
        chunk_idx = frame_idx // frames_per_chunk
        frame_in_chunk = frame_idx % frames_per_chunk
        
        # Zarr v3 chunk path
        chunk_file = array_path / "c" / str(chunk_idx) / "0" / "0"
        
        if not chunk_file.exists():
            print(f"Warning: chunk {chunk_idx} not found")
            return np.zeros((height, width), dtype=np.uint8)
        
        try:
            # Read the chunk file
            with open(chunk_file, 'rb') as f:
                # Each frame is height * width bytes
                offset = frame_in_chunk * height * width
                f.seek(offset)
                data = f.read(height * width)
                
                if len(data) < height * width:
                    print(f"Warning: incomplete data for frame {frame_idx}")
                    return np.zeros((height, width), dtype=np.uint8)
                
                frame = np.frombuffer(data, dtype=np.uint8).reshape(height, width)
                return frame
        except Exception as e:
            print(f"Error loading frame {frame_idx}: {e}")
            return np.zeros((height, width), dtype=np.uint8)
    
    # Test load
    print("Loading first frame...")
    frame0 = load_frame(0)
    print(f"Frame 0: min={frame0.min()}, max={frame0.max()}, mean={frame0.mean():.1f}")
    
    # Create figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    plt.subplots_adjust(bottom=0.15)
    
    # Display first frame
    im = ax.imshow(frame0, cmap='gray', vmin=0, vmax=255)
    ax.set_title(f'Frame 0 / {n_frames - 1}')
    ax.axis('off')
    
    # Add slider
    ax_slider = plt.axes([0.1, 0.05, 0.65, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, n_frames - 1, 
                    valinit=0, valstep=1, valfmt='%d')
    
    # Add jump buttons
    ax_m100 = plt.axes([0.80, 0.05, 0.05, 0.03])
    btn_m100 = plt.Button(ax_m100, '-100')
    
    ax_p100 = plt.axes([0.90, 0.05, 0.05, 0.03])
    btn_p100 = plt.Button(ax_p100, '+100')
    
    current_frame = [0]  # Use list for mutability in closure
    
    def update_display(frame_idx):
        """Update the displayed frame."""
        frame_idx = max(0, min(frame_idx, n_frames - 1))
        current_frame[0] = frame_idx
        
        print(f"Loading frame {frame_idx}...")
        frame = load_frame(frame_idx)
        
        im.set_data(frame)
        ax.set_title(f'Frame {frame_idx} / {n_frames - 1}')
        
        # Update slider without triggering event
        slider.set_val(frame_idx)
        fig.canvas.draw_idle()
    
    def on_slider(val):
        """Handle slider change."""
        new_frame = int(slider.val)
        if new_frame != current_frame[0]:
            update_display(new_frame)
    
    def on_minus_100(event):
        """Jump back 100 frames."""
        update_display(current_frame[0] - 100)
    
    def on_plus_100(event):
        """Jump forward 100 frames."""
        update_display(current_frame[0] + 100)
    
    # Connect events
    slider.on_changed(on_slider)
    btn_m100.on_clicked(on_minus_100)
    btn_p100.on_clicked(on_plus_100)
    
    # Keyboard shortcuts
    def on_key(event):
        if event.key == 'left':
            update_display(current_frame[0] - 1)
        elif event.key == 'right':
            update_display(current_frame[0] + 1)
        elif event.key == 'pageup':
            update_display(current_frame[0] - 10)
        elif event.key == 'pagedown':
            update_display(current_frame[0] + 10)
        elif event.key == 'home':
            update_display(0)
        elif event.key == 'end':
            update_display(n_frames - 1)
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    print("\nControls:")
    print("  Slider: Navigate frames")
    print("  Buttons: Jump ±100 frames")
    print("  ← / →: Previous/Next frame")
    print("  PageUp/PageDown: Jump ±10 frames")
    print("  Home/End: First/Last frame")
    
    plt.show()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python viewer.py /path/to/zarr")
        sys.exit(1)
    
    view_zarr_manual(sys.argv[1])



    #!/usr/bin/env python3
# """
# Minimal viewer using only PIL/Pillow and tkinter.
# """

# from pathlib import Path
# import tkinter as tk
# from tkinter import ttk
# from PIL import Image, ImageTk
# import struct

# class ZarrViewer:
#     def __init__(self, zarr_path):
#         self.zarr_path = Path(zarr_path)
#         self.array_path = self.zarr_path / "raw_video" / "images_full"
        
#         # Array parameters
#         self.n_frames = 45627
#         self.height = 4512
#         self.width = 4512
#         self.frames_per_chunk = 64
#         self.current_frame = 0
        
#         # Create GUI
#         self.root = tk.Tk()
#         self.root.title(f"Zarr Viewer - {zarr_path}")
        
#         # Scale factor for display (since 4512x4512 is huge)
#         self.scale = 0.2  # Show at 20% size
#         self.display_size = int(self.width * self.scale)
        
#         # Canvas for image
#         self.canvas = tk.Canvas(
#             self.root, 
#             width=self.display_size, 
#             height=self.display_size,
#             bg='black'
#         )
#         self.canvas.pack()
        
#         # Frame info label
#         self.info_label = tk.Label(self.root, text="")
#         self.info_label.pack()
        
#         # Slider
#         self.slider = ttk.Scale(
#             self.root,
#             from_=0,
#             to=self.n_frames-1,
#             orient=tk.HORIZONTAL,
#             length=self.display_size,
#             command=self.on_slider_change
#         )
#         self.slider.set(0)
#         self.slider.pack()
        
#         # Buttons
#         button_frame = tk.Frame(self.root)
#         button_frame.pack()
        
#         tk.Button(button_frame, text="<<", command=lambda: self.jump(-100)).pack(side=tk.LEFT)
#         tk.Button(button_frame, text="<", command=lambda: self.jump(-1)).pack(side=tk.LEFT)
#         tk.Button(button_frame, text=">", command=lambda: self.jump(1)).pack(side=tk.LEFT)
#         tk.Button(button_frame, text=">>", command=lambda: self.jump(100)).pack(side=tk.LEFT)
        
#         # Bind keyboard
#         self.root.bind('<Left>', lambda e: self.jump(-1))
#         self.root.bind('<Right>', lambda e: self.jump(1))
#         self.root.bind('<Prior>', lambda e: self.jump(-10))  # Page Up
#         self.root.bind('<Next>', lambda e: self.jump(10))   # Page Down
#         self.root.bind('<Home>', lambda e: self.show_frame(0))
#         self.root.bind('<End>', lambda e: self.show_frame(self.n_frames-1))
        
#         # Load first frame
#         self.show_frame(0)
    
#     def load_frame_data(self, frame_idx):
#         """Load raw frame data from chunk file."""
#         chunk_idx = frame_idx // self.frames_per_chunk
#         frame_in_chunk = frame_idx % self.frames_per_chunk
        
#         chunk_file = self.array_path / "c" / str(chunk_idx) / "0" / "0"
        
#         if not chunk_file.exists():
#             return bytes(self.height * self.width)  # Return zeros
        
#         with open(chunk_file, 'rb') as f:
#             offset = frame_in_chunk * self.height * self.width
#             f.seek(offset)
#             data = f.read(self.height * self.width)
#             return data
    
#     def show_frame(self, frame_idx):
#         """Display a frame."""
#         frame_idx = max(0, min(frame_idx, self.n_frames - 1))
#         self.current_frame = frame_idx
        
#         # Load raw data
#         raw_data = self.load_frame_data(frame_idx)
        
#         # Create PIL Image from raw bytes
#         img = Image.frombytes('L', (self.width, self.height), raw_data)
        
#         # Resize for display
#         img_resized = img.resize(
#             (self.display_size, self.display_size), 
#             Image.Resampling.LANCZOS
#         )
        
#         # Convert to PhotoImage
#         self.photo = ImageTk.PhotoImage(img_resized)
        
#         # Update canvas
#         self.canvas.delete("all")
#         self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        
#         # Update info
#         self.info_label.config(text=f"Frame {frame_idx} / {self.n_frames - 1}")
        
#         # Update slider without triggering event
#         self.slider.set(frame_idx)
    
#     def jump(self, delta):
#         """Jump by delta frames."""
#         self.show_frame(self.current_frame + delta)
    
#     def on_slider_change(self, value):
#         """Handle slider change."""
#         new_frame = int(float(value))
#         if new_frame != self.current_frame:
#             self.show_frame(new_frame)
    
#     def run(self):
#         """Start the GUI."""
#         self.root.mainloop()

# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) != 2:
#         print("Usage: python viewer.py /path/to/zarr")
#         sys.exit(1)
    
#     viewer = ZarrViewer(sys.argv[1])
#     viewer.run()
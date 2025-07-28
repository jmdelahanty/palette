#!/usr/bin/env python3
"""
Interactive Parameter Visualizer
Real-time adjustment of fish detection parameters with live preview.
"""

import cv2
import numpy as np
import zarr
import argparse
from pathlib import Path
from skimage.morphology import disk, erosion, dilation
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from matplotlib.patches import Rectangle, Circle
import matplotlib.patches as patches

class InteractiveParameterVisualizer:
    def __init__(self, zarr_path, start_frame=0):
        """Initialize the interactive visualizer."""
        self.zarr_path = zarr_path
        self.current_frame = start_frame
        
        # Load Zarr data
        self.root = zarr.open(zarr_path, mode='r')
        self.images_ds = self.root['raw_video/images_ds']
        self.images_full = self.root['raw_video/images_full']
        self.background_ds = self.root['background_models/background_ds'][:]
        self.background_full = self.root['background_models/background_full'][:]
        
        self.num_frames = self.images_ds.shape[0]
        
        # Default parameters (from enhanced_unified_tracker.py)
        self.params = {
            'ds_thresh': 55,        # Detection threshold for downsampled image
            'roi_thresh': 115,      # Detection threshold for ROI
            'roi_size': 320,        # ROI size (square)
            'min_area': 5,          # Minimum area for keypoints
            'margin_factor': 1.5,   # Bounding box margin factor
            'se1_size': 1,          # Morphological structuring element 1
            'se2_size': 2,          # Morphological structuring element 2  
            'se4_size': 4           # Morphological structuring element 4
        }
        
        # Current detection results
        self.detection_results = None
        self.roi_image = None
        self.roi_coords = None
        
        # Setup matplotlib
        self.setup_figure()
        self.update_display()
    
    def setup_figure(self):
        """Set up the matplotlib figure with controls."""
        self.fig = plt.figure(figsize=(18, 12))
        
        # Create subplot layout - 4x5 grid to accommodate background panels
        # Top row: Original image, Background, Background subtraction, Thresholded, After morphology
        self.ax_original = plt.subplot2grid((4, 5), (0, 0), colspan=1, rowspan=1)
        self.ax_original.set_title('Original Frame')
        
        self.ax_background = plt.subplot2grid((4, 5), (0, 1), colspan=1, rowspan=1)
        self.ax_background.set_title('Background Model')
        
        self.ax_diff = plt.subplot2grid((4, 5), (0, 2), colspan=1, rowspan=1)
        self.ax_diff.set_title('Background Subtraction')
        
        self.ax_thresh = plt.subplot2grid((4, 5), (0, 3), colspan=1, rowspan=1)
        self.ax_thresh.set_title('After Threshold')
        
        self.ax_morph = plt.subplot2grid((4, 5), (0, 4), colspan=1, rowspan=1)
        self.ax_morph.set_title('After Morphology')
        
        # Second row: Main detection result, ROI crop, ROI processed, ROI keypoints, Info
        self.ax_main = plt.subplot2grid((4, 5), (1, 0), colspan=2, rowspan=1)
        self.ax_main.set_title('Fish Detection Result')
        
        self.ax_roi = plt.subplot2grid((4, 5), (1, 2), colspan=1, rowspan=1)
        self.ax_roi.set_title('ROI Crop')
        
        self.ax_roi_proc = plt.subplot2grid((4, 5), (1, 3), colspan=1, rowspan=1)
        self.ax_roi_proc.set_title('ROI Processed')
        
        self.ax_info = plt.subplot2grid((4, 5), (1, 4), colspan=1, rowspan=1)
        self.ax_info.set_title('Detection Results')
        self.ax_info.axis('off')
        
        # Third row: ROI background subtraction pipeline
        self.ax_roi_bg = plt.subplot2grid((4, 5), (2, 0), colspan=1, rowspan=1)
        self.ax_roi_bg.set_title('ROI Background')
        
        self.ax_roi_diff = plt.subplot2grid((4, 5), (2, 1), colspan=1, rowspan=1)
        self.ax_roi_diff.set_title('ROI Diff')
        
        self.ax_roi_thresh = plt.subplot2grid((4, 5), (2, 2), colspan=1, rowspan=1)
        self.ax_roi_thresh.set_title('ROI Threshold')
        
        self.ax_roi_morph = plt.subplot2grid((4, 5), (2, 3), colspan=1, rowspan=1)
        self.ax_roi_morph.set_title('ROI Morphology')
        
        self.ax_keypoints = plt.subplot2grid((4, 5), (2, 4), colspan=1, rowspan=1)
        self.ax_keypoints.set_title('Final Keypoints')
        
        # Bottom row: Parameter controls
        self.ax_controls = plt.subplot2grid((4, 5), (3, 0), colspan=5, rowspan=1)
        self.ax_controls.axis('off')
        
        # Create sliders
        self.create_sliders()
        
        # Create buttons
        self.create_buttons()
        
        plt.tight_layout()
    
    def create_sliders(self):
        """Create parameter adjustment sliders."""
        slider_height = 0.03
        slider_width = 0.15
        x_start = 0.05
        y_start = 0.05
        x_spacing = 0.2
        y_spacing = 0.08
        
        # Define slider parameters
        slider_configs = [
            ('ds_thresh', 'DS Threshold', 5, 150, self.params['ds_thresh']),
            ('roi_thresh', 'ROI Threshold', 50, 200, self.params['roi_thresh']),
            ('roi_size', 'ROI Size', 200, 400, self.params['roi_size']),
            ('min_area', 'Min Area', 1, 20, self.params['min_area']),
            ('margin_factor', 'Margin Factor', 1.0, 3.0, self.params['margin_factor']),
            ('se1_size', 'SE1 Size', 1, 5, self.params['se1_size']),
            ('se2_size', 'SE2 Size', 1, 5, self.params['se2_size']),
            ('se4_size', 'SE4 Size', 1, 8, self.params['se4_size'])
        ]
        
        self.sliders = {}
        
        for i, (param_name, label, min_val, max_val, initial) in enumerate(slider_configs):
            x_pos = x_start + (i % 4) * x_spacing
            y_pos = y_start + (i // 4) * y_spacing
            
            # Create slider axis
            ax_slider = plt.axes([x_pos, y_pos, slider_width, slider_height])
            
            # Create slider
            if param_name == 'margin_factor':
                slider = Slider(ax_slider, label, min_val, max_val, valinit=initial, 
                              valfmt='%.2f')
            else:
                slider = Slider(ax_slider, label, min_val, max_val, valinit=initial, 
                              valfmt='%d', valstep=1)
            
            slider.on_changed(self.on_parameter_changed)
            self.sliders[param_name] = slider
    
    def create_buttons(self):
        """Create control buttons."""
        button_width = 0.08
        button_height = 0.03
        button_spacing = 0.1
        y_button = 0.22
        
        # Previous frame button
        ax_prev = plt.axes([0.05, y_button, button_width, button_height])
        self.btn_prev = Button(ax_prev, '◀ Prev')
        self.btn_prev.on_clicked(self.prev_frame)
        
        # Next frame button
        ax_next = plt.axes([0.05 + button_spacing, y_button, button_width, button_height])
        self.btn_next = Button(ax_next, 'Next ▶')
        self.btn_next.on_clicked(self.next_frame)
        
        # Reset parameters button
        ax_reset = plt.axes([0.05 + 2 * button_spacing, y_button, button_width, button_height])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self.reset_parameters)
        
        # Save parameters button
        ax_save = plt.axes([0.05 + 3 * button_spacing, y_button, button_width, button_height])
        self.btn_save = Button(ax_save, 'Save Params')
        self.btn_save.on_clicked(self.save_parameters)
        
        # Frame selection
        ax_frame = plt.axes([0.5, y_button, 0.15, button_height])
        self.slider_frame = Slider(ax_frame, 'Frame', 0, self.num_frames-1, 
                                  valinit=self.current_frame, valfmt='%d', valstep=1)
        self.slider_frame.on_changed(self.on_frame_changed)
    
    def on_parameter_changed(self, val):
        """Handle parameter slider changes."""
        # Update parameters from sliders
        for param_name, slider in self.sliders.items():
            if param_name == 'margin_factor':
                self.params[param_name] = slider.val
            else:
                self.params[param_name] = int(slider.val)
        
        # Update display
        self.update_display()
    
    def on_frame_changed(self, val):
        """Handle frame slider changes."""
        self.current_frame = int(val)
        self.update_display()
    
    def prev_frame(self, event):
        """Go to previous frame."""
        if self.current_frame > 0:
            self.current_frame -= 1
            self.slider_frame.set_val(self.current_frame)
            self.update_display()
    
    def next_frame(self, event):
        """Go to next frame."""
        if self.current_frame < self.num_frames - 1:
            self.current_frame += 1
            self.slider_frame.set_val(self.current_frame)
            self.update_display()
    
    def reset_parameters(self, event):
        """Reset all parameters to defaults."""
        defaults = {
            'ds_thresh': 55,
            'roi_thresh': 115,
            'roi_size': 320,
            'min_area': 5,
            'margin_factor': 1.5,
            'se1_size': 1,
            'se2_size': 2,
            'se4_size': 4
        }
        
        for param_name, default_val in defaults.items():
            self.params[param_name] = default_val
            self.sliders[param_name].set_val(default_val)
        
        self.update_display()
    
    def save_parameters(self, event):
        """Save current parameters to file."""
        param_file = Path(self.zarr_path).parent / 'optimized_parameters.txt'
        
        with open(param_file, 'w') as f:
            f.write("# Optimized Fish Detection Parameters\n")
            f.write("# Generated by Interactive Parameter Visualizer\n\n")
            
            for param_name, value in self.params.items():
                f.write(f"{param_name} = {value}\n")
        
        print(f"✅ Parameters saved to: {param_file}")
    
    def detect_fish_in_downsampled(self, image):
        """Detect fish in downsampled image with current parameters."""
        try:
            # Background subtraction
            diff_ds = np.clip(self.background_ds.astype(np.int16) - image.astype(np.int16), 
                             0, 255).astype(np.uint8)
            
            # Thresholding
            im_binary = diff_ds >= self.params['ds_thresh']
            
            # Morphological operations
            se1 = disk(self.params['se1_size'])
            se4 = disk(self.params['se4_size'])
            
            im_eroded = erosion(im_binary, se1)
            im_processed = dilation(im_eroded, se4)
            
            # Find connected components
            labeled = label(im_processed)
            regions = regionprops(labeled)
            
            if not regions:
                return None, diff_ds, im_binary, im_eroded, im_processed
            
            # Get largest region (assumed to be fish)
            largest_region = max(regions, key=lambda r: r.area)
            
            # Calculate fish center in normalized coordinates
            centroid_norm = np.array(largest_region.centroid)[::-1] / np.array(image.shape)
            
            return {
                'centroid_norm': centroid_norm,
                'centroid_px': largest_region.centroid[::-1],
                'area': largest_region.area,
                'bbox': largest_region.bbox
            }, diff_ds, im_binary, im_eroded, im_processed
            
        except Exception as e:
            print(f"Error in fish detection: {e}")
            return None, None, None, None, None
    
    def extract_and_process_roi(self, fish_detection):
        """Extract ROI and process for keypoint detection."""
        if fish_detection is None:
            return None, None, None
        
        try:
            # Get current frame
            img_full = self.images_full[self.current_frame]
            
            # Calculate ROI position
            ds_shape = self.images_ds.shape[1:]  # (640, 640)
            full_shape = img_full.shape  # (4512, 4512)
            
            # Scale centroid to full image
            centroid_full = fish_detection['centroid_norm'] * np.array(full_shape)
            
            # Calculate ROI bounds
            roi_size = self.params['roi_size']
            roi_half = roi_size // 2
            
            x1 = int(centroid_full[0] - roi_half)
            y1 = int(centroid_full[1] - roi_half)
            x2 = x1 + roi_size
            y2 = y1 + roi_size
            
            # Ensure bounds are within image
            x1 = max(0, min(x1, full_shape[1] - roi_size))
            y1 = max(0, min(y1, full_shape[0] - roi_size))
            x2 = x1 + roi_size
            y2 = y1 + roi_size
            
            # Extract ROI
            roi = img_full[y1:y2, x1:x2]
            
            if roi.shape != (roi_size, roi_size):
                return None, None, None
            
            # Process ROI for keypoint detection
            background_roi = self.background_full[y1:y2, x1:x2]
            diff_roi = np.clip(background_roi.astype(np.int16) - roi.astype(np.int16), 
                              0, 255).astype(np.uint8)
            
            # Apply ROI threshold and morphology
            im_roi = diff_roi >= self.params['roi_thresh']
            
            se1 = disk(self.params['se1_size'])
            se2 = disk(self.params['se2_size'])
            
            im_roi = erosion(im_roi, se1)
            im_roi = dilation(im_roi, se2)
            im_roi = erosion(im_roi, se1)
            
            # Find keypoints
            labeled_roi = label(im_roi)
            roi_regions = [r for r in regionprops(labeled_roi) if r.area >= self.params['min_area']]
            
            return roi, im_roi, roi_regions, (x1, y1)
            
        except Exception as e:
            print(f"Error in ROI processing: {e}")
            return None, None, None, None
    
    def update_display(self):
        """Update all display panels with current parameters."""
        # Clear all axes
        axes_to_clear = [self.ax_original, self.ax_background, self.ax_diff, self.ax_thresh, 
                        self.ax_morph, self.ax_main, self.ax_roi, self.ax_roi_proc, 
                        self.ax_roi_bg, self.ax_roi_diff, self.ax_roi_thresh, 
                        self.ax_roi_morph, self.ax_keypoints]
        
        for ax in axes_to_clear:
            ax.clear()
        
        # Get current frame
        if self.current_frame >= self.num_frames:
            self.current_frame = 0
        
        img_ds = self.images_ds[self.current_frame]
        
        # === DOWNSAMPLED IMAGE PROCESSING PIPELINE ===
        
        # Display original frame
        self.ax_original.imshow(img_ds, cmap='gray')
        self.ax_original.set_title(f'Frame {self.current_frame}')
        
        # Display background model
        self.ax_background.imshow(self.background_ds, cmap='gray')
        self.ax_background.set_title('Background Model')
        
        # Detect fish in downsampled image
        fish_detection, diff_image, thresh_image, eroded_image, final_image = self.detect_fish_in_downsampled(img_ds)
        
        # Display background subtraction pipeline
        if diff_image is not None:
            self.ax_diff.imshow(diff_image, cmap='hot', vmin=0, vmax=255)
            self.ax_diff.set_title(f'BG Subtraction')
            
        if thresh_image is not None:
            self.ax_thresh.imshow(thresh_image, cmap='gray')
            self.ax_thresh.set_title(f'Thresh ≥{self.params["ds_thresh"]}')
            
        if final_image is not None:
            self.ax_morph.imshow(final_image, cmap='gray') 
            self.ax_morph.set_title(f'Morph (E{self.params["se1_size"]}/D{self.params["se4_size"]})')
        
        # === MAIN DETECTION RESULT ===
        
        self.ax_main.imshow(img_ds, cmap='gray')
        self.ax_main.set_title(f'Detection Result')
        
        detection_info = []
        
        if fish_detection is not None:
            # Draw fish detection on main image
            centroid_px = fish_detection['centroid_px']
            self.ax_main.plot(centroid_px[0], centroid_px[1], 'r+', markersize=15, markeredgewidth=3)
            
            # Draw ROI outline
            roi_size_display = self.params['roi_size'] * (img_ds.shape[0] / 4512)  # Scale to display
            roi_rect = Rectangle(
                (centroid_px[0] - roi_size_display/2, centroid_px[1] - roi_size_display/2),
                roi_size_display, roi_size_display,
                linewidth=2, edgecolor='yellow', facecolor='none'
            )
            self.ax_main.add_patch(roi_rect)
            
            detection_info.append(f"✅ Fish detected!")
            detection_info.append(f"DS Area: {fish_detection['area']} px")
            detection_info.append(f"Center: ({centroid_px[0]:.1f}, {centroid_px[1]:.1f})")
            
            # === ROI PROCESSING PIPELINE ===
            
            roi_result = self.extract_and_process_roi(fish_detection)
            if roi_result and len(roi_result) == 9:
                (roi, roi_bg, roi_diff, roi_thresh, roi_eroded, 
                 roi_dilated, roi_final, keypoints, roi_coords) = roi_result
                
                # Display ROI pipeline
                self.ax_roi.imshow(roi, cmap='gray')
                self.ax_roi.set_title(f'ROI Original')
                
                self.ax_roi_bg.imshow(roi_bg, cmap='gray')
                self.ax_roi_bg.set_title('ROI Background')
                
                self.ax_roi_diff.imshow(roi_diff, cmap='hot', vmin=0, vmax=255)
                self.ax_roi_diff.set_title('ROI BG Sub')
                
                self.ax_roi_thresh.imshow(roi_thresh, cmap='gray')
                self.ax_roi_thresh.set_title(f'ROI Thresh ≥{self.params["roi_thresh"]}')
                
                self.ax_roi_morph.imshow(roi_final, cmap='gray')
                self.ax_roi_morph.set_title(f'ROI Morph (E{self.params["se1_size"]}/D{self.params["se2_size"]}/E{self.params["se1_size"]})')
                
                # Display final keypoints
                self.ax_keypoints.imshow(roi, cmap='gray')
                self.ax_keypoints.set_title('Final Keypoints')
                
                # Draw keypoints on multiple panels
                colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta']
                
                for i, kp in enumerate(keypoints[:7]):  # Show up to 7 keypoints
                    y, x = kp.centroid
                    color = colors[i % len(colors)]
                    
                    # Draw on original ROI
                    circle1 = Circle((x, y), 5, color=color, fill=False, linewidth=2)
                    self.ax_roi.add_patch(circle1)
                    
                    # Draw on final keypoints view
                    circle2 = Circle((x, y), 8, color=color, fill=False, linewidth=3)
                    self.ax_keypoints.add_patch(circle2)
                    self.ax_keypoints.text(x+10, y, f'{i+1}', color=color, fontweight='bold', fontsize=10)
                    
                    # Draw on processed ROI
                    circle3 = Circle((x, y), 3, color=color, fill=False, linewidth=2)
                    self.ax_roi_morph.add_patch(circle3)
                
                detection_info.append(f"ROI Keypoints: {len(keypoints)}")
                if len(keypoints) >= 3:
                    detection_info.append("✅ Sufficient for tracking")
                    # Sort keypoints by area for triangle analysis
                    sorted_kp = sorted(keypoints, key=lambda k: k.area, reverse=True)[:3]
                    detection_info.append("Top 3 keypoints:")
                    for i, kp in enumerate(sorted_kp):
                        detection_info.append(f"  {i+1}: area={kp.area}")
                else:
                    detection_info.append("❌ Need ≥3 keypoints")
                    detection_info.append("Try adjusting:")
                    detection_info.append("• ROI Threshold")
                    detection_info.append("• Min Area")
                    detection_info.append("• SE sizes")
            else:
                # ROI processing failed
                detection_info.append("❌ ROI processing failed")
                
                # Show what we can
                if diff_image is not None:
                    self.ax_roi.imshow(diff_image, cmap='hot')
                    self.ax_roi.set_title('No ROI (show DS diff)')
                
        else:
            detection_info.append("❌ No fish detected")
            detection_info.append("Try adjusting:")
            detection_info.append("• Lower DS Threshold")
            detection_info.append("• Adjust SE sizes")
            detection_info.append("• Check background quality")
            
            # Still show processing steps
            if diff_image is not None:
                self.ax_roi.imshow(diff_image, cmap='hot')
                self.ax_roi.set_title('No fish - DS diff')
                
                self.ax_roi_bg.imshow(self.background_ds, cmap='gray')
                self.ax_roi_bg.set_title('DS Background')
        
        # === UPDATE INFO PANEL ===
        
        self.ax_info.clear()
        self.ax_info.axis('off')
        info_text = '\n'.join(detection_info)
        self.ax_info.text(0.05, 0.95, info_text, transform=self.ax_info.transAxes,
                         verticalalignment='top', fontfamily='monospace', fontsize=8)
        
        # Add parameter summary
        param_text = f"Current Parameters:\n"
        param_text += f"DS Thresh: {self.params['ds_thresh']}\n"
        param_text += f"ROI Thresh: {self.params['roi_thresh']}\n"
        param_text += f"ROI Size: {self.params['roi_size']}\n"
        param_text += f"Min Area: {self.params['min_area']}\n"
        param_text += f"Margin: {self.params['margin_factor']:.2f}\n"
        param_text += f"SE: {self.params['se1_size']}/{self.params['se2_size']}/{self.params['se4_size']}"
        
        self.ax_info.text(0.05, 0.35, param_text, transform=self.ax_info.transAxes,
                         verticalalignment='top', fontfamily='monospace', fontsize=7,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        # Add tips
        tips_text = "💡 Tips:\n"
        tips_text += "• Watch BG subtraction\n"
        tips_text += "• Bright = fish pixels\n" 
        tips_text += "• Threshold removes noise\n"
        tips_text += "• Morphology cleans up\n"
        tips_text += "• Need 3+ keypoints"
        
        self.ax_info.text(0.05, 0.05, tips_text, transform=self.ax_info.transAxes,
                         verticalalignment='bottom', fontfamily='monospace', fontsize=7,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
        
        # Refresh display
        self.fig.canvas.draw_idle()

def main():
    parser = argparse.ArgumentParser(
        description="Interactive parameter visualizer for fish detection",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Controls:
  • Use sliders to adjust detection parameters in real-time
  • Use ◀ Prev / Next ▶ buttons to navigate frames
  • Use Frame slider to jump to specific frames
  • Reset button restores default parameters
  • Save Params button saves optimized parameters to file

Tips:
  • If no fish detected, try lowering DS Threshold
  • If too noisy, increase SE sizes or thresholds
  • ROI size affects keypoint detection quality
  • Min Area filters out small noise regions
        """
    )
    
    parser.add_argument("zarr_path", type=str, help="Path to the Zarr file")
    parser.add_argument("--start-frame", type=int, default=0, 
                       help="Starting frame number (default: 0)")
    
    args = parser.parse_args()
    
    zarr_path = Path(args.zarr_path)
    if not zarr_path.exists():
        print(f"❌ Error: Zarr file not found: {zarr_path}")
        return
    
    print(f"🎛️  Starting Interactive Parameter Visualizer")
    print(f"📁 Zarr file: {zarr_path}")
    print(f"🎯 Use sliders to adjust parameters and see real-time effects")
    print(f"💾 Click 'Save Params' to save optimized parameters")
    
    # Create and run visualizer
    try:
        visualizer = InteractiveParameterVisualizer(args.zarr_path, args.start_frame)
        plt.show()
        
    except Exception as e:
        print(f"❌ Error running visualizer: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
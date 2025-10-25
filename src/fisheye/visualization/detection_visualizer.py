import argparse
import os, sys, shutil, subprocess
from pathlib import Path

quality_flags = None
blip_frames = None
jump_frames = None
current_artifact_index = 0
all_artifact_frames = []
detection_history = []
HISTORY_WINDOW = 10

REFINED_DETECT_GROUP = "refined_detect_runs"
LEGACY_REFINED_DETECT_GROUP = "refined_runs"

def load_quality_data(zarr_root, detect_run_name):
    """Load quality report data and the jump threshold used for analysis."""

    global quality_flags, blip_frames, jump_frames, all_artifact_frames, jump_threshold_px
    
    try:
        detect_group = zarr_root[f'detect_runs/{detect_run_name}']
        
        if 'quality_reports' not in detect_group:
            print("No quality reports found for this detect run")
            return False
        
        latest_quality = detect_group['quality_reports'].attrs.get('latest')
        if not latest_quality:
            print("No quality reports found")
            return False
        
        quality_group = detect_group[f'quality_reports/{latest_quality}']
        
        # Load jump threshold
        if 'artifact_detection_params' in quality_group.attrs:
            params = quality_group.attrs['artifact_detection_params']
            jump_threshold_px = params.get('jump_threshold', 0.0)
        
        # Load quality flags array
        quality_flags = quality_group['quality_flags'][:]
        
        # Compute artifact frame lists from quality_flags
        blip_frames = list(np.where(quality_flags == 2)[0])
        jump_frames = list(np.where(quality_flags == 3)[0])
        
        # Combine and sort all artifact frames
        all_artifact_frames = sorted(set(blip_frames + jump_frames))
        
        print(f"\n✓ Quality Report Loaded:")
        if jump_threshold_px > 0:
            print(f"  - Jump Threshold: {jump_threshold_px:.2f} pixels")
        print(f"  - Blips: {len(blip_frames)}")
        print(f"  - Jumps: {len(jump_frames)}")
        print(f"  - Total artifacts: {len(all_artifact_frames)}")
        
        return True
        
    except Exception as e:
        print(f"Could not load quality data: {e}")
        return False


def pick_zarr_path_textual(start_dir: str) -> str | None:
    """Open a terminal UI to pick a Zarr directory. Returns path or None."""
    start_dir = os.path.expanduser(start_dir)
    try:
        from textual.app import App, ComposeResult
        from textual.widgets import DirectoryTree, Footer, Header, Static
        from textual.containers import Horizontal
        from textual.reactive import reactive
    except Exception as e:
        print(f"⚠️ Textual not available ({e}). Try: pip install textual rich")
        return None

    class ZarrPickerApp(App[str]):
        CSS = """
        Screen { layout: vertical; }
        #status { height: 3; }
        #debug { height: 4; border: solid yellow; padding: 0 1; }
        #bar { height: 3; }
        DirectoryTree { height: 1fr; }
        """
        BINDINGS = [
            ("q", "quit", "Quit"),
            ("escape", "quit", "Quit"),
            ("enter", "toggle", "Open/close"),
            ("right", "open_dir", "Expand"),
            ("left", "close_dir", "Collapse"),
            ("l", "open_dir", "Open"),
            ("h", "close_dir", "Close"),
            ("s", "select", "Select"),
        ]
        current_path = reactive("")
        debug_text = reactive("Debug: Waiting for navigation...")

        def __init__(self, start_dir: str):
            super().__init__()
            self._start_dir = start_dir

        def compose(self) -> ComposeResult:
            yield Header(show_clock=False)
            yield Static(
                "↑↓ Navigate | →/Enter Expand | ← Collapse | S Select Zarr | Q/Esc Quit",
                id="status",
            )
            yield DirectoryTree(self._start_dir, id="tree")
            yield Static(self.debug_text, id="debug")
            yield Horizontal(Static("Navigate: ↑↓   Expand: →/Enter   Collapse: ←   Select: S   Quit: Q/Esc", id="bar"))
            yield Footer()

        def watch_debug_text(self, value: str) -> None:
            """Update debug panel when debug_text changes."""
            try:
                debug_widget = self.query_one("#debug", Static)
                debug_widget.update(value)
            except:
                pass

        def on_mount(self) -> None:
            tree = self.query_one("#tree", DirectoryTree)
            tree.focus()
            tree.can_focus = True
            self.debug_text = "Debug: Tree mounted, waiting for navigation..."
        
        def on_tree_node_highlighted(self, event) -> None:
            """Alternative event name."""
            self.debug_text = "Debug: on_tree_node_highlighted fired!"
            self._handle_highlight(event)
        
        def on_directory_tree_node_highlighted(self, event) -> None:
            """Standard event name."""
            self.debug_text = "Debug: on_directory_tree_node_highlighted fired!"
            self._handle_highlight(event)
        
        def on_key(self, event) -> None:
            """Catch any key press to update debug and check cursor."""
            tree = self.query_one("#tree", DirectoryTree)
            cursor = tree.cursor_node if hasattr(tree, 'cursor_node') else None
            
            if cursor:
                label = str(cursor.label) if hasattr(cursor, 'label') else "no label"
                self.debug_text = f"Debug: Key '{event.key}' pressed | Cursor: {label}"
                
                if label.endswith(".zarr"):
                    status = self.query_one("#status", Static)
                    status.update(f"[b green]✓ Zarr found:[/b green] {label} - Press S to select")
                    self.current_path = label
                else:
                    status = self.query_one("#status", Static)
                    status.update("↑↓ Navigate | →/Enter Expand | ← Collapse | S Select Zarr | Q/Esc Quit")
            else:
                self.debug_text = f"Debug: Key '{event.key}' pressed | No cursor node"
        
        def _handle_highlight(self, event) -> None:
            """Common highlighting logic."""
            debug_info = []
            node = event.node if hasattr(event, 'node') else None
            debug_info.append(f"Event has node: {node is not None}")
            
            if node is None:
                tree = self.query_one("#tree", DirectoryTree)
                node = tree.cursor_node if hasattr(tree, 'cursor_node') else None
                debug_info.append(f"Using cursor_node: {node is not None}")
            
            status = self.screen.query_one("#status", Static)
            
            if node:
                label = str(node.label) if hasattr(node, 'label') else "NO LABEL"
                debug_info.append(f"Label: {label}")
                
                if hasattr(node, 'data') and node.data:
                    if hasattr(node.data, 'path'):
                        path = str(node.data.path)
                        debug_info.append(f"Path: {path}")
                        self.current_path = path
                        
                        if path.endswith(".zarr"):
                            status.update(f"[b green]✓ Zarr found:[/b green] {os.path.basename(path)} - Press S to select")
                            self.debug_text = f"Debug: {' | '.join(debug_info)} | ZARR DETECTED!"
                            return
                    else:
                        debug_info.append("data exists but no path")
                else:
                    debug_info.append("No data attribute")
                
                if label.endswith(".zarr"):
                    status.update(f"[b green]✓ Zarr found:[/b green] {label} - Press S to select")
                    self.current_path = label
                    self.debug_text = f"Debug: {' | '.join(debug_info)} | ZARR DETECTED (by label)!"
                    return
            else:
                debug_info.append("No node found!")
            
            self.debug_text = f"Debug: {' | '.join(debug_info)}"
            status.update("↑↓ Navigate | →/Enter Expand | ← Collapse | S Select Zarr | Q/Esc Quit")
            if not self.current_path:
                self.current_path = ""

        def action_toggle(self) -> None:
            tree = self.query_one("#tree", DirectoryTree)
            if tree.cursor_node:
                tree.cursor_node.toggle()

        def action_open_dir(self) -> None:
            tree = self.query_one("#tree", DirectoryTree)
            if tree.cursor_node:
                if tree.cursor_node.allow_expand and not tree.cursor_node.is_expanded:
                    tree.cursor_node.expand()

        def action_close_dir(self) -> None:
            tree = self.query_one("#tree", DirectoryTree)
            if tree.cursor_node:
                if tree.cursor_node.is_expanded:
                    tree.cursor_node.collapse()

        def action_select(self) -> None:
            tree = self.query_one("#tree", DirectoryTree)
            node = tree.cursor_node
            if node:
                path = None
                if hasattr(node, "data") and node.data and hasattr(node.data, "path"):
                    path = str(node.data.path)
                elif hasattr(node, "path"):
                    path = str(node.path)
                elif hasattr(node, "label"):
                    label = str(node.label)
                    if self.current_path and self.current_path.endswith(label):
                        path = self.current_path
                    else:
                        path = label
                else:
                    path = self.current_path or self._start_dir
                
                if path and os.path.exists(path) and os.path.isdir(path):
                    if path.endswith(".zarr") or \
                       os.path.isfile(os.path.join(path, ".zarray")) or \
                       os.path.isfile(os.path.join(path, ".zgroup")):
                        print(f"✅ Selected: {path}")
                        self.exit(path)
                        return
                    
                    parent_path = os.path.dirname(path)
                    if parent_path.endswith(".zarr") or \
                       os.path.isfile(os.path.join(parent_path, ".zarray")) or \
                       os.path.isfile(os.path.join(parent_path, ".zgroup")):
                        status = self.screen.query_one("#status", Static)
                        status.update(
                            f"[b yellow]Hint:[/b yellow] Navigate to parent directory and press S to select the Zarr root"
                        )
                        return
                
                self.screen.query_one("#status", Static).update(
                    f"[b red]⚠ Not a Zarr root:[/b red] {os.path.basename(path) if path else 'Unknown'} - Look for .zarr folders"
                )

        def on_directory_tree_node_selected(self, event) -> None:
            self.action_toggle()

    try:
        return ZarrPickerApp(start_dir).run()
    except KeyboardInterrupt:
        return None
    
def is_zarr_root(p: str) -> bool:
    p = os.path.expanduser(p)
    if os.path.isdir(p) and p.endswith(".zarr"):
        return True
    return os.path.isfile(os.path.join(p, ".zarray")) or os.path.isfile(os.path.join(p, ".zgroup"))


# --- Backend selection must happen before importing pyplot ---
def configure_matplotlib(inline_mode: str):
    """
    Decide on a matplotlib backend so plots can render inline in VS Code/Jupyter.
    """
    import importlib

    def in_ipython_kernel() -> bool:
        try:
            import IPython
            ip = IPython.get_ipython()
            return ip is not None and getattr(ip, "kernel", None) is not None
        except Exception:
            return False

    def try_use_ipympl():
        try:
            import ipympl
            import matplotlib
            matplotlib.use("module://ipympl.backend_nbagg")
            return True
        except Exception:
            return False

    def use_inline_static():
        import matplotlib
        matplotlib.use("module://matplotlib_inline.backend_inline")

    if inline_mode not in {"auto", "widget", "static", "off"}:
        inline_mode = "auto"

    running_in_kernel = in_ipython_kernel()

    if inline_mode == "off":
        return

    if inline_mode == "widget":
        if not try_use_ipympl():
            use_inline_static()
        return

    if inline_mode == "static":
        use_inline_static()
        return

    # auto
    if running_in_kernel:
        if try_use_ipympl():
            return
        use_inline_static()

# Parse minimal args early to know inline preference
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--inline", choices=["auto", "widget", "static", "off"], default="auto",
                         help="Inline rendering mode for VS Code/Jupyter.")
_pre_args, _ = _pre_parser.parse_known_args()

configure_matplotlib(_pre_args.inline)

# Now safe to import pyplot & the rest
import zarr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider

try:
    from decord import VideoReader, cpu, gpu
    _HAVE_DECORD = True
except Exception as decord_exc:  # pragma: no cover - import guard
    VideoReader = None
    cpu = gpu = None
    _HAVE_DECORD = False
    _DECORD_ERROR = decord_exc

# Global variables
fig, ax = plt.subplots(figsize=(10, 10))
zarr_root = None
images_ds = None
frame_indices = None
bbox_coords = None
detection_ids = None
frame_to_detection_map = None  # NEW: map frame -> detection indices
output_dir = None
frame_slider = None
jump_threshold_px = 0.0

# Optional refined detections overlay
refined_enabled = False
refined_run_name = None
refined_bbox_coords = None
refined_frame_indices = None
refined_frame_map = None
refined_detection_source = None
refined_interpolated_count = 0


class VideoFrameSource:
    """Random-access video reader that mimics a minimal NumPy/Zarr interface."""

    def __init__(self, video_path: Path):
        self.path = Path(video_path).expanduser()
        if not self.path.exists():
            raise FileNotFoundError(f"Video file not found: {self.path}")

        if not _HAVE_DECORD:
            raise RuntimeError(
                f"Decord is required for video playback but could not be imported ({_DECORD_ERROR}). "
                "Install with `pip install decord` or ensure it is available in your environment."
            )

        self._cached_first_frame = None
        self._reader = self._init_reader(str(self.path))
        if self._reader is None:
            raise ValueError(f"Failed to open video with decord: {self.path}")

        # Prime metadata from the first frame
        try:
            raw_first = self._reader[0]
            first_frame = raw_first.asnumpy()
        except Exception as exc:
            raise ValueError(f"Unable to read first frame with decord: {exc}") from exc

        first_frame = np.array(first_frame, copy=True)
        if first_frame.ndim != 3 or first_frame.shape[2] not in (1, 3):
            raise ValueError(f"Unexpected frame shape from decord: {first_frame.shape}")

        self._frame_count = len(self._reader)
        if self._frame_count <= 0:
            raise ValueError(f"Video has zero frames or unknown length: {self.path}")

        self._height, self._width = first_frame.shape[:2]
        # Ensure frames are RGB; decord returns RGB by default.
        if first_frame.shape[2] == 1:
            first_frame = np.repeat(first_frame, 3, axis=2)

        self._cached_first_frame = first_frame
        self._shape = (self._frame_count, self._height, self._width, 3)

    @staticmethod
    def _init_reader(path: str):
        ctx_candidates = []
        if gpu is not None:
            try:
                ctx_candidates.append(gpu(0))
            except Exception:
                pass
        if cpu is not None:
            ctx_candidates.append(cpu(0))

        last_error = None
        for ctx in ctx_candidates:
            try:
                return VideoReader(path, ctx=ctx)
            except Exception as exc:
                last_error = exc
                continue
        if last_error is not None:
            raise RuntimeError(f"Decord failed to open video: {last_error}") from last_error
        return None

    @property
    def shape(self):
        return self._shape

    def __len__(self) -> int:
        return self._frame_count

    def __getitem__(self, idx: int) -> np.ndarray:
        if isinstance(idx, slice):
            raise TypeError("VideoFrameSource does not support slice access")
        idx = int(idx)
        if idx < 0 or idx >= self._frame_count:
            raise IndexError(f"Frame index out of range: {idx} (0..{self._frame_count - 1})")

        if idx == 0 and self._cached_first_frame is not None:
            return self._cached_first_frame.copy()

        try:
            frame = self._reader[idx]
        except Exception as exc:
            raise ValueError(f"Failed to read frame {idx} from {self.path} via decord: {exc}") from exc

        frame_np = frame.asnumpy()
        if frame_np.ndim == 2:
            frame_np = np.expand_dims(frame_np, axis=-1)
        if frame_np.shape[-1] == 1:
            frame_np = np.repeat(frame_np, 3, axis=2)
        return frame_np

    def release(self) -> None:
        self._reader = None
        self._cached_first_frame = None

    def __del__(self):
        self.release()


def update_frame(frame_idx):
    """
    Called when the slider moves. Draws a circle on 'clean' frames and dots on previous detections.
    """
    global detection_history, refined_enabled
    
    frame_idx = int(frame_idx)
    
    # When jumping (large frame gap), ensure we have at least the previous detection
    if detection_history:
        last_frame_in_history = detection_history[-1][0]
        if frame_idx - last_frame_in_history > HISTORY_WINDOW:
            # We jumped - find the most recent detection before current frame
            detection_history = []
            for prev_frame in range(frame_idx - 1, max(0, frame_idx - HISTORY_WINDOW), -1):
                if prev_frame in frame_to_detection_map:
                    # Found a previous detection, add it to history
                    det_indices = frame_to_detection_map[prev_frame]
                    prev_bbox = bbox_coords[det_indices[0]]  # Use first detection
                    center_x_norm, center_y_norm = prev_bbox[0], prev_bbox[1]
                    img_height, img_width = images_ds[prev_frame].shape[:2]
                    center_x = float(center_x_norm) * img_width
                    center_y = float(center_y_norm) * img_height
                    
                    # Check if it was a clean frame
                    prev_flag = 0
                    if quality_flags is not None and prev_frame < len(quality_flags):
                        prev_flag = quality_flags[prev_frame]
                    is_clean = (prev_flag == 0)
                    
                    detection_history.append((prev_frame, center_x, center_y, is_clean))
                    break  # Only need the most recent one
    
    ax.clear()

    # Load and display the frame
    image = images_ds[frame_idx]
    ax.imshow(image, cmap='gray')

    # Get detections in this frame
    det_indices = frame_to_detection_map.get(frame_idx, [])
    num_dets_in_frame = len(det_indices)
    
    # Build title with quality flag info
    title = f"Frame: {frame_idx} | Detections: {num_dets_in_frame}"
    
    # Get the quality flag for the current frame
    flag = 0 # Default to 'good' if flags are not loaded
    if quality_flags is not None and frame_idx < len(quality_flags):
        flag = quality_flags[frame_idx]
    
    flag_labels = {0: "", 2: " [BLIP]", 3: " [JUMP]", 4: " [MULTI-DET]"}
    flag_colors = {0: 'black', 2: 'orange', 3: 'magenta', 4: 'yellow'}
    
    if flag > 0:
        title += flag_labels.get(flag, "")
        ax.set_title(title, fontsize=12, color=flag_colors.get(flag, 'black'), fontweight='bold')
    else:
        # Add a label for clean frames to make it obvious
        title += " [CLEAN]"
        ax.set_title(title, fontsize=12, color='green', fontweight='bold')

    if num_dets_in_frame > 0:
        frame_bboxes = bbox_coords[det_indices]
        frame_ids = detection_ids[det_indices] if detection_ids is not None else [-1] * num_dets_in_frame

        for i, bbox in enumerate(frame_bboxes):
            assigned_id = int(frame_ids[i]) if detection_ids is not None else -1
            
            # Use green for clean frames, red for flagged frames
            box_color = 'lime' if flag == 0 else 'red'

            center_x_norm, center_y_norm, width_norm, height_norm = bbox
            
            img_height, img_width = image.shape[:2]
            center_x = float(center_x_norm) * img_width
            center_y = float(center_y_norm) * img_height
            box_w = float(width_norm) * img_width
            box_h = float(height_norm) * img_height

            x1 = center_x - (box_w / 2)
            y1 = center_y - (box_h / 2)

            rect = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=box_color, facecolor='none')
            ax.add_patch(rect)

            id_text = f"ID: {assigned_id}" if assigned_id != -1 else "Unassigned"
            # Keep label inside the frame by flipping below the box if needed
            label_y = y1 - 5 if y1 >= 15 else y1 + box_h + 15
            ax.text(
                x1,
                label_y,
                id_text,
                color='black',
                fontsize=10,
                fontweight='bold',
                bbox=dict(facecolor=box_color, alpha=0.8, edgecolor='none', pad=2)
            )
            
            # Draw circle for good frames with threshold
            if flag == 0 and jump_threshold_px > 0:
                circle = patches.Circle((center_x, center_y), jump_threshold_px,
                                        linewidth=1.5,
                                        linestyle='-',
                                        edgecolor='lime',
                                        facecolor='none')
                ax.add_patch(circle)
            
            # Add current detection to history (first detection only)
            if i == 0:
                is_clean = (flag == 0)
                detection_history.append((frame_idx, center_x, center_y, is_clean))

    # Overlay refined detections if available
    if refined_enabled and refined_frame_map is not None:
        refined_indices = refined_frame_map.get(frame_idx, [])
        if refined_indices:
            refined_bboxes = refined_bbox_coords[refined_indices]
            refined_sources = refined_detection_source[refined_indices] if refined_detection_source is not None else None

            for i, bbox in enumerate(refined_bboxes):
                center_x_norm, center_y_norm, width_norm, height_norm = bbox
                img_height, img_width = image.shape[:2]
                center_x = float(center_x_norm) * img_width
                center_y = float(center_y_norm) * img_height
                box_w = float(width_norm) * img_width
                box_h = float(height_norm) * img_height

                x1 = center_x - (box_w / 2)
                y1 = center_y - (box_h / 2)

                is_interpolated = refined_sources[i] == 1 if refined_sources is not None else False
                color = 'orange' if is_interpolated else 'cyan'
                label = "Interpolated" if is_interpolated else "Refined"

                rect = patches.Rectangle((x1, y1), box_w, box_h, linewidth=1.5,
                                         edgecolor=color, facecolor='none', linestyle='--')
                ax.add_patch(rect)
                label_y = y1 + box_h + 15
                ax.text(x1, label_y, label, color='black', fontsize=9,
                        bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', pad=2))

    # Clean up old history entries outside the window
    detection_history = [(f, x, y, clean) for f, x, y, clean in detection_history 
                        if frame_idx - f < HISTORY_WINDOW]
    
    # Draw dots for previous detections (excluding current frame)
    for past_frame, past_x, past_y, was_clean in detection_history:
        if past_frame < frame_idx:  # Don't draw dot on current frame
            # Calculate age and alpha for fade effect
            age = frame_idx - past_frame
            alpha = 1.0 - (age / HISTORY_WINDOW)
            
            # Use cyan for clean detections, orange for flagged ones
            dot_color = 'cyan' if was_clean else 'orange'
            
            # Draw a small dot
            ax.plot(past_x, past_y, 'o', 
                   color=dot_color, 
                   markersize=4, 
                   alpha=alpha, 
                   zorder=10)

    ax.axis('off')
    fig.canvas.draw_idle()


def jump_to_next_artifact():
    """Jump to the next artifact frame."""
    global current_artifact_index, frame_slider, all_artifact_frames
    
    if not all_artifact_frames:
        print("No artifacts to navigate to")
        return
    
    current_frame = int(frame_slider.val)
    
    # Find next artifact after current frame
    next_artifacts = [f for f in all_artifact_frames if f > current_frame]
    
    if next_artifacts:
        frame_slider.set_val(next_artifacts[0])
        print(f"Jumped to artifact at frame {next_artifacts[0]}")
    else:
        # Wrap around to first artifact
        frame_slider.set_val(all_artifact_frames[0])
        print(f"Wrapped to first artifact at frame {all_artifact_frames[0]}")


def jump_to_prev_artifact():
    """Jump to the previous artifact frame."""
    global current_artifact_index, frame_slider, all_artifact_frames
    
    if not all_artifact_frames:
        print("No artifacts to navigate to")
        return
    
    current_frame = int(frame_slider.val)
    
    # Find previous artifact before current frame
    prev_artifacts = [f for f in all_artifact_frames if f < current_frame]
    
    if prev_artifacts:
        frame_slider.set_val(prev_artifacts[-1])
        print(f"Jumped to artifact at frame {prev_artifacts[-1]}")
    else:
        # Wrap around to last artifact
        frame_slider.set_val(all_artifact_frames[-1])
        print(f"Wrapped to last artifact at frame {all_artifact_frames[-1]}")


def on_key_press(event):
    global frame_slider
    if event.key == 's':
        save_current_frame()
    elif event.key == 'right' and frame_slider is not None:
        new_val = min(frame_slider.val + 1, frame_slider.valmax)
        frame_slider.set_val(new_val)
    elif event.key == 'left' and frame_slider is not None:
        new_val = max(frame_slider.val - 1, frame_slider.valmin)
        frame_slider.set_val(new_val)
    elif event.key == 'n':  # Next artifact
        jump_to_next_artifact()
    elif event.key == 'p':  # Previous artifact
        jump_to_prev_artifact()
    elif event.key in ('q', 'escape'):
        print("Closing figure...")
        plt.close(fig)


def save_current_frame():
    global output_dir, frame_slider
    if output_dir is None:
        print("Cannot save: Please specify an output directory using --output-dir.")
        return

    current_frame_idx = int(frame_slider.val) if frame_slider is not None else 0
    save_path = Path(output_dir) / f"detection_frame_{current_frame_idx:06d}.png"
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
    print(f"Frame {current_frame_idx} saved to: {save_path}")


def main(args):
    global zarr_root, images_ds, frame_indices, bbox_coords, frame_to_detection_map
    global output_dir, frame_slider, detection_ids
    global refined_enabled, refined_run_name, refined_bbox_coords, refined_frame_indices
    global refined_frame_map, refined_detection_source, refined_interpolated_count

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saved frames will be stored in: {output_dir}")

    try:
        # Open zarr file
        zarr_root = zarr.open(args.zarr_path, mode='r')
        
        # Try to get the latest detection run
        if 'detect_runs' in zarr_root:
            latest_detect_run = zarr_root['detect_runs'].attrs.get('latest')
            if latest_detect_run:
                detect_group = zarr_root[f'detect_runs/{latest_detect_run}']
            else:
                print("No completed detect runs found!")
                return
        else:
            print("No detect_runs found!")
            return

        # Load video frames - try different locations
        if 'raw_video/images_ds' in zarr_root:
            images_ds = zarr_root['raw_video/images_ds']
            print("Using downsampled video from raw_video/images_ds")
        elif 'raw_video/images_full' in zarr_root:
            images_ds = zarr_root['raw_video/images_full']
            print("Using full resolution video from raw_video/images_full")
        else:
            # Attempt to load frames from an external video file (e.g. YOLO-only runs)
            candidate_paths = []
            if args.video:
                candidate_paths.append(Path(args.video))

            root_attrs = getattr(zarr_root, "attrs", {})
            attr_path = root_attrs.get('source_video_path') or root_attrs.get('source_path')
            if attr_path:
                candidate_paths.append(Path(attr_path))

            attr_name = root_attrs.get('source_video')
            if attr_name:
                candidate_paths.append(Path(attr_name))
                candidate_paths.append(Path(args.zarr_path).resolve().parent / attr_name)

            resolved_path = None
            candidate_errors = []
            for candidate in candidate_paths:
                if candidate is None:
                    continue
                candidate = candidate.expanduser()
                if not candidate.is_absolute():
                    candidate = (Path(args.zarr_path).resolve().parent / candidate).resolve()
                if not candidate.exists():
                    candidate_errors.append(f"{candidate} (missing)")
                    continue
                try:
                    images_ds = VideoFrameSource(candidate)
                    resolved_path = candidate
                    break
                except Exception as exc:
                    candidate_errors.append(f"{candidate} ({exc})")
                    continue

            if images_ds is not None:
                print(f"Streaming frames from original video: {resolved_path}")
            else:
                search_summary = "; ".join(candidate_errors) if candidate_errors else "no candidates available"
                raise ValueError(
                    "No video data found in zarr file and unable to open source video "
                    f"(checked: {search_summary}).\n"
                    "Pass --video /path/to/video.mp4 if the original file is stored elsewhere."
                )

        # Load detection data - NEW STRUCTURE
        if 'frame_indices' in detect_group:
            frame_indices = detect_group['frame_indices'][:]
            print(f"Loaded frame_indices array with {len(frame_indices)} detections")
        else:
            raise ValueError("No frame_indices array found!")
            
        if 'bbox_norm_coords' in detect_group:
            bbox_coords = detect_group['bbox_norm_coords'][:]
            print(f"Loaded {len(bbox_coords)} bounding boxes in normalized format")
        else:
            raise ValueError("No bbox_norm_coords array found!")

        # Build frame -> detection indices mapping
        print("Building frame to detection mapping...")
        frame_to_detection_map = {}
        for det_idx, frame_idx in enumerate(frame_indices):
            if frame_idx not in frame_to_detection_map:
                frame_to_detection_map[frame_idx] = []
            frame_to_detection_map[frame_idx].append(det_idx)

        # Try to load detection IDs if available
        detection_ids = None
        if 'id_assignment_runs' in zarr_root:
            latest_id_run = zarr_root['id_assignment_runs'].attrs.get('latest')
            if latest_id_run:
                id_group = zarr_root[f'id_assignment_runs/{latest_id_run}']
                if 'detection_ids' in id_group:
                    detection_ids = id_group['detection_ids'][:]
                    print(f"Loaded detection IDs from {latest_id_run}")
                    
                    # Print ID distribution
                    unique_ids, counts = np.unique(detection_ids, return_counts=True)
                    print(f"  ID distribution: {dict(zip(unique_ids, counts))}")

                    # Verify provenance matches
                    id_source_run = id_group.attrs.get('source_detect_run')
                    if id_source_run and id_source_run != latest_detect_run:
                        print(f"  ⚠️ ID assignment was generated from detect run '{id_source_run}',")
                        print(f"     but visualizer is using '{latest_detect_run}'. IDs will be hidden.")
                        detection_ids = None
                    elif detection_ids.size != bbox_coords.shape[0]:
                        print("  ⚠️ detection_ids length does not match number of detections.")
                        print(f"     IDs length: {len(detection_ids)}, detections: {len(bbox_coords)}")
                        detection_ids = None
                    elif args.debug_ids:
                        print("\n🔍 DEBUG: ID Alignment")
                        print(f"  bbox_coords length:    {len(bbox_coords)}")
                        print(f"  frame_indices length:  {len(frame_indices)}")
                        print(f"  detection_ids length:  {len(detection_ids)}")

                        sample_frame = args.start_frame
                        if sample_frame in frame_to_detection_map:
                            det_indices = frame_to_detection_map[sample_frame]
                            print(f"\n  Frame {sample_frame}:")
                            print(f"    Detection indices: {det_indices}")
                            print(f"    Assigned IDs:      {detection_ids[det_indices]}")
                        else:
                            print(f"\n  Frame {sample_frame} has no detections")
            else:
                print("No completed ID assignment runs found")
        else:
            print("No ID assignment data found. Will only display bounding boxes.")

        # Load refined detections if requested
        refined_enabled = False
        refined_run_name = None
        refined_bbox_coords = np.empty((0, 4), dtype=np.float64)
        refined_frame_indices = np.empty(0, dtype=np.int64)
        refined_frame_map = {}
        refined_detection_source = None
        refined_interpolated_count = 0

        if args.show_refined:
            if REFINED_DETECT_GROUP in zarr_root:
                refined_root = zarr_root[REFINED_DETECT_GROUP]
            elif LEGACY_REFINED_DETECT_GROUP in zarr_root:
                refined_root = zarr_root[LEGACY_REFINED_DETECT_GROUP]
            else:
                refined_root = None

            if refined_root is None:
                print("No refined runs found; cannot overlay interpolated detections.")
            else:
                candidate_run = args.refined_run or refined_root.attrs.get('latest')
                if not candidate_run or candidate_run not in refined_root:
                    print(f"Refined run '{candidate_run}' not found.")
                else:
                    refined_group_root = refined_root[candidate_run]
                    source_detect = refined_group_root.attrs.get('source_detect_run')
                    if source_detect != latest_detect_run:
                        print(f"⚠️ Refined run '{candidate_run}' was generated from detect run '{source_detect}',")
                        print(f"   but current visualization uses '{latest_detect_run}'. Skipping refined overlay.")
                    elif 'interpolated' not in refined_group_root:
                        print(f"Refined run '{candidate_run}' does not contain an 'interpolated' stage.")
                    else:
                        interp_group = refined_group_root['interpolated']
                        refined_bbox_coords = interp_group['bbox_norm_coords'][:]
                        refined_frame_indices = interp_group['frame_indices'][:]
                        if 'detection_source' in interp_group:
                            refined_detection_source = interp_group['detection_source'][:]
                        else:
                            refined_detection_source = None

                        refined_frame_map = {}
                        for det_idx, frame_idx in enumerate(refined_frame_indices):
                            refined_frame_map.setdefault(int(frame_idx), []).append(det_idx)

                        refined_enabled = True
                        refined_run_name = candidate_run
                        total_refined = refined_bbox_coords.shape[0]
                        refined_interpolated_count = int(np.sum(refined_detection_source == 1)) if refined_detection_source is not None else total_refined
                        print(f"Refined detections loaded from {candidate_run} (total={total_refined}, interpolated={refined_interpolated_count}).")
        
        # Load quality data
        if 'detect_runs' in zarr_root:
            latest_detect_run = zarr_root['detect_runs'].attrs.get('latest')
            if latest_detect_run:
                load_quality_data(zarr_root, latest_detect_run)

        # Print summary
        num_frames = images_ds.shape[0]
        total_detections = len(frame_indices)
        frames_with_detections = len(frame_to_detection_map)
        print(f"\n📊 Data Summary:")
        print(f"  - Frames: {num_frames}")
        avg_per_frame = (total_detections/frames_with_detections) if frames_with_detections else 0.0
        print(f"  - Total detections: {total_detections}")
        print(f"  - Frames with detections: {frames_with_detections}")
        print(f"  - Average detections per frame (with dets): {avg_per_frame:.2f}")
        print(f"  - Frames without detections: {num_frames - frames_with_detections}")
        if refined_enabled:
            print(f"  - Refined overlay: {refined_run_name} (interpolated: {refined_interpolated_count})")

    except Exception as e:
        print(f"Error opening Zarr file or finding data: {e}")
        print("\nTrying to print zarr structure for debugging:")
        try:
            zarr_root = zarr.open(args.zarr_path, mode='r')
            print(zarr_root.tree())
        except:
            pass
        return

    # Create slider for frame selection
    num_frames = images_ds.shape[0]
    plt.subplots_adjust(bottom=0.2)
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    frame_slider = Slider(
        ax=ax_slider,
        label='Frame',
        valmin=0,
        valmax=num_frames - 1,
        valinit=max(0, min(args.start_frame, num_frames - 1)),
        valstep=1
    )

    # Connect events
    frame_slider.on_changed(update_frame)
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    
    # Display initial frame
    update_frame(args.start_frame)

    print("\n Starting Detection Visualizer...")
    print("Controls:")
    print("  - Use slider or arrow keys to navigate frames")
    print("  - Press 'n' to jump to NEXT artifact frame")
    print("  - Press 'p' to jump to PREVIOUS artifact frame")
    print("  - Press 's' to save the current view as a PNG")
    print("  - Press 'q' or 'Esc' to close the figure")

    plt.show()


if __name__ == "__main__":
    import sys, os
    from pathlib import Path

    in_ipykernel = "ipykernel" in sys.modules

    # If running in an IPython kernel and no CLI args supplied, open a non-blocking chooser
    if in_ipykernel and len(sys.argv) == 1:
        try:
            from ipyfilechooser import FileChooser
            from IPython.display import display, clear_output
            import ipywidgets as widgets

            start_dir = os.path.expanduser("~/Desktop")
            print(f"📂 Select a Zarr file to visualize… (starting in {start_dir})")

            fc = FileChooser(start_dir)
            fc.title = "Choose a .zarr folder"
            fc.show_only_dirs = False
            display(fc)

            go = widgets.Button(description="Load selected Zarr", button_style="primary")
            status = widgets.HTML(value="Choose a folder, then click <b>Load selected Zarr</b>.")
            display(go, status)

            def _on_click(_):
                sel = fc.selected or ""
                if not sel:
                    status.value = "<span style='color:crimson'>No selection.</span>"
                    return
                if not is_zarr_root(sel):
                    status.value = "<span style='color:crimson'>Not a Zarr root. Select a *.zarr folder.</span>"
                    return

                clear_output()
                print(f"✅ Selected: {sel}")

                ns = argparse.Namespace(
                    zarr_path=sel,
                    start_frame=0,
                    output_dir="detection_snapshots",
                    inline=_pre_args.inline,
                    debug_ids=False,
                    show_refined=False,
                    refined_run=None
                )
                main(ns)

            go.on_click(_on_click)
            raise SystemExit

        except Exception as e:
            print(
                f"⚠️ ipyfilechooser/ipywidgets issue: {e}. "
                "Install with `pip install ipyfilechooser ipywidgets`, "
                "or run with a path explicitly."
            )

    # Normal CLI flow
    parser = argparse.ArgumentParser(description="Visualize fish detections from new zarr structure.",
                                     parents=[_pre_parser])
    parser.add_argument("zarr_path", nargs="?", type=str,
                        help="Path to the Zarr folder.")
    parser.add_argument("--pick-textual", action="store_true",
                        help="Open a Textual TUI to choose a Zarr folder (works over SSH).")
    parser.add_argument("--start-dir", type=str, default="~/Desktop",
                        help="Start directory for pickers (default: ~/Desktop).")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="Frame to start on.")
    parser.add_argument("--output-dir", type=str, default="detection_snapshots",
                        help="Directory to save snapshot images.")
    parser.add_argument("--debug-ids", action="store_true",
                        help="Print detailed ID alignment diagnostics.")
    parser.add_argument("--show-refined", action="store_true",
                        help="Overlay detections from the latest (or specified) refined run.")
    parser.add_argument("--refined-run", type=str,
                        help="Specific refined run name to use for overlay.")
    parser.add_argument("--video", type=str,
                        help="Explicit path to source video when raw frames are absent in the Zarr.")
    args = parser.parse_args()

    # If textual picker requested or no path provided, launch TUI
    if args.pick_textual or (args.zarr_path is None):
        chosen = pick_zarr_path_textual(args.start_dir)
        if not chosen:
            print(" No valid Zarr selected. Exiting.")
            sys.exit(1)
        args.zarr_path = chosen

    main(args)

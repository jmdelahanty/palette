import argparse
import json
import os, sys, shutil, subprocess
from pathlib import Path
from typing import Optional

quality_flags = None
blip_frames = None
jump_frames = None
current_artifact_index = 0
all_artifact_frames = []
detection_history = []
HISTORY_WINDOW = 10

REFINED_DETECT_GROUP = "refined_detect_runs"
LEGACY_REFINED_DETECT_GROUP = "refined_runs"
flag_file_path = None
frame_flag_file_path = None
current_zarr_path = None
current_zarr_dir = None


def _resolve_manual_label(refined_group_root) -> Optional[str]:
    manual_label = refined_group_root.attrs.get("manual_review_latest")
    if manual_label and manual_label in refined_group_root:
        return str(manual_label)
    if "manual" in refined_group_root:
        return "manual"
    return None


def _pick_refined_group(refined_group_root, requested: Optional[str]) -> Optional[str]:
    if requested and requested != "auto":
        if requested == "manual":
            return _resolve_manual_label(refined_group_root)
        if requested in refined_group_root:
            return requested
        return None
    manual_label = _resolve_manual_label(refined_group_root)
    if manual_label:
        return manual_label
    if "interpolated" in refined_group_root:
        return "interpolated"
    if "filtered" in refined_group_root:
        return "filtered"
    return None


def _append_flagged_path() -> None:
    if not flag_file_path or not current_zarr_path:
        print("No flag file configured. Pass --flag-file to enable flagging.")
        return
    try:
        flag_path = Path(flag_file_path)
        flag_path.parent.mkdir(parents=True, exist_ok=True)
        existing = set()
        if flag_path.exists():
            with open(flag_path, "r", encoding="utf-8") as handle:
                existing = {line.strip() for line in handle if line.strip()}
        if current_zarr_path in existing:
            print(f"Already flagged: {current_zarr_path}")
            return
        with open(flag_path, "a", encoding="utf-8") as handle:
            handle.write(f"{current_zarr_path}\n")
        print(f"Flagged for retune: {current_zarr_path}")
        if current_zarr_dir:
            print(f"Flag file: {flag_path} (cwd: {current_zarr_dir})")
        else:
            print(f"Flag file: {flag_path}")
    except Exception as exc:
        print(f"Failed to flag path: {exc}")


def _load_frame_flags(path: Path) -> dict[str, list[int]]:
    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
        if not raw.strip():
            return {}
        data = json.loads(raw)
    except Exception as exc:
        raise RuntimeError(f"Failed to load frame flags from {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Frame flag file must contain a JSON object: {path}")
    parsed: dict[str, list[int]] = {}
    for key, value in data.items():
        if isinstance(value, list):
            frames: list[int] = []
            for item in value:
                try:
                    frames.append(int(item))
                except (TypeError, ValueError):
                    continue
            parsed[str(key)] = frames
    return parsed


def _append_flagged_frame(frame_idx: int) -> None:
    if not frame_flag_file_path or not current_zarr_path:
        print("No frame flag file configured. Pass --frame-flag-file to enable frame flagging.")
        return
    try:
        flag_path = Path(frame_flag_file_path)
        flag_path.parent.mkdir(parents=True, exist_ok=True)
        data = _load_frame_flags(flag_path)
        frames = set(data.get(current_zarr_path, []))
        frames.add(int(frame_idx))
        data[current_zarr_path] = sorted(frames)
        flag_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Flagged frame {frame_idx} for retune: {current_zarr_path}")
        if current_zarr_dir:
            print(f"Frame flag file: {flag_path} (cwd: {current_zarr_dir})")
        else:
            print(f"Frame flag file: {flag_path}")
    except Exception as exc:
        print(f"Failed to flag frame: {exc}")


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
refined_variant_label = None
refined_is_manual = False
using_refined_as_primary = False  # Track if we're using refined detections as primary source
primary_detection_source = None  # Source array for primary detections (0=original, 1=interpolated)


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
    global detection_history, refined_enabled, using_refined_as_primary, primary_detection_source
    
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

            # Determine if this detection is interpolated (when using refined as primary)
            det_idx = det_indices[i]
            is_interpolated = False
            if using_refined_as_primary and primary_detection_source is not None and det_idx < len(primary_detection_source):
                is_interpolated = (primary_detection_source[det_idx] == 1)

            # Use orange for interpolated, green for clean original, red for flagged
            if is_interpolated:
                box_color = 'orange'
            else:
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

            # Add interpolation indicator to label if applicable
            if assigned_id != -1:
                id_text = f"ID: {assigned_id}"
                if is_interpolated:
                    id_text += " [INTERP]"
            else:
                id_text = "Unassigned"
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

                if refined_is_manual:
                    color = 'magenta'
                    if refined_variant_label and refined_variant_label != "manual":
                        label = f"Manual ({refined_variant_label})"
                    else:
                        label = "Manual"
                else:
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
    key = (event.key or "").lower()
    if key == 's':
        save_current_frame()
    elif key == 'right' and frame_slider is not None:
        new_val = min(frame_slider.val + 1, frame_slider.valmax)
        frame_slider.set_val(new_val)
    elif key == 'left' and frame_slider is not None:
        new_val = max(frame_slider.val - 1, frame_slider.valmin)
        frame_slider.set_val(new_val)
    elif key == 'n':  # Next artifact
        jump_to_next_artifact()
    elif key == 'p':  # Previous artifact
        jump_to_prev_artifact()
    elif key == 'f':
        _append_flagged_path()
    elif key == 'b':
        if frame_slider is not None:
            _append_flagged_frame(int(frame_slider.val))
    elif key in ('q', 'escape'):
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
    global refined_variant_label, refined_is_manual
    global flag_file_path, frame_flag_file_path, current_zarr_path, current_zarr_dir
    global using_refined_as_primary, primary_detection_source

    current_zarr_path = str(Path(args.zarr_path).expanduser().resolve(strict=False))
    current_zarr_dir = os.getcwd()
    flag_file_path = args.flag_file
    frame_flag_file_path = args.frame_flag_file

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

        # Load arena assignments (without validating yet - we'll check against the appropriate detection source)
        detection_ids = None
        raw_detection_ids = None
        id_source_run = None
        id_source_refined_run = None
        latest_id_run = None
        if 'arena_assignment_runs' in zarr_root:
            latest_id_run = zarr_root['arena_assignment_runs'].attrs.get('latest')
            if latest_id_run:
                id_group = zarr_root[f'arena_assignment_runs/{latest_id_run}']
                if 'arena_ids' in id_group:
                    raw_detection_ids = id_group['arena_ids'][:]
                    print(f"Loaded arena assignments from {latest_id_run}")

                    # Print arena distribution
                    unique_ids, counts = np.unique(raw_detection_ids, return_counts=True)
                    print(f"  Arena distribution: {dict(zip(unique_ids, counts))}")

                    # Get provenance info
                    id_source_run = id_group.attrs.get('source_detect_run')
                    id_source_refined_run = id_group.attrs.get('source_refined_run')
            else:
                print("No completed arena assignment runs found")
        else:
            print("No arena assignment data found. Will only display bounding boxes.")

        # Initialize detection source variables (will be set based on what we're using)
        bbox_coords = None
        frame_indices = None
        frame_to_detection_map = None

        # Initialize refined detection variables
        refined_enabled = False
        refined_run_name = None
        refined_bbox_coords = np.empty((0, 4), dtype=np.float64)
        refined_frame_indices = np.empty(0, dtype=np.int64)
        refined_frame_map = {}
        refined_detection_source = None
        refined_interpolated_count = 0
        refined_variant_label = None
        refined_is_manual = False

        # Determine which detection source to use as primary
        # Strategy: When --show-refined is set, prioritize using refined detections if they exist and match IDs
        if args.refined_only:
            args.show_refined = True
        if args.show_refined:
            print("\n🔍 Checking for refined detections to use as primary source...")

            # Try to load refined detections
            if REFINED_DETECT_GROUP in zarr_root:
                refined_root = zarr_root[REFINED_DETECT_GROUP]
            elif LEGACY_REFINED_DETECT_GROUP in zarr_root:
                refined_root = zarr_root[LEGACY_REFINED_DETECT_GROUP]
            else:
                refined_root = None

            refined_available = False
            if refined_root is None:
                print("  No refined runs found - will use original detections")
            else:
                candidate_run = args.refined_run or refined_root.attrs.get('latest')
                if not candidate_run or candidate_run not in refined_root:
                    print(f"  Refined run '{candidate_run}' not found - will use original detections")
                else:
                    refined_group_root = refined_root[candidate_run]
                    source_detect = refined_group_root.attrs.get('source_detect_run')

                    if source_detect != latest_detect_run:
                        print(f"  ⚠️ Refined run '{candidate_run}' was generated from detect run '{source_detect}',")
                        print(f"     but original detections are from '{latest_detect_run}'")
                        # Don't skip - this is expected when using refined detections

                    manual_label = _resolve_manual_label(refined_group_root)
                    refined_group_name = _pick_refined_group(refined_group_root, args.refined_variant)
                    if refined_group_name is None:
                        print(
                            f"  Refined run '{candidate_run}' does not contain requested group "
                            f"({args.refined_variant}) - will use original detections"
                        )
                    else:
                        refined_group = refined_group_root[refined_group_name]
                        # Load refined detection data
                        refined_bbox_coords = refined_group['bbox_norm_coords'][:]
                        refined_frame_indices = refined_group['frame_indices'][:]
                        refined_detection_source = refined_group.get('detection_source', None)
                        if refined_detection_source is not None:
                            refined_detection_source = refined_detection_source[:]

                        refined_run_name = candidate_run
                        refined_variant_label = refined_group_name
                        refined_is_manual = manual_label is not None and refined_group_name == manual_label
                        total_refined = refined_bbox_coords.shape[0]
                        refined_interpolated_count = int(np.sum(refined_detection_source == 1)) if refined_detection_source is not None else 0

                        print(f"  Loaded refined detections from {candidate_run}/{refined_group_name}:")
                        print(f"    • Total: {total_refined}")
                        if refined_detection_source is not None:
                            print(f"    • Interpolated: {refined_interpolated_count}")

                        refined_available = True

                        # Check if arena assignments match refined detections
                        ids_match_refined = (raw_detection_ids is not None and len(raw_detection_ids) == total_refined)
                        source_matches_refined = False
                        if id_source_refined_run:
                            source_matches_refined = (id_source_refined_run == candidate_run)
                        elif id_source_run:
                            source_matches_refined = (
                                id_source_run == candidate_run or
                                id_source_run.startswith(REFINED_DETECT_GROUP) or
                                id_source_run.startswith(LEGACY_REFINED_DETECT_GROUP)
                            )

                        if ids_match_refined:
                            print(f"  ✓ Arena assignments match refined detection count ({len(raw_detection_ids)} == {total_refined})")
                            if source_matches_refined:
                                print(f"  ✓ Arena assignment source run matches refined detections")
                            else:
                                source_label = id_source_refined_run or id_source_run
                                print(f"  ⚠️ Arena assignment source run is '{source_label}' but using anyway since count matches")

                            # Use refined as primary source
                            print(f"\n✅ Using refined detections as primary source\n")
                            bbox_coords = refined_bbox_coords
                            frame_indices = refined_frame_indices
                            using_refined_as_primary = True
                            primary_detection_source = refined_detection_source
                            detection_ids = raw_detection_ids
                            refined_enabled = False  # Not showing as overlay since it's primary
                        else:
                            print(
                                f"  ⚠️ Arena assignments do not match refined detections "
                                f"(Assignments: {len(raw_detection_ids) if raw_detection_ids is not None else 0}, Refined: {total_refined})"
                            )
                            if args.refined_only:
                                print("  Using refined detections as primary (forced by --refined-only); arena assignments disabled.")
                                bbox_coords = refined_bbox_coords
                                frame_indices = refined_frame_indices
                                using_refined_as_primary = True
                                primary_detection_source = refined_detection_source
                                detection_ids = None
                                refined_enabled = False
                            else:
                                print("  Will use original detections as primary and show refined as overlay")

                                # Build refined frame map for overlay
                                refined_frame_map = {}
                                for det_idx, frame_idx in enumerate(refined_frame_indices):
                                    refined_frame_map.setdefault(int(frame_idx), []).append(det_idx)
                                refined_enabled = True

        # If we haven't set bbox_coords yet, use original detections as primary
        if bbox_coords is None:
            print("\n📦 Using original detections as primary source")
            original_frame_indices = detect_group['frame_indices'][:]
            original_bbox_coords = detect_group['bbox_norm_coords'][:]
            print(f"Loaded original detections: {len(original_bbox_coords)} bounding boxes")
            bbox_coords = original_bbox_coords
            frame_indices = original_frame_indices
            using_refined_as_primary = False
            primary_detection_source = None

            # Validate arena assignments against original detections
            if raw_detection_ids is not None:
                # Check provenance
                if id_source_run and id_source_run != latest_detect_run:
                    print(f"  ⚠️ Arena assignment source run '{id_source_run}' does not match detect run '{latest_detect_run}'")
                    print(f"     Arena assignments will not be displayed")
                    detection_ids = None
                # Check size
                elif len(raw_detection_ids) != len(bbox_coords):
                    print(f"  ⚠️ Arena assignment count ({len(raw_detection_ids)}) does not match detection count ({len(bbox_coords)})")
                    print(f"     Arena assignments will not be displayed")
                    detection_ids = None
                else:
                    print(f"  ✓ Arena assignments validated against original detections")
                    detection_ids = raw_detection_ids

        # Build frame to detection mapping for primary source
        if frame_to_detection_map is None:
            print("Building frame to detection mapping...")
            frame_to_detection_map = {}
            for det_idx, frame_idx in enumerate(frame_indices):
                if frame_idx not in frame_to_detection_map:
                    frame_to_detection_map[frame_idx] = []
                frame_to_detection_map[frame_idx].append(det_idx)

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
            label = refined_run_name or "unknown"
            if refined_variant_label:
                label = f"{label}/{refined_variant_label}"
            if refined_detection_source is not None:
                print(f"  - Refined overlay: {label} (interpolated: {refined_interpolated_count})")
            else:
                print(f"  - Refined overlay: {label}")
        if using_refined_as_primary:
            interpolated_count = int(np.sum(primary_detection_source == 1)) if primary_detection_source is not None else 0
            original_count = total_detections - interpolated_count
            print(f"  - Using refined detections as primary source:")
            if primary_detection_source is not None:
                print(f"    • Original detections: {original_count}")
                print(f"    • Interpolated detections: {interpolated_count} (shown in orange)")
            if refined_variant_label:
                print(f"    • Source run: {refined_run_name}/{refined_variant_label}")
            else:
                print(f"    • Source run: {refined_run_name}")

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
    print("  - Press 'f' to flag this recording for retune (requires --flag-file)")
    print("  - Press 'b' to flag the current frame for retune (requires --frame-flag-file)")
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
                    refined_run=None,
                    refined_variant="auto",
                    refined_only=False,
                    flag_file=None,
                    frame_flag_file=None
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
    parser.add_argument(
        "--refined-variant",
        choices=["auto", "interpolated", "filtered", "manual"],
        default="auto",
        help="Refined group to visualize (default: auto prefers manual if available).",
    )
    parser.add_argument(
        "--refined-only",
        action="store_true",
        help="Use refined detections as primary when available (disables IDs if they don't match).",
    )
    parser.add_argument(
        "--flag-file",
        type=str,
        help="Append current zarr path to this file when pressing 'f'.",
    )
    parser.add_argument(
        "--frame-flag-file",
        type=str,
        help="Append current frame index to this JSON file when pressing 'b'.",
    )
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

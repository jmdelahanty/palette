import argparse
import os, sys, shutil, subprocess
from pathlib import Path

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
            ("right", "open_dir", "Expand"),     # Right arrow expands
            ("left", "close_dir", "Collapse"),   # Left arrow collapses
            ("l", "open_dir", "Open"),           # vi-style
            ("h", "close_dir", "Close"),
            ("s", "select", "Select"),           # S selects the highlighted dir
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

        # Give keyboard focus to the tree so arrows work immediately
        def on_mount(self) -> None:
            tree = self.query_one("#tree", DirectoryTree)
            tree.focus()
            # Also ensure the tree handles arrow keys natively
            tree.can_focus = True
            # Set initial debug text
            self.debug_text = "Debug: Tree mounted, waiting for navigation..."
        
        # Try multiple event handlers to catch cursor movement
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
                
                # Check if we're on a zarr
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
            # Debug info
            debug_info = []
            
            # Try to get the node directly from the event
            node = event.node if hasattr(event, 'node') else None
            debug_info.append(f"Event has node: {node is not None}")
            
            if node is None:
                # Fallback to cursor node
                tree = self.query_one("#tree", DirectoryTree)
                node = tree.cursor_node if hasattr(tree, 'cursor_node') else None
                debug_info.append(f"Using cursor_node: {node is not None}")
            
            status = self.screen.query_one("#status", Static)
            
            if node:
                # Get the label (filename) from the node
                label = str(node.label) if hasattr(node, 'label') else "NO LABEL"
                debug_info.append(f"Label: {label}")
                
                # Check for data.path
                if hasattr(node, 'data') and node.data:
                    if hasattr(node.data, 'path'):
                        path = str(node.data.path)
                        debug_info.append(f"Path: {path}")
                        self.current_path = path
                        
                        # Check if it's a zarr
                        if path.endswith(".zarr"):
                            status.update(f"[b green]✓ Zarr found:[/b green] {os.path.basename(path)} - Press S to select")
                            self.debug_text = f"Debug: {' | '.join(debug_info)} | ZARR DETECTED!"
                            return
                    else:
                        debug_info.append("data exists but no path")
                else:
                    debug_info.append("No data attribute")
                
                # Simple label check as fallback
                if label.endswith(".zarr"):
                    status.update(f"[b green]✓ Zarr found:[/b green] {label} - Press S to select")
                    self.current_path = label
                    self.debug_text = f"Debug: {' | '.join(debug_info)} | ZARR DETECTED (by label)!"
                    return
            else:
                debug_info.append("No node found!")
            
            # Update debug display
            self.debug_text = f"Debug: {' | '.join(debug_info)}"
            
            # Default navigation instructions
            status.update("↑↓ Navigate | →/Enter Expand | ← Collapse | S Select Zarr | Q/Esc Quit")
            if not self.current_path:
                self.current_path = ""

        # Enter toggles expand/collapse
        def action_toggle(self) -> None:
            tree = self.query_one("#tree", DirectoryTree)
            if tree.cursor_node:
                tree.cursor_node.toggle()

        # Right arrow or 'l' expands directory
        def action_open_dir(self) -> None:
            tree = self.query_one("#tree", DirectoryTree)
            if tree.cursor_node:
                if tree.cursor_node.allow_expand and not tree.cursor_node.is_expanded:
                    tree.cursor_node.expand()

        # Left arrow or 'h' collapses directory  
        def action_close_dir(self) -> None:
            tree = self.query_one("#tree", DirectoryTree)
            if tree.cursor_node:
                if tree.cursor_node.is_expanded:
                    tree.cursor_node.collapse()

        # Select only when user presses S
        def action_select(self) -> None:
            tree = self.query_one("#tree", DirectoryTree)
            node = tree.cursor_node
            if node:
                # Try multiple ways to get the path
                path = None
                if hasattr(node, "data") and node.data and hasattr(node.data, "path"):
                    path = str(node.data.path)
                elif hasattr(node, "path"):
                    path = str(node.path)
                elif hasattr(node, "label"):
                    # Use label if that's all we have
                    label = str(node.label)
                    # If stored current_path matches the label, use it
                    if self.current_path and self.current_path.endswith(label):
                        path = self.current_path
                    else:
                        path = label
                else:
                    path = self.current_path or self._start_dir
                
                # Check if it's a valid Zarr root
                if path and os.path.exists(path) and os.path.isdir(path):
                    # Check for .zarr extension or zarr structure files
                    if path.endswith(".zarr") or \
                       os.path.isfile(os.path.join(path, ".zarray")) or \
                       os.path.isfile(os.path.join(path, ".zgroup")):
                        print(f"✅ Selected: {path}")
                        self.exit(path)
                        return
                    
                    # Also check parent directory if we're on a child node
                    # (e.g., highlighting .zarray inside a zarr directory)
                    parent_path = os.path.dirname(path)
                    if parent_path.endswith(".zarr") or \
                       os.path.isfile(os.path.join(parent_path, ".zarray")) or \
                       os.path.isfile(os.path.join(parent_path, ".zgroup")):
                        status = self.screen.query_one("#status", Static)
                        status.update(
                            f"[b yellow]Hint:[/b yellow] Navigate to parent directory and press S to select the Zarr root"
                        )
                        return
                
                # Not a Zarr root
                self.screen.query_one("#status", Static).update(
                    f"[b red]⚠ Not a Zarr root:[/b red] {os.path.basename(path) if path else 'Unknown'} - Look for .zarr folders"
                )

        # Handle double-click by just toggling expand/collapse
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
    Modes:
      - 'widget': ipympl interactive backend (requires `pip install ipympl`)
      - 'static': matplotlib_inline static PNGs inline
      - 'off':    normal GUI backend (uses your system display)
      - 'auto':   pick best based on environment & availability
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
            import ipympl  # noqa: F401
            import matplotlib
            matplotlib.use("module://ipympl.backend_nbagg")
            return True
        except Exception:
            return False

    def use_inline_static():
        import matplotlib
        # Works in Jupyter/VS Code notebooks/Interactive
        matplotlib.use("module://matplotlib_inline.backend_inline")

    if inline_mode not in {"auto", "widget", "static", "off"}:
        inline_mode = "auto"

    running_in_kernel = in_ipython_kernel()

    if inline_mode == "off":
        return  # default GUI backend

    if inline_mode == "widget":
        if not try_use_ipympl():
            # Fallback to static if ipympl not available
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
    # else: not in a kernel → keep GUI backend (off)

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

# Global variables
fig, ax = plt.subplots(figsize=(10, 10))
zarr_root = None
images_ds = None
n_detections = None
bbox_coords = None
detection_ids = None
cumulative_detections = None
output_dir = None
frame_slider = None


def update_frame(frame_idx):
    """
    Called when the slider moves. Draws the frame, detections, and IDs.
    """
    frame_idx = int(frame_idx)
    ax.clear()

    image = images_ds[frame_idx]
    ax.imshow(image, cmap='gray')

    num_dets_in_frame = int(n_detections[frame_idx])
    ax.set_title(f"Frame: {frame_idx} | Detections: {num_dets_in_frame}", fontsize=12)

    if num_dets_in_frame > 0:
        start_idx = int(cumulative_detections[frame_idx])
        end_idx = int(cumulative_detections[frame_idx + 1])

        frame_bboxes = bbox_coords[start_idx:end_idx]
        frame_ids = detection_ids[start_idx:end_idx] if detection_ids is not None else [-1] * num_dets_in_frame

        for i, bbox in enumerate(frame_bboxes):
            assigned_id = int(frame_ids[i]) if detection_ids is not None else -1
            box_color = 'lime' if assigned_id != -1 else 'red'

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
            ax.text(x1, y1 - 5, id_text, color=box_color, fontsize=10, fontweight='bold')

    ax.axis('off')
    fig.canvas.draw_idle()


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
    global zarr_root, images_ds, n_detections, bbox_coords, cumulative_detections, output_dir, frame_slider, detection_ids

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saved frames will be stored in: {output_dir}")

    try:
        zarr_root = zarr.open(args.zarr_path, mode='r')
        latest_detect_run = zarr_root['detect_runs'].attrs['latest']
        detect_group = zarr_root[f'detect_runs/{latest_detect_run}']

        images_ds = zarr_root['raw_video/images_ds']
        n_detections = detect_group['n_detections'][:]
        bbox_coords = detect_group['bbox_norm_coords'][:]
        cumulative_detections = np.cumsum(np.insert(n_detections, 0, 0))

        # Load detection IDs if available
        if 'id_assignments_runs' in zarr_root and 'latest' in zarr_root['id_assignments_runs'].attrs:
            latest_id_run = zarr_root['id_assignments_runs'].attrs['latest']
            id_group = zarr_root[f'id_assignments_runs/{latest_id_run}']
            detection_ids = id_group['detection_ids'][:]
            print("Loaded detection IDs.")
        else:
            print("No ID assignment data found. Will only display bounding boxes.")

    except Exception as e:
        print(f"Error opening Zarr file or finding data: {e}")
        return

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

    frame_slider.on_changed(update_frame)
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    update_frame(args.start_frame)

    print("\n🚀 Starting Detection Visualizer...")
    print("Controls:")
    print("  - Press 's' to save the current view as a PNG.")
    print("  - Press 'q' or 'Esc' to close the figure.")
    print("  - Close the window to quit (if GUI).")

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
            fc.title = "Choose a .zarr folder or a directory containing .zarray/.zgroup"
            fc.show_only_dirs = False  # allow clicking a *.zarr directory by name
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
                    status.value = "<span style='color:crimson'>Not a Zarr root. Select a *.zarr folder or a dir with .zarray/.zgroup.</span>"
                    return

                # Clean up UI then launch the visualizer with the chosen path
                clear_output()
                print(f"✅ Selected: {sel}")

                # Build args namespace and call main() directly (no argparse)
                ns = argparse.Namespace(
                    zarr_path=sel,
                    start_frame=0,
                    output_dir="detection_snapshots",
                    inline=_pre_args.inline  # whatever was chosen for inline ("auto" by default)
                )
                main(ns)

            go.on_click(_on_click)

            # IMPORTANT: return so argparse doesn't run right now.
            # The callback above will call main() when the user clicks the button.
            raise SystemExit

        except Exception as e:
            print(
                f"⚠️ ipyfilechooser/ipywidgets issue: {e}. "
                "Install with `pip install ipyfilechooser ipywidgets`, "
                "or run with a path explicitly."
            )

    # --- normal CLI flow below ---
    parser = argparse.ArgumentParser(description="Visualize fish detections and assigned IDs.",
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
    args = parser.parse_args()

    # If textual picker requested or no path provided, launch TUI
    if args.pick_textual or (args.zarr_path is None):
        chosen = pick_zarr_path_textual(args.start_dir)
        if not chosen:
            print("❌ No valid Zarr selected. Exiting.")
            sys.exit(1)
        args.zarr_path = chosen

    main(args)
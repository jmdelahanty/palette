# src/fisheye/cli/interactive_launcher.py
"""
Interactive Textual UI for launching the FishEye pipeline.

Provides a user-friendly interface for:
- Browsing and selecting files with DirectoryTree
- Choosing pipeline stages
- Running tuners
"""

import os
from pathlib import Path
from typing import Optional, List
import subprocess
import sys

try:
    from textual.app import App, ComposeResult
    from textual.widgets import (
        DirectoryTree, Footer, Header, Static, Button, 
        Checkbox, Label, Input, Select
    )
    from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
    from textual.reactive import reactive
    from textual.binding import Binding
except ImportError:
    print("⚠️ Textual not installed. Install with: pip install textual rich")
    sys.exit(1)


# Pipeline stage information
STAGE_INFO = {
    'import': {
        'desc': 'Import video into zarr format',
        'requires': [],
        'requires_video': True,
        'color': 'cyan'
    },
    'downsample': {
        'desc': 'Create downsampled video array',
        'requires': ['import'],
        'color': 'blue'
    },
    'background': {
        'desc': 'Compute background model',
        'requires': ['import'],
        'color': 'green'
    },
    'detect': {
        'desc': 'Detect fish in each frame',
        'requires': ['background'],
        'color': 'yellow'
    },
    'crop': {
        'desc': 'Crop regions around detections',
        'requires': ['detect'],
        'color': 'magenta'
    },
    'keypoints': {
        'desc': 'Detect anatomical keypoints',
        'requires': ['crop', 'background'],
        'color': 'cyan'
    },
    'track': {
        'desc': 'Track fish across frames',
        'requires': ['keypoints'],
        'color': 'green'
    },
    'refine': {
        'desc': 'Refine tracking results',
        'requires': ['keypoints'],
        'color': 'blue'
    },
    'assign_ids': {
        'desc': 'Assign consistent IDs',
        'requires': ['detect'],
        'color': 'yellow'
    }
}

STAGE_ORDER = ['import', 'downsample', 'background', 'detect', 'crop', 'keypoints', 'track', 'refine', 'assign_ids']

TUNER_INFO = {
    'mask': 'Tune dish mask detection (Hough circles)',
    'detect': 'Tune fish detection thresholds',
    'threshold': 'Alias for detect tuner',
}


class PipelineLauncherApp(App):
    """Interactive TUI for launching the FishEye pipeline."""
    
    CSS = """
    Screen {
        layout: vertical;
    }
    
    #header_info {
        height: 3;
        background: $primary;
        color: $text;
        content-align: center middle;
        text-style: bold;
    }
    
    #main_container {
        height: 1fr;
        layout: horizontal;
    }
    
    #left_panel {
        width: 1fr;
        border: solid $primary;
        padding: 1;
    }
    
    #right_panel {
        width: 2fr;
        border: solid $accent;
        padding: 1;
    }
    
    #status_bar {
        height: 3;
        background: $surface;
        color: $text;
        padding: 0 1;
    }
    
    .bookmark_bar {
        height: auto;
        margin: 0 0 1 0;
    }
    
    .bookmark_bar Button {
        margin: 0 1 0 0;
        min-width: 10;
    }
    
    .stage_checkbox {
        margin: 0 2;
    }
    
    .section_header {
        text-style: bold;
        color: $accent;
        margin: 1 0;
    }
    
    Button {
        margin: 1 2;
    }
    
    .file_input {
        margin: 0 2;
    }
    
    .info_text {
        color: $text-muted;
        margin: 0 2;
    }
    """
    
    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("r", "run_pipeline", "Run Pipeline", show=True),
        Binding("t", "run_tuner", "Run Tuner", show=True),
        Binding("a", "toggle_all_stages", "Toggle All", show=True),
        Binding("s", "select_file", "Select File", show=True),
    ]
    
    selected_zarr = reactive("")
    selected_video = reactive("")
    selected_config = reactive("")
    status_message = reactive("Ready to launch pipeline")
    
    def __init__(self):
        super().__init__()
        self.stage_checkboxes = {}
        self.config = self._load_launcher_config()
        self.start_dir = self._get_start_directory()
    
    def _load_launcher_config(self) -> dict:
        """Load launcher configuration."""
        config_path = Path("configs/launcher.yaml")
        
        if config_path.exists():
            try:
                import yaml
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                print(f"Warning: Could not load launcher config: {e}")
        
        # Default config
        return {
            'start_directory': '~/Desktop',
            'bookmarks': [
                {'name': 'Desktop', 'path': '~/Desktop'},
                {'name': 'Home', 'path': '~'},
            ]
        }
    
    def _get_start_directory(self) -> str:
        """Get starting directory from config."""
        start_dir = self.config.get('start_directory', '~/Desktop')
        return os.path.expanduser(start_dir)
    
    def compose(self) -> ComposeResult:
        """Create the UI layout."""
        yield Header(show_clock=True)
        
        yield Static("🐟 FishEye Pipeline Launcher", id="header_info")
        
        with Container(id="main_container"):
            # Left panel - File browsing
            with Vertical(id="left_panel"):
                yield Label("📁 File Browser", classes="section_header")
                
                # Bookmark buttons
                bookmarks = self.config.get('bookmarks', [])
                if bookmarks:
                    yield Label("Quick Jump:", classes="info_text")
                    with Horizontal(classes="bookmark_bar"):
                        for bookmark in bookmarks:
                            yield Button(
                                bookmark['name'], 
                                id=f"bookmark_{bookmark['name']}", 
                                variant="default"
                            )
                
                yield Label("Navigate with arrows, press 'S' or click to select", classes="info_text")
                yield DirectoryTree(self.start_dir, id="file_tree")
                
                yield Label("\nSelected Files:", classes="section_header")
                yield Static("Zarr: None", id="zarr_display")
                yield Static("Video: None", id="video_display")
                yield Static("Config: configs/fisheye/default.yaml", id="config_display")
            
            # Right panel - Stage selection and actions
            with ScrollableContainer(id="right_panel"):
                yield Label("🔧 Pipeline Stages", classes="section_header")
                yield Label("Select stages to run:", classes="info_text")
                
                # Create checkboxes for each stage
                for stage in STAGE_ORDER:
                    info = STAGE_INFO[stage]
                    label = f"{stage}: {info['desc']}"
                    if info['requires']:
                        label += f" (requires: {', '.join(info['requires'])})"
                    
                    yield Checkbox(label, id=f"stage_{stage}", classes="stage_checkbox")
                
                yield Label("\n🎛️ Tuning Tools", classes="section_header")
                yield Label("Run parameter tuners:", classes="info_text")
                
                yield Select(
                    [(name, name) for name in TUNER_INFO.keys()],
                    prompt="Select tuner...",
                    id="tuner_select"
                )
                yield Button("🎨 Run Tuner", id="run_tuner_btn", variant="success")
                
                yield Label("\n🚀 Actions", classes="section_header")
                yield Button("▶️  Run Pipeline", id="run_pipeline_btn", variant="success")
                yield Button("🔍 Dry Run (show plan)", id="dry_run_btn", variant="default")
        
        yield Static(self.status_message, id="status_bar")
        yield Footer()
    
    def on_mount(self) -> None:
        """Set up initial state."""
        # Store references to checkboxes
        for stage in STAGE_ORDER:
            try:
                checkbox = self.query_one(f"#stage_{stage}", Checkbox)
                self.stage_checkboxes[stage] = checkbox
            except:
                pass
        
        # Set default config
        self.selected_config = "configs/fisheye/default.yaml"
    
    def on_directory_tree_file_selected(self, event) -> None:
        """Handle file selection from directory tree (click or Enter)."""
        self._handle_file_selection(event.path)
    
    def action_select_file(self) -> None:
        """Select the currently highlighted file in the tree (S key)."""
        try:
            tree = self.query_one("#file_tree", DirectoryTree)
            if tree.cursor_node and hasattr(tree.cursor_node, 'data'):
                path = tree.cursor_node.data.path
                self._handle_file_selection(path)
        except Exception as e:
            self.status_message = f"Could not select file: {e}"
    
    def _handle_file_selection(self, path) -> None:
        """Handle selection of a file from the tree."""
        path_str = str(path)
        
        # Check what type of file was selected
        if path_str.endswith('.zarr') or self._is_zarr_dir(path_str):
            self.selected_zarr = path_str
            zarr_display = self.query_one("#zarr_display", Static)
            zarr_display.update(f"[green]Zarr: {os.path.basename(path_str)}[/green]")
            self.status_message = f"✓ Selected zarr: {os.path.basename(path_str)}"
        elif self._is_video_file(path_str):
            self.selected_video = path_str
            video_display = self.query_one("#video_display", Static)
            video_display.update(f"[green]Video: {os.path.basename(path_str)}[/green]")
            self.status_message = f"✓ Selected video: {os.path.basename(path_str)}"
        elif path_str.endswith(('.yaml', '.yml')):
            self.selected_config = path_str
            config_display = self.query_one("#config_display", Static)
            config_display.update(f"[green]Config: {os.path.basename(path_str)}[/green]")
            self.status_message = f"✓ Selected config: {os.path.basename(path_str)}"
        else:
            self.status_message = "ℹ️ Select a .zarr directory, video file (.mp4, .avi, etc), or .yaml config"
    
    def _is_zarr_dir(self, path: str) -> bool:
        """Check if path is a zarr directory."""
        if not os.path.isdir(path):
            return False
        return (
            os.path.exists(os.path.join(path, '.zarray')) or
            os.path.exists(os.path.join(path, '.zgroup'))
        )
    
    def _is_video_file(self, path: str) -> bool:
        """Check if path is a video file."""
        video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.mpeg', '.mpg', '.wmv', '.flv', '.webm'}
        return Path(path).suffix.lower() in video_exts
    
    def watch_status_message(self, value: str) -> None:
        """Update status bar when message changes."""
        try:
            status = self.query_one("#status_bar", Static)
            status.update(value)
        except:
            pass
    
    def watch_selected_zarr(self, value: str) -> None:
        """Update zarr input when selection changes."""
        try:
            zarr_input = self.query_one("#zarr_input", PathInput)
            zarr_input.value = value
        except:
            pass
    
    def watch_selected_video(self, value: str) -> None:
        """Update video input when selection changes."""
        try:
            video_input = self.query_one("#video_input", PathInput)
            video_input.value = value
        except:
            pass
    
    def watch_selected_config(self, value: str) -> None:
        """Update config input when selection changes."""
        try:
            config_input = self.query_one("#config_input", PathInput)
            config_input.value = value
        except:
            pass
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        # Check if it's a bookmark button
        if button_id and button_id.startswith("bookmark_"):
            bookmark_name = button_id.replace("bookmark_", "")
            self._jump_to_bookmark(bookmark_name)
        elif button_id == "run_pipeline_btn":
            self.action_run_pipeline()
        elif button_id == "dry_run_btn":
            self._run_dry_run()
        elif button_id == "run_tuner_btn":
            self.action_run_tuner()
    
    def _jump_to_bookmark(self, bookmark_name: str) -> None:
        """Jump to a bookmarked location."""
        bookmarks = self.config.get('bookmarks', [])
        for bookmark in bookmarks:
            if bookmark['name'] == bookmark_name:
                path = os.path.expanduser(bookmark['path'])
                if os.path.exists(path):
                    # Update the tree
                    tree = self.query_one("#file_tree", DirectoryTree)
                    tree.path = path
                    self.status_message = f"Jumped to: {bookmark['name']}"
                else:
                    self.status_message = f"Path not found: {path}"
                return
    
    async def _browse_zarr(self) -> None:
        """Open zarr file browser."""
        from .file_browsers import FileBrowserApp, is_zarr_root
        
        # Exit this app and launch the browser
        browser = FileBrowserApp(
            start_dir=self.start_dir,
            title="Select Zarr Directory",
            filter_func=is_zarr_root,
            instructions="↑↓ Navigate | →/Enter Expand | ← Collapse | S Select Zarr | Q/Esc Quit"
        )
        
        # Run browser in suspended mode
        selected = await self.app.suspend(lambda: browser.run())
        
        if selected:
            self.selected_zarr = selected
            self.status_message = f"✓ Selected zarr: {os.path.basename(selected)}"
        else:
            self.status_message = "Zarr selection cancelled"
    
    async def _browse_video(self) -> None:
        """Open video file browser."""
        from .file_browsers import FileBrowserApp, is_video_file
        
        browser = FileBrowserApp(
            start_dir=self.start_dir,
            title="Select Video File",
            filter_func=is_video_file,
            instructions="↑↓ Navigate | →/Enter Expand | ← Collapse | S Select Video | Q/Esc Quit"
        )
        
        selected = await self.app.suspend(lambda: browser.run())
        
        if selected:
            self.selected_video = selected
            self.status_message = f"✓ Selected video: {os.path.basename(selected)}"
        else:
            self.status_message = "Video selection cancelled"
    
    async def _browse_config(self) -> None:
        """Open config file browser."""
        from .file_browsers import FileBrowserApp, is_yaml_file
        
        browser = FileBrowserApp(
            start_dir=os.path.dirname(self.start_dir) if os.path.isfile(self.start_dir) else self.start_dir,
            title="Select Config File",
            filter_func=is_yaml_file,
            instructions="↑↓ Navigate | →/Enter Expand | ← Collapse | S Select YAML | Q/Esc Quit"
        )
        
        selected = await self.app.suspend(lambda: browser.run())
        
        if selected:
            self.selected_config = selected
            self.status_message = f"✓ Selected config: {os.path.basename(selected)}"
        else:
            self.status_message = "Config selection cancelled"
    
    def _get_selected_stages(self) -> List[str]:
        """Get list of selected stages."""
        selected = []
        for stage, checkbox in self.stage_checkboxes.items():
            if checkbox.value:
                selected.append(stage)
        return selected
    
    def _build_command(self, dry_run: bool = False) -> Optional[List[str]]:
        """Build the pipeline command."""
        zarr_path = self.selected_zarr  # ← Changed
        video_path = self.selected_video  # ← Changed
        config_path = self.selected_config  # ← Changed
        
        if not zarr_path:
            self.status_message = "❌ Error: Select a zarr file from the tree!"
            return None
        
        stages = self._get_selected_stages()
        if not stages:
            self.status_message = "❌ Error: Select at least one stage!"
            return None
        
        # Check if import is selected and video path provided
        if 'import' in stages and not video_path:
            self.status_message = "❌ Error: Select a video file for import stage!"
            return None
        
        # Build command - use sys.executable instead of "python"
        cmd = [sys.executable, "-m", "fisheye", zarr_path]  # ← Changed
        
        if video_path:
            cmd.extend(["--video-path", video_path])
        
        if config_path:
            cmd.extend(["--config", config_path])
        
        cmd.extend(["--stages"] + stages)
        
        if dry_run:
            cmd.append("--dry-run")
        
        return cmd

    def action_run_tuner(self) -> None:
        """Run a tuner."""
        zarr_path = self.selected_zarr
        
        if not zarr_path:
            self.status_message = "❌ Error: Select a zarr file first!"
            return
        
        try:
            tuner_select = self.query_one("#tuner_select", Select)
            tuner_name = tuner_select.value
            
            if not tuner_name or tuner_name == Select.BLANK:
                self.status_message = "❌ Error: Select a tuner first!"
                return
            
            config_path = self.selected_config
            
            # Build tuner command
            cmd = [sys.executable, "-m", "fisheye", zarr_path, "--tune", str(tuner_name)]  # ← Changed
            
            if config_path:
                cmd.extend(["--config", config_path])
            
            self.status_message = f"Launching {tuner_name} tuner..."
            
            # Suspend the TUI while running the tuner
            with self.suspend():  # ← Changed - keeps launcher open
                try:
                    result = subprocess.run(cmd, check=True)
                    self.status_message = f"✓ Tuner completed successfully!"
                except subprocess.CalledProcessError as e:
                    self.status_message = f"❌ Tuner failed with exit code {e.returncode}"
                except KeyboardInterrupt:
                    self.status_message = "⚠️ Tuner interrupted"
            
        except Exception as e:
            self.status_message = f"❌ Error: {e}"
    
    def action_toggle_all_stages(self) -> None:
        """Toggle all stage checkboxes."""
        # Check if any are selected
        any_selected = any(cb.value for cb in self.stage_checkboxes.values())
        
        # If any selected, deselect all. Otherwise select all.
        new_state = not any_selected
        
        for checkbox in self.stage_checkboxes.values():
            checkbox.value = new_state
        
        self.status_message = f"{'✓' if new_state else '○'} {'Selected' if new_state else 'Deselected'} all stages"


def run_interactive_launcher(start_dir: str = "~/Desktop") -> Optional[List[str]]:
    """
    Run the interactive pipeline launcher.
    
    Returns:
        Command to execute, or None if cancelled
    """
    app = PipelineLauncherApp()
    app.start_dir = os.path.expanduser(start_dir)
    
    try:
        result = app.run()
        return result if isinstance(result, list) else None
    except KeyboardInterrupt:
        return None


if __name__ == "__main__":
    import sys
    
    cmd = run_interactive_launcher()
    
    if cmd:
        print("\n" + "="*60)
        print("🚀 Launching pipeline with command:")
        print(" ".join(cmd))
        print("="*60 + "\n")
        
        # Run the command
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Pipeline failed with exit code {e.returncode}")
            sys.exit(e.returncode)
        except KeyboardInterrupt:
            print("\n⚠️ Pipeline interrupted by user")
            sys.exit(130)
    else:
        print("\n❌ Pipeline launch cancelled")
        sys.exit(0)
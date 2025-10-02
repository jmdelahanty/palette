# src/fisheye/cli/interactive_launcher.py
"""
Interactive Textual UI for launching the FishEye pipeline.

Provides a user-friendly interface for:
- Browsing and selecting files with DirectoryTree
- Choosing pipeline stages
- Running tuners
- Real-time pipeline progress display
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
        Checkbox, Label, Input, Select, RichLog
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
        layout: vertical;
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
    
    #progress_panel {
        height: 15;
        border: solid yellow;
        padding: 1;
        margin-top: 1;
    }
    
    #progress_log {
        height: 1fr;
        border: solid gray;
    }
    """
    
    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("r", "run_pipeline", "Run Pipeline", show=True),
        Binding("t", "run_tuner", "Run Tuner", show=True),
        Binding("a", "toggle_all_stages", "Toggle All", show=True),
        Binding("s", "select_file", "Select File", show=True),
        Binding("c", "clear_log", "Clear Log", show=True),
    ]
    
    selected_zarr = reactive("")
    selected_video = reactive("")
    selected_config = reactive("")
    status_message = reactive("Ready to launch pipeline")
    is_running = reactive(False)
    current_stage = reactive("")
    
    def __init__(self):
        super().__init__()
        self.stage_checkboxes = {}
        self.config = self._load_launcher_config()
        self.start_dir = self._get_start_directory()
        self.progress_log = None
    
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
                
                for stage in STAGE_ORDER:
                    info = STAGE_INFO[stage]
                    label = f"{stage}: {info['desc']}"
                    if info['requires']:
                        label += f" (requires: {', '.join(info['requires'])})"
                    
                    yield Checkbox(label, id=f"stage_{stage}", classes="stage_checkbox")
                
                yield Label("\n🔧 Tuning Tools", classes="section_header")
                yield Label("Run parameter tuners:", classes="info_text")
                
                yield Select(
                    [(name, name) for name in TUNER_INFO.keys()],
                    prompt="Select tuner...",
                    id="tuner_select"
                )
                yield Button("🔧 Run Tuner", id="run_tuner_btn", variant="success")
                
                # NEW: Advanced Settings Section
                yield Label("\n⚙️ Advanced Settings", classes="section_header")
                yield Label("Execution scheduler:", classes="info_text")
                yield Select(
                    [
                        ("Processes (default)", "processes"),
                        ("Distributed (cluster)", "distributed"),
                        ("Threads", "threads"),
                        ("Single-threaded", "single-threaded")
                    ],
                    prompt="Select scheduler...",
                    value="processes",
                    id="scheduler_select"
                )
                
                yield Label("\n▶ Actions", classes="section_header")
                yield Button("▶ Run Pipeline", id="run_pipeline_btn", variant="success")
                yield Button("🔍 Dry Run (show plan)", id="dry_run_btn", variant="default")
                
                yield Label("\n📊 Pipeline Output", classes="section_header")
                with Container(id="progress_panel"):
                    yield RichLog(id="progress_log", wrap=True, highlight=True, markup=True)
        
        yield Static(self.status_message, id="status_bar")
        yield Footer()
    
    def on_mount(self) -> None:
        """Set up initial state."""
        for stage in STAGE_ORDER:
            try:
                checkbox = self.query_one(f"#stage_{stage}", Checkbox)
                self.stage_checkboxes[stage] = checkbox
            except:
                pass
        
        self.selected_config = "configs/fisheye/default.yaml"
        
        self.progress_log = self.query_one("#progress_log", RichLog)
        self.progress_log.write("[dim]Pipeline output will appear here...[/dim]")
    
    def on_directory_tree_file_selected(self, event) -> None:
        """Handle file selection from directory tree."""
        self._handle_file_selection(event.path)
    
    def action_select_file(self) -> None:
        """Select the currently highlighted file in the tree."""
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
        
        if path_str.endswith('.zarr') or self._is_zarr_dir(path_str):
            self.selected_zarr = path_str
            zarr_display = self.query_one("#zarr_display", Static)
            zarr_display.update(f"[green]Zarr: {os.path.basename(path_str)}[/green]")
            self.status_message = f"✓ Selected zarr: {os.path.basename(path_str)}"
            if self.progress_log:
                self.progress_log.write(f"[green]✓[/] Selected zarr: {os.path.basename(path_str)}")
        elif self._is_video_file(path_str):
            self.selected_video = path_str
            video_display = self.query_one("#video_display", Static)
            video_display.update(f"[green]Video: {os.path.basename(path_str)}[/green]")
            self.status_message = f"✓ Selected video: {os.path.basename(path_str)}"
            if self.progress_log:
                self.progress_log.write(f"[green]✓[/] Selected video: {os.path.basename(path_str)}")
        elif path_str.endswith(('.yaml', '.yml')):
            self.selected_config = path_str
            config_display = self.query_one("#config_display", Static)
            config_display.update(f"[green]Config: {os.path.basename(path_str)}[/green]")
            self.status_message = f"✓ Selected config: {os.path.basename(path_str)}"
            if self.progress_log:
                self.progress_log.write(f"[green]✓[/] Selected config: {os.path.basename(path_str)}")
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
            if self.is_running:
                zarr_file = os.path.basename(self.selected_zarr) if self.selected_zarr else "..."
                if self.current_stage:
                    status.update(f"[yellow]⏳ RUNNING[/yellow] | [cyan]{self.current_stage}[/cyan] on [dim]{zarr_file}[/dim]")
                else:
                    status.update(f"[yellow]⏳ RUNNING[/yellow] | [dim]{zarr_file}[/dim]")
            else:
                status.update(value)
        except:
            pass
    
    def watch_is_running(self, value: bool) -> None:
        """Update status bar when running state changes."""
        self.watch_status_message(self.status_message)
    
    def watch_current_stage(self, value: str) -> None:
        """Update status bar when current stage changes."""
        if self.is_running:
            self.watch_status_message(self.status_message)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
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
                    tree = self.query_one("#file_tree", DirectoryTree)
                    tree.path = path
                    self.status_message = f"Jumped to: {bookmark['name']}"
                else:
                    self.status_message = f"Path not found: {path}"
                return
    
    def _get_selected_stages(self) -> List[str]:
        """Get list of selected stages."""
        selected = []
        for stage, checkbox in self.stage_checkboxes.items():
            if checkbox.value:
                selected.append(stage)
        return selected
    
    def _build_command(self, dry_run: bool = False) -> Optional[List[str]]:
        """Build the pipeline command."""
        zarr_path = self.selected_zarr
        video_path = self.selected_video
        config_path = self.selected_config
        
        if not zarr_path:
            self.status_message = "❌ Error: Select a zarr file from the tree!"
            if self.progress_log:
                self.progress_log.write("[red]❌ Error: Select a zarr file first![/]")
            return None
        
        stages = self._get_selected_stages()
        if not stages:
            self.status_message = "❌ Error: Select at least one stage!"
            if self.progress_log:
                self.progress_log.write("[red]❌ Error: Select at least one stage![/]")
            return None
        
        if 'import' in stages and not video_path:
            self.status_message = "❌ Error: Select a video file for import stage!"
            if self.progress_log:
                self.progress_log.write("[red]❌ Error: Import stage requires a video file![/]")
            return None
        
        cmd = [sys.executable, "-m", "fisheye", zarr_path]
        
        if video_path:
            cmd.extend(["--video-path", video_path])
        
        if config_path:
            cmd.extend(["--config", config_path])
        
        cmd.extend(["--stages"] + stages)
        
        # Add scheduler selection
        try:
            scheduler_select = self.query_one("#scheduler_select", Select)
            if scheduler_select.value and scheduler_select.value != Select.BLANK:
                cmd.extend(["--scheduler", str(scheduler_select.value)])
        except:
            pass  # Scheduler selection not available, use default
        
        if dry_run:
            cmd.append("--dry-run")
        
        return cmd
    
    def _run_subprocess_with_output(self, cmd: List[str]) -> None:
        """Run subprocess and capture output in real-time."""
        self.progress_log.write(f"\n[bold cyan]▶ Running:[/] {' '.join(cmd)}\n")
        
        zarr_file = os.path.basename(self.selected_zarr) if self.selected_zarr else "unknown"
        stages = self._get_selected_stages()
        
        try:
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env
            )
            
            for line in process.stdout:
                line = line.rstrip()
                if line:
                    line_lower = line.lower()
                    for stage in stages:
                        if stage in line_lower or f"running {stage}" in line_lower or f"{stage} stage" in line_lower:
                            self.current_stage = stage
                            self.status_message = f"Running {stage} on {zarr_file}"
                            break
                    
                    self.progress_log.write(line)
            
            process.wait()
            
            self.current_stage = ""
            
            if process.returncode == 0:
                self.progress_log.write("\n[bold green]✓ Pipeline completed successfully![/]\n")
                self.status_message = f"✓ Completed: {zarr_file} ({', '.join(stages)})"
            else:
                self.progress_log.write(f"\n[bold red]❌ Pipeline failed (exit code {process.returncode})[/]\n")
                self.status_message = f"❌ Failed: {zarr_file} (exit {process.returncode})"
        
        except Exception as e:
            self.progress_log.write(f"\n[bold red]❌ Error: {e}[/]\n")
            self.status_message = f"❌ Error: {e}"
        
        finally:
            self.is_running = False
            self.current_stage = ""
            self.call_later(self.refresh)
    
    def action_run_pipeline(self) -> None:
        """Run the pipeline with live output display."""
        if self.is_running:
            if self.progress_log:
                self.progress_log.write("[yellow]⚠ Pipeline already running[/]")
            return
        
        cmd = self._build_command(dry_run=False)
        if not cmd:
            return
        
        self.is_running = True
        zarr_file = os.path.basename(self.selected_zarr) if self.selected_zarr else "..."
        stages = self._get_selected_stages()
        self.status_message = f"Starting pipeline: {', '.join(stages)}"
        
        self.run_worker(lambda: self._run_subprocess_with_output(cmd), thread=True)

    def _run_dry_run(self) -> None:
        """Show what would be run without executing."""
        cmd = self._build_command(dry_run=True)
        if cmd:
            if self.progress_log:
                self.progress_log.write(f"\n[yellow]Would run:[/] {' '.join(cmd)}\n")
            self.status_message = "Dry run - see log for command"

    def action_run_tuner(self) -> None:
        """Run a tuner."""
        zarr_path = self.selected_zarr
        
        if not zarr_path:
            self.status_message = "❌ Error: Select a zarr file first!"
            if self.progress_log:
                self.progress_log.write("[red]❌ Error: Select a zarr file first![/]")
            return
        
        try:
            tuner_select = self.query_one("#tuner_select", Select)
            tuner_name = tuner_select.value
            
            if not tuner_name or tuner_name == Select.BLANK:
                self.status_message = "❌ Error: Select a tuner first!"
                if self.progress_log:
                    self.progress_log.write("[red]❌ Error: Select a tuner first![/]")
                return
            
            config_path = self.selected_config
            
            cmd = [sys.executable, "-m", "fisheye", zarr_path, "--tune", str(tuner_name)]
            
            if config_path:
                cmd.extend(["--config", config_path])
            
            self.status_message = f"Launching {tuner_name} tuner..."
            if self.progress_log:
                self.progress_log.write(f"\n[cyan]Launching {tuner_name} tuner...[/]\n")
            
            with self.suspend():
                try:
                    result = subprocess.run(cmd, check=True)
                    self.status_message = f"✓ Tuner completed successfully!"
                except subprocess.CalledProcessError as e:
                    self.status_message = f"❌ Tuner failed with exit code {e.returncode}"
                except KeyboardInterrupt:
                    self.status_message = "⚠ Tuner interrupted"
            
            if self.progress_log:
                self.progress_log.write(f"[cyan]Tuner session ended: {self.status_message}[/]\n")
            
        except Exception as e:
            self.status_message = f"❌ Error: {e}"
            if self.progress_log:
                self.progress_log.write(f"[red]❌ Error: {e}[/]\n")
    
    def action_clear_log(self) -> None:
        """Clear the progress log."""
        if self.progress_log:
            self.progress_log.clear()
            self.progress_log.write("[dim]Log cleared[/dim]")
            self.status_message = "Log cleared"
    
    def action_toggle_all_stages(self) -> None:
        """Toggle all stage checkboxes."""
        any_selected = any(cb.value for cb in self.stage_checkboxes.values())
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
        print("▶ Launching pipeline with command:")
        print(" ".join(cmd))
        print("="*60 + "\n")
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"\n❌Pipeline failed with exit code {e.returncode}")
            sys.exit(e.returncode)
        except KeyboardInterrupt:
            print("\n⚠ Pipeline interrupted by user")
            sys.exit(130)
    else:
        print("\n❌ Pipeline launch cancelled")
        sys.exit(0)
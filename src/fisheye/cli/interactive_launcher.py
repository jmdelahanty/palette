# src/fisheye/cli/interactive_launcher.py
"""
Interactive Textual UI for launching the FishEye pipeline.

Provides a user-friendly interface for:
- Browsing and selecting files with DirectoryTree
- Choosing pipeline stages with real-time status display
- Running tuners
- Real-time pipeline progress display
"""

import os
from pathlib import Path
from typing import Optional, List
import subprocess
import sys
import zarr

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
        'desc': 'Assign consistent IDs (auto for single-dish)',
        'requires': ['detect'],
        'color': 'yellow'
    }
}

STAGE_ORDER = ['import', 'downsample', 'background', 'detect', 'crop', 'keypoints', 'track', 'refine', 'assign_ids']

TUNER_INFO = {
    'mask': 'Tune dish mask detection (Hough circles)',
    'subdish': 'Define sub-dish masks for spatial ID assignment (multi-dish only)',
    'detect': 'Tune fish detection thresholds',
    'threshold': 'Alias for detect tuner',
    'keypoints': 'Tune anatomical keypoint detection (swim bladder & eyes)',
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
        width: 40%;
        border: solid $primary;
        padding: 1;
    }

    #file_tree {
        height: 15;
        min-height: 8;
    }

    #right_panel {
        width: 60%;
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

    #experiment_info_panel {
        height: auto;
        border: solid cyan;
        padding: 1;
        margin-top: 1;
        background: $surface;
    }

    #stage_status_panel {
        height: auto;
        border: solid green;
        padding: 1;
        margin-top: 1;
        background: $surface;
    }
    """
    
    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("r", "run_pipeline", "Run Pipeline", show=True),
        Binding("t", "run_tuner", "Run Tuner", show=True),
        Binding("e", "configure_experiment", "Setup Experiment", show=True),
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
    
    def _check_zarr_stage_status(self, zarr_path: str) -> dict:
        """
        Check which stages are complete in a zarr file.
        
        Returns:
            Dictionary mapping stage names to completion status
        """
        import zarr
        from pathlib import Path
        
        status = {stage: '○ Not Run' for stage in STAGE_ORDER}
        
        if not zarr_path or not Path(zarr_path).exists():
            return status
        
        try:
            root = zarr.open_group(zarr_path, mode='r')
            
            # Check import
            if 'raw_video' in root and 'images_full' in root['raw_video']:
                status['import'] = '✓ Complete'
                
            # Check downsample
            if 'raw_video' in root and 'images_ds' in root['raw_video']:
                status['downsample'] = '✓ Complete'
                
            # Check background
            if 'background_runs' in root and len(list(root['background_runs'].group_keys())) > 0:
                latest = root['background_runs'].attrs.get('latest')
                status['background'] = f'✓ Complete ({latest})' if latest else '✓ Complete'
                
            # Check detect
            if 'detect_runs' in root and len(list(root['detect_runs'].group_keys())) > 0:
                latest = root['detect_runs'].attrs.get('latest')
                detect_group = root[f'detect_runs/{latest}'] if latest else None
                if detect_group and 'summary_statistics' in detect_group.attrs:
                    stats = detect_group.attrs['summary_statistics']
                    n_detections = stats.get('total_detections', 0)
                    frames_with_detections = stats.get('frames_with_detections', 0)
                    total_frames = stats.get('total_frames', 0)
                    if total_frames > 0:
                        percent = (frames_with_detections / total_frames) * 100
                        status['detect'] = f'✓ Complete ({n_detections} detections, {percent:.1f}%)'
                    else:
                        status['detect'] = f'✓ Complete ({n_detections} detections)'
                else:
                    status['detect'] = '✓ Complete'
                    
            # Check crop
            if 'crop_runs' in root and len(list(root['crop_runs'].group_keys())) > 0:
                latest = root['crop_runs'].attrs.get('latest')
                crop_group = root[f'crop_runs/{latest}'] if latest else None
                if crop_group and 'summary_statistics' in crop_group.attrs:
                    stats = crop_group.attrs['summary_statistics']
                    n_crops = stats.get('total_rois_cropped', 0)
                    frames_with_crops = stats.get('frames_with_crops', 0)
                    total_frames = stats.get('total_frames', 0)
                    if total_frames > 0:
                        percent = (frames_with_crops / total_frames) * 100
                        status['crop'] = f'✓ Complete ({n_crops} ROIs, {percent:.1f}%)'
                    else:
                        status['crop'] = f'✓ Complete ({n_crops} ROIs)'
                else:
                    status['crop'] = '✓ Complete'
                    
            # Check keypoints
            keypoint_group_name = None
            if 'keypoints_runs' in root and len(list(root['keypoints_runs'].group_keys())) > 0:
                keypoint_group_name = 'keypoints_runs'
            
            if keypoint_group_name:
                latest = root[keypoint_group_name].attrs.get('latest')
                keypoint_group = root[f'{keypoint_group_name}/{latest}'] if latest else None
                if keypoint_group and 'summary_statistics' in keypoint_group.attrs:
                    n_success = keypoint_group.attrs['summary_statistics'].get('successful_detections', 0)
                    success_rate = keypoint_group.attrs['summary_statistics'].get('success_rate_percent', 0)
                    status['keypoints'] = f'✓ Complete ({n_success} successful, {success_rate:.1f}%)'
                else:
                    status['keypoints'] = '✓ Complete'
                    
            # Check track
            if 'tracking_runs' in root and len(list(root['tracking_runs'].group_keys())) > 0:
                latest = root['tracking_runs'].attrs.get('latest')
                status['track'] = f'✓ Complete ({latest})' if latest else '✓ Complete'
                
            # Check refine
            if 'refine_runs' in root and len(list(root['refine_runs'].group_keys())) > 0:
                latest = root['refine_runs'].attrs.get('latest')
                status['refine'] = f'✓ Complete ({latest})' if latest else '✓ Complete'
                
            # Check assign_ids
            if 'id_assignment_runs' in root:
                try:
                    latest = root['id_assignment_runs'].attrs.get('latest')
                    if latest:
                        assign_group = root[f'id_assignment_runs/{latest}']
                        if 'summary_statistics' in assign_group.attrs:
                            stats = assign_group.attrs['summary_statistics']
                            assigned = stats.get('assigned_detections', 0)
                            total = stats.get('total_detections', 0)
                            setup_type = stats.get('setup_type', 'unknown')
                            if total > 0:
                                percent = (assigned / total) * 100
                                status['assign_ids'] = f'✓ Complete ({setup_type}, {percent:.0f}% assigned)'
                            else:
                                status['assign_ids'] = f'✓ Complete ({setup_type})'
                        else:
                            status['assign_ids'] = '✓ Complete'
                except Exception as e:
                    pass  # Silently fail and leave as "Not Run"


            experiment_setup = root.attrs.get('experiment_setup', {})
            if experiment_setup:
                setup_type = experiment_setup.get('setup_type', 'unknown')
                num_dishes = experiment_setup.get('num_dishes', '?')
                
        except Exception as e:
            # If there's an error reading zarr, mark all as unknown
            for stage in STAGE_ORDER:
                status[stage] = f'? Error: {str(e)[:30]}'
        
        return status
    
    def _update_experiment_info_display(self) -> None:
        """Update the experiment setup information panel."""
        import zarr
        from pathlib import Path
        
        experiment_panel = self.query_one("#experiment_info_panel", Static)
        
        if not self.selected_zarr or not Path(self.selected_zarr).exists():
            experiment_panel.update("[dim]Select a zarr file to view experiment configuration[/dim]")
            return
        
        try:
            root = zarr.open(self.selected_zarr, mode='r')
            experiment_setup = root.attrs.get('experiment_setup', {})
            
            if not experiment_setup:
                # Check if this is a valid zarr that just hasn't been configured
                if 'raw_video' in root or 'detect_runs' in root:
                    experiment_panel.update(
                        "[yellow]⚠ No experiment setup configured[/]\n"
                        "[dim]Press 'e' to configure[/dim]"
                    )
                else:
                    experiment_panel.update("[dim]Not a valid FishEye zarr file[/dim]")
                return
            
            # Build formatted display
            setup_type = experiment_setup.get('setup_type', 'unknown')
            num_dishes = experiment_setup.get('num_dishes', '?')
            fish_per_dish = experiment_setup.get('fish_per_dish', '?')
            total_fish = experiment_setup.get('total_expected_fish', '?')
            source = experiment_setup.get('source', 'unknown')
            configured_at = experiment_setup.get('configured_at', 'unknown')
            
            # Color code based on setup type
            if setup_type == 'single_dish':
                type_color = 'cyan'
                type_icon = ''
            elif setup_type == 'multi_dish':
                type_color = 'yellow'
                type_icon = ''
            else:
                type_color = 'red'
                type_icon = '❓'
            
            info_text = (
                f"{type_icon} [bold {type_color}]{setup_type.replace('_', ' ').title()}[/bold {type_color}]\n"
                f"[bold]Dishes:[/] {num_dishes}\n"
                f"[bold]Fish/dish:[/] {fish_per_dish}\n"
                f"[bold]Total fish:[/] {total_fish}\n"
                f"[dim]Source: {source}[/dim]"
            )
            
            # Add validation status if assign_ids has run
            if 'id_assignment_runs' in root and root['id_assignment_runs'].attrs.get('latest'):
                latest = root['id_assignment_runs'].attrs['latest']
                assign_group = root[f'id_assignment_runs/{latest}']
                if 'summary_statistics' in assign_group.attrs:
                    stats = assign_group.attrs['summary_statistics']
                    num_rois = stats.get('num_masks', 0)
                    
                    if num_rois == num_dishes:
                        info_text += f"\n[green]✓ {num_rois} ROIs tracked[/green]"
                    else:
                        info_text += f"\n[yellow]⚠ {num_rois} ROIs tracked (expected {num_dishes})[/yellow]"
            
            experiment_panel.update(info_text)
            
        except Exception as e:
            experiment_panel.update(f"[red]Error reading zarr: {e}[/red]")
    
    def compose(self) -> ComposeResult:
        """Create the UI layout."""
        yield Header(show_clock=True)
        
        yield Static("🐟 FishEye Pipeline Launcher", id="header_info")
        
        with Container(id="main_container"):
            # Left panel - File browsing
            with ScrollableContainer(id="left_panel"):
                yield Label("📁 File Browser", classes="section_header")
                
                bookmarks = self.config.get('bookmarks', [])
                if bookmarks:
                    yield Label("Quick Access:", classes="info_text")
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

                yield Label("\n Experiment Setup:", classes="section_header")
                yield Static(
                    "[dim]Select a zarr file to view experiment configuration[/dim]",
                    id="experiment_info_panel",
                    classes="info_text"
                )

                # Stage Status Display
                yield Label("\n📊 Stage Status:", classes="section_header")
                yield Static(
                    "[dim]Select a zarr file to view stage status[/dim]", 
                    id="stage_status_panel"
                )
            
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
                
                # Advanced Settings Section
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
        
        if self._is_zarr_dir(path_str):
            self.selected_zarr = path_str
            self.status_message = f"Selected zarr: {os.path.basename(path_str)}"
        elif self._is_video_file(path_str):
            self.selected_video = path_str
            self.status_message = f"Selected video: {os.path.basename(path_str)}"
        else:
            self.status_message = f"Selected: {os.path.basename(path_str)} (not zarr or video)"
    
    def _is_zarr_dir(self, path: str) -> bool:
        """Check if path is a zarr directory."""
        if not os.path.isdir(path):
            return False
        
        # Check if it's a zarr root by looking for:
        # 1. .zarray or .zgroup files directly
        # 2. .zarr extension
        # 3. Common zarr subdirectories (raw_video, detect_runs, etc.)
        if (os.path.exists(os.path.join(path, '.zarray')) or
            os.path.exists(os.path.join(path, '.zgroup'))):
            return True
        
        # Check for .zarr extension
        if path.endswith('.zarr'):
            return True
        
        # Check for common pipeline zarr structure
        zarr_indicators = ['raw_video', 'detect_runs', 'background_runs', 'crop_runs', 'keypoints_runs']
        for indicator in zarr_indicators:
            if os.path.isdir(os.path.join(path, indicator)):
                return True
        
        return False
    
    def _is_video_file(self, path: str) -> bool:
        """Check if path is a video file."""
        video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.mpeg', '.mpg', '.wmv', '.flv', '.webm'}
        return Path(path).suffix.lower() in video_exts
    
    def watch_selected_zarr(self, value: str) -> None:
        """Update zarr display when selection changes."""
        try:
            display = self.query_one("#zarr_display", Static)
            if value:
                display.update(f"Zarr: [cyan]{os.path.basename(value)}[/cyan]")
                # Update both stage status and experiment info
                self._update_stage_status_display()
                self._update_experiment_info_display()
            else:
                display.update("Zarr: [dim]None[/dim]")
                self._update_stage_status_display()
                self._update_experiment_info_display()
        except Exception:
            pass
    
    def watch_selected_video(self, value: str) -> None:
        """Update video display when selection changes."""
        try:
            display = self.query_one("#video_display", Static)
            if value:
                display.update(f"Video: [cyan]{os.path.basename(value)}[/cyan]")
            else:
                display.update("Video: [dim]None[/dim]")
        except:
            pass
    
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
    
    def _update_stage_status_display(self) -> None:
        """Update the stage status display panel."""
        status_panel = self.query_one("#stage_status_panel", Static)
        
        if not self.selected_zarr:
            status_panel.update("[dim]Select a zarr file to view stage status[/dim]")
            return
        
        # Also update experiment info when updating stage status
        self._update_experiment_info_display()
        
        status = self._check_zarr_stage_status(self.selected_zarr)
        
        # Build formatted status display
        status_lines = []
        for stage in STAGE_ORDER:
            stage_status = status.get(stage, '○ Not Run')
            
            # Color code based on status
            if '✓' in stage_status:
                color = 'green'
            elif '?' in stage_status:
                color = 'red'
            else:
                color = 'dim'
            
            status_lines.append(f"[{color}]{stage}:[/{color}] {stage_status}")
        
        status_text = "\n".join(status_lines)
        status_panel.update(status_text)
    
    def _run_subprocess_with_output(self, cmd: List[str]) -> None:
        """Run subprocess and capture output in real-time."""
        if self.progress_log:
            self.progress_log.clear()
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
                # Refresh stage status after completion
                self._update_stage_status_display()
            else:
                self.progress_log.write(f"\n[bold red]❌ Pipeline failed (exit code {process.returncode})[/]\n")
                self.status_message = f"❌ Failed: {zarr_file} (exit {process.returncode})"
        
        except Exception as e:
            self.progress_log.write(f"\n[bold red]❌ Error: {e}[/]\n")
            self.status_message = f"❌ Error: {e}"
        
        finally:
            self.is_running = False
            self.current_stage = ""
            self.call_later(self._update_stage_status_display)
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
                    # Refresh stage status after tuning
                    self._update_stage_status_display()
                except subprocess.CalledProcessError as e:
                    self.status_message = f"❌ Tuner failed with exit code {e.returncode}"
                except KeyboardInterrupt:
                    self.status_message = "Tuner interrupted"
            
            if self.progress_log:
                self.progress_log.write(f"[cyan]Tuner session ended: {self.status_message}[/]\n")
            
        except Exception as e:
            self.status_message = f"❌ Error: {e}"
            if self.progress_log:
                self.progress_log.write(f"[red]❌ Error: {e}[/]\n")
    
    def action_configure_experiment(self) -> None:
        """Open experiment setup configuration."""
        if not self.selected_zarr:
            self.status_message = "❌ Please select a zarr file first"
            if self.progress_log:
                self.progress_log.write("[red]❌ Please select a zarr file first[/]")
            return
        
        # Check if zarr exists
        from pathlib import Path
        if not Path(self.selected_zarr).exists():
            self.status_message = "❌ Zarr file doesn't exist yet - run import first"
            if self.progress_log:
                self.progress_log.write("[red]❌ Zarr doesn't exist - run import first[/]")
            return
        
        # Run setup tool
        setup_script = Path(__file__).parent.parent.parent / "setup_experiment_metadata.py"
        
        self.status_message = "Running experiment setup..."
        if self.progress_log:
            self.progress_log.write("\n[cyan]Opening experiment setup tool...[/]\n")
        
        with self.suspend():  # Suspend TUI while running interactive tool
            try:
                result = subprocess.run([
                    sys.executable,
                    str(setup_script),
                    self.selected_zarr,
                    "--interactive"
                ])
                
                if result.returncode == 0:
                    self.status_message = "✓ Experiment setup configured"
                    if self.progress_log:
                        self.progress_log.write("[green]✓ Experiment setup configured[/]\n")
                else:
                    self.status_message = "❌ Setup cancelled or failed"
                    if self.progress_log:
                        self.progress_log.write("[yellow]Setup cancelled[/]\n")
            except Exception as e:
                self.status_message = f"❌ Error: {e}"
                if self.progress_log:
                    self.progress_log.write(f"[red]❌ Error: {e}[/]\n")
        
        # Refresh displays after configuration
        self._update_experiment_info_display()  # UPDATE experiment info
        self._update_stage_status_display()     # UPDATE stage status
        self.refresh()
    
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
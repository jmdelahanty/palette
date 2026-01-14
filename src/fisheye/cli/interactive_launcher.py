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
import time
from typing import Set, Dict, Any

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
REFINED_DETECT_GROUP = "refined_detect_runs"
LEGACY_REFINED_DETECT_GROUP = "refined_runs"

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
        'desc': 'Refine detections (filter & interpolate)',
        'requires': ['detect'],
        'color': 'blue'
    },
    'assign_ids': {
        'desc': 'Assign consistent IDs (auto for single-dish)',
        'requires': ['detect'],
        'color': 'yellow'
    }
}

STAGE_ORDER = [
    'import',
    'downsample',
    'background',
    'detect',
    'refine',
    'assign_ids',
    'crop',
    'keypoints',
    'track'
]

TUNER_INFO = {
    'mask': 'Tune dish mask detection (Hough circles)',
    'subdish': 'Define sub-dish masks for spatial ID assignment (multi-dish only)',
    'detect': 'Tune fish detection thresholds',
    'threshold': 'Alias for detect tuner',
    'keypoints': 'Tune anatomical keypoint detection (swim bladder & eyes)',
}

VIZ_INFO = {
    'detections': {
        'script': 'visualization/detection_visualizer.py',
        'desc': 'Interactive detection viewer with ID labels',
        'requires': ['detect']
    }
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

    #batch_display {
        height: auto;
        border: solid magenta;
        padding: 1;
        margin-top: 1;
        background: $surface;
        min-height: 5;
        max-height: 12;
    }
    
    .batch_mode_active #batch_display {
        border: solid yellow;
        background: $panel;
    }

    .stage_option {
        margin-left: 2;
        margin-bottom: 1;
        width: 90%;
    }
    """
    
    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("r", "run_pipeline", "Run Pipeline", show=True),
        Binding("b", "toggle_batch_mode", "Batch Mode", show=True),
        Binding("B", "run_batch", "Run Batch", show=True),
        Binding("x", "clear_batch", "Clear Batch", show=True),
        Binding("i", "inspect_zarr", "Inspect Zarr", show=True),
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
    batch_mode = reactive(False)
    training_data_enabled = reactive(False)
    frame_step_value = reactive("")
    
    def __init__(self):
        super().__init__()
        self.stage_checkboxes = {}
        self.config = self._load_launcher_config()
        self.start_dir = self._get_start_directory()
        self.progress_log = None
        self.batch_queue: Set[str] = set()
    
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
                {'name': 'nvme-local', 'path': '/nvme1'}
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
                    total_frames = stats.get('total_frames', 1)
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
                    total_frames = stats.get('total_frames', 1)
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
            elif 'keypoint_runs' in root and len(list(root['keypoint_runs'].group_keys())) > 0:
                keypoint_group_name = 'keypoint_runs'
            
            if keypoint_group_name:
                latest = root[keypoint_group_name].attrs.get('latest')
                keypoint_group = root[f'{keypoint_group_name}/{latest}'] if latest else None
                if keypoint_group and 'summary_statistics' in keypoint_group.attrs:
                    stats = keypoint_group.attrs['summary_statistics']
                    n_success = stats.get('successful_detections', 0)
                    success_rate = stats.get('success_rate_percent', 0)
                    status['keypoints'] = f'✓ Complete ({n_success} successful, {success_rate:.1f}%)'
                else:
                    status['keypoints'] = '✓ Complete'
                    
            # Check track
            if 'tracking_runs' in root and len(list(root['tracking_runs'].group_keys())) > 0:
                latest = root['tracking_runs'].attrs.get('latest')
                status['track'] = f'✓ Complete ({latest})' if latest else '✓ Complete'
                
            # Check refine - WITH DETAILED SUB-ITEMS
            refined_root = None
            if REFINED_DETECT_GROUP in root:
                refined_root = root[REFINED_DETECT_GROUP]
            elif LEGACY_REFINED_DETECT_GROUP in root:
                refined_root = root[LEGACY_REFINED_DETECT_GROUP]
            if refined_root and len(list(refined_root.group_keys())) > 0:
                latest = refined_root.attrs.get('latest')
                if latest and latest in refined_root:
                    refined_group = refined_root[latest]
                
                # Build detailed status with sub-items
                status_lines = ['✓ Complete']
                
                # Check filtered stage
                if 'filtered' in refined_group:
                    filtered_grp = refined_group['filtered']
                    filtered_attrs = filtered_grp.attrs
                    
                    total_dets = filtered_attrs.get('total_detections', 0)
                    dropped_dets = filtered_attrs.get('dropped_detections', 0)
                    
                    status_lines.append(f'  └─ filtered: {total_dets:,} detections ({dropped_dets} jumps removed)')
                
                # Check interpolated stage
                if 'interpolated' in refined_group:
                    interp_grp = refined_group['interpolated']
                    interp_attrs = interp_grp.attrs
                    
                    total_dets = interp_attrs.get('total_detections', 0)
                    original_dets = interp_attrs.get('original_detections', 0)
                    interpolated_dets = interp_attrs.get('interpolated_detections', 0)
                    gaps_filled = interp_attrs.get('gaps_filled', 0)
                    
                    status_lines.append(f'  └─ interpolated: {total_dets:,} detections ({interpolated_dets} added, {gaps_filled} gaps filled)')
                
                # Join all lines
                status['refine'] = '\n'.join(status_lines)
            
            # Check assign_ids
            if 'id_assignment_runs' in root and len(list(root['id_assignment_runs'].group_keys())) > 0:
                latest = root['id_assignment_runs'].attrs.get('latest')
                assign_group = root[f'id_assignment_runs/{latest}'] if latest else None
                
                if assign_group and 'summary_statistics' in assign_group.attrs:
                    stats = assign_group.attrs['summary_statistics']
                    
                    # Check experiment setup for type
                    experiment_setup = root.attrs.get('experiment_setup', {})
                    setup_type = experiment_setup.get('setup_type', 'unknown')
                    
                    if setup_type == 'single_dish':
                        # For single dish, just show basic stats
                        assigned = stats.get('assigned_detections', 0)
                        total = stats.get('total_detections', 1)
                        percent = stats.get('assignment_rate_percent', 0)
                        status['assign_ids'] = f'✓ Complete (single_dish, {percent:.0f}% assigned)'
                    else:
                        # For multi-dish, show number of ROIs
                        num_masks = stats.get('num_masks', 0)
                        percent = stats.get('assignment_rate_percent', 0)
                        status['assign_ids'] = f'✓ Complete ({num_masks} ROIs, {percent:.0f}% assigned)'
                else:
                    status['assign_ids'] = '✓ Complete'
        
        except Exception as e:
            # If there's an error, keep defaults
            pass
        
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
                        "[yellow] No experiment setup configured[/]\n"
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


                yield Label("\n📋 Batch Queue", classes="section_header")
                yield Static("[dim]No files in batch queue[/dim]", id="batch_display")
            
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

                    if stage == 'import':
                        yield Label("  └─ Training data sampling:", classes="info_text")
                        yield Checkbox(
                            "      Enable frame sampling",
                            value=False,
                            id="training_data_checkbox",
                            classes="stage_option"
                        )
                        yield Label("      Sample every Nth frame:", classes="info_text")
                        yield Input(
                            placeholder="e.g. 100",
                            value="",
                            id="frame_step_input",
                            classes="stage_option"
                        )
                        yield Static(
                            "      [dim]Leave empty to import all frames[/dim]",
                            classes="info_text"
                        )

                    if stage == 'crop':
                        yield Label("  └─ Crop source:", classes="info_text")
                        yield Select(
                        [
                            ("Original detections", "detect"),
                            ("Filtered (jumps removed)", "filtered"),  
                            ("Gaps filled (interpolated)", "interpolated")
                        ],
                        value="detect",
                        id="crop_source_select",
                        classes="stage_option"
                    )
                    if stage == 'refine':
                        yield Label("  └─ Refinement parameters:", classes="info_text")
                        yield Label("      Max gap (frames):", classes="info_text")
                        yield Input(
                            placeholder="e.g. 50",
                            value="50",
                            id="refine_max_gap_input",
                            classes="stage_option"
                        )
                        yield Label("      Interpolation method:", classes="info_text")
                        yield Select(
                            [
                                ("Linear", "linear"),
                            ],
                            value="linear",
                            id="refine_method_select",
                            classes="stage_option"
                        )
                        yield Checkbox(
                            "      Remove jumps",
                            value=True,
                            id="refine_remove_jumps_checkbox",
                            classes="stage_option"
                        )
                        yield Checkbox(
                            "      Remove blips",
                            value=False,
                            id="refine_remove_blips_checkbox",
                            classes="stage_option"
                        )
                
                yield Label("\n🔧 Tuning Tools", classes="section_header")
                yield Label("Run parameter tuners:", classes="info_text")
                
                yield Select(
                    [(name, name) for name in TUNER_INFO.keys()],
                    prompt="Select tuner...",
                    id="tuner_select"
                )
                yield Button("🔧 Run Tuner", id="run_tuner_btn", variant="success")


                yield Label("\n📊 Visualization Tools", classes="section_header")
                yield Label("View and inspect results:", classes="info_text")

                yield Select(
                    [(name, name) for name in VIZ_INFO.keys()],
                    prompt="Choose visualization",
                    id="viz_select"
                )

                yield Button("Open Visualizer", id="run_viz_button", variant="primary")
                yield Checkbox(
                    "Show refined/interpolated detections (orange overlay)",
                    value=True,
                    id="show_refined_checkbox",
                    classes="stage_option"
                )
                
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
            if self.batch_mode:
                # Batch mode: add/remove from queue
                if path_str in self.batch_queue:
                    self.batch_queue.remove(path_str)
                    self.status_message = f"Removed from batch: {os.path.basename(path_str)} | {len(self.batch_queue)} in queue"
                    if self.progress_log:
                        self.progress_log.write(f"[yellow]Removed:[/yellow] {os.path.basename(path_str)}\n")
                else:
                    self.batch_queue.add(path_str)
                    self.status_message = f"Added to batch: {os.path.basename(path_str)} | {len(self.batch_queue)} in queue"
                    if self.progress_log:
                        self.progress_log.write(f"[green]Added:[/green] {os.path.basename(path_str)}\n")

                # Update selected_zarr so panels show info for this file
                self.selected_zarr = path_str

                self._update_batch_display()
            else:
                # Normal mode: set as selected file
                self.selected_zarr = path_str
                self.status_message = f"Selected zarr: {os.path.basename(path_str)}"

        elif self._is_video_file(path_str):
            # Handle video file selection
            if self.batch_mode:
                # Batch mode: add/remove videos from queue
                if path_str in self.batch_queue:
                    self.batch_queue.remove(path_str)
                    self.status_message = f"Removed from batch: {os.path.basename(path_str)} | {len(self.batch_queue)} in queue"
                    if self.progress_log:
                        self.progress_log.write(f"[yellow]Removed:[/yellow] {os.path.basename(path_str)}\n")
                else:
                    self.batch_queue.add(path_str)
                    self.status_message = f"Added to batch: {os.path.basename(path_str)} | {len(self.batch_queue)} in queue"
                    if self.progress_log:
                        self.progress_log.write(f"[green]Added:[/green] {os.path.basename(path_str)}\n")

                # Update selected_video so panels show info
                self.selected_video = path_str
                self._update_batch_display()
            else:
                # Normal mode: set as selected file
                self.selected_video = path_str
                video_path = Path(path_str)
                proposed_zarr = video_path.with_suffix('.zarr')

                # Only auto-set zarr if no zarr is currently selected
                if not self.selected_zarr:
                    self.selected_zarr = str(proposed_zarr)
                    self.status_message = f"Selected video: {video_path.name} → will create {proposed_zarr.name}"
                else:
                    self.status_message = f"Selected video: {video_path.name}"

        elif path_str.endswith('.yaml') or path_str.endswith('.yml'):
            # Handle config file selection
            self.selected_config = path_str
            self.status_message = f"Selected config: {os.path.basename(path_str)}"
    
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
    
    def _get_crop_source_selection(self) -> str:
        """Get the selected crop source from dropdown."""
        try:
            crop_source_select = self.query_one("#crop_source_select", Select)
            return crop_source_select.value if crop_source_select.value != Select.BLANK else "detect"
        except Exception:
            # Fallback if widget not found
            return "detect"
    
    def _get_training_data_options(self) -> Dict[str, Optional[Any]]:
        """Read training data sampling options from the UI."""
        options: Dict[str, Optional[Any]] = {
            "enabled": False,
            "frame_step": None,
        }
        try:
            training_checkbox = self.query_one("#training_data_checkbox", Checkbox)
            options["enabled"] = training_checkbox.value

            if options["enabled"]:
                frame_step_input = self.query_one("#frame_step_input", Input)
                frame_step_str = frame_step_input.value.strip()
                if frame_step_str:
                    try:
                        options["frame_step"] = int(frame_step_str)
                    except ValueError:
                        pass  # Will be validated later
        except Exception:
            pass  # Fall back to defaults

        return options

    def _get_refine_options(self) -> Dict[str, Optional[Any]]:
        """Read refinement parameter overrides from the UI."""
        options: Dict[str, Optional[Any]] = {
            "max_gap": None,
            "method": None,
            "remove_jumps": None,
            "remove_blips": None,
        }
        try:
            max_gap_input = self.query_one("#refine_max_gap_input", Input)
            max_gap_str = max_gap_input.value.strip()
            if max_gap_str:
                try:
                    options["max_gap"] = max(0, int(max_gap_str))
                except ValueError:
                    self.status_message = "⚠️ Refinement max gap must be an integer."
                    if self.progress_log:
                        self.progress_log.write("[yellow]⚠️ Refinement max gap must be an integer.[/yellow]\n")
            method_select = self.query_one("#refine_method_select", Select)
            if method_select.value and method_select.value != Select.BLANK:
                options["method"] = method_select.value
            jumps_checkbox = self.query_one("#refine_remove_jumps_checkbox", Checkbox)
            options["remove_jumps"] = bool(jumps_checkbox.value)
            blips_checkbox = self.query_one("#refine_remove_blips_checkbox", Checkbox)
            options["remove_blips"] = bool(blips_checkbox.value)
        except Exception:
            # If widgets not present, keep defaults
            pass
        return options

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
                # Check if this is batch processing or single file
                if "/" in str(self.current_stage) and self.current_stage.split("/")[0].isdigit():
                    # Batch processing - current_stage is like "1/3"
                    status.update(f"[yellow]⏳ BATCH RUNNING[/yellow] | {value}")
                else:
                    # Single file processing
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
    
    def watch_batch_mode(self, value: bool) -> None:
        """Update display when batch mode changes."""
        self._update_batch_display()
        if value:
            self.status_message = f"[yellow]BATCH MODE[/yellow] | {len(self.batch_queue)} files queued"
        else:
            self.status_message = f"Ready | {len(self.batch_queue)} files in batch queue"
    
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
            self._action_run_tuner()
        elif button_id == "run_viz_button":
            self._action_run_visualizer()

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
        video_path = self.selected_video
        config_path = self.selected_config

        stages = self._get_selected_stages()
        if not stages:
            self.status_message = " Error: Select at least one stage!"
            if self.progress_log:
                self.progress_log.write("[red] Error: Select at least one stage![/]")
            return None

        # Special case: import stage
        if 'import' in stages:
            if not video_path:
                self.status_message = " Error: Select a video file for import stage!"
                if self.progress_log:
                    self.progress_log.write("[red] Error: Import stage requires a video file![/]")
                return None

            # Auto-create zarr path if not specified
            zarr_path = self.selected_zarr
            if not zarr_path:
                video_path_obj = Path(video_path)
                zarr_path = str(video_path_obj.with_suffix('.zarr'))
                self.selected_zarr = zarr_path  # Update the selection
                if self.progress_log:
                    self.progress_log.write(f"[yellow]Auto-generated zarr path: {zarr_path}[/yellow]\n")
        else:
            # For non-import stages, require existing zarr
            zarr_path = self.selected_zarr
            if not zarr_path:
                self.status_message = " Error: Select a zarr file from the tree!"
                if self.progress_log:
                    self.progress_log.write("[red] Error: Select a zarr file first![/]")
                return None

        cmd = [sys.executable, "-m", "fisheye", zarr_path]
        
        if video_path:
            cmd.extend(["--video-path", video_path])
        
        if config_path:
            cmd.extend(["--config", config_path])
        
        cmd.extend(["--stages"] + stages)

        # Add training data parameters if import stage is selected
        if 'import' in stages:
            training_opts = self._get_training_data_options()
            if training_opts["enabled"]:
                frame_step = training_opts["frame_step"]
                if frame_step is not None:
                    if frame_step < 1:
                        self.status_message = " Error: Frame step must be >= 1"
                        if self.progress_log:
                            self.progress_log.write("[red] Error: Frame step must be >= 1[/red]\n")
                        return None
                    cmd.extend(["--training-data", "--frame-step", str(frame_step)])
                    if self.progress_log:
                        self.progress_log.write(
                            f"[yellow]Training data mode: sampling every {frame_step}th frame[/yellow]\n"
                        )
                else:
                    self.status_message = " Error: Frame step required when training mode enabled"
                    if self.progress_log:
                        self.progress_log.write("[red] Error: Enter a frame step value[/red]\n")
                    return None

        # Add crop source if crop stage is selected
        if 'crop' in stages:
            crop_source = self._get_crop_source_selection()
            
            # Check if refined source is available
            if crop_source in ['filtered', 'interpolated']:
                if not self._check_refined_runs_exist(zarr_path):
                    self.status_message = f" Error: No refined detection runs found for '{crop_source}' source"
                    if self.progress_log:
                        self.progress_log.write(
                            f"[red] Error: '{crop_source}' source requires refined detections.[/red]\n"
                            "[yellow]Run the 'refine' stage first or select 'detect' source.[/yellow]\n"
                        )
                    return None
            
            cmd.extend(["--crop-source", crop_source])
            
            if self.progress_log and not dry_run:
                self.progress_log.write(f"[cyan]  Crop source: {crop_source}[/cyan]\n")
        
        if 'refine' in stages:
            refine_opts = self._get_refine_options()
            if refine_opts['max_gap'] is not None:
                cmd.extend(["--refine-max-gap", str(refine_opts['max_gap'])])
            if refine_opts['method']:
                cmd.extend(["--refine-method", refine_opts['method']])
            if refine_opts['remove_jumps'] is not None:
                cmd.append("--refine-remove-jumps" if refine_opts['remove_jumps'] else "--refine-keep-jumps")
            if refine_opts['remove_blips'] is not None:
                cmd.append("--refine-remove-blips" if refine_opts['remove_blips'] else "--refine-keep-blips")

            if self.progress_log and not dry_run:
                msg = "[cyan]  Refinement overrides:";
                parts = []
                if refine_opts['max_gap'] is not None:
                    parts.append(f"max_gap={refine_opts['max_gap']}")
                if refine_opts['method']:
                    parts.append(f"method={refine_opts['method']}")
                if refine_opts['remove_jumps'] is not None:
                    parts.append("remove_jumps=" + ("true" if refine_opts['remove_jumps'] else "false"))
                if refine_opts['remove_blips'] is not None:
                    parts.append("remove_blips=" + ("true" if refine_opts['remove_blips'] else "false"))
                msg += " " + (", ".join(parts) if parts else "(defaults)")
                self.progress_log.write(msg + "\n")
        
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
    
    def _check_refined_runs_exist(self, zarr_path: str) -> bool:
        """Check if refined detection runs exist in zarr file."""
        try:
            if not Path(zarr_path).exists():
                return False
            
            root = zarr.open(zarr_path, mode='r')
            if REFINED_DETECT_GROUP in root:
                return root[REFINED_DETECT_GROUP].attrs.get('latest') is not None
            if LEGACY_REFINED_DETECT_GROUP in root:
                return root[LEGACY_REFINED_DETECT_GROUP].attrs.get('latest') is not None
            return False
        except Exception:
            return False
    
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
            
            # Check if status has newlines (multi-line status like refine)
            if '\n' in stage_status:
                # Split into main line and sub-items
                lines = stage_status.split('\n')
                main_status = lines[0]
                sub_items = lines[1:]
                
                # Color code main status
                if '✓' in main_status:
                    color = 'green'
                elif '?' in main_status:
                    color = 'red'
                else:
                    color = 'dim'
                
                # Add main line
                status_lines.append(f"[{color}]{stage}:[/{color}] {main_status}")
                
                # Add sub-items (already formatted with └─)
                for sub_item in sub_items:
                    status_lines.append(f"[dim]{sub_item}[/dim]")
            else:
                # Single-line status (normal case)
                if '✓' in stage_status:
                    color = 'green'
                elif '?' in stage_status:
                    color = 'red'
                else:
                    color = 'dim'
                
                status_lines.append(f"[{color}]{stage}:[/{color}] {stage_status}")
        
        status_text = "\n".join(status_lines)
        status_panel.update(status_text)
    
    def _update_batch_display(self) -> None:
        """Update the batch queue display panel."""
        try:
            batch_display = self.query_one("#batch_display", Static)
            
            if not self.batch_queue:
                batch_display.update("[dim]No files in batch queue[/dim]")
            else:
                lines = [f"[bold cyan]Batch Queue ({len(self.batch_queue)} files):[/bold cyan]"]
                for i, path in enumerate(sorted(self.batch_queue), 1):
                    filename = os.path.basename(path)
                    lines.append(f"  {i}. [cyan]{filename}[/cyan]")
                batch_display.update("\n".join(lines))
        except:
            pass

    def _build_command_for_file(self, zarr_path: str, stages: List[str]) -> Optional[List[str]]:
        """Build command for a specific file in batch."""
        cmd = [sys.executable, "-m", "fisheye", zarr_path]
        
        cmd.extend(["--stages"] + stages)
        
        if self.selected_config:
            cmd.extend(["--config", self.selected_config])
        
        try:
            scheduler_select = self.query_one("#scheduler_select", Select)
            scheduler = scheduler_select.value
            if scheduler and scheduler != Select.BLANK:
                cmd.extend(["--scheduler", str(scheduler)])
        except:
            pass
        
        return cmd

    def _build_command_for_video_import(self, video_path: str, zarr_path: str, stages: List[str]) -> Optional[List[str]]:
        """Build command for video import in batch mode."""
        cmd = [sys.executable, "-m", "fisheye", zarr_path]
        cmd.extend(["--video-path", video_path])

        if self.selected_config:
            cmd.extend(["--config", self.selected_config])

        cmd.extend(["--stages"] + stages)

        # Add training data parameters if enabled
        training_opts = self._get_training_data_options()
        if training_opts["enabled"] and training_opts["frame_step"] is not None:
            cmd.extend(["--training-data", "--frame-step", str(training_opts["frame_step"])])

        return cmd

    def _run_batch_worker(self, files: List[str], stages: List[str]) -> None:
        """Worker thread for batch processing."""
        results = {
            'total': len(files),
            'success': 0,
            'failed': 0,
            'errors': []
        }

        # Detect if batch contains videos or zarrs
        is_video_batch = self._is_video_file(files[0]) if files else False

        for i, file_path in enumerate(files, 1):
            file_name = os.path.basename(file_path)
            self.current_stage = f"{i}/{len(files)}"

            if is_video_batch:
                # For video imports, auto-generate zarr path
                video_path = Path(file_path)
                zarr_path = str(video_path.with_suffix('.zarr'))

                if self.progress_log:
                    self.progress_log.write(
                        f"\n[bold cyan]▶ Processing {i}/{len(files)}: {file_name}[/bold cyan]\n"
                        f"  [yellow]→ Output: {os.path.basename(zarr_path)}[/yellow]"
                    )

                cmd = self._build_command_for_video_import(file_path, zarr_path, stages)
            else:
                # For zarr files, use existing logic
                zarr_path = file_path

                if self.progress_log:
                    self.progress_log.write(f"\n[bold cyan]▶ Processing {i}/{len(files)}: {file_name}[/bold cyan]")

                cmd = self._build_command_for_file(zarr_path, stages)
            
            if not cmd:
                results['failed'] += 1
                results['errors'].append(f"{file_name}: Failed to build command")
                if self.progress_log:
                    self.progress_log.write(f"[red]  ✗ Failed to build command[/red]")
                continue
            
            try:
                start_time = time.time()
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                
                for line in process.stdout:
                    line = line.rstrip()
                    if line and self.progress_log:
                        if "Running stage:" in line:
                            stage_name = line.split("Running stage:")[-1].strip()
                            self.progress_log.write(f"  [cyan]{stage_name}[/cyan]")
                        else:
                            self.progress_log.write(f"  {line}")
                
                return_code = process.wait()
                duration = time.time() - start_time
                
                if return_code == 0:
                    results['success'] += 1
                    if self.progress_log:
                        self.progress_log.write(f"[green]  ✓ Completed in {duration:.1f}s[/green]\n")
                else:
                    results['failed'] += 1
                    results['errors'].append(f"{zarr_name}: Exit code {return_code}")
                    if self.progress_log:
                        self.progress_log.write(f"[red]  ✗ Failed with exit code {return_code}[/red]\n")
                    
            except Exception as e:
                results['failed'] += 1
                results['errors'].append(f"{zarr_name}: {str(e)}")
                if self.progress_log:
                    self.progress_log.write(f"[red]  ✗ Error: {e}[/red]\n")
        
        self.is_running = False
        self.current_stage = ""
        
        if self.progress_log:
            self.progress_log.write(f"\n[bold cyan]═══════════════════════════════════[/bold cyan]")
            self.progress_log.write(f"[bold cyan]  BATCH COMPLETE[/bold cyan]")
            self.progress_log.write(f"[bold cyan]═══════════════════════════════════[/bold cyan]")
            self.progress_log.write(f"Total:      {results['total']}")
            self.progress_log.write(f"[green]Success:    {results['success']}[/green]")
            self.progress_log.write(f"[red]Failed:     {results['failed']}[/red]")
            
            if results['errors']:
                self.progress_log.write("\n[yellow]Errors:[/yellow]")
                for error in results['errors']:
                    self.progress_log.write(f"  [red]•[/red] {error}")
        
        self.status_message = f"Batch complete: {results['success']}/{results['total']} succeeded"

    def action_toggle_batch_mode(self) -> None:
        """Toggle batch selection mode."""
        self.batch_mode = not self.batch_mode
        
        if self.batch_mode:
            self.status_message = f"[yellow]BATCH MODE[/yellow] - Select files with 's' or Enter | {len(self.batch_queue)} in queue"
            if self.progress_log:
                self.progress_log.write("\n[yellow]══════ BATCH MODE ACTIVATED ══════[/yellow]")
                self.progress_log.write("[dim]Select multiple zarr files to process sequentially[/dim]")
                self.progress_log.write("[dim]Press 's' or Enter to add files to batch queue[/dim]")
                self.progress_log.write("[dim]Press 'B' to run batch or 'x' to clear queue[/dim]\n")
        else:
            self.status_message = f"Batch mode OFF | {len(self.batch_queue)} files queued"
            if self.progress_log:
                self.progress_log.write(f"\n[cyan]Batch mode deactivated - {len(self.batch_queue)} files in queue[/cyan]\n")
    
    def action_clear_batch(self) -> None:
        """Clear the batch queue."""
        count = len(self.batch_queue)
        self.batch_queue.clear()
        self.status_message = f"Cleared {count} files from batch queue"
        
        if self.progress_log:
            self.progress_log.write(f"[yellow]Cleared {count} files from batch queue[/yellow]\n")
        
        self._update_batch_display()
    
    def action_run_batch(self) -> None:
        """Run pipeline on all files in batch queue."""
        if not self.batch_queue:
            self.status_message = "❌ Batch queue is empty!"
            if self.progress_log:
                self.progress_log.write("[red]❌ Batch queue is empty! Add files with batch mode first.[/red]\n")
            return
        
        if self.is_running:
            self.status_message = "❌ Pipeline already running!"
            return
        
        stages = self._get_selected_stages()
        if not stages:
            self.status_message = "❌ No stages selected!"
            if self.progress_log:
                self.progress_log.write("[red]❌ Select at least one stage first![/red]\n")
            return

        # Check if batch contains videos (for import) or zarrs (for other stages)
        batch_files = list(self.batch_queue)
        first_file = batch_files[0]
        is_video_batch = self._is_video_file(first_file)

        if is_video_batch and 'import' not in stages:
            self.status_message = "❌ Video batch requires 'import' stage!"
            if self.progress_log:
                self.progress_log.write("[red]❌ Video files in batch require 'import' stage[/red]\n")
            return

        if not is_video_batch and 'import' in stages:
            self.status_message = "❌ Import stage requires video files in batch!"
            if self.progress_log:
                self.progress_log.write("[red]❌ Import stage requires video files, not zarr files[/red]\n")
            return

        self.is_running = True
        
        if self.progress_log:
            self.progress_log.write(f"\n[bold cyan]═══════════════════════════════════[/bold cyan]")
            self.progress_log.write(f"[bold cyan]  BATCH PROCESSING: {len(batch_files)} FILES[/bold cyan]")
            self.progress_log.write(f"[bold cyan]═══════════════════════════════════[/bold cyan]\n")
            self.progress_log.write(f"Stages: [yellow]{', '.join(stages)}[/yellow]\n")
        
        self.run_worker(
            lambda: self._run_batch_worker(batch_files, stages),
            thread=True
        )
    
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

    def _action_run_tuner(self) -> None:
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
            self.status_message = f"Error: {e}"
            if self.progress_log:
                self.progress_log.write(f"[red]Error: {e}[/]\n")
    
    def _action_run_visualizer(self) -> None:
        """Run the selected visualization tool."""
        viz_select = self.query_one("#viz_select", Select)
        
        if not self.selected_zarr:
            self.status_message = "Error: Please select a zarr file first!"
            if self.progress_log:
                self.progress_log.write("[red]Error: Please select a zarr file first![/red]\n")
            return
        
        zarr_path = self.selected_zarr
        if not Path(zarr_path).exists():
            self.status_message = f"Error: Zarr path does not exist: {zarr_path}"
            if self.progress_log:
                self.progress_log.write(f"[red]Error: Zarr path does not exist: {zarr_path}[/red]\n")
            return
        
        viz_name = str(viz_select.value)
        if viz_name == Select.BLANK:
            self.status_message = "Error: Please select a visualization tool"
            return

        viz_info = VIZ_INFO.get(viz_name)
        if not viz_info:
            self.status_message = f"Error: Unknown visualizer: {viz_name}"
            return

        # Use the friendly description in messages
        self.status_message = f"Launching {viz_info['desc']}..."
        
        # Check requirements
        status = self._check_zarr_stage_status(zarr_path)
        missing_reqs = [req for req in viz_info['requires'] if 'Complete' not in status.get(req, '')]
        
        if missing_reqs:
            self.status_message = f"❌ Cannot run {viz_name}: missing stages {', '.join(missing_reqs)}"
            if self.progress_log:
                self.progress_log.write(f"[red]❌ Missing required stages: {', '.join(missing_reqs)}[/red]\n")
            return
        
        # Launch the visualizer
        viz_script = Path(__file__).parent.parent / viz_info['script']
        
        if not viz_script.exists():
            self.status_message = f"❌ Visualizer script not found: {viz_script}"
            if self.progress_log:
                self.progress_log.write(f"[red]❌ Script not found: {viz_script}[/red]\n")
            return
        
        self.status_message = f"Launching {viz_info['desc']}..."
        if self.progress_log:
            self.progress_log.write(f"\n[cyan]Launching {viz_info['desc']}...[/cyan]\n")

        # Build command with optional flags
        cmd = [sys.executable, str(viz_script), zarr_path]

        # Check if refined overlay is enabled
        try:
            show_refined_checkbox = self.query_one("#show_refined_checkbox", Checkbox)
            if show_refined_checkbox.value:
                cmd.append("--show-refined")
                if self.progress_log:
                    self.progress_log.write("[dim]  └─ With refined/interpolated overlay[/dim]\n")
        except Exception:
            pass  # Checkbox might not exist for all visualizers

        try:
            # Run in background so TUI stays responsive
            subprocess.Popen(cmd)
            self.status_message = f"✓ Visualizer launched! Check your display."
            if self.progress_log:
                self.progress_log.write("[green]✓ Visualizer window opened[/green]\n")
        except Exception as e:
            self.status_message = f"❌ Failed to launch visualizer: {e}"
            if self.progress_log:
                self.progress_log.write(f"[red]❌ Error: {e}[/red]\n")
    
    def action_configure_experiment(self) -> None:
        """Open experiment setup configuration."""
        if not self.selected_zarr:
            self.status_message = "❌ Please select a zarr file first"
            if self.progress_log:
                self.progress_log.write("[red]❌ Please select a zarr file first[/]")
            return
        
        # Check if zarr exists
        if not Path(self.selected_zarr).exists():
            self.status_message = "❌ Zarr file doesn't exist yet - run import first"
            if self.progress_log:
                self.progress_log.write("[red]❌ Zarr doesn't exist - run import first[/]")
            return
        
        # Run setup tool
        setup_script = Path(__file__).parent.parent / "utils" / "setup_experiment_metadata.py"
        
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

    def action_inspect_zarr(self) -> None:
        """Open zarr inspector for the selected file."""
        if not self.selected_zarr:
            self.status_message = "❌ Please select a zarr file first"
            if self.progress_log:
                self.progress_log.write("[red]❌ Please select a zarr file first[/red]\n")
            return
        
        # Check if zarr exists
        if not Path(self.selected_zarr).exists():
            self.status_message = "❌ Zarr file doesn't exist"
            if self.progress_log:
                self.progress_log.write("[red]❌ Zarr file doesn't exist[/red]\n")
            return
        
        # Find zarr inspector script
        inspector_script = Path(__file__).resolve().parent.parent.parent / "zarr_inspector.py"
        
        if not inspector_script.exists():
            self.status_message = "❌ zarr_inspector.py not found"
            if self.progress_log:
                self.progress_log.write(f"[red]❌ Inspector script not found at: {inspector_script}[/red]\n")
            return
        
        self.status_message = "Opening zarr inspector..."
        if self.progress_log:
            self.progress_log.write(f"\n[cyan]Opening inspector for: {os.path.basename(self.selected_zarr)}[/cyan]\n")
        
        # Suspend TUI and run inspector
        with self.suspend():
            try:
                result = subprocess.run([
                    sys.executable,
                    str(inspector_script),
                    self.selected_zarr
                ])
                
                if result.returncode == 0:
                    self.status_message = "✓ Inspector closed"
                else:
                    self.status_message = "Inspector exited with errors"
                    
            except Exception as e:
                self.status_message = f"❌ Error running inspector: {e}"
                if self.progress_log:
                    self.progress_log.write(f"[red]❌ Error: {e}[/red]\n")
        
        if self.progress_log:
            self.progress_log.write("[cyan]Returned to TUI[/cyan]\n")
        
        # Refresh displays in case anything changed
        self.refresh()


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

#!/usr/bin/env python
#
# textual_inspector.py (Textual TUI Version)
# Interactive terminal application to inspect Zarr archives with advanced performance visualizations.
#
# Usage: python textual_inspector.py /path/to/your/data.zarr
# Requirements: pip install textual-plotext

import zarr
import argparse
import json
import os
from pathlib import Path
from datetime import datetime
import numpy as np

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.json import JSON
from rich.text import Text
from rich.align import Align
from rich.progress import Progress, BarColumn, TextColumn

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, Tree, TabbedContent, TabPane
from textual.containers import VerticalScroll, Horizontal, Grid
from textual.reactive import reactive

try:
    from textual_plotext import PlotextPlot
    PLOTEXT_AVAILABLE = True
except ImportError:
    PLOTEXT_AVAILABLE = False
    PlotextPlot = None

# --- Helper & Analysis Functions ---
def format_bytes(size: int) -> str:
    """Converts bytes to a human-readable format (KB, MB, GB)."""
    if size is None or size == 0: return "0 B"
    power, n = 1024, 0
    power_labels = {0: '', 1: 'K', 2: 'M', 3: 'G', 4: 'T'}
    while size >= power and n < len(power_labels) - 1:
        size /= power; n += 1
    return f"{size:.2f} {power_labels[n]}B"

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable format."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def detect_data_format(root):
    """Detects the data format based on tracking result columns."""
    if 'tracking_runs' in root and 'latest' in root['tracking_runs'].attrs:
        latest_run = root['tracking_runs'].attrs['latest']
        if f'tracking_runs/{latest_run}/tracking_results' in root:
            tracking_results = root[f'tracking_runs/{latest_run}/tracking_results']
            if 'column_names' in tracking_results.attrs:
                column_names = tracking_results.attrs['column_names']
                if len(column_names) >= 20 and 'bbox_x_norm_ds' in column_names:
                    return 'multi-scale'
    return 'unknown'

def create_custom_bar_chart(data_dict: dict, title: str, max_width: int = 40) -> Text:
    """Create a custom ASCII bar chart using Rich Text."""
    if not data_dict:
        return Text("No data available")
    
    max_value = max(data_dict.values()) if data_dict.values() else 1
    chart_lines = []
    
    for label, value in data_dict.items():
        # Calculate bar length
        bar_length = int((value / max_value) * max_width) if max_value > 0 else 0
        bar = "█" * bar_length
        
        # Color coding
        if "crop" in label.lower():
            color = "green"
        elif "track" in label.lower():
            color = "blue"
        else:
            color = "cyan"
        
        # Format the line
        line = f"{label:<15} [{color}]{bar}[/{color}] {value:.1f}%"
        chart_lines.append(line)
    
    chart_text = Text()
    chart_text.append(f"{title}\n\n", style="bold")
    chart_text.append("\n".join(chart_lines))
    
    return chart_text

def extract_timing_data(root) -> dict:
    """Extract timing information from zarr metadata based on the actual pipeline structure."""
    timing_data = {}
    
    # Check each pipeline stage for timing data
    stages_to_check = ['background_runs', 'crop_runs', 'refine_runs', 'tracking_runs']
    
    for stage_group_name in stages_to_check:
        if stage_group_name not in root:
            continue
            
        stage_group = root[stage_group_name]
        
        # Get the latest run if available
        if 'latest' in stage_group.attrs:
            latest_run_name = stage_group.attrs['latest']
            if latest_run_name in stage_group:
                run_group = stage_group[latest_run_name]
                
                # Extract duration from run attributes
                if 'duration_seconds' in run_group.attrs:
                    duration = float(run_group.attrs['duration_seconds'])
                    stage_name = stage_group_name.replace('_runs', '').title()
                    timing_data[stage_name] = {'duration': duration, 'count': 1}
                    
                # Also check for any additional timing info
                for attr_name, attr_value in run_group.attrs.items():
                    if 'time' in attr_name.lower() and attr_name != 'duration_seconds':
                        try:
                            timing_data[f"{stage_name}_{attr_name}"] = {
                                'duration': float(attr_value), 
                                'count': 1
                            }
                        except (ValueError, TypeError):
                            continue
    
    # Add total pipeline time if available
    if 'total_pipeline_duration_seconds' in root.attrs:
        timing_data['Total_Pipeline'] = {
            'duration': float(root.attrs['total_pipeline_duration_seconds']),
            'count': 1
        }
    
    # Calculate throughput metrics if we have frame counts
    try:
        if 'raw_video' in root and 'images_full' in root['raw_video']:
            total_frames = root['raw_video/images_full'].shape[0]
            
            # Add frames per second for each stage
            for stage_name, stage_data in timing_data.items():
                if stage_name != 'Total_Pipeline' and stage_data['duration'] > 0:
                    fps = total_frames / stage_data['duration']
                    timing_data[f"{stage_name}_FPS"] = {
                        'duration': fps,
                        'count': 1
                    }
    except:
        pass
    
    return timing_data

def extract_size_data(root) -> dict:
    """Extract data size information from zarr."""
    size_data = {}
    
    def collect_sizes(group, prefix=""):
        for name, obj in group.items():
            full_name = f"{prefix}/{name}" if prefix else name
            if isinstance(obj, zarr.core.Array):
                size_data[full_name] = obj.nbytes
            elif isinstance(obj, zarr.hierarchy.Group):
                collect_sizes(obj, full_name)
    
    collect_sizes(root)
    return size_data

def extract_confidence_data(root) -> dict:
    """Extract confidence score data from tracking results."""
    confidence_data = {}
    
    # Look for confidence data in tracking runs
    if 'tracking_runs' in root:
        # Check the latest/best run first
        if 'latest' in root['tracking_runs'].attrs:
            latest_run = root['tracking_runs'].attrs['latest']
            run_path = f'tracking_runs/{latest_run}/tracking_results'
            
            if run_path in root:
                try:
                    tracking_results = root[run_path]
                    
                    # Look for confidence column
                    if 'column_names' in tracking_results.attrs:
                        columns = tracking_results.attrs['column_names']
                        
                        # Find confidence-related columns
                        conf_columns = [col for col in columns if 'conf' in col.lower() or 'confidence' in col.lower()]
                        
                        if conf_columns and hasattr(tracking_results, 'shape') and len(tracking_results.shape) > 1:
                            # Get confidence column index
                            conf_col_idx = columns.index(conf_columns[0])
                            
                            # Extract confidence scores (sample subset for performance)
                            total_rows = tracking_results.shape[0]
                            sample_size = min(10000, total_rows)  # Sample max 10k points
                            step = max(1, total_rows // sample_size)
                            
                            confidence_scores = tracking_results[::step, conf_col_idx]
                            
                            # Filter out invalid values (NaN, negative, > 1)
                            valid_scores = []
                            for score in confidence_scores:
                                if isinstance(score, (int, float)) and 0 <= score <= 1:
                                    valid_scores.append(float(score))
                            
                            if valid_scores:
                                confidence_data['scores'] = valid_scores
                                confidence_data['column_name'] = conf_columns[0]
                                confidence_data['total_detections'] = total_rows
                                confidence_data['sampled_detections'] = len(valid_scores)
                                
                except Exception as e:
                    print(f"Error extracting confidence data: {e}")
    
    return confidence_data

class PerformanceChart(PlotextPlot):
    """A custom PlotextPlot widget for performance metrics."""
    
    def __init__(self, chart_type: str, **kwargs):
        super().__init__(**kwargs)
        self.chart_type = chart_type
        self.data = {}
        
    def update_data(self, data: dict):
        """Update the chart data."""
        self.data = data
        self.refresh_chart()
        
    def refresh_chart(self):
        """Refresh the chart with current data."""
        if not self.data:
            return
            
        self.plt.clear_data()
        
        if self.chart_type == "success_rates":
            labels = list(self.data.keys())
            values = list(self.data.values())
            self.plt.bar(labels, values)
            self.plt.title("Success Rates (%)")
            self.plt.ylabel("Percentage")
            
        elif self.chart_type == "timing":
            if not self.data:
                self.plt.text("No timing data found", 0.5, 0.5)
                self.plt.title("Timing Analysis")
            else:
                # Separate stage timings from FPS data
                stage_data = {k: v for k, v in self.data.items() if not k.endswith('_FPS')}
                
                if stage_data:
                    labels = list(stage_data.keys())
                    values = [stage_data[label]['duration'] for label in labels]
                    
                    # Use different colors for different stages
                    colors = ['blue', 'green', 'orange', 'red', 'purple']
                    
                    self.plt.bar(labels, values)
                    self.plt.title("Processing Time by Pipeline Stage")
                    self.plt.ylabel("Duration (seconds)")
                    
                    # Add total time in title if available
                    if 'Total_Pipeline' in stage_data:
                        total_time = stage_data['Total_Pipeline']['duration']
                        self.plt.title(f"Pipeline Stage Timing (Total: {total_time:.1f}s)")
                else:
                    self.plt.text("No stage timing data available", 0.5, 0.5)
                    self.plt.title("Timing Analysis")
            
        elif self.chart_type == "data_sizes":
            # Sort by size and take top 10
            sorted_items = sorted(self.data.items(), key=lambda x: x[1], reverse=True)[:10]
            labels = [item[0].split('/')[-1][:10] for item in sorted_items]
            values = [item[1] / (1024*1024) for item in sorted_items]  # Convert to MB
            self.plt.bar(labels, values)
            self.plt.title("Largest Data Components (MB)")
            self.plt.ylabel("Size (MB)")
            
        elif self.chart_type == "confidence_dist":
            if 'scores' in self.data and self.data['scores']:
                scores = self.data['scores']
                
                # Create histogram bins
                n_bins = 20
                hist, bin_edges = np.histogram(scores, bins=n_bins, range=(0, 1))
                
                # Create bin centers for plotting
                bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]
                
                # Plot histogram
                self.plt.bar(bin_centers, hist)
                self.plt.title(f"Confidence Score Distribution")
                self.plt.xlabel("Confidence Score")
                self.plt.ylabel("Frequency")
                
                # Add statistics as subtitle
                mean_conf = np.mean(scores)
                median_conf = np.median(scores)
                stats_text = f"Mean: {mean_conf:.3f}, Median: {median_conf:.3f}, n={len(scores)}"
                # Note: plotext doesn't have subtitle, so we'll include it in title
                self.plt.title(f"Confidence Score Distribution\n{stats_text}")
            else:
                self.plt.text("No confidence data available", 0.5, 0.5)
                self.plt.title("Confidence Score Distribution")

def create_performance_charts(root) -> Group:
    """Creates performance visualization charts."""
    charts = []
    
    # Performance Metrics Bar Chart
    success_data = {}
    if 'crop_runs' in root and 'best' in root['crop_runs'].attrs:
        crop_success = root['crop_runs'].attrs['best'].get('percent_cropped', 0)
        success_data['Cropping'] = crop_success
    
    if 'tracking_runs' in root and 'best' in root['tracking_runs'].attrs:
        track_success = root['tracking_runs'].attrs['best'].get('percent_tracked', 0)
        success_data['Tracking'] = track_success
    
    if success_data:
        bar_chart = create_custom_bar_chart(success_data, "Success Rates (%)")
        charts.append(Panel(bar_chart, title="Performance Overview", border_style="cyan"))
    
    # Fallback message if plotext widgets not available
    if not PLOTEXT_AVAILABLE:
        charts.append(Panel(
            "Advanced charts require 'textual-plotext' package.\nRun: pip install textual-plotext", 
            title="Enhanced Visualizations", 
            border_style="yellow"
        ))
    
    return Group(*charts) if charts else Text("No performance data available")

def get_initial_summary(root) -> Group:
    """Creates the initial summary view with performance and readiness panels."""
    data_format = detect_data_format(root)
    
    # YOLO Readiness Panel
    if data_format == 'multi-scale':
        yolo_content = "[bold green]Multi-scale tracking data found.[/bold green]\nReady for flexible YOLO dataset generation."
        border_style = "green"
    else:
        yolo_content = "[yellow]Single-scale or unknown format detected.[/yellow]"
        border_style = "yellow"
    yolo_panel = Panel(yolo_content, title="YOLO Training Readiness", border_style=border_style, expand=False)

    # Performance Tables
    perf_tables = []
    if 'crop_runs' in root and 'best' in root['crop_runs'].attrs:
        stats = root['crop_runs'].attrs['best']
        table = Table(title="Best Cropping Performance", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Run Name", stats.get('run_name', 'N/A').split('/')[-1])
        table.add_row("Success Rate", f"{stats.get('percent_cropped', 0):.2f}%")
        
        # Add timing info if available
        if 'duration' in stats:
            table.add_row("Duration", format_duration(stats['duration']))
        
        perf_tables.append(table)

    if 'tracking_runs' in root and 'best' in root['tracking_runs'].attrs:
        stats = root['tracking_runs'].attrs['best']
        table = Table(title="Best Tracking Performance", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Run Name", stats.get('run_name', 'N/A').split('/')[-1])
        table.add_row("Success Rate", f"{stats.get('percent_tracked', 0):.2f}%")
        
        if 'confidence_stats' in stats:
            table.add_row("Avg Confidence", f"{stats['confidence_stats'].get('mean', 0):.3f}")
        
        # Add timing info if available
        if 'duration' in stats:
            table.add_row("Duration", format_duration(stats['duration']))
            
        perf_tables.append(table)
        
    return Group(yolo_panel, *perf_tables)

def get_node_details(node_path: str, root: zarr.hierarchy.Group) -> Group:
    """Gets the detailed view for a specific Zarr group or array."""
    obj = root[node_path]
    
    # Attributes Table
    attributes_table = Table(title=f"Attributes for '{node_path}'", box=box.ROUNDED, show_header=True, header_style="bold magenta")
    attributes_table.add_column("Attribute Key", style="dim", width=35)
    attributes_table.add_column("Value")
    if not obj.attrs:
        attributes_table.add_row("[dim]No attributes[/dim]", "")
    for key, value in obj.attrs.items():
        if isinstance(value, dict):
            attributes_table.add_row(key, JSON.from_data(value))
        else:
            attributes_table.add_row(key, str(value))
        
    if isinstance(obj, zarr.hierarchy.Group):
        datasets = {name: d for name, d in obj.items() if isinstance(d, zarr.core.Array)}
        datasets_table = Table(title=f"Datasets in '{node_path}'", box=box.ROUNDED, show_header=True, header_style="bold cyan")
        if not datasets:
            datasets_table.add_row("[dim]No datasets in this group[/dim]", "")
        else:
            datasets_table.add_column("Name")
            datasets_table.add_column("Shape")
            datasets_table.add_column("Chunks")
            datasets_table.add_column("Dtype")
            datasets_table.add_column("Size", justify="right")
            for name, dset in datasets.items():
                datasets_table.add_row(name, str(dset.shape), str(dset.chunks), str(dset.dtype), format_bytes(dset.nbytes))
        return Group(attributes_table, datasets_table)
    
    elif isinstance(obj, zarr.core.Array):
        return Group(attributes_table)

class ZarrInspectorApp(App):
    """A Textual application to inspect Zarr files with advanced performance visualizations."""

    CSS = """
    .sidebar { 
        width: 30%; 
        height: 100%; 
        dock: left; 
        overflow: auto; 
    }
    .main-content { 
        width: 70%; 
        height: 100%; 
        overflow: auto; 
        padding: 1; 
    }
    .chart-grid {
        grid-size: 2;
        grid-gutter: 1;
    }
    .chart-widget {
        height: 20;
        border: solid $primary;
    }
    """
    BINDINGS = [("q", "quit", "Quit"), ("p", "toggle_performance", "Performance")]
    
    def __init__(self, zarr_path):
        super().__init__()
        self.zarr_path = zarr_path
        self.root = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Tree("Zarr Structure", id="tree", classes="sidebar")
        with VerticalScroll(classes="main-content"):
            with TabbedContent():
                with TabPane("Overview", id="overview"):
                    yield Static(id="details")
                with TabPane("Performance", id="performance"):
                    yield Static(id="performance_overview")
                    if PLOTEXT_AVAILABLE:
                        with Grid(classes="chart-grid"):
                            yield PerformanceChart("success_rates", id="success_chart", classes="chart-widget")
                            yield PerformanceChart("confidence_dist", id="confidence_chart", classes="chart-widget")
                            yield PerformanceChart("timing", id="timing_chart", classes="chart-widget")
                            yield PerformanceChart("data_sizes", id="size_chart", classes="chart-widget")
        yield Footer()

    def on_mount(self) -> None:
        """Called when the app is first mounted."""
        try:
            self.root = zarr.open(self.zarr_path, mode='r')
            self.title = f"Zarr Inspector: {os.path.basename(self.zarr_path)}"
            
            tree = self.query_one(Tree)
            tree.root.data = "/"
            self.build_zarr_tree(self.root, tree.root)
            tree.root.expand()

            # Load overview
            details_pane = self.query_one("#details", Static)
            details_pane.update(get_initial_summary(self.root))
            
            # Load performance overview
            performance_pane = self.query_one("#performance_overview", Static)
            performance_pane.update(create_performance_charts(self.root))
            
            # Update interactive charts if available
            if PLOTEXT_AVAILABLE:
                self.update_performance_charts()

        except Exception as e:
            error_msg = f"[bold red]Error opening Zarr file:[/bold red]\n{e}"
            self.query_one("#details", Static).update(error_msg)
            performance_pane = self.query_one("#performance_overview", Static)
            performance_pane.update(error_msg)

    def update_performance_charts(self):
        """Update the interactive performance charts."""
        if not PLOTEXT_AVAILABLE:
            return
            
        # Success rates chart
        success_data = {}
        if 'crop_runs' in self.root and 'best' in self.root['crop_runs'].attrs:
            success_data['Cropping'] = self.root['crop_runs'].attrs['best'].get('percent_cropped', 0)
        if 'tracking_runs' in self.root and 'best' in self.root['tracking_runs'].attrs:
            success_data['Tracking'] = self.root['tracking_runs'].attrs['best'].get('percent_tracked', 0)
        
        success_chart = self.query_one("#success_chart", PerformanceChart)
        success_chart.update_data(success_data)
        
        # Confidence distribution chart
        confidence_data = extract_confidence_data(self.root)
        confidence_chart = self.query_one("#confidence_chart", PerformanceChart)
        confidence_chart.update_data(confidence_data)
        
        # Timing chart
        timing_data = extract_timing_data(self.root)
        timing_chart = self.query_one("#timing_chart", PerformanceChart)
        timing_chart.update_data(timing_data)
        
        # Data sizes chart
        size_data = extract_size_data(self.root)
        size_chart = self.query_one("#size_chart", PerformanceChart)
        size_chart.update_data(size_data)

    def build_zarr_tree(self, group, tree_node):
        """Recursively builds the Textual Tree from the Zarr hierarchy."""
        for name, grp_obj in sorted(list(group.groups())):
            child_node = tree_node.add(f"[bold blue]{name}/[/bold blue]", data=grp_obj.path)
            self.build_zarr_tree(grp_obj, child_node)
        
        for name, arr_obj in sorted(list(group.arrays())):
            tree_node.add_leaf(
                f"[green]{name}[/green] [dim]({arr_obj.shape}, {arr_obj.dtype})[/dim]", 
                data=arr_obj.path
            )

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Called when a node in the tree is clicked."""
        details_pane = self.query_one("#details", Static)
        node_path = event.node.data
        if node_path:
            details_pane.update(get_node_details(node_path, self.root))

    def action_toggle_performance(self) -> None:
        """Toggle to performance tab."""
        tabbed_content = self.query_one(TabbedContent)
        tabbed_content.active = "performance"

def main():
    parser = argparse.ArgumentParser(description="Interactive inspector for Zarr archives with advanced performance visualizations.")
    parser.add_argument("zarr_path", type=str, help="Path to the Zarr archive directory.")
    args = parser.parse_args()

    if not Path(args.zarr_path).exists():
        print(f"Error: Path does not exist -> {args.zarr_path}")
        return

    if not PLOTEXT_AVAILABLE:
        print("Note: For enhanced visualizations, install textual-plotext:")
        print("  pip install textual-plotext")
        print()

    app = ZarrInspectorApp(zarr_path=args.zarr_path)
    app.run()

if __name__ == "__main__":
    main()
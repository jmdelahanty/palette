#!/usr/bin/env python3
"""
Enhanced inspector for kvikIO/GDS-imported Zarr video arrays with full metadata display.
Shows comprehensive system information, HPC job details, and environment tracking.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from rich.panel import Panel
from rich import box
from rich.columns import Columns
from rich.syntax import Syntax

def format_bytes(size_bytes):
    """Convert bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def format_timestamp(iso_str):
    """Format ISO timestamp to readable format."""
    try:
        dt = datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return iso_str

def count_chunks(array_path):
    """Count actual chunk files on disk for Zarr v3."""
    if not isinstance(array_path, Path):
        array_path = Path(array_path)
    
    chunk_dir = array_path / "c"
    if not chunk_dir.exists():
        return 0, 0
    
    chunk_count = 0
    total_size = 0
    
    # In this Zarr v3 structure, chunks are stored as c/0/0/0, c/1/0/0, etc.
    # The pattern is c/{chunk_index}/0/0
    for chunk_index_dir in chunk_dir.iterdir():
        if chunk_index_dir.is_dir() and chunk_index_dir.name.isdigit():
            # Look for the actual chunk file at {chunk_index}/0/0
            chunk_file = chunk_index_dir / "0" / "0"
            if chunk_file.exists():
                chunk_count += 1
                if chunk_file.is_file():
                    total_size += chunk_file.stat().st_size
                elif chunk_file.is_dir():
                    # If it's still a directory, sum all files within
                    for f in chunk_file.rglob('*'):
                        if f.is_file():
                            total_size += f.stat().st_size
    
    return chunk_count, total_size

def display_system_metadata(attrs, console):
    """Display comprehensive system metadata."""
    
    # System Information Table
    system_table = Table(title="System Information", box=box.ROUNDED)
    system_table.add_column("Property", style="cyan")
    system_table.add_column("Value", style="yellow")
    
    # Basic system info
    if 'system_hostname' in attrs:
        system_table.add_row("Hostname", attrs.get('system_hostname', 'Unknown'))
        system_table.add_row("FQDN", attrs.get('system_fqdn', 'Unknown'))
    
    system_table.add_row("OS", f"{attrs.get('system_os', 'Unknown')} {attrs.get('system_os_release', '')}")
    system_table.add_row("Machine", attrs.get('system_machine', 'Unknown'))
    system_table.add_row("Username", attrs.get('system_username', 'Unknown'))
    system_table.add_row("Python", attrs.get('system_python_version', 'Unknown'))
    system_table.add_row("CPU Cores", str(attrs.get('system_cpu_cores', 'Unknown')))
    
    # CPU details
    if 'cpu_model' in attrs:
        system_table.add_row("CPU Model", attrs.get('cpu_model', 'Unknown'))
        if 'cpu_arch' in attrs:
            system_table.add_row("CPU Arch", attrs.get('cpu_arch', 'Unknown'))
    
    # Memory info
    if 'memory_total_gb' in attrs:
        mem_total = attrs.get('memory_total_gb', 0)
        mem_avail = attrs.get('memory_available_gb', 0)
        mem_used = attrs.get('memory_percent_used', 0)
        system_table.add_row("Memory", f"{mem_total:.1f} GB total, {mem_avail:.1f} GB free ({mem_used:.1f}% used)")
    
    # Disk info
    if 'disk_total_gb' in attrs:
        disk_total = attrs.get('disk_total_gb', 0)
        disk_avail = attrs.get('disk_available_gb', 0)
        disk_used = attrs.get('disk_percent_used', 0)
        disk_path = attrs.get('disk_path', 'Unknown')
        system_table.add_row("Disk", f"{disk_total:.1f} GB total, {disk_avail:.1f} GB free ({disk_used:.1f}% used)")
        system_table.add_row("Disk Path", disk_path)
    
    console.print(system_table)
    
    # HPC Scheduler Information (if present)
    if attrs.get('hpc_scheduler'):
        scheduler = attrs['hpc_scheduler']
        hpc_table = Table(title=f"HPC Job Information ({scheduler})", box=box.ROUNDED)
        hpc_table.add_column("Property", style="cyan")
        hpc_table.add_column("Value", style="yellow")
        
        if scheduler == "LSF":
            hpc_table.add_row("Job ID", attrs.get('lsf_job_id', 'Unknown'))
            hpc_table.add_row("Job Name", attrs.get('lsf_job_name', 'Unknown'))
            hpc_table.add_row("Queue", attrs.get('lsf_queue', 'Unknown'))
            hpc_table.add_row("Hosts", attrs.get('lsf_hosts', 'Unknown'))
        elif scheduler == "SLURM":
            hpc_table.add_row("Job ID", attrs.get('slurm_job_id', 'Unknown'))
            hpc_table.add_row("Job Name", attrs.get('slurm_job_name', 'Unknown'))
            hpc_table.add_row("Node List", attrs.get('slurm_node_list', 'Unknown'))
        
        console.print(hpc_table)
    
    # GPU Information
    if attrs.get('gpu_available'):
        gpu_table = Table(title="GPU Information", box=box.ROUNDED)
        gpu_table.add_column("Property", style="cyan")
        gpu_table.add_column("Value", style="yellow")
        
        gpu_table.add_row("Available", "Yes")
        gpu_table.add_row("Backend", attrs.get('gpu_backend', 'Unknown'))
        gpu_table.add_row("Count", str(attrs.get('gpu_count', 1)))
        
        if 'cuda_version' in attrs:
            gpu_table.add_row("CUDA Version", attrs.get('cuda_version', 'Unknown'))
        
        gpu_table.add_row("GPU Name", attrs.get('gpu_name', 'Unknown'))
        gpu_table.add_row("Compute Capability", attrs.get('gpu_compute_capability', 'Unknown'))
        gpu_table.add_row("Memory", f"{attrs.get('gpu_memory_total_gb', 0):.1f} GB")
        
        # Runtime telemetry if available
        if 'gpu_temperature_c' in attrs:
            gpu_table.add_row("Temperature", f"{attrs.get('gpu_temperature_c', 0)}°C")
        if 'gpu_power_draw_w' in attrs:
            gpu_table.add_row("Power Draw", f"{attrs.get('gpu_power_draw_w', 0):.1f} W")
        if 'gpu_utilization_percent' in attrs:
            gpu_table.add_row("Utilization", f"{attrs.get('gpu_utilization_percent', 0)}%")
        
        console.print(gpu_table)
    
    # Git Information
    if 'git_commit_hash' in attrs and attrs['git_commit_hash'] != 'unknown':
        git_table = Table(title="Git Repository", box=box.ROUNDED)
        git_table.add_column("Property", style="cyan")
        git_table.add_column("Value", style="yellow")
        
        git_table.add_row("Commit", attrs.get('git_short_hash', 'Unknown'))
        git_table.add_row("Branch", attrs.get('git_branch', 'Unknown'))
        git_table.add_row("Dirty", "Yes" if attrs.get('git_is_dirty', False) else "No")
        
        remote_url = attrs.get('git_remote_url', 'Unknown')
        if remote_url != 'Unknown' and len(remote_url) > 50:
            # Truncate long URLs
            remote_url = remote_url[:47] + "..."
        git_table.add_row("Remote", remote_url)
        
        console.print(git_table)
    
    # Environment Information
    if 'environment_type' in attrs:
        env_table = Table(title="Environment", box=box.ROUNDED)
        env_table.add_column("Property", style="cyan")
        env_table.add_column("Value", style="yellow")
        
        env_table.add_row("Type", attrs.get('environment_type', 'Unknown'))
        env_table.add_row("Name", attrs.get('environment_name', 'Unknown'))
        env_table.add_row("Total Packages", str(attrs.get('total_packages', 0)))
        
        if 'deep_learning_framework' in attrs:
            env_table.add_row("DL Framework", attrs.get('deep_learning_framework', 'Unknown'))
        
        console.print(env_table)
        
        # Display key packages if available
        if 'key_packages_json' in attrs:
            try:
                key_packages = json.loads(attrs['key_packages_json'])
                if key_packages:
                    pkg_table = Table(title="Key Packages", box=box.SIMPLE)
                    pkg_table.add_column("Package", style="cyan", width=20)
                    pkg_table.add_column("Version", style="yellow", width=15)
                    
                    # Sort and display up to 20 most relevant packages
                    sorted_packages = sorted(key_packages.items())[:20]
                    for pkg_name, version in sorted_packages:
                        pkg_table.add_row(pkg_name, version)
                    
                    console.print(pkg_table)
            except json.JSONDecodeError:
                pass

def display_full_environment_info(attrs, console):
    """Display complete environment info if available."""
    if '_full_environment_info' in attrs:
        try:
            full_info = json.loads(attrs['_full_environment_info'])
            
            # Create a nicely formatted JSON display
            console.print("\n[bold]Full Environment Information (JSON)[/bold]")
            
            # Pretty print with syntax highlighting
            json_str = json.dumps(full_info, indent=2, default=str)
            syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False)
            
            # Put in a panel for better visibility
            panel = Panel(syntax, title="Complete Metadata", border_style="dim", expand=False)
            console.print(panel)
            
            return True
        except json.JSONDecodeError:
            return False
    return False

def display_background_metadata(zarr_path, console):
    """Display background computation metadata."""
    try:
        import zarr
        root = zarr.open(zarr_path, mode='r')
        
        if 'background_runs' not in root:
            return False
        
        bg_runs = root['background_runs']
        
        # Get list of all runs
        run_names = [name for name in bg_runs.group_keys()]
        
        if not run_names:
            return False
        
        # Display summary table
        bg_table = Table(title=" Background Computations", box=box.ROUNDED)
        bg_table.add_column("Run", style="cyan")
        bg_table.add_column("Timestamp", style="yellow")
        bg_table.add_column("Method", style="magenta")
        bg_table.add_column("Frames", style="green")
        bg_table.add_column("Duration", style="blue")
        
        latest_run = bg_runs.attrs.get('latest', '')
        
        for run_name in sorted(run_names):
            run = bg_runs[run_name]
            attrs = run.attrs
            
            # Mark latest run
            if run_name == latest_run:
                run_display = f"{run_name} [bold green]★[/bold green]"
            else:
                run_display = run_name
            
            timestamp = format_timestamp(attrs.get('background_timestamp_utc', ''))
            method = attrs.get('parameters', {}).get('method', 'unknown')
            frames = attrs.get('n_frames_used', 0)
            duration = attrs.get('duration_seconds', 0)
            
            bg_table.add_row(
                run_display,
                timestamp,
                method,
                str(frames),
                f"{duration:.1f}s"
            )
        
        console.print(bg_table)
        
        # Show details of latest run
        if latest_run and latest_run in bg_runs:
            latest = bg_runs[latest_run]
            params = latest.attrs.get('parameters', {})
            
            detail_table = Table(title=f"Latest Background Parameters ({latest_run})", box=box.SIMPLE)
            detail_table.add_column("Parameter", style="cyan")
            detail_table.add_column("Value", style="yellow")
            
            detail_table.add_row("Sample size", str(params.get('sample_size', 'unknown')))
            detail_table.add_row("Random seed", str(params.get('seed', 'unknown')))
            detail_table.add_row("Method", params.get('method', 'unknown'))
            
            # Check which backgrounds exist
            has_full = 'background_full' in latest
            has_ds = 'background_ds' in latest
            
            if has_full:
                detail_table.add_row("Full resolution", f"✓ {latest['background_full'].shape}")
            if has_ds:
                detail_table.add_row("Downsampled", f"✓ {latest['background_ds'].shape}")
            
            console.print(detail_table)
        
        return True
        
    except Exception as e:
        console.print(f"[yellow]Could not load background metadata: {e}[/yellow]")
        return False

def display_detection_metadata(zarr_path, console):
    """Display fish detection run metadata."""
    try:
        import zarr
        root = zarr.open(zarr_path, mode='r')

        if 'detect_runs' not in root:
            return False

        detect_runs = root['detect_runs']
        run_names = [name for name in detect_runs.group_keys()]

        if not run_names:
            return False

        # Summary table of all runs
        summary_table = Table(title=" Fish Detection Runs", box=box.ROUNDED)
        summary_table.add_column("Run", style="cyan")
        summary_table.add_column("Timestamp", style="yellow")
        summary_table.add_column("Total Detections", style="green")
        summary_table.add_column("Frames w/ Detections", style="magenta")
        summary_table.add_column("Source BG", style="blue")
        summary_table.add_column("Duration", style="blue")

        latest_run = detect_runs.attrs.get('latest', '')

        for run_name in sorted(run_names):
            run = detect_runs[run_name]
            attrs = run.attrs
            stats = attrs.get('summary_statistics', {})

            run_display = f"{run_name} [bold green]★[/bold green]" if run_name == latest_run else run_name

            timestamp = format_timestamp(attrs.get('detect_timestamp_utc', ''))
            total_detections = stats.get('total_detections', 0)
            frames_with_detections = stats.get('frames_with_detections', 0)
            percent_frames = stats.get('percent_frames_with_detections', 0)
            source_bg = attrs.get('source_background_run', 'unknown').replace('background_', '')
            duration = attrs.get('duration_seconds', 0)

            frames_display = f"{frames_with_detections} ({percent_frames}%)"

            summary_table.add_row(
                run_display,
                timestamp,
                str(total_detections),
                frames_display,
                source_bg,
                f"{duration:.1f}s"
            )
        
        console.print(summary_table)

        # Detail view for the latest run
        if latest_run and latest_run in detect_runs:
            latest = detect_runs[latest_run]
            params = latest.attrs.get('parameters', {})
            
            detail_table = Table(title=f"Latest Detection Parameters ({latest_run})", box=box.SIMPLE)
            detail_table.add_column("Parameter", style="cyan")
            detail_table.add_column("Value", style="yellow")

            for key, value in params.items():
                detail_table.add_row(str(key), str(value))
            
            # Also show array info
            if 'n_detections' in latest:
                detail_table.add_row("Counts array", f"✓ {latest['n_detections'].shape} | {latest['n_detections'].dtype}")
            if 'bbox_norm_coords' in latest:
                detail_table.add_row("BBox array", f"✓ {latest['bbox_norm_coords'].shape} | {latest['bbox_norm_coords'].dtype}")

            console.print(detail_table)

        return True

    except Exception as e:
        console.print(f"[yellow]Could not load detection metadata: {e}[/yellow]")
        return False

def display_analysis_metadata(zarr_path, console):
    """Display analysis metadata including mask and detection tuning."""
    try:
        import zarr
        root = zarr.open(zarr_path, mode='r')
        
        if 'analysis_metadata' not in root:
            return False
        
        analysis_meta = root['analysis_metadata']
        
        tables_shown = False
        
        # Check for dish mask tuning
        if 'dish_mask' in analysis_meta.attrs:
            mask_data = analysis_meta.attrs['dish_mask']
            
            mask_table = Table(title="🎯 Dish Mask Calibration", box=box.ROUNDED)
            mask_table.add_column("Property", style="cyan")
            mask_table.add_column("Value", style="yellow")
            
            # Basic mask info
            circle = mask_data.get('detected_circle', {})
            mask_table.add_row("Center", f"({circle.get('center', [0,0])[0]}, {circle.get('center', [0,0])[1]})")
            mask_table.add_row("Radius", f"{circle.get('radius', 0)} pixels")
            
            # Tuning details
            mask_table.add_row("Tuned on array", mask_data.get('tuned_on_array', 'Unknown'))
            mask_table.add_row("Tuned on frame", str(mask_data.get('tuned_on_frame', 0)))
            
            # Hough parameters
            mask_table.add_row("Hough param1", str(circle.get('hough_param1', 0)))
            mask_table.add_row("Hough param2", str(circle.get('hough_param2', 0)))
            mask_table.add_row("Radius adjustment", str(circle.get('radius_adjustment', 0)))
            
            # Metadata
            if 'tuned_timestamp' in mask_data:
                mask_table.add_row("Tuned at", format_timestamp(mask_data['tuned_timestamp']))
            
            console.print(mask_table)
            tables_shown = True
        
        # Check for detection tuning
        if 'detection_tuning' in analysis_meta.attrs:
            detect_data = analysis_meta.attrs['detection_tuning']
            params = detect_data.get('tuned_parameters', {})
            
            detect_table = Table(title="🐠 Detection Parameters", box=box.ROUNDED)
            detect_table.add_column("Parameter", style="cyan")
            detect_table.add_column("Value", style="yellow")
            
            detect_table.add_row("Threshold", str(params.get('ds_thresh', 'unknown')))
            detect_table.add_row("Erosion radius", str(params.get('se1_radius', 'unknown')))
            detect_table.add_row("Dilation radius", str(params.get('se4_radius', 'unknown')))
            detect_table.add_row("Min area", str(params.get('min_area', 'unknown')))
            detect_table.add_row("Max area", str(params.get('max_area', 'unknown')))
            
            if 'tuned_on_frame' in detect_data:
                detect_table.add_row("Tuned on frame", str(detect_data['tuned_on_frame']))
            
            if 'tuned_timestamp' in detect_data:
                detect_table.add_row("Tuned at", format_timestamp(detect_data['tuned_timestamp']))
            
            console.print(detect_table)
            tables_shown = True
        
        return tables_shown
            
    except Exception as e:
        console.print(f"[yellow]Could not load analysis metadata: {e}[/yellow]")
        return False

def inspect_video_zarr(zarr_path, show_full_env=False):
    """
    Inspect a video Zarr archive created by kvikIO import.
    
    Args:
        zarr_path: Path to the Zarr archive
        show_full_env: If True, display complete environment JSON at the end
    """
    console = Console()
    
    zarr_path = Path(zarr_path)
    if not zarr_path.exists():
        console.print(f"[red]Error: Path does not exist: {zarr_path}[/red]")
        return
    
    # Header
    console.rule("[bold cyan]Enhanced Video Zarr Inspector[/bold cyan]")
    console.print(f"[dim]Path: {zarr_path.absolute()}[/dim]\n")
    
    # Check root metadata
    root_json = zarr_path / "zarr.json"
    if root_json.exists():
        with open(root_json) as f:
            root_meta = json.load(f)
            if root_meta.get('node_type') == 'group':
                console.print("[green]Valid Zarr v3 group[/green]")
    
    analysis_meta_path = zarr_path / "analysis_metadata"
    if analysis_meta_path.exists():
        console.print("[green] Analysis metadata found[/green]")
        display_analysis_metadata(zarr_path, console)
        console.print()
    
    # Check raw_video group
    raw_video_path = zarr_path / "raw_video"
    raw_video_json = raw_video_path / "zarr.json"
    
    if not raw_video_json.exists():
        console.print("[yellow] raw_video group metadata not found[/yellow]")
        return
    
    with open(raw_video_json) as f:
        raw_meta = json.load(f)
    
    # Get attributes
    attrs = raw_meta.get('attributes', {})
    
    # Display import metadata with resolution info
    import_table = Table(title="Import Configuration", box=box.ROUNDED)
    import_table.add_column("Parameter", style="cyan")
    import_table.add_column("Value", style="yellow")
    
    # Basic info
    import_table.add_row("Source video", attrs.get('source_video', 'Unknown'))
    import_table.add_row("Import method", attrs.get('import_method', 'Unknown'))
    
    # Resolution mode info
    resolutions_mode = attrs.get('resolutions_mode', 'full')
    import_table.add_row("Resolution mode", resolutions_mode.title())
    
    import_table.add_row("Total frames", str(attrs.get('total_frames', 0)))
    
    # Show original resolution
    orig_res = attrs.get('original_resolution', [0, 0])
    import_table.add_row("Original resolution", f"{orig_res[1]}×{orig_res[0]}")
    
    # Show downsampled resolution if present
    if attrs.get('has_downsampled'):
        down_res = attrs.get('downsampled_resolution', [0, 0])
        import_table.add_row("Downsampled resolution", f"{down_res[1]}×{down_res[0]}")
        import_table.add_row("Downsample method", attrs.get('downsample_method', 'unknown'))
    
    import_table.add_row("FPS", f"{attrs.get('fps', 0):.1f}")
    import_table.add_row("Duration", f"{attrs.get('video_duration_seconds', 0):.1f} seconds")
    
    # Import settings
    import_table.add_row("", "")  # Separator
    import_table.add_row("Chunk size", f"{attrs.get('chunk_size', 0)} frames")
    import_table.add_row("IO batch size", f"{attrs.get('io_batch_size', 0)} frames")
    import_table.add_row("Device", attrs.get('device', 'Unknown'))
    
    if attrs.get('gpu_fp16'):
        import_table.add_row("GPU FP16", "Yes")
    
    if attrs.get('sharding_enabled'):
        import_table.add_row("Sharding", f"Yes ({attrs.get('chunks_per_shard', 0)} chunks/shard)")
    
    console.print(import_table)
    
    # Display performance metrics
    perf_table = Table(title="Performance Metrics", box=box.ROUNDED)
    perf_table.add_column("Metric", style="cyan")
    perf_table.add_column("Value", style="yellow")
    
    duration = attrs.get('import_duration_seconds', 0)
    perf_table.add_row("Import time", f"{duration:.1f} seconds ({duration/60:.1f} minutes)")
    perf_table.add_row("Throughput", f"{attrs.get('throughput_gbps', 0):.2f} GB/s")
    perf_table.add_row("Processing speed", f"{attrs.get('frames_per_second', 0):.1f} fps")
    perf_table.add_row("Data size", f"{attrs.get('data_size_gb', 0):.1f} GB")
    
    if 'import_timestamp' in attrs:
        perf_table.add_row("Imported at", format_timestamp(attrs['import_timestamp']))
    
    console.print(perf_table)
    
    # Display comprehensive system metadata
    console.print("\n[bold]Processing Environment[/bold]")
    display_system_metadata(attrs, console)
    
    # Check BOTH arrays if they exist (UPDATED)
    arrays_to_check = []
    if attrs.get('has_full_resolution', True):  # Default to True for backward compat
        arrays_to_check.append(('images_full', 'Full Resolution'))
    if attrs.get('has_downsampled', False):
        arrays_to_check.append(('images_ds', 'Downsampled'))
    
    for array_name, array_label in arrays_to_check:
        array_path = raw_video_path / array_name
        array_json = array_path / "zarr.json"
        
        if not array_json.exists():
            console.print(f"\n[yellow] {array_label} array ({array_name}) not found[/yellow]")
            continue
        
        with open(array_json) as f:
            array_meta = json.load(f)
        
        console.print(f"\n[bold]{array_label} Array Structure[/bold]")
        array_table = Table(box=box.ROUNDED)
        array_table.add_column("Property", style="cyan")
        array_table.add_column("Value", style="yellow")
        
        # Array properties
        shape = array_meta.get('shape', [])
        if shape:
            array_table.add_row("Shape", f"{shape[0]} × {shape[1]} × {shape[2]}")
            
            # Calculate theoretical size
            dtype_size = 1  # uint8
            total_elements = shape[0] * shape[1] * shape[2]
            theoretical_size = total_elements * dtype_size
            array_table.add_row("Theoretical size", format_bytes(theoretical_size))
        
        array_table.add_row("Data type", array_meta.get('data_type', 'Unknown'))
        array_table.add_row("Zarr format", str(array_meta.get('zarr_format', 'Unknown')))
        
        # Chunk configuration
        if 'chunk_grid' in array_meta:
            chunk_shape = array_meta['chunk_grid']['configuration']['chunk_shape']
            array_table.add_row("Chunk shape", f"{chunk_shape[0]} × {chunk_shape[1]} × {chunk_shape[2]}")
            
            # Calculate expected chunks
            if shape and chunk_shape:
                expected_chunks = -(-shape[0] // chunk_shape[0])  # Ceiling division
                array_table.add_row("Expected chunks", str(expected_chunks))
        
        console.print(array_table)
        
        # Storage analysis for this array
        chunk_count, total_chunk_size = count_chunks(array_path)
        
        storage_table = Table(title=f"{array_label} Storage", box=box.ROUNDED)
        storage_table.add_column("Metric", style="cyan")
        storage_table.add_column("Value", style="yellow")
        
        storage_table.add_row("Chunk files found", str(chunk_count))
        storage_table.add_row("Total chunk size", format_bytes(total_chunk_size))
        
        if chunk_count > 0 and shape and chunk_shape:
            expected_chunks = -(-shape[0] // chunk_shape[0])
            completion = (chunk_count / expected_chunks) * 100
            storage_table.add_row("Completion", f"{completion:.1f}%")
            
            avg_chunk_size = total_chunk_size / chunk_count
            storage_table.add_row("Average chunk size", format_bytes(avg_chunk_size))
        
        console.print(storage_table)
    
    # Total Zarr size (across all arrays)
    console.print("\n[bold]Overall Storage[/bold]")
    total_zarr_size = sum(f.stat().st_size for f in zarr_path.rglob('*') if f.is_file())
    
    overall_table = Table(box=box.ROUNDED)
    overall_table.add_column("Metric", style="cyan")
    overall_table.add_column("Value", style="yellow")
    overall_table.add_row("Total Zarr size", format_bytes(total_zarr_size))
    overall_table.add_row("Arrays present", ", ".join([name for name, _ in arrays_to_check]))
    
    console.print(overall_table)
    
    # Directory tree (compact view)
    console.print("\n[bold]Directory Structure[/bold]")
    tree = Tree(f"[cyan]{zarr_path.name}[/cyan]")
    
    def add_tree_node(parent_node, path, max_depth=3, current_depth=0):
        if current_depth >= max_depth:
            parent_node.add("[dim]...[/dim]")
            return
        
        items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        
        for item in items[:20]:  # Limit items shown
            if item.is_dir():
                if item.name == 'c':  # This is a chunks directory
                    chunk_count = len(list(item.iterdir()))
                    node = parent_node.add(f"[blue]{item.name}/[/blue] [dim]({chunk_count} chunks)[/dim]")
                elif item.name == 'analysis_metadata':
                    # Special highlight for analysis metadata
                    node = parent_node.add(f"[magenta]{item.name}/[/magenta] [dim](calibration data)[/dim]")
                    add_tree_node(node, item, max_depth, current_depth + 1)
                elif item.name in ['background_runs', 'detect_runs']: # HIGHLIGHT RUNS
                    node = parent_node.add(f"[yellow]{item.name}/[/yellow] [dim](processing runs)[/dim]")
                    add_tree_node(node, item, max_depth, current_depth + 1)
                elif item.name == 'raw_video':
                    node = parent_node.add(f"[green]{item.name}/[/green] [dim](video data)[/dim]")
                    add_tree_node(node, item, max_depth, current_depth + 1)
                elif item.name in ['images_full', 'images_ds']:
                    if (item / 'zarr.json').exists():
                        node = parent_node.add(f"[green]{item.name}/[/green] [dim](array)[/dim]")
                    else:
                        node = parent_node.add(f"[yellow]{item.name}/[/yellow] [dim](no metadata)[/dim]")
                    add_tree_node(node, item, max_depth, current_depth + 1)
                else:
                    node = parent_node.add(f"[blue]{item.name}/[/blue]")
                    add_tree_node(node, item, max_depth, current_depth + 1)
            else:
                size_str = format_bytes(item.stat().st_size)
                if item.suffix == '.json':
                    parent_node.add(f"[green]{item.name}[/green] [dim]({size_str})[/dim]")
                else:
                    parent_node.add(f"{item.name} [dim]({size_str})[/dim]")
    
    add_tree_node(tree, zarr_path, max_depth=3)
    console.print(tree)

    background_path = zarr_path / "background_runs"
    if background_path.exists():
        console.print("[green] Background computations found[/green]")
        display_background_metadata(zarr_path, console)
        console.print()
        
    detect_path = zarr_path / "detect_runs"
    if detect_path.exists():
        console.print("[green] Fish detection runs found[/green]")
        display_detection_metadata(zarr_path, console)
        console.print()
        
    # Optionally display full environment info
    if show_full_env:
        display_full_environment_info(attrs, console)
    
    # Validation summary (UPDATED for multi-resolution)
    console.print("\n[bold]Validation Summary[/bold]")
    
    issues = []
    warnings = []
    
    # Check expected arrays exist
    if attrs.get('has_full_resolution', True) and not (raw_video_path / 'images_full' / 'zarr.json').exists():
        issues.append("Full resolution array metadata missing")
    
    if attrs.get('has_downsampled', False) and not (raw_video_path / 'images_ds' / 'zarr.json').exists():
        issues.append("Downsampled array metadata missing")
    
    # Check for mismatched resolution config
    if resolutions_mode == 'both' and not (attrs.get('has_full_resolution') and attrs.get('has_downsampled')):
        warnings.append("Resolution mode is 'both' but not all arrays are marked present")
    
    if issues:
        console.print("[red]Issues found:[/red]")
        for issue in issues:
            console.print(f"  • {issue}")
    
    if warnings:
        console.print("[yellow]Warnings:[/yellow]")
        for warning in warnings:
            console.print(f"  • {warning}")
    
    if not issues and not warnings:
        console.print("[green]Array structure looks good![/green]")

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Inspect kvikIO/GDS Zarr video archives")
    parser.add_argument("zarr_path", help="Path to Zarr archive")
    parser.add_argument("--full-env", action="store_true", 
                       help="Display complete environment information JSON")
    
    args = parser.parse_args()
    
    inspect_video_zarr(args.zarr_path, show_full_env=args.full_env)
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

REFINED_DETECT_GROUP = "refined_detect_runs"
LEGACY_REFINED_DETECT_GROUP = "refined_runs"
REFINED_KEYPOINT_GROUP = "refined_keypoints_runs"
LEGACY_REFINED_KEYPOINT_GROUP = "keypoints_refined_runs"

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


def _coverage_from_group(group):
    frame_counts = group.get("frame_counts") or group.get("n_detections")
    if frame_counts is None:
        return None
    try:
        counts = frame_counts[:]
    except Exception:
        return None
    if counts.shape[0] == 0:
        return None
    present = (counts > 0).sum()
    return float(present) / float(counts.shape[0]) * 100.0


def _refined_detect_mode(run):
    attrs = run.attrs
    parameters = attrs.get("parameters")
    refine_mode = parameters.get("refine_mode") if isinstance(parameters, dict) else None
    operations = attrs.get("operations")
    if refine_mode == "passthrough" or operations == ["passthrough"]:
        return "passthrough"
    comparison = attrs.get("coverage_comparison")
    if isinstance(comparison, dict):
        filtered = comparison.get("filtered")
        interpolated = comparison.get("interpolated")
        if isinstance(filtered, dict) and isinstance(interpolated, dict):
            removed = filtered.get("detections_removed")
            added = interpolated.get("detections_added")
            if removed == 0 and added == 0:
                return "unchanged"
    if "interpolated" in run:
        return "interpolated"
    if "filtered" in run:
        return "filtered"
    return "unknown"


def _normalize_attr(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", "ignore")
    return str(value)


def _format_review_status(status: Optional[dict]) -> str:
    if not isinstance(status, dict):
        return "none"
    parts = []
    state = status.get("state")
    method = status.get("method")
    intended = status.get("intended_use")
    resolved = status.get("resolved_group") or status.get("target_group")
    if state:
        parts.append(f"state={state}")
    if method:
        parts.append(f"method={method}")
    if intended:
        parts.append(f"use={intended}")
    if resolved:
        parts.append(f"group={resolved}")
    return ", ".join(parts) if parts else "set"
    
def show_parameter_provenance(zarr_path, console):
    """Show where parameters came from for each completed stage."""
    try:
        import zarr
        root = zarr.open(zarr_path, mode='r')
        
        prov_table = Table(title=" Parameter Provenance", box=box.ROUNDED)
        prov_table.add_column("Stage", style="cyan")
        prov_table.add_column("Run", style="yellow")
        prov_table.add_column("Source", style="magenta")
        prov_table.add_column("Details", style="dim")
        
        stages_checked = []
        
        # Check each possible stage
        stage_groups = {
            'background': 'background_runs',
            'detect': 'detect_runs',
            'refine_detect': REFINED_DETECT_GROUP,
            'crop': 'crop_runs',
            'keypoints': 'keypoints_runs',
            'refine_keypoints': REFINED_KEYPOINT_GROUP,
            'track': 'tracking_runs',
            'assign_ids': 'id_assignment_runs',
        }
        tuning_key_map = {
            'detect': 'detection_tuning',
            'keypoints': 'keypoint_tuning',
        }
        
        for stage_name, group_name in stage_groups.items():
            if group_name not in root:
                if stage_name == "refine_detect" and LEGACY_REFINED_DETECT_GROUP in root:
                    group_name = LEGACY_REFINED_DETECT_GROUP
                else:
                    continue
            
            stage_group = root[group_name]
            latest = stage_group.attrs.get('latest')
            
            if not latest or latest not in stage_group:
                continue
            
            run = stage_group[latest]
            params = run.attrs.get('parameters', {})
            
            # Determine source
            param_source = "Unknown"
            details = ""
            
            # Check if from tuning
            if 'analysis_metadata' in root:
                analysis = root['analysis_metadata']
                tuning_key = tuning_key_map.get(stage_name, f"{stage_name}_tuning")
                if tuning_key in analysis.attrs:
                    param_source = "Zarr (tuned)"
                    tuning_data = analysis.attrs[tuning_key]
                    if 'tuned_timestamp' in tuning_data:
                        details = format_timestamp(tuning_data['tuned_timestamp'])
            
            # Check if from config
            if 'pipeline_params' in root:
                pipeline_params = root['pipeline_params']
                if stage_name in pipeline_params.attrs:
                    if param_source == "Unknown":
                        param_source = "Config file"
            
            # Default fallback
            if param_source == "Unknown":
                if stage_name == "refine_detect" and isinstance(params, dict):
                    source_label = params.get("parameter_source")
                    if source_label:
                        param_source = f"Run ({source_label})"
                provenance = run.attrs.get('provenance')
                if isinstance(provenance, dict):
                    param_info = provenance.get('parameters', {})
                    source_label = param_info.get('parameter_source')
                    if source_label:
                        param_source = f"Run ({source_label})"
                        created_at = provenance.get('created_at_utc')
                        if created_at:
                            details = format_timestamp(created_at)
                if param_source == "Unknown":
                    param_source = "Code defaults"
            
            prov_table.add_row(
                stage_name,
                latest,
                param_source,
                details
            )
            stages_checked.append(stage_name)
        
        if stages_checked:
            console.print(prov_table)
            return True
        return False
        
    except Exception as e:
        console.print(f"[dim]Could not load parameter provenance: {e}[/dim]")
        return False

def show_quick_summary(zarr_path, console):
    """Show a quick 5-line summary of the zarr status."""
    try:
        import zarr
        root = zarr.open(zarr_path, mode='r')
        
        # Count completed stages
        stages = ['import', 'background', 'detect', 'refine_detect', 'crop', 'keypoints', 'track', 'assign_ids']
        completed = []
        
        if 'raw_video' in root and 'images_full' in root['raw_video']:
            completed.append('import')
        if 'background_runs' in root and root['background_runs'].attrs.get('latest'):
            completed.append('background')
        if 'detect_runs' in root and root['detect_runs'].attrs.get('latest'):
            completed.append('detect')
        if (REFINED_DETECT_GROUP in root and root[REFINED_DETECT_GROUP].attrs.get('latest')) or (
            LEGACY_REFINED_DETECT_GROUP in root and root[LEGACY_REFINED_DETECT_GROUP].attrs.get('latest')
        ):
            completed.append('refine_detect')
        if 'crop_runs' in root and root['crop_runs'].attrs.get('latest'):
            completed.append('crop')
        if 'keypoints_runs' in root and root['keypoints_runs'].attrs.get('latest'):
            completed.append('keypoints')
        if 'tracking_runs' in root and root['tracking_runs'].attrs.get('latest'):
            completed.append('track')
        if 'id_assignment_runs' in root and root['id_assignment_runs'].attrs.get('latest'):
            completed.append('assign_ids')
        
        # Get detection coverage if available
        coverage_str = "N/A"
        if 'detect_runs' in root:
            latest = root['detect_runs'].attrs.get('latest')
            if latest and latest in root['detect_runs']:
                stats = root['detect_runs'][latest].attrs.get('summary_statistics', {})
                coverage_str = f"{stats.get('percent_frames_with_detections', 0):.1f}%"

        refined_coverage_str = "N/A"
        refined_parent = root.get(REFINED_DETECT_GROUP) or root.get(LEGACY_REFINED_DETECT_GROUP)
        if refined_parent is not None:
            latest_refined = refined_parent.attrs.get("latest")
            if latest_refined and latest_refined in refined_parent:
                refined_run = refined_parent[latest_refined]
                manual_label = _normalize_attr(refined_run.attrs.get("manual_review_latest"))
                base_group = None
                label = None
                if manual_label and manual_label in refined_run:
                    base_group = refined_run[manual_label]
                    label = f"manual:{manual_label}"
                elif "interpolated" in refined_run:
                    base_group = refined_run["interpolated"]
                    label = "interpolated"
                elif "filtered" in refined_run:
                    base_group = refined_run["filtered"]
                    label = "filtered"
                if base_group is not None:
                    cov = _coverage_from_group(base_group)
                    if cov is not None:
                        refined_coverage_str = f"{cov:.1f}%"
                        if label:
                            refined_coverage_str = f"{refined_coverage_str} ({label})"
        
        # Get experiment setup
        exp_setup = root.attrs.get('experiment_setup', {})
        setup_type = exp_setup.get('setup_type', 'Not configured')
        
        # Get last modified
        zarr_path_obj = Path(zarr_path)
        last_mod = datetime.fromtimestamp(zarr_path_obj.stat().st_mtime)
        
        # Build summary
        summary_panel = Panel(
            f"[cyan]Stages:[/cyan] {len(completed)}/{len(stages)} completed ({', '.join(completed) if completed else 'none'})\n"
            f"[cyan]Detection Coverage:[/cyan] {coverage_str}\n"
            f"[cyan]Refined Coverage:[/cyan] {refined_coverage_str}\n"
            f"[cyan]Experiment Type:[/cyan] {setup_type}\n"
            f"[cyan]Last Modified:[/cyan] {last_mod.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"[cyan]Size:[/cyan] {format_bytes(sum(f.stat().st_size for f in zarr_path_obj.rglob('*') if f.is_file()))}",
            title="Quick Summary",
            border_style="cyan",
            box=box.ROUNDED
        )
        
        console.print(summary_panel)
        return True
        
    except Exception as e:
        console.print(f"[red]Error generating summary: {e}[/red]")
        return False

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


def display_refined_detect_metadata(zarr_path, console):
    """Display refined detection run metadata (including manual/retune groups)."""
    try:
        import zarr

        root = zarr.open(zarr_path, mode="r")

        refined_parent = root.get(REFINED_DETECT_GROUP) or root.get(LEGACY_REFINED_DETECT_GROUP)
        if refined_parent is None:
            return False

        run_names = [name for name in refined_parent.group_keys()]
        if not run_names:
            return False

        latest_run = refined_parent.attrs.get("latest", "")

        summary_table = Table(title=" Refined Detection Runs", box=box.ROUNDED)
        summary_table.add_column("Run", style="cyan")
        summary_table.add_column("Timestamp", style="yellow")
        summary_table.add_column("Mode", style="magenta")
        summary_table.add_column("Coverage", style="green")
        summary_table.add_column("Manual Group", style="blue")
        summary_table.add_column("Source Detect", style="blue")

        for run_name in sorted(run_names):
            run = refined_parent[run_name]
            attrs = run.attrs

            run_display = f"{run_name} [bold green]★[/bold green]" if run_name == latest_run else run_name
            timestamp = format_timestamp(attrs.get("refinement_timestamp", attrs.get("created_at", "")))
            mode = _refined_detect_mode(run)

            manual_group_name = _normalize_attr(attrs.get("manual_review_latest"))
            manual_group = run.get(manual_group_name) if manual_group_name else None
            coverage = None
            coverage_label = None
            if manual_group is not None:
                coverage = _coverage_from_group(manual_group)
                coverage_label = f"manual:{manual_group_name}"
            elif "interpolated" in run:
                coverage = _coverage_from_group(run["interpolated"])
                coverage_label = "interpolated"
            elif "filtered" in run:
                coverage = _coverage_from_group(run["filtered"])
                coverage_label = "filtered"

            coverage_str = "N/A"
            if coverage is not None:
                coverage_str = f"{coverage:.1f}%"
                if coverage_label:
                    coverage_str = f"{coverage_str} ({coverage_label})"

            source_detect = str(attrs.get("source_detect_run", "unknown"))

            summary_table.add_row(
                run_display,
                timestamp,
                mode,
                coverage_str,
                str(manual_group_name or "-"),
                source_detect,
            )

        console.print(summary_table)

        if latest_run and latest_run in refined_parent:
            latest = refined_parent[latest_run]
            attrs = latest.attrs

            detail_table = Table(title=f"Latest Refined Detect Details ({latest_run})", box=box.SIMPLE)
            detail_table.add_column("Field", style="cyan")
            detail_table.add_column("Value", style="yellow")

            detail_table.add_row("Source detect", str(attrs.get("source_detect_run", "unknown")))
            detail_table.add_row("Source quality", str(attrs.get("source_quality_run", "unknown")))
            detail_table.add_row("Operations", str(attrs.get("operations", "unknown")))

            parameters = attrs.get("parameters", {})
            if isinstance(parameters, dict):
                detail_table.add_row("Refine mode", str(parameters.get("refine_mode", "unknown")))
                detail_table.add_row("Parameter source", str(parameters.get("parameter_source", "unknown")))
                detail_table.add_row("Interpolation method", str(parameters.get("interpolation_method", "unknown")))
                detail_table.add_row("Max gap", str(parameters.get("max_gap", "unknown")))
                detail_table.add_row("Filters", str(parameters.get("filters_applied", [])))
                detail_table.add_row("Sampled import", str(parameters.get("sampled_import", False)))

            detail_table.add_row("Coverage source", str(attrs.get("coverage_frame_source", "unknown")))
            detail_table.add_row("Coverage frames total", str(attrs.get("coverage_frames_total", "unknown")))
            detail_table.add_row("Coverage frames full", str(attrs.get("coverage_frames_full", "unknown")))

            manual_group_name = _normalize_attr(attrs.get("manual_review_latest"))
            detail_table.add_row("Manual group", str(manual_group_name or "none"))

            retune_params = attrs.get("retune_params")
            retune_count = len(retune_params) if isinstance(retune_params, dict) else 0
            detail_table.add_row("Retune parameter sets", str(retune_count))

            console.print(detail_table)

            if manual_group_name and manual_group_name in latest:
                manual = latest[manual_group_name]
                manual_attrs = manual.attrs
                manual_table = Table(title=f"Manual/Retune Details ({manual_group_name})", box=box.SIMPLE)
                manual_table.add_column("Field", style="cyan")
                manual_table.add_column("Value", style="yellow")

                manual_table.add_row("Detection source type", str(manual_attrs.get("detection_source_type", "unknown")))
                manual_table.add_row("Detection source path", str(manual_attrs.get("detection_source_path", "unknown")))
                manual_table.add_row("Retune base group", str(manual_attrs.get("retune_base_group", "unknown")))
                manual_table.add_row("Retune parameter source", str(manual_attrs.get("retune_parameter_source", "unknown")))
                manual_table.add_row("Retune score", str(manual_attrs.get("retune_score", "unknown")))
                manual_table.add_row("Retune class id", str(manual_attrs.get("retune_class_id", "unknown")))
                manual_table.add_row("Retune frames requested", str(manual_attrs.get("retune_frames_requested", "unknown")))
                manual_table.add_row("Retune detections added", str(manual_attrs.get("retune_detections_added", "unknown")))
                manual_table.add_row("Retune timestamp", format_timestamp(manual_attrs.get("retune_timestamp", "")))

                manual_coverage = _coverage_from_group(manual)
                if manual_coverage is not None:
                    manual_table.add_row("Manual coverage", f"{manual_coverage:.1f}%")

                console.print(manual_table)

        return True

    except Exception as e:
        console.print(f"[yellow]Could not load refined detect metadata: {e}[/yellow]")
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
            mask_attr = analysis_meta.attrs['dish_mask']
            mask_data = dict(mask_attr)
            shape = mask_data.get('shape')
            if not shape:
                if 'detected_circle' in mask_data:
                    shape = 'circle'
                elif 'rectangle' in mask_data:
                    shape = 'rectangle'
            mask_table = Table(title=" Dish Mask Calibration", box=box.ROUNDED)
            mask_table.add_column("Property", style="cyan")
            mask_table.add_column("Value", style="yellow")
            
            mask_table.add_row("Shape", shape or "unknown")
            if 'tuned_timestamp' in mask_data:
                mask_table.add_row("Tuned at", format_timestamp(mask_data['tuned_timestamp']))
            source_info = mask_data.get('source', {})
            if source_info:
                mask_table.add_row("Source array", str(source_info.get('array', 'unknown')))
                if 'frame' in source_info:
                    mask_table.add_row("Source frame", str(source_info['frame']))
            
            if shape == 'circle' and 'detected_circle' in mask_data:
                circle = mask_data['detected_circle']
                center = circle.get('center', [0, 0])
                mask_table.add_row("Center", f"({center[0]}, {center[1]})")
                mask_table.add_row("Radius", f"{circle.get('radius', 0)} px")
                hough = mask_data.get('hough_params', {})
                if hough:
                    mask_table.add_row("Hough param1", str(hough.get('param1', '')))
                    mask_table.add_row("Hough param2", str(hough.get('param2', '')))
                    mask_table.add_row("Radius adjustment", str(hough.get('radius_adjustment', 0)))
            elif shape == 'rectangle' and 'rectangle' in mask_data:
                roi = mask_data['rectangle'].get('roi', [0, 0, 0, 0])
                mask_table.add_row("ROI", f"x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
            
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

        # Check for keypoint tuning
        if 'keypoint_tuning' in analysis_meta.attrs:
            keypoint_data = analysis_meta.attrs['keypoint_tuning']
            params = keypoint_data.get('tuned_parameters', {})

            keypoint_table = Table(title="Keypoint Parameters", box=box.ROUNDED)
            keypoint_table.add_column("Parameter", style="cyan")
            keypoint_table.add_column("Value", style="yellow")

            if isinstance(params, dict) and params:
                for key in sorted(params.keys()):
                    keypoint_table.add_row(str(key), str(params.get(key)))
            else:
                keypoint_table.add_row("tuned_parameters", "missing")

            if 'tuned_on_frame' in keypoint_data:
                keypoint_table.add_row("Tuned on frame", str(keypoint_data['tuned_on_frame']))

            if 'tuned_timestamp' in keypoint_data:
                keypoint_table.add_row("Tuned at", format_timestamp(keypoint_data['tuned_timestamp']))

            console.print(keypoint_table)
            tables_shown = True
        
        return tables_shown
            
    except Exception as e:
        console.print(f"[yellow]Could not load analysis metadata: {e}[/yellow]")
        return False
    

def display_crop_metadata(zarr_path, console):
    """Display crop run metadata."""
    try:
        import zarr
        root = zarr.open(zarr_path, mode='r')

        if 'crop_runs' not in root:
            return False

        crop_runs = root['crop_runs']
        run_names = [name for name in crop_runs.group_keys()]

        if not run_names:
            return False

        summary_table = Table(title=" Crop Runs", box=box.ROUNDED)
        summary_table.add_column("Run", style="cyan")
        summary_table.add_column("Timestamp", style="yellow")
        summary_table.add_column("Total ROIs", style="green")
        summary_table.add_column("Frames w/ Crops", style="magenta")
        summary_table.add_column("ROI Size", style="blue")
        summary_table.add_column("Duration", style="blue")

        latest_run = crop_runs.attrs.get('latest', '')

        for run_name in sorted(run_names):
            run = crop_runs[run_name]
            attrs = run.attrs
            stats = attrs.get('summary_statistics', {})

            run_display = f"{run_name} [bold green]★[/bold green]" if run_name == latest_run else run_name

            timestamp = format_timestamp(attrs.get('crop_timestamp_utc', ''))
            total_rois = stats.get('total_rois_cropped', 0)
            frames_with_crops = stats.get('frames_with_crops', 0)
            roi_size = stats.get('roi_size', [0, 0])
            duration = attrs.get('duration_seconds', 0)

            summary_table.add_row(
                run_display,
                timestamp,
                str(total_rois),
                str(frames_with_crops),
                f"{roi_size[0]}×{roi_size[1]}",
                f"{duration:.1f}s"
            )
        
        console.print(summary_table)

        # Detail view for latest run
        if latest_run and latest_run in crop_runs:
            latest = crop_runs[latest_run]
            params = latest.attrs.get('parameters', {})

            detail_table = Table(title=f"Latest Crop Parameters ({latest_run})", box=box.SIMPLE)
            detail_table.add_column("Parameter", style="cyan")
            detail_table.add_column("Value", style="yellow")

            for key, value in params.items():
                detail_table.add_row(str(key), str(value))

            console.print(detail_table)

            meta_table = Table(title=f"Latest Crop Details ({latest_run})", box=box.SIMPLE)
            meta_table.add_column("Field", style="cyan")
            meta_table.add_column("Value", style="yellow")

            meta_table.add_row("Detection source type", str(latest.attrs.get("detection_source_type", "unknown")))
            meta_table.add_row("Detection source path", str(latest.attrs.get("detection_source_path", "unknown")))
            meta_table.add_row("Preferred policy", str(latest.attrs.get("detection_preferred_policy", "none")))
            meta_table.add_row("Review status", _format_review_status(latest.attrs.get("detect_review_status")))
            meta_table.add_row("Review status ref", str(latest.attrs.get("detect_review_status_ref", "none")))
            meta_table.add_row("Source detect run", str(latest.attrs.get("source_detect_run", "unknown")))
            meta_table.add_row("Source refined run", str(latest.attrs.get("source_refined_run", "unknown")))

            console.print(meta_table)

        return True

    except Exception as e:
        console.print(f"[yellow]Could not load crop metadata: {e}[/yellow]")
        return False


def display_keypoints_metadata(zarr_path, console):
    """Display keypoints run metadata."""
    try:
        import zarr
        root = zarr.open(zarr_path, mode='r')

        if 'keypoints_runs' not in root:
            return False

        kp_runs = root['keypoints_runs']
        run_names = [name for name in kp_runs.group_keys()]

        if not run_names:
            return False

        summary_table = Table(title=" Keypoints Runs", box=box.ROUNDED)
        summary_table.add_column("Run", style="cyan")
        summary_table.add_column("Timestamp", style="yellow")
        summary_table.add_column("Success Rate", style="green")
        summary_table.add_column("Successful", style="magenta")
        summary_table.add_column("Duration", style="blue")

        latest_run = kp_runs.attrs.get('latest', '')

        for run_name in sorted(run_names):
            run = kp_runs[run_name]
            attrs = run.attrs
            stats = attrs.get('summary_statistics', {})

            run_display = f"{run_name} [bold green]★[/bold green]" if run_name == latest_run else run_name

            timestamp = format_timestamp(attrs.get('keypoints_timestamp_utc', ''))
            success_rate = stats.get('success_rate_percent', 0)
            successful = stats.get('successful_detections', 0)
            duration = attrs.get('duration_seconds', 0)

            summary_table.add_row(
                run_display,
                timestamp,
                f"{success_rate:.1f}%",
                str(successful),
                f"{duration:.1f}s"
            )
        
        console.print(summary_table)

        # Detail view for latest run
        if latest_run and latest_run in kp_runs:
            latest = kp_runs[latest_run]
            params = latest.attrs.get('parameters', {})
            
            detail_table = Table(title=f"Latest Keypoint Parameters ({latest_run})", box=box.SIMPLE)
            detail_table.add_column("Parameter", style="cyan")
            detail_table.add_column("Value", style="yellow")

            for key, value in params.items():
                detail_table.add_row(str(key), str(value))
            
            console.print(detail_table)

        return True

    except Exception as e:
        console.print(f"[yellow]Could not load keypoints metadata: {e}[/yellow]")
        return False


def _split_keypoint_summary(stats):
    if not isinstance(stats, dict):
        return {}, {}
    if isinstance(stats.get("refine"), dict):
        refine_stats = stats.get("refine", {})
        post_stats = stats.get("postprocess", {})
        if not isinstance(post_stats, dict):
            post_stats = {}
        return refine_stats, post_stats
    return stats, {}


def _format_rate(value):
    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return str(value)


def display_refined_keypoints_metadata(zarr_path, console):
    """Display refined keypoints run metadata."""
    try:
        import zarr
        root = zarr.open(zarr_path, mode='r')

        refined_parent = None
        for group_name in (REFINED_KEYPOINT_GROUP, LEGACY_REFINED_KEYPOINT_GROUP):
            if group_name in root:
                refined_parent = root[group_name]
                break

        if refined_parent is None:
            return False

        run_names = [name for name in refined_parent.group_keys()]
        if not run_names:
            return False

        summary_table = Table(title=" Refined Keypoints Runs", box=box.ROUNDED)
        summary_table.add_column("Run", style="cyan")
        summary_table.add_column("Timestamp", style="yellow")
        summary_table.add_column("Success Rate", style="green")
        summary_table.add_column("Refined", style="magenta")
        summary_table.add_column("Usable", style="magenta")
        summary_table.add_column("Manual", style="blue")
        summary_table.add_column("Retune", style="blue")

        latest_run = refined_parent.attrs.get('latest', '')

        for run_name in sorted(run_names):
            run = refined_parent[run_name]
            attrs = run.attrs
            stats = attrs.get('summary_statistics', {})
            refine_stats, post_stats = _split_keypoint_summary(stats)

            use_stats = post_stats or refine_stats

            run_display = f"{run_name} [bold green]★[/bold green]" if run_name == latest_run else run_name

            timestamp = format_timestamp(
                attrs.get('created_utc', attrs.get('created_at', ''))
            )
            success_rate = use_stats.get(
                'success_rate_percent',
                refine_stats.get('pass_rate_percent', 0),
            )
            refined_success = use_stats.get('refined_success', refine_stats.get('refined_success', 0))
            usable = use_stats.get('usable_keypoints', refine_stats.get('usable_keypoints', 0))
            manual = post_stats.get('manual_corrections', 0)
            retune_total = post_stats.get('retune_total', 0)

            summary_table.add_row(
                run_display,
                timestamp,
                _format_rate(success_rate),
                str(refined_success),
                str(usable),
                str(manual),
                str(retune_total),
            )

        console.print(summary_table)

        if latest_run and latest_run in refined_parent:
            latest = refined_parent[latest_run]
            attrs = latest.attrs

            detail_table = Table(title=f"Latest Refined Keypoint Details ({latest_run})", box=box.SIMPLE)
            detail_table.add_column("Field", style="cyan")
            detail_table.add_column("Value", style="yellow")

            detail_table.add_row("Source keypoints", str(attrs.get("source_keypoints_run", "unknown")))
            detail_table.add_row("Source crop", str(attrs.get("source_crop_run", "unknown")))
            detail_table.add_row("Source detect", str(attrs.get("source_detect_run", "unknown")))

            retune_params = attrs.get("retune_params")
            retune_count = len(retune_params) if isinstance(retune_params, dict) else 0
            detail_table.add_row("Retune parameter sets", str(retune_count))

            summary_stats = attrs.get("summary_statistics", {})
            if isinstance(summary_stats, dict):
                post_updated = summary_stats.get("postprocess_updated_utc")
            else:
                post_updated = None
            if post_updated:
                detail_table.add_row("Postprocess updated", format_timestamp(post_updated))

            provenance = attrs.get("provenance", {})
            param_info = provenance.get("parameters", {}) if isinstance(provenance, dict) else {}
            if param_info:
                detail_table.add_row("Parameter source", str(param_info.get("parameter_source", "unknown")))
                detail_table.add_row("Chunk size", str(param_info.get("chunk_size", "unknown")))
                detail_table.add_row("Scheduler", str(param_info.get("scheduler", "unknown")))

            console.print(detail_table)

            stats = attrs.get("summary_statistics", {})
            refine_stats, post_stats = _split_keypoint_summary(stats)

            if refine_stats:
                refine_table = Table(title="Refine Summary", box=box.SIMPLE)
                refine_table.add_column("Metric", style="cyan")
                refine_table.add_column("Value", style="yellow")
                for key in (
                    "total_rois",
                    "source_success",
                    "source_failures",
                    "refined_success",
                    "flips_corrected",
                    "low_confidence",
                    "confidence_missing",
                    "geometry_issues",
                    "clean",
                    "usable_keypoints",
                    "pass_rate_percent",
                    "duration_seconds",
                ):
                    if key in refine_stats:
                        value = refine_stats[key]
                        if key.endswith("percent"):
                            value = _format_rate(value)
                        refine_table.add_row(str(key), str(value))
                console.print(refine_table)

            if post_stats:
                post_table = Table(title="Postprocess Summary", box=box.SIMPLE)
                post_table.add_column("Metric", style="cyan")
                post_table.add_column("Value", style="yellow")
                for key in (
                    "total_rois",
                    "refined_success",
                    "remaining_failures",
                    "success_rate_percent",
                    "usable_keypoints",
                    "retune_total",
                    "manual_corrections",
                ):
                    if key in post_stats:
                        value = post_stats[key]
                        if key.endswith("percent"):
                            value = _format_rate(value)
                        post_table.add_row(str(key), str(value))
                console.print(post_table)

        return True

    except Exception as e:
        console.print(f"[yellow]Could not load refined keypoints metadata: {e}[/yellow]")
        return False


def display_tracking_metadata(zarr_path, console):
    """Display tracking run metadata."""
    try:
        import zarr
        root = zarr.open(zarr_path, mode='r')

        if 'tracking_runs' not in root:
            return False

        track_runs = root['tracking_runs']
        run_names = [name for name in track_runs.group_keys()]

        if not run_names:
            return False

        summary_table = Table(title=" Tracking Runs", box=box.ROUNDED)
        summary_table.add_column("Run", style="cyan")
        summary_table.add_column("Timestamp", style="yellow")
        summary_table.add_column("Method", style="magenta")
        summary_table.add_column("Duration", style="blue")

        latest_run = track_runs.attrs.get('latest', '')

        for run_name in sorted(run_names):
            run = track_runs[run_name]
            attrs = run.attrs

            run_display = f"{run_name} [bold green]★[/bold green]" if run_name == latest_run else run_name

            timestamp = format_timestamp(attrs.get('tracking_timestamp_utc', ''))
            method = attrs.get('parameters', {}).get('method', 'unknown')
            duration = attrs.get('duration_seconds', 0)

            summary_table.add_row(
                run_display,
                timestamp,
                method,
                f"{duration:.1f}s"
            )
        
        console.print(summary_table)

        return True

    except Exception as e:
        console.print(f"[yellow]Could not load tracking metadata: {e}[/yellow]")
        return False


def display_id_assignment_metadata(zarr_path, console):
    """Display ID assignment run metadata."""
    try:
        import zarr
        root = zarr.open(zarr_path, mode='r')

        if 'id_assignment_runs' not in root:
            return False

        id_runs = root['id_assignment_runs']
        run_names = [name for name in id_runs.group_keys()]

        if not run_names:
            return False

        summary_table = Table(title=" ID Assignment Runs", box=box.ROUNDED)
        summary_table.add_column("Run", style="cyan")
        summary_table.add_column("Timestamp", style="yellow")
        summary_table.add_column("Setup Type", style="magenta")
        summary_table.add_column("Assigned", style="green")
        summary_table.add_column("Duration", style="blue")

        latest_run = id_runs.attrs.get('latest', '')

        for run_name in sorted(run_names):
            run = id_runs[run_name]
            attrs = run.attrs
            stats = attrs.get('summary_statistics', {})

            run_display = f"{run_name} [bold green]★[/bold green]" if run_name == latest_run else run_name

            timestamp = format_timestamp(attrs.get('assign_ids_timestamp_utc', ''))
            setup_type = stats.get('setup_type', 'unknown')
            assigned = stats.get('assigned_detections', 0)
            total = stats.get('total_detections', 0)
            duration = attrs.get('duration_seconds', 0)

            assigned_str = f"{assigned}/{total}"
            if total > 0:
                pct = (assigned / total) * 100
                assigned_str += f" ({pct:.0f}%)"

            summary_table.add_row(
                run_display,
                timestamp,
                setup_type,
                assigned_str,
                f"{duration:.1f}s"
            )
        
        console.print(summary_table)

        return True

    except Exception as e:
        console.print(f"[yellow]Could not load ID assignment metadata: {e}[/yellow]")
        return False


def display_import_metadata(zarr_path, console):
    """Display import/raw_video metadata."""
    try:
        raw_video_path = zarr_path / "raw_video"
        raw_video_json = raw_video_path / "zarr.json"
        
        if not raw_video_json.exists():
            return False
        
        with open(raw_video_json) as f:
            raw_meta = json.load(f)
        
        attrs = raw_meta.get('attributes', {})
        
        # Import info
        import_table = Table(title=" Import Information", box=box.ROUNDED)
        import_table.add_column("Property", style="cyan")
        import_table.add_column("Value", style="yellow")
        
        import_table.add_row("Source video", attrs.get('source_video', 'Unknown'))
        import_table.add_row("Import method", attrs.get('import_method', 'Unknown'))
        import_table.add_row("Total frames", str(attrs.get('total_frames', 0)))
        
        orig_res = attrs.get('original_resolution', [0, 0])
        import_table.add_row("Resolution", f"{orig_res[1]}×{orig_res[0]}")
        
        import_table.add_row("FPS", f"{attrs.get('fps', 0):.1f}")
        import_table.add_row("Duration", f"{attrs.get('video_duration_seconds', 0):.1f}s")
        import_table.add_row("Device", attrs.get('device', 'Unknown'))
        
        if attrs.get('import_timestamp'):
            import_table.add_row("Imported at", format_timestamp(attrs['import_timestamp']))
        
        console.print(import_table)
        
        # Performance
        perf_table = Table(title="Performance", box=box.SIMPLE)
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value", style="yellow")
        
        duration = attrs.get('import_duration_seconds', 0)
        perf_table.add_row("Import time", f"{duration:.1f}s")
        perf_table.add_row("Throughput", f"{attrs.get('throughput_gbps', 0):.2f} GB/s")
        perf_table.add_row("Processing speed", f"{attrs.get('frames_per_second', 0):.1f} fps")
        
        console.print(perf_table)
        
        return True
        
    except Exception as e:
        console.print(f"[yellow]Could not load import metadata: {e}[/yellow]")
        return False

def inspect_video_zarr(zarr_path, mode='normal', show_full_env=False, focus_stage=None):
    """
    Inspect a video Zarr archive.
    
    Args:
        zarr_path: Path to the Zarr archive
        show_full_env: If True, display complete environment JSON at the end
    """
    console = Console()
    
    zarr_path = Path(zarr_path)
    if not zarr_path.exists():
        console.print(f"[red]Error: Path does not exist: {zarr_path}[/red]")
        return
    
    if mode == 'quick':
       console.rule(f"[bold cyan]Quick Summary: {zarr_path.name}[/bold cyan]")
       show_quick_summary(zarr_path, console)
    
    # Header
    if mode != 'quick':
        console.rule("[bold cyan]Zarr Inspector[/bold cyan]")
        console.print(f"[dim]Path: {zarr_path.absolute()}[/dim]")
        if focus_stage:
            console.print(f"[dim]Focusing on: {focus_stage}[/dim]")
        console.print()
    
    # If focusing on specific stage, show only that stage's details
    if focus_stage:
        console.print(f"[bold cyan]Stage: {focus_stage}[/bold cyan]\n")
        
        stage_shown = False
        
        if focus_stage == 'background':
            stage_shown = display_background_metadata(zarr_path, console)
            
        elif focus_stage == 'detect':
            stage_shown = display_detection_metadata(zarr_path, console)

        elif focus_stage in ('refined_detect', 'refine_detect'):
            stage_shown = display_refined_detect_metadata(zarr_path, console)
            
        elif focus_stage == 'crop':
            stage_shown = display_crop_metadata(zarr_path, console)
            
        elif focus_stage == 'keypoints':
            stage_shown = display_keypoints_metadata(zarr_path, console)
            refined_shown = display_refined_keypoints_metadata(zarr_path, console)
            stage_shown = stage_shown or refined_shown
            
        elif focus_stage == 'track':
            stage_shown = display_tracking_metadata(zarr_path, console)
            
        elif focus_stage == 'assign_ids':
            stage_shown = display_id_assignment_metadata(zarr_path, console)
            
        elif focus_stage == 'import':
            # Show import/raw_video info
            stage_shown = display_import_metadata(zarr_path, console)
            
        else:
            console.print(f"[yellow]Unknown stage: {focus_stage}[/yellow]")
            console.print("[dim]Valid stages: import, background, detect, refined_detect, crop, keypoints, track, assign_ids[/dim]")
        
        if not stage_shown:
            console.print(f"[yellow]Stage '{focus_stage}' has not been run yet[/yellow]")
        
        console.print("\n[dim]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/dim]")
        console.print("[dim]Press any key to return to TUI...[/dim]")
        return
    
    # Check root metadata
    root_json = zarr_path / "zarr.json"
    if root_json.exists():
        with open(root_json) as f:
            root_meta = json.load(f)
            if root_meta.get('node_type') == 'group':
                console.print("[green]Valid Zarr v3 group[/green]")
    
    # Show quick summary even in normal mode
    show_quick_summary(zarr_path, console)
    console.print()
    
    # Show parameter provenance
    show_parameter_provenance(zarr_path, console)
    console.print()
    
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
                elif item.name in ['background_runs', 'detect_runs', REFINED_DETECT_GROUP]: # HIGHLIGHT RUNS
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

    refined_detect_path = zarr_path / REFINED_DETECT_GROUP
    legacy_refined_detect_path = zarr_path / LEGACY_REFINED_DETECT_GROUP
    if refined_detect_path.exists() or legacy_refined_detect_path.exists():
        console.print("[green] Refined detection runs found[/green]")
        display_refined_detect_metadata(zarr_path, console)
        console.print()

    keypoints_path = zarr_path / "keypoints_runs"
    if keypoints_path.exists():
        console.print("[green] Keypoint runs found[/green]")
        display_keypoints_metadata(zarr_path, console)
        console.print()

    refined_keypoints_path = zarr_path / REFINED_KEYPOINT_GROUP
    legacy_refined_keypoints_path = zarr_path / LEGACY_REFINED_KEYPOINT_GROUP
    if refined_keypoints_path.exists() or legacy_refined_keypoints_path.exists():
        console.print("[green] Refined keypoint runs found[/green]")
        display_refined_keypoints_metadata(zarr_path, console)
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

    # Add a helpful footer for TUI users
    console.print("\n[dim]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/dim]")
    console.print("[dim]Press any key to return to TUI...[/dim]")

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Inspect Zarr video archives",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Display Modes:
  quick   - Essential info only (stages, coverage, parameters)
  normal  - Standard inspection (default)
  full    - Everything including system metadata and environment

Examples:
  # Quick summary
  python zarr_inspector.py data.zarr --mode quick
  
  # Normal inspection
  python zarr_inspector.py data.zarr
  
  # Full details with environment
  python zarr_inspector.py data.zarr --mode full --full-env
  
  # Focus on specific stage
  python zarr_inspector.py data.zarr --stage detect
        """
    )
    
    parser.add_argument("zarr_path", help="Path to Zarr archive")
    parser.add_argument("--mode", choices=['quick', 'normal', 'full'],
                       default='normal', help="Display verbosity level")
    parser.add_argument("--full-env", action="store_true", 
                       help="Display complete environment information JSON")
    parser.add_argument(
        "--stage",
        help="Focus on specific stage (import, background, detect, refined_detect, crop, keypoints, track, assign_ids).",
    )
    
    args = parser.parse_args()
    
    inspect_video_zarr(
        args.zarr_path, 
        mode=args.mode,
        show_full_env=args.full_env,
        focus_stage=args.stage
    )

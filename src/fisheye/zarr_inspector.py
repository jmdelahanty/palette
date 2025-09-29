#!/usr/bin/env python3
"""
Inspector for kvikIO/GDS-imported Zarr video arrays.
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
    """Count actual chunk files on disk."""
    chunk_dir = array_path / "c"
    if not chunk_dir.exists():
        return 0, 0
    
    chunk_count = 0
    total_size = 0
    
    # Iterate through chunk directories
    for chunk_idx in chunk_dir.iterdir():
        if chunk_idx.is_dir():
            # Check for the actual chunk file at c/idx/0/0
            chunk_file = chunk_idx / "0" / "0"
            if chunk_file.exists():
                chunk_count += 1
                total_size += chunk_file.stat().st_size
    
    return chunk_count, total_size

def inspect_video_zarr(zarr_path):
    """Inspect a video Zarr archive created by kvikIO import."""
    console = Console()
    
    zarr_path = Path(zarr_path)
    if not zarr_path.exists():
        console.print(f"[red]Error: Path does not exist: {zarr_path}[/red]")
        return
    
    # Header
    console.rule("[bold cyan]Video Zarr Inspector[/bold cyan]")
    console.print(f"[dim]Path: {zarr_path.absolute()}[/dim]\n")
    
    # Check root metadata
    root_json = zarr_path / "zarr.json"
    if root_json.exists():
        with open(root_json) as f:
            root_meta = json.load(f)
            if root_meta.get('node_type') == 'group':
                console.print("[green]✓[/green] Valid Zarr v3 group")
    
    # Check raw_video group
    raw_video_path = zarr_path / "raw_video"
    raw_video_json = raw_video_path / "zarr.json"
    
    if not raw_video_json.exists():
        console.print("[yellow]⚠[/yellow] raw_video group metadata not found")
        return
    
    with open(raw_video_json) as f:
        raw_meta = json.load(f)
    
    # Display import metadata
    attrs = raw_meta.get('attributes', {})
    
    import_table = Table(title="Import Configuration", box=box.ROUNDED)
    import_table.add_column("Parameter", style="cyan")
    import_table.add_column("Value", style="yellow")
    
    # Basic info
    import_table.add_row("Source video", attrs.get('source_video', 'Unknown'))
    import_table.add_row("Import method", attrs.get('import_method', 'Unknown'))
    import_table.add_row("Total frames", str(attrs.get('total_frames', 0)))
    import_table.add_row("Resolution", f"{attrs.get('video_width', 0)}×{attrs.get('video_height', 0)}")
    import_table.add_row("FPS", f"{attrs.get('fps', 0):.1f}")
    import_table.add_row("Duration", f"{attrs.get('video_duration_seconds', 0):.1f} seconds")
    
    # Import settings
    import_table.add_row("", "")  # Separator
    import_table.add_row("Chunk size", f"{attrs.get('chunk_size', 0)} frames")
    import_table.add_row("IO batch size", f"{attrs.get('io_batch_size', 0)} frames")
    import_table.add_row("Device", attrs.get('device', 'Unknown'))
    import_table.add_row("Compression", attrs.get('compression', 'none'))
    
    if attrs.get('gpu_fp16'):
        import_table.add_row("GPU FP16", "Yes")
    
    if attrs.get('sharding_enabled'):
        import_table.add_row("Sharding", f"Yes ({attrs.get('chunks_per_shard', 0)} chunks/shard)")
    
    # Performance
    import_table.add_row("", "")  # Separator
    duration = attrs.get('import_duration_seconds', 0)
    import_table.add_row("Import time", f"{duration:.1f} seconds ({duration/60:.1f} minutes)")
    import_table.add_row("Throughput", f"{attrs.get('throughput_gbps', 0):.2f} GB/s")
    import_table.add_row("Processing speed", f"{attrs.get('frames_per_second', 0):.1f} fps")
    import_table.add_row("Data size", f"{attrs.get('data_size_gb', 0):.1f} GB")
    
    # Timestamps
    if 'import_timestamp' in attrs:
        import_table.add_row("Imported at", format_timestamp(attrs['import_timestamp']))
    
    console.print(import_table)
    
    # Check array metadata and actual chunks
    array_path = raw_video_path / "images_full"
    array_json = array_path / "zarr.json"
    
    if array_json.exists():
        with open(array_json) as f:
            array_meta = json.load(f)
        
        array_table = Table(title="Array Structure", box=box.ROUNDED)
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
    
    # Check actual chunks on disk
    console.print("\n[bold]Storage Analysis[/bold]")
    
    chunk_count, total_chunk_size = count_chunks(array_path)
    
    storage_table = Table(box=box.ROUNDED)
    storage_table.add_column("Metric", style="cyan")
    storage_table.add_column("Value", style="yellow")
    
    storage_table.add_row("Chunk files found", str(chunk_count))
    storage_table.add_row("Total store size", format_bytes(total_chunk_size))
    
    if chunk_count > 0 and 'shape' in array_meta:
        expected_chunks = -(-shape[0] // chunk_shape[0])
        completion = (chunk_count / expected_chunks) * 100
        storage_table.add_row("Completion", f"{completion:.1f}%")
        
        avg_chunk_size = total_chunk_size / chunk_count
        storage_table.add_row("Average chunk size", format_bytes(avg_chunk_size))
    
    # Check for other files
    total_zarr_size = sum(f.stat().st_size for f in zarr_path.rglob('*') if f.is_file())
    storage_table.add_row("Total Zarr size", format_bytes(total_zarr_size))
    
    console.print(storage_table)
    
    # Directory tree
    console.print("\n[bold]Directory Structure[/bold]")
    tree = Tree(f"[cyan]{zarr_path.name}[/cyan]")
    
    def add_tree_node(parent_node, path, max_depth=3, current_depth=0):
        if current_depth >= max_depth:
            parent_node.add("[dim]...[/dim]")
            return
        
        items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        
        for item in items[:20]:  # Limit items shown
            if item.is_dir():
                if item.name == 'c' and current_depth == 1:
                    # Special handling for chunk directory
                    chunk_count, _ = count_chunks(path)
                    node = parent_node.add(f"[blue]{item.name}/[/blue] [dim]({chunk_count} chunks)[/dim]")
                    # Don't expand chunk dirs
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
    
    # Validation summary
    console.print("\n[bold]Validation Summary[/bold]")
    
    issues = []
    warnings = []
    
    if not array_json.exists():
        issues.append("Array metadata (zarr.json) missing")
    
    if chunk_count == 0:
        issues.append("No chunk files found")
    elif 'shape' in locals():
        expected_chunks = -(-shape[0] // chunk_shape[0])
        if chunk_count < expected_chunks:
            warnings.append(f"Only {chunk_count}/{expected_chunks} chunks present")
    
    if attrs.get('import_method') == 'kvikio_zarr' and attrs.get('compression') != 'none':
        warnings.append("kvikIO arrays should not use compression")
    
    if issues:
        console.print("[red]Issues found:[/red]")
        for issue in issues:
            console.print(f"  • {issue}")
    
    if warnings:
        console.print("[yellow]Warnings:[/yellow]")
        for warning in warnings:
            console.print(f"  • {warning}")
    
    if not issues and not warnings:
        console.print("[green]✓ Array structure looks good![/green]")
    
    # Tips
    console.print("\n[dim]Tips:[/dim]")
    console.print("[dim]• Use kvikIO GDSStore to read this array with GPU[/dim]")
    console.print("[dim]• Array can be read with standard Zarr if metadata exists[/dim]")
    console.print("[dim]• Each chunk file contains 64 frames of 4512×4512 pixels[/dim]")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python zarr_inspector.py /path/to/zarr")
        sys.exit(1)
    
    inspect_video_zarr(sys.argv[1])
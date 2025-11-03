#!/usr/bin/env python3
"""
Verify enum storage format in a zarr file.

Usage:
    python verify_enum_format.py /path/to/file.zarr
"""

import sys
from pathlib import Path
import zarr
from rich.console import Console
from rich.tree import Tree

def inspect_enum_structure(zarr_path: Path):
    """Show the structure of enum storage in a zarr file."""
    console = Console()

    console.print(f"\n[bold]Inspecting:[/bold] {zarr_path}")
    console.print("=" * 80)

    try:
        root = zarr.open(str(zarr_path), mode='r')
    except Exception as e:
        console.print(f"[red]Error opening zarr file:[/red] {e}")
        return

    # Check for enums
    analysis = root.get('analysis')
    if analysis is None:
        console.print("[yellow]No 'analysis' group found[/yellow]")
        return

    enums_group = analysis.get('enums')
    if enums_group is None:
        console.print("[yellow]No 'enums' group found in analysis[/yellow]")
        return

    console.print(f"\n[green]✓ Found enums at:[/green] analysis/enums")

    # Build tree visualization
    tree = Tree("📁 analysis/enums/")

    for enum_name in enums_group.keys():
        node = enums_group[enum_name]

        if isinstance(node, zarr.Group):
            # Columnar format
            enum_tree = tree.add(f"📁 [cyan]{enum_name}/[/cyan] (columnar format ✓)")

            if 'id' in node:
                id_array = node['id']
                enum_tree.add(f"📊 id: {id_array.dtype} {id_array.shape} ({id_array.nbytes} bytes)")

            if 'name' in node:
                name_array = node['name']
                enum_tree.add(f"📊 name: {name_array.dtype} {name_array.shape} ({name_array.nbytes} bytes)")

            # Show attributes
            if hasattr(node, 'attrs') and node.attrs:
                attrs_str = ", ".join(f"{k}={v}" for k, v in node.attrs.items())
                enum_tree.add(f"[dim]attrs: {attrs_str}[/dim]")

            # Sample data
            if 'id' in node and 'name' in node:
                ids = node['id'][:5]
                names = node['name'][:5]
                sample_tree = enum_tree.add("[dim]Sample (first 5):[/dim]")
                for id_val, name_val in zip(ids, names):
                    sample_tree.add(f"{id_val} → {name_val}")

        elif isinstance(node, zarr.Array):
            # Structured array format (legacy)
            array_tree = tree.add(f"📊 [yellow]{enum_name}[/yellow] (structured array - legacy)")
            array_tree.add(f"dtype: {node.dtype}")
            array_tree.add(f"shape: {node.shape}")
            array_tree.add(f"size: {node.nbytes} bytes")

            # Show field names if structured
            if node.dtype.names:
                array_tree.add(f"fields: {', '.join(node.dtype.names)}")

            # Sample data
            if node.size > 0:
                sample_data = node[:5]
                sample_tree = array_tree.add("[dim]Sample (first 5):[/dim]")
                for record in sample_data:
                    if node.dtype.names and 'id' in node.dtype.names and 'name' in node.dtype.names:
                        id_val = record['id']
                        name_val = record['name']
                        if isinstance(name_val, bytes):
                            name_val = name_val.decode('utf-8', errors='ignore').rstrip('\x00')
                        sample_tree.add(f"{id_val} → {name_val}")
                    else:
                        sample_tree.add(str(record))

    console.print(tree)

    # Summary
    console.print("\n" + "=" * 80)
    console.print("[bold]Summary:[/bold]")

    columnar_count = sum(1 for name in enums_group.keys() if isinstance(enums_group[name], zarr.Group))
    structured_count = sum(1 for name in enums_group.keys() if isinstance(enums_group[name], zarr.Array))

    console.print(f"  Columnar format (new): {columnar_count} tables")
    console.print(f"  Structured format (legacy): {structured_count} tables")

    if columnar_count > 0 and structured_count == 0:
        console.print("\n[green]✓ All enums in columnar format![/green]")
    elif columnar_count > 0 and structured_count > 0:
        console.print("\n[yellow]⚠ Mixed format - migration in progress[/yellow]")
    elif structured_count > 0:
        console.print("\n[yellow]⚠ All enums in legacy structured format[/yellow]")
        console.print("   Run import again to convert to columnar format")

    console.print()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python verify_enum_format.py /path/to/file.zarr")
        sys.exit(1)

    zarr_path = Path(sys.argv[1])
    if not zarr_path.exists():
        print(f"Error: {zarr_path} does not exist")
        sys.exit(1)

    inspect_enum_structure(zarr_path)

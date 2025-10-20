"""Zarr schema utilities."""

import zarr
from typing import Optional, Tuple
from rich.console import Console


def get_run_group(
    root: zarr.Group,
    stage_name: str,
    *,
    console: Optional[Console] = None,
    create_new: bool = False,
) -> Tuple[zarr.Group, str]:
    """Get or create a run group."""
    # Placeholder implementation
    import time
    run_name = f"{stage_name}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    parent_name = f"{stage_name}_runs"
    parent = root.require_group(parent_name)
    run_group = parent.create_group(run_name)
    parent.attrs["latest"] = run_name
    if console:
        console.print(f"Created run group: [cyan]{parent_name}/{run_name}[/cyan]")
    return run_group, run_name

#!/usr/bin/env python3
"""
Patch existing Zarr stores to add pose_schema metadata to keypoint runs.

This script adds the pose_schema attribute to existing keypoints_runs and
refined_keypoints_runs that don't have it yet. It's useful for updating
older zarr archives created before pose schema propagation was implemented.

Prerequisites:
    - Run in the palette Python environment with zarr and rich installed
    - Or: conda activate palette (or whatever your environment is named)

Usage:
    python patch_pose_schema.py /path/to/data.zarr
    python patch_pose_schema.py /path/to/data.zarr --run keypoints_2024-01-15_10-30-00
    python patch_pose_schema.py /path/to/data.zarr --schema traditional_feret_v1
    python patch_pose_schema.py /path/to/data.zarr --dry-run  # Preview changes
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import zarr
from rich.console import Console
from rich.table import Table


def load_pose_schema(schema_name: str) -> Dict:
    """Load pose schema from package."""
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from fisheye.pose.schema import schema_from_package

    schema = schema_from_package(schema_name)
    return {
        "name": schema.name,
        "nodes": schema.node_names,
        "edges": schema.edges,
        "metadata": schema.metadata,
        "source": f"configs/fisheye/pose_schemas/{schema_name}.json"
    }


def detect_schema_from_keypoint_labels(labels: List[str]) -> Optional[str]:
    """Try to detect which schema based on keypoint labels."""
    if not labels:
        return None

    # Traditional 3-point schema
    if len(labels) == 3 and set(labels) == {"bladder", "eye_left", "eye_right"}:
        return "traditional_v1"

    # Traditional Feret 11-point schema
    if len(labels) == 11 and "bladder" in labels and "feret" in str(labels).lower():
        return "traditional_feret_v1"

    return None


def get_keypoint_runs(root: zarr.Group) -> Dict[str, List[str]]:
    """Get all keypoint run paths in the zarr store."""
    runs = {
        "keypoints_runs": [],
        "refined_keypoints_runs": []
    }

    if "keypoints_runs" in root:
        kp_group = root["keypoints_runs"]
        for name in kp_group.keys():
            if not name.startswith("."):  # Skip hidden groups
                runs["keypoints_runs"].append(f"keypoints_runs/{name}")

    if "refined_keypoints_runs" in root:
        ref_group = root["refined_keypoints_runs"]
        for name in ref_group.keys():
            if not name.startswith("."):
                runs["refined_keypoints_runs"].append(f"refined_keypoints_runs/{name}")

    return runs


def check_run_schema_status(root: zarr.Group, run_path: str) -> Dict:
    """Check if a run has pose_schema and what it contains."""
    kp_group = root[run_path]

    status = {
        "path": run_path,
        "has_schema": False,
        "schema_name": None,
        "keypoint_labels": None,
        "detected_schema": None,
        "num_keypoints": None
    }

    # Check for existing schema
    if "pose_schema" in kp_group.attrs:
        status["has_schema"] = True
        schema = kp_group.attrs["pose_schema"]
        status["schema_name"] = schema.get("name")

    # Check keypoint labels
    if "keypoint_labels" in kp_group.attrs:
        status["keypoint_labels"] = kp_group.attrs["keypoint_labels"]
        status["detected_schema"] = detect_schema_from_keypoint_labels(status["keypoint_labels"])

    # Get number of keypoints from data
    if "keypoints_roi" in kp_group:
        kp_array = kp_group["keypoints_roi"]
        if len(kp_array.shape) >= 2:
            status["num_keypoints"] = kp_array.shape[1]

    return status


def patch_run(root: zarr.Group, run_path: str, schema_dict: Dict, dry_run: bool = False) -> bool:
    """Add pose_schema to a keypoint run."""
    kp_group = root[run_path]

    if "pose_schema" in kp_group.attrs:
        return False  # Already has schema

    if not dry_run:
        kp_group.attrs["pose_schema"] = schema_dict

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Patch existing Zarr stores to add pose_schema metadata.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Patch latest runs with default schema
  python patch_pose_schema.py /path/to/data.zarr

  # Patch specific run
  python patch_pose_schema.py /path/to/data.zarr --run keypoints_2024-01-15_10-30-00

  # Use different schema
  python patch_pose_schema.py /path/to/data.zarr --schema traditional_feret_v1

  # Preview changes without modifying
  python patch_pose_schema.py /path/to/data.zarr --dry-run

  # Patch all runs (not just latest)
  python patch_pose_schema.py /path/to/data.zarr --all-runs
        """
    )
    parser.add_argument(
        "zarr_path",
        type=Path,
        help="Path to Zarr archive to patch"
    )
    parser.add_argument(
        "--run",
        help="Specific run name to patch (e.g., 'keypoints_2024-01-15_10-30-00'). "
             "If not specified, patches latest keypoints and refined runs."
    )
    parser.add_argument(
        "--schema",
        default="traditional_v1",
        help="Pose schema to use (default: traditional_v1)"
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Patch all keypoint runs, not just latest"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying the zarr store"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing pose_schema if present"
    )

    args = parser.parse_args()
    console = Console()

    # Validate zarr path
    if not args.zarr_path.exists():
        console.print(f"[red]✗ Zarr path not found:[/red] {args.zarr_path}")
        sys.exit(1)

    if not args.zarr_path.is_dir():
        console.print(f"[red]✗ Path is not a directory:[/red] {args.zarr_path}")
        sys.exit(1)

    # Open zarr store
    try:
        root = zarr.open(str(args.zarr_path), mode="r" if args.dry_run else "a")
    except Exception as e:
        console.print(f"[red]✗ Failed to open zarr store:[/red] {e}")
        sys.exit(1)

    # Load pose schema
    try:
        schema_dict = load_pose_schema(args.schema)
        console.print(f"[cyan]Loaded schema:[/cyan] {args.schema}")
        console.print(f"  Nodes: {schema_dict['nodes']}")
        console.print(f"  Edges: {schema_dict['edges']}\n")
    except Exception as e:
        console.print(f"[red]✗ Failed to load schema '{args.schema}':[/red] {e}")
        sys.exit(1)

    # Get runs to patch
    if args.run:
        # Specific run
        runs_to_check = []
        for prefix in ["keypoints_runs", "refined_keypoints_runs"]:
            full_path = f"{prefix}/{args.run}"
            if full_path in root:
                runs_to_check.append(full_path)
        if not runs_to_check:
            console.print(f"[red]✗ Run not found:[/red] {args.run}")
            sys.exit(1)
    elif args.all_runs:
        # All runs
        all_runs = get_keypoint_runs(root)
        runs_to_check = all_runs["keypoints_runs"] + all_runs["refined_keypoints_runs"]
        if not runs_to_check:
            console.print("[yellow]No keypoint runs found in zarr store[/yellow]")
            sys.exit(0)
    else:
        # Latest runs only
        runs_to_check = []
        if "keypoints_runs" in root:
            latest = root["keypoints_runs"].attrs.get("latest")
            if latest:
                runs_to_check.append(f"keypoints_runs/{latest}")
        if "refined_keypoints_runs" in root:
            latest = root["refined_keypoints_runs"].attrs.get("latest")
            if latest:
                runs_to_check.append(f"refined_keypoints_runs/{latest}")

        if not runs_to_check:
            console.print("[yellow]No latest keypoint runs found[/yellow]")
            sys.exit(0)

    # Check status of all runs
    console.print(f"[bold]Checking {len(runs_to_check)} run(s)...[/bold]\n")

    table = Table(title="Keypoint Run Status")
    table.add_column("Run Path", style="cyan")
    table.add_column("Has Schema?", style="magenta")
    table.add_column("Schema Name", style="yellow")
    table.add_column("Keypoints", justify="right")
    table.add_column("Detected Schema", style="green")
    table.add_column("Action", style="bold")

    runs_to_patch = []
    for run_path in runs_to_check:
        status = check_run_schema_status(root, run_path)

        has_schema_str = "✓" if status["has_schema"] else "✗"
        schema_name_str = status["schema_name"] or "—"
        num_kp_str = str(status["num_keypoints"]) if status["num_keypoints"] else "—"
        detected_str = status["detected_schema"] or "—"

        if status["has_schema"] and not args.force:
            action = "[dim]skip (has schema)[/dim]"
        elif status["detected_schema"] and status["detected_schema"] != args.schema:
            action = f"[yellow]⚠ schema mismatch[/yellow]"
            runs_to_patch.append(run_path)
        else:
            action = "[green]patch[/green]"
            runs_to_patch.append(run_path)

        table.add_row(
            run_path,
            has_schema_str,
            schema_name_str,
            num_kp_str,
            detected_str,
            action
        )

    console.print(table)
    console.print()

    if not runs_to_patch:
        console.print("[green]✓ All runs already have pose_schema[/green]")
        return

    # Confirm if not dry-run
    if not args.dry_run:
        console.print(f"[yellow]About to patch {len(runs_to_patch)} run(s) with schema '{args.schema}'[/yellow]")
        response = console.input("Continue? [y/N] ")
        if response.lower() != "y":
            console.print("[dim]Cancelled[/dim]")
            return

    # Apply patches
    console.print()
    patched_count = 0
    for run_path in runs_to_patch:
        try:
            was_patched = patch_run(root, run_path, schema_dict, dry_run=args.dry_run)
            if was_patched:
                if args.dry_run:
                    console.print(f"[dim]Would patch:[/dim] {run_path}")
                else:
                    console.print(f"[green]✓ Patched:[/green] {run_path}")
                patched_count += 1
            else:
                console.print(f"[dim]Skipped (already has schema):[/dim] {run_path}")
        except Exception as e:
            console.print(f"[red]✗ Failed to patch {run_path}:[/red] {e}")

    console.print()
    if args.dry_run:
        console.print(f"[cyan]Dry run: Would patch {patched_count} run(s)[/cyan]")
        console.print("[dim]Run without --dry-run to apply changes[/dim]")
    else:
        console.print(f"[green]✓ Successfully patched {patched_count} run(s)[/green]")


if __name__ == "__main__":
    main()

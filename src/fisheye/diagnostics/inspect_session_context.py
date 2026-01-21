"""
Diagnostic for session context metadata.

Prints analysis_metadata.session_context from a Zarr archive.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import zarr
from rich.console import Console
from rich.panel import Panel


def _load_session_context(root: zarr.Group) -> Optional[Dict[str, Any]]:
    analysis = root.get("analysis_metadata")
    if analysis is None:
        return None
    raw = analysis.attrs.get("session_context")
    if raw is None:
        return None
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "ignore")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Zarr session context metadata.")
    parser.add_argument("zarr_path", type=str, help="Path to Palette Zarr archive.")
    args = parser.parse_args()

    console = Console()
    zarr_path = Path(args.zarr_path)
    console.print(f"[bold]Inspecting[/bold] {zarr_path}")

    if not zarr_path.exists():
        console.print(f"[red]Zarr path not found:[/red] {zarr_path}")
        return

    root = zarr.open(str(zarr_path), mode="r")
    context = _load_session_context(root)
    if not context:
        console.print("[yellow]No session_context found in analysis_metadata.[/yellow]")
        console.print("[dim]Run import_stimulus_to_zarr to mirror session context from H5.[/dim]")
        return

    payload = json.dumps(context, indent=2, sort_keys=True)
    console.print(Panel(payload, title="Session Context", expand=False))


if __name__ == "__main__":
    main()

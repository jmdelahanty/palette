# src/fisheye/tune/dispatcher.py
"""
Tuner dispatcher - routes tuning requests to appropriate tuner modules.
"""

from pathlib import Path
from typing import Optional
from rich.console import Console


def run_tuner(
    tuner_name: str,
    zarr_path: str,
    config_path: Optional[str] = None,
    frame_idx: Optional[int] = None,
    use_full_res: bool = False,
    console: Optional[Console] = None
) -> int:
    """
    Run a specific tuner.
    
    Args:
        tuner_name: Name of the tuner to run ('mask', 'detect', 'threshold', 'keypoint', etc.)
        zarr_path: Path to zarr file
        config_path: Path to config YAML
        frame_idx: Specific frame index to use
        use_full_res: Use full resolution instead of downsampled
        console: Rich console for output
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    if console is None:
        console = Console()
    
    # Validate zarr path
    if not Path(zarr_path).exists():
        console.print(f"[red]Error: Zarr file not found: {zarr_path}[/red]")
        return 1
    
    # Route to appropriate tuner
    try:
        if tuner_name == 'mask':
            from .mask_tuner import main as mask_main
            console.print("[bold cyan]🎭 Starting Mask Tuner[/bold cyan]")
            console.print(f"Zarr: {zarr_path}")
            if config_path:
                console.print(f"Config: {config_path}")
            print()  # blank line for tuner output
            
            mask_main(
                zarr_path=zarr_path,
                use_full_res=use_full_res,
                frame_idx=frame_idx,
                config_path=config_path
            )
            return 0
    
        elif tuner_name == 'subdish':
            from .subdish_mask_tuner import main as subdish_main
            console.print("[bold cyan]🎭 Starting Sub-Dish Mask Tuner[/bold cyan]")
            console.print(f"Zarr: {zarr_path}")
            print()  # blank line for tuner output
            
            subdish_main(
                zarr_path=zarr_path,
                use_full_res=use_full_res
            )
            return 0
            
        elif tuner_name == 'detect' or tuner_name == 'threshold':
            from .detect_threshold_tuner import main as detect_main
            console.print("[bold cyan]🔍 Starting Detection Threshold Tuner[/bold cyan]")
            console.print(f"Zarr: {zarr_path}")
            if config_path:
                console.print(f"Config: {config_path}")
            print()  # blank line for tuner output
            
            detect_main(
                zarr_path=zarr_path,
                config_path=config_path,
                start_frame=frame_idx if frame_idx else 1
            )
            return 0
            
        elif tuner_name == 'keypoint' or tuner_name == 'keypoints':
            from .keypoint_tuner import main as keypoint_main
            console.print("[bold cyan] Starting Keypoint Detection Tuner[/bold cyan]")
            console.print(f"Zarr: {zarr_path}")
            if config_path:
                console.print(f"Config: {config_path}")
            print()  # blank line for tuner output
            
            keypoint_main(
                zarr_path=zarr_path,
                start_frame=frame_idx if frame_idx else 1
            )
            return 0

        elif tuner_name in {'keypoint-review', 'keypoint_review', 'review'}:
            console.print(
                "[yellow]Keypoint review uses a separate entrypoint.[/yellow]"
            )
            console.print(
                "Run: `python -m fisheye.tune.keypoint_review <zarr> --retune|--manual|--audit`"
            )
            return 0
        elif tuner_name in {'eye-mask-review', 'eye_mask_review', 'eye-mask-reviewer'}:
            console.print(
                "[yellow]Eye mask review uses a separate entrypoint.[/yellow]"
            )
            console.print(
                "Run: `python -m fisheye.tune.eye_mask_review <zarr> --retune|--manual|--audit`"
            )
            return 0
            
        else:
            console.print(f"[red]Error: Unknown tuner '{tuner_name}'[/red]")
            console.print("\n[yellow]Available tuners:[/yellow]")
            console.print("  • mask      - Tune dish mask detection (Hough circles)")
            console.print("  • subdish   - Define sub-dish masks for spatial ID assignment")
            console.print("  • detect    - Tune fish detection thresholds")
            console.print("  • threshold - Alias for 'detect'")
            console.print("  • keypoint  - Tune anatomical keypoint detection (swim bladder & eyes)")
            console.print("  • keypoints - Alias for 'keypoint'")
            console.print("  • eye-mask-review - Retune/manual review for refined eye masks")
            return 1
            
    except ImportError as e:
        console.print(f"[red]Error importing tuner module: {e}[/red]")
        return 1
    except KeyboardInterrupt:
        console.print("\n[yellow]Tuner interrupted by user[/yellow]")
        return 130  # Standard Unix exit code for Ctrl+C
    except Exception as e:
        console.print(f"[red]Error running tuner: {e}[/red]")
        return 1


def list_tuners(console: Optional[Console] = None):
    """Print available tuners and their descriptions."""
    if console is None:
        console = Console()
    
    console.print("\n[bold]Available Tuners:[/bold]\n")
    
    tuners = [
        ("mask", "Tune dish mask detection using Hough circle detection"),
        ("subdish", "Define sub-dish masks for spatial ID assignment"),
        ("detect", "Tune fish detection thresholds and morphological parameters"),
        ("threshold", "Alias for 'detect' tuner"),
        ("keypoints", "Tune anatomical keypoint detection (swim bladder and eyes)"),
        ("keypoint-review", "Review refined keypoints (retune/manual/audit)"),
        ("eye-mask-review", "Review refined eye masks (retune/manual/audit)"),
    ]
    
    for name, description in tuners:
        console.print(f"  [cyan]{name:12}[/cyan] {description}")
    
    console.print("\n[dim]Usage: python -m fisheye --tune <tuner_name> --zarr-path <path>[/dim]")
    console.print("\n[bold]Examples:[/bold]")
    console.print("  python -m fisheye data.zarr --tune mask")
    console.print("  python -m fisheye data.zarr --tune detect --frame 100")
    console.print("  python -m fisheye data.zarr --tune keypoints --frame 50")
    console.print("  python -m fisheye data.zarr --tune subdish")
    console.print()

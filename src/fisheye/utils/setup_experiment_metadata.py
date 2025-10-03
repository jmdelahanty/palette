#!/usr/bin/env python3
"""
Add or update experiment setup metadata in a Zarr file.

This tool allows you to specify the number of dishes and expected fish
per dish for an experiment. This metadata is used by the assign_ids stage
to automatically determine whether to use single-dish or multi-dish tracking.

Usage:
    # Single dish with 1 fish
    python setup_experiment_metadata.py data.zarr --num-dishes 1 --fish-per-dish 1
    
    # Multi-dish with 12 fish
    python setup_experiment_metadata.py data.zarr --num-dishes 12 --fish-per-dish 1
    
    # View current setup
    python setup_experiment_metadata.py data.zarr --show
    
    # Interactive mode
    python setup_experiment_metadata.py data.zarr --interactive
"""

import argparse
import zarr
from datetime import datetime, timezone
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box


def show_current_setup(zarr_path: str, console: Console):
    """Display current experiment setup metadata."""
    try:
        root = zarr.open(zarr_path, mode='r')
        
        if 'experiment_setup' not in root.attrs:
            console.print("[yellow]No experiment setup metadata found.[/yellow]")
            console.print("Run with --num-dishes and --fish-per-dish to configure.")
            return False
        
        setup = root.attrs['experiment_setup']
        
        table = Table(title="Experiment Setup", box=box.ROUNDED)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="yellow")
        
        table.add_row("Number of dishes", str(setup.get('num_dishes', 'N/A')))
        table.add_row("Fish per dish", str(setup.get('fish_per_dish', 'N/A')))
        table.add_row("Total expected fish", str(setup.get('total_expected_fish', 'N/A')))
        table.add_row("Setup type", setup.get('setup_type', 'N/A'))
        table.add_row("Source", setup.get('source', 'N/A'))
        
        if 'configured_at' in setup:
            table.add_row("Configured at", setup['configured_at'])
        
        console.print(table)
        return True
        
    except Exception as e:
        console.print(f"[red]Error reading zarr file: {e}[/red]")
        return False


def configure_setup(zarr_path: str, num_dishes: int, fish_per_dish: int, 
                   console: Console, force: bool = False):
    """Add or update experiment setup metadata."""
    try:
        root = zarr.open(zarr_path, mode='r+')
        
        # Check if metadata already exists
        if 'experiment_setup' in root.attrs and not force:
            console.print("[yellow]Experiment setup already configured.[/yellow]")
            console.print("Current configuration:")
            show_current_setup(zarr_path, console)
            console.print("\n[yellow]Use --force to overwrite.[/yellow]")
            return False
        
        # Validate inputs
        if num_dishes < 1:
            console.print("[red]Error: num_dishes must be at least 1[/red]")
            return False
        
        if fish_per_dish < 1:
            console.print("[red]Error: fish_per_dish must be at least 1[/red]")
            return False
        
        # Determine setup type
        setup_type = 'single_dish' if num_dishes == 1 else 'multi_dish'
        total_expected = num_dishes * fish_per_dish
        
        # Create metadata
        setup_metadata = {
            'num_dishes': num_dishes,
            'fish_per_dish': fish_per_dish,
            'total_expected_fish': total_expected,
            'setup_type': setup_type,
            'source': 'manual_cli',
            'configured_at': datetime.now(timezone.utc).isoformat()
        }
        
        # Store in zarr
        root.attrs['experiment_setup'] = setup_metadata
        
        # Display result
        console.print(Panel.fit(
            f"[green]✓ Experiment setup configured successfully![/green]\n\n"
            f"Setup type: [cyan]{setup_type}[/cyan]\n"
            f"Dishes: [yellow]{num_dishes}[/yellow]\n"
            f"Fish per dish: [yellow]{fish_per_dish}[/yellow]\n"
            f"Total expected fish: [yellow]{total_expected}[/yellow]",
            title="Configuration Complete",
            border_style="green"
        ))
        
        # Show next steps
        if setup_type == 'single_dish':
            console.print("\n[cyan]Next steps:[/cyan]")
            console.print("  - The assign_ids stage will automatically use the dish mask as a single ROI")
            console.print("  - No need to run subdish mask tuner")
        else:
            console.print("\n[cyan]Next steps:[/cyan]")
            console.print("  1. Run the subdish mask tuner:")
            console.print(f"     python -m fisheye {zarr_path} --tune subdish")
            console.print("  2. Then run assign_ids stage:")
            console.print(f"     python tracker.py {zarr_path} --stage assign_ids")
        
        return True
        
    except Exception as e:
        console.print(f"[red]Error configuring zarr file: {e}[/red]")
        return False


def interactive_mode(zarr_path: str, console: Console):
    """Interactive configuration mode."""
    console.print(Panel.fit(
        "Interactive Experiment Setup Configuration",
        border_style="cyan"
    ))
    
    # Show current setup if exists
    console.print("\n[cyan]Current configuration:[/cyan]")
    has_setup = show_current_setup(zarr_path, console)
    
    if has_setup:
        console.print()
        overwrite = input("Configuration exists. Overwrite? (y/N): ").strip().lower()
        if overwrite != 'y':
            console.print("[yellow]Configuration cancelled.[/yellow]")
            return False
    
    # Prompt for inputs
    console.print("\n[cyan]Enter experiment setup:[/cyan]")
    
    try:
        num_dishes = int(input("Number of dishes in camera view: ").strip())
        fish_per_dish = int(input("Expected fish per dish: ").strip())
    except ValueError:
        console.print("[red]Error: Please enter valid numbers[/red]")
        return False
    
    # Confirm
    total = num_dishes * fish_per_dish
    setup_type = 'single_dish' if num_dishes == 1 else 'multi_dish'
    
    console.print(f"\n[cyan]Configuration summary:[/cyan]")
    console.print(f"  Setup type: {setup_type}")
    console.print(f"  Dishes: {num_dishes}")
    console.print(f"  Fish per dish: {fish_per_dish}")
    console.print(f"  Total expected fish: {total}")
    
    confirm = input("\nSave this configuration? (Y/n): ").strip().lower()
    if confirm == 'n':
        console.print("[yellow]Configuration cancelled.[/yellow]")
        return False
    
    return configure_setup(zarr_path, num_dishes, fish_per_dish, console, force=True)


def main():
    parser = argparse.ArgumentParser(
        description="Add or update experiment setup metadata in a Zarr file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Configure single dish experiment
  %(prog)s data.zarr --num-dishes 1 --fish-per-dish 1
  
  # Configure multi-dish experiment
  %(prog)s data.zarr --num-dishes 12 --fish-per-dish 1
  
  # View current configuration
  %(prog)s data.zarr --show
  
  # Interactive mode
  %(prog)s data.zarr --interactive
  
  # Overwrite existing configuration
  %(prog)s data.zarr --num-dishes 1 --fish-per-dish 1 --force
        """
    )
    
    parser.add_argument(
        "zarr_path",
        type=str,
        help="Path to the Zarr file"
    )
    
    parser.add_argument(
        "--num-dishes",
        type=int,
        help="Number of dishes in the camera view"
    )
    
    parser.add_argument(
        "--fish-per-dish",
        type=int,
        help="Expected number of fish per dish"
    )
    
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show current experiment setup configuration"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive configuration mode"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing configuration"
    )
    
    args = parser.parse_args()
    
    # Check that zarr path exists
    if not Path(args.zarr_path).exists():
        print(f"Error: Zarr file not found: {args.zarr_path}")
        return 1
    
    console = Console()
    
    # Show mode
    if args.show:
        show_current_setup(args.zarr_path, console)
        return 0
    
    # Interactive mode
    if args.interactive:
        success = interactive_mode(args.zarr_path, console)
        return 0 if success else 1
    
    # Command-line mode
    if args.num_dishes is not None and args.fish_per_dish is not None:
        success = configure_setup(
            args.zarr_path, 
            args.num_dishes, 
            args.fish_per_dish,
            console,
            args.force
        )
        return 0 if success else 1
    
    # No action specified
    console.print("[yellow]No action specified.[/yellow]")
    console.print("Use --show, --interactive, or provide --num-dishes and --fish-per-dish")
    parser.print_help()
    return 1


if __name__ == "__main__":
    exit(main())
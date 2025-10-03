#!/usr/bin/env python3
"""
Batch processing script for running the FishEye pipeline on multiple files.

This script can:
1. Process all zarr files in a directory
2. Process a list of specific files
3. Handle errors gracefully and continue processing
4. Generate a summary report

Usage:
    python batch_runner.py /path/to/zarr/files --stages detect track
    python batch_runner.py --file-list files.txt --stages all
    python batch_runner.py /path/to/dir --pattern "*.zarr" --recursive
"""

import argparse
import sys
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import time

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.panel import Panel
    from rich.live import Live
except ImportError:
    print("⚠️ Rich not installed. Install with: pip install rich")
    sys.exit(1)


console = Console()


class BatchResult:
    """Store results for a single batch item."""
    
    def __init__(self, zarr_path: str):
        self.zarr_path = zarr_path
        self.status = "pending"  # pending, running, success, failed, skipped
        self.start_time = None
        self.end_time = None
        self.error_message = None
        self.stages_completed = []
        
    @property
    def duration(self) -> Optional[float]:
        """Get processing duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON export."""
        return {
            'zarr_path': self.zarr_path,
            'status': self.status,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'error_message': self.error_message,
            'stages_completed': self.stages_completed
        }


class BatchRunner:
    """Run pipeline on multiple files sequentially."""
    
    def __init__(
        self,
        stages: List[str],
        config_path: Optional[str] = None,
        video_dir: Optional[str] = None,
        scheduler: str = "processes",
        continue_on_error: bool = True
    ):
        self.stages = stages
        self.config_path = config_path
        self.video_dir = video_dir
        self.scheduler = scheduler
        self.continue_on_error = continue_on_error
        self.results: List[BatchResult] = []
        
    def find_files(
        self,
        search_path: Path,
        pattern: str = "*.zarr",
        recursive: bool = False
    ) -> List[Path]:
        """Find zarr files matching pattern."""
        if recursive:
            files = list(search_path.rglob(pattern))
        else:
            files = list(search_path.glob(pattern))
        
        # Filter to only directories (zarr stores)
        zarr_files = [f for f in files if f.is_dir()]
        
        return sorted(zarr_files)
    
    def load_file_list(self, file_list_path: Path) -> List[Path]:
        """Load list of files from text file (one per line)."""
        with open(file_list_path) as f:
            paths = [line.strip() for line in f if line.strip()]
        return [Path(p) for p in paths]
    
    def build_command(self, zarr_path: Path) -> List[str]:
        """Build pipeline command for a single file."""
        cmd = [sys.executable, "-m", "fisheye", str(zarr_path)]
        
        # Add stages
        cmd.extend(["--stages"] + self.stages)
        
        # Add optional parameters
        if self.config_path:
            cmd.extend(["--config", self.config_path])
        
        if self.video_dir:
            # Try to find matching video file
            video_file = self._find_video_for_zarr(zarr_path)
            if video_file:
                cmd.extend(["--video-path", str(video_file)])
        
        cmd.extend(["--scheduler", self.scheduler])
        
        return cmd
    
    def _find_video_for_zarr(self, zarr_path: Path) -> Optional[Path]:
        """Try to find a matching video file for the zarr."""
        if not self.video_dir:
            return None
        
        video_dir = Path(self.video_dir)
        zarr_name = zarr_path.stem
        
        # Common video extensions
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            video_path = video_dir / f"{zarr_name}{ext}"
            if video_path.exists():
                return video_path
        
        return None
    
    def run_single(self, zarr_path: Path) -> BatchResult:
        """Run pipeline on a single file."""
        result = BatchResult(str(zarr_path))
        result.status = "running"
        result.start_time = time.time()
        
        cmd = self.build_command(zarr_path)
        
        try:
            # Run subprocess and capture output
            process = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            
            result.status = "success"
            result.stages_completed = self.stages.copy()
            
        except subprocess.CalledProcessError as e:
            result.status = "failed"
            result.error_message = f"Exit code {e.returncode}: {e.stderr[:200]}"
            
        except Exception as e:
            result.status = "failed"
            result.error_message = str(e)
        
        finally:
            result.end_time = time.time()
        
        return result
    
    def run_batch(self, zarr_files: List[Path]) -> List[BatchResult]:
        """Run pipeline on all files in batch."""
        console.print(f"\n[bold cyan]Starting batch processing of {len(zarr_files)} files[/bold cyan]\n")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            overall_task = progress.add_task(
                "[cyan]Overall progress",
                total=len(zarr_files)
            )
            
            for i, zarr_path in enumerate(zarr_files, 1):
                progress.update(
                    overall_task,
                    description=f"[cyan]Processing {i}/{len(zarr_files)}: {zarr_path.name}"
                )
                
                result = self.run_single(zarr_path)
                self.results.append(result)
                
                # Log result
                if result.status == "success":
                    console.print(f"  [green]✓[/green] {zarr_path.name} completed in {result.duration:.1f}s")
                elif result.status == "failed":
                    console.print(f"  [red]✗[/red] {zarr_path.name} failed: {result.error_message}")
                    if not self.continue_on_error:
                        console.print("[yellow]Stopping batch due to error[/yellow]")
                        break
                
                progress.update(overall_task, advance=1)
        
        return self.results
    
    def generate_summary(self) -> None:
        """Generate and display summary report."""
        total = len(self.results)
        success = sum(1 for r in self.results if r.status == "success")
        failed = sum(1 for r in self.results if r.status == "failed")
        
        # Create summary table
        table = Table(title="Batch Processing Summary", show_header=True)
        table.add_column("File", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Duration", justify="right")
        table.add_column("Error", style="red")
        
        for result in self.results:
            status_str = {
                "success": "[green]✓ Success[/green]",
                "failed": "[red]✗ Failed[/red]",
                "skipped": "[yellow]○ Skipped[/yellow]"
            }.get(result.status, result.status)
            
            duration_str = f"{result.duration:.1f}s" if result.duration else "-"
            error_str = result.error_message[:50] if result.error_message else ""
            
            table.add_row(
                Path(result.zarr_path).name,
                status_str,
                duration_str,
                error_str
            )
        
        console.print("\n")
        console.print(table)
        
        # Summary stats
        total_time = sum(r.duration for r in self.results if r.duration)
        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"  Total files: {total}")
        console.print(f"  [green]Successful: {success}[/green]")
        console.print(f"  [red]Failed: {failed}[/red]")
        console.print(f"  Total time: {total_time:.1f}s")
        
        if success > 0:
            avg_time = total_time / success
            console.print(f"  Average time per file: {avg_time:.1f}s")
    
    def save_report(self, output_path: Path) -> None:
        """Save detailed report to JSON."""
        report = {
            'batch_info': {
                'timestamp': datetime.now().isoformat(),
                'stages': self.stages,
                'config_path': self.config_path,
                'scheduler': self.scheduler
            },
            'results': [r.to_dict() for r in self.results],
            'summary': {
                'total': len(self.results),
                'success': sum(1 for r in self.results if r.status == "success"),
                'failed': sum(1 for r in self.results if r.status == "failed")
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        console.print(f"\n[green]Report saved to:[/green] {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch process multiple zarr files through the FishEye pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all zarr files in a directory
  python batch_runner.py /path/to/zarr/files --stages detect track
  
  # Process with recursion
  python batch_runner.py /path/to/data --stages all --recursive
  
  # Process from file list
  python batch_runner.py --file-list my_files.txt --stages background detect
  
  # Generate report
  python batch_runner.py /path/to/files --stages all --report results.json
        """
    )
    
    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "directory",
        type=str,
        nargs='?',
        help="Directory containing zarr files to process"
    )
    input_group.add_argument(
        "--file-list",
        type=str,
        help="Text file with list of zarr paths (one per line)"
    )
    
    # Processing options
    parser.add_argument(
        "--stages",
        nargs='+',
        required=True,
        help="Pipeline stages to run (e.g., detect track, or 'all')"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        help="Directory to search for matching video files"
    )
    parser.add_argument(
        "--scheduler",
        choices=['processes', 'threads', 'single-threaded', 'distributed'],
        default='processes',
        help="Dask scheduler to use"
    )
    
    # File discovery options
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.zarr",
        help="File pattern to match (default: *.zarr)"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively in subdirectories"
    )
    
    # Behavior options
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop batch if any file fails (default: continue)"
    )
    parser.add_argument(
        "--report",
        type=str,
        help="Path to save JSON report"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without running"
    )
    
    args = parser.parse_args()
    
    # Find files to process
    if args.file_list:
        zarr_files = BatchRunner([], None).load_file_list(Path(args.file_list))
    else:
        search_path = Path(args.directory).expanduser()
        if not search_path.exists():
            console.print(f"[red]Error:[/red] Directory not found: {search_path}")
            sys.exit(1)
        
        runner = BatchRunner([], None)
        zarr_files = runner.find_files(search_path, args.pattern, args.recursive)
    
    if not zarr_files:
        console.print("[yellow]No zarr files found![/yellow]")
        sys.exit(0)
    
    # Display files to be processed
    console.print(f"\n[bold]Found {len(zarr_files)} zarr files:[/bold]")
    for i, f in enumerate(zarr_files, 1):
        console.print(f"  {i}. {f}")
    
    if args.dry_run:
        console.print("\n[yellow]Dry run mode - no processing performed[/yellow]")
        sys.exit(0)
    
    # Confirm before proceeding
    console.print(f"\n[bold]Will run stages:[/bold] {', '.join(args.stages)}")
    if args.config:
        console.print(f"[bold]Config:[/bold] {args.config}")
    
    try:
        response = input("\nProceed? [Y/n]: ").strip().lower()
        if response and response != 'y':
            console.print("[yellow]Cancelled[/yellow]")
            sys.exit(0)
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled[/yellow]")
        sys.exit(0)
    
    # Run batch processing
    runner = BatchRunner(
        stages=args.stages,
        config_path=args.config,
        video_dir=args.video_dir,
        scheduler=args.scheduler,
        continue_on_error=not args.stop_on_error
    )
    
    try:
        runner.run_batch(zarr_files)
        runner.generate_summary()
        
        if args.report:
            runner.save_report(Path(args.report))
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Batch processing interrupted![/yellow]")
        runner.generate_summary()
        sys.exit(130)


if __name__ == "__main__":
    main()
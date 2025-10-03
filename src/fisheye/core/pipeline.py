"""
Core pipeline orchestrator for FishEye tracking system.
Manages the execution of video processing stages in sequence.
"""

import sys
import yaml
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timezone
from dataclasses import dataclass, field
import zarr
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

# Import stage modules
from ..capture.import_video import import_video, get_import_stats
from ..preprocessing.background import compute_background
from ..detection.detect_traditional import detect_fish
from ..detection.detect_keypoints_traditional import detect_keypoints
from ..tracking.crop import crop_detections
from ..shared.zarr.schema import validate_zarr_structure


@dataclass
class PipelineConfig:
    """Configuration for the pipeline."""
    zarr_path: str
    video_path: Optional[str] = None
    config_path: str = "configs/fisheye/default.yaml"
    stages: List[str] = field(default_factory=lambda: ["all"])
    scheduler: str = "processes"
    num_workers: Optional[int] = None
    use_gpu: bool = True
    force_cpu: bool = False
    verbose: bool = False
    dry_run: bool = False
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "PipelineConfig":
        """Create config from command line arguments."""
        return cls(
            zarr_path=args.zarr_path,
            video_path=getattr(args, 'video_path', None),
            config_path=getattr(args, 'config', 'configs/fisheye/default.yaml'),
            stages=getattr(args, 'stages', ['all']),
            scheduler=getattr(args, 'scheduler', 'processes'),
            num_workers=getattr(args, 'num_workers', None),
            use_gpu=not getattr(args, 'no_gpu', False),
            force_cpu=getattr(args, 'force_cpu', False),
            verbose=getattr(args, 'verbose', False),
            dry_run=getattr(args, 'dry_run', False)
        )


class Pipeline:
    """
    Main pipeline orchestrator for FishEye tracking.
    
    Manages the execution of processing stages and data flow between them.
    """
    
    # Define stage order and dependencies
    STAGE_ORDER = [
        'import',
        'downsample',
        'background', 
        'detect',
        'crop',
        'keypoints',
        'track',
        'refine',
        'assign_ids'
    ]
    
    STAGE_DEPENDENCIES = {
        'import': [],
        'downsample': ['import'],
        'background': ['import'],
        'detect': ['background'],
        'crop': ['detect'],
        'keypoints': ['crop', 'background'],
        'track': ['keypoints'],
        'refine': ['keypoints'],
        'assign_ids': ['detect']
    }

    ANALYSIS_STAGES = {'background', 'detect', 'track', 'refine'}
    DATA_STAGES = {'import', 'downsample'}
    
    def __init__(
        self,
        config: Union[PipelineConfig, Dict[str, Any]],
        console: Optional[Console] = None
    ):
        """
        Initialize the pipeline.
        
        Args:
            config: Pipeline configuration object or dict
            console: Rich console for output
        """
        if isinstance(config, dict):
            self.config = PipelineConfig(**config)
        else:
            self.config = config
            
        self.console = console or Console()
        self.pipeline_params = self._load_pipeline_params()
        self.zarr_root = None
        self.stage_timings = {}
        
    def _load_pipeline_params(self) -> Dict[str, Any]:
        """Load pipeline parameters from YAML config file."""
        config_path = Path(self.config.config_path)
        
        if not config_path.exists():
            self.console.print(f"[yellow]Config file not found: {config_path}[/yellow]")
            self.console.print("[yellow]Using default parameters[/yellow]")
            return self._get_default_params()
        
        with open(config_path, 'r') as f:
            params = yaml.safe_load(f)
            
        self.console.print(f"Loaded config from: [green]{config_path}[/green]")
        return params
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default pipeline parameters."""
        return {
            'import': {
                'downsample_size': [640, 640],
                'chunk_size': 32,
                'batch_size': 16
            },
            'background': {
                'sample_size': 100,
                'seed': 42,
                'method': 'mode'
            },
            'detect': {
                'ds_thresh': 30,
                'se1_radius': 3,
                'se4_radius': 7,
                'min_area': 10,
                'max_area': 500,
                'max_fish': 20
            },
            'crop': {
                'roi_sz': [256, 256]
            },
            'keypoints': {  # ← Added
                'roi_thresh': 50,
                'se1_radius': 1,
                'se2_radius': 2,
                'min_area': 5,
                'adaptive_steps': 5,
                'thresh_decrement': 5,
                'scheduler': 'processes',
                'num_workers': None
            },
            'track': {
                'roi_thresh': 25,
                'se1_radius': 3,
                'se2_radius': 5
            }
        }
        
    def run(
        self,
        stages: Optional[List[str]] = None,
        validate: bool = True
    ) -> zarr.Group:
        """
        Run the pipeline stages.
        
        Args:
            stages: List of stages to run, or None for config default
            validate: Whether to validate outputs after each stage
            
        Returns:
            Root zarr group
        """
        stages = stages or self.config.stages
        
        # Handle 'all' keyword
        if 'all' in stages:
            stages = self.STAGE_ORDER
        
        # Validate stage names
        invalid_stages = set(stages) - set(self.STAGE_ORDER)
        if invalid_stages:
            raise ValueError(f"Invalid stages: {invalid_stages}")
        
        # Check dependencies
        stages_to_run = self._resolve_dependencies(stages)
        
        # Display pipeline plan
        self._display_pipeline_plan(stages_to_run)
        
        if self.config.dry_run:
            self.console.print("[yellow]Dry run mode - no processing performed[/yellow]")
            return None
        
        # Initialize timing
        pipeline_start = time.perf_counter()

        for stage in stages_to_run:
            self._run_stage(stage)
        
        # Display summary
        total_time = time.perf_counter() - pipeline_start
        self._display_summary(stages_to_run, total_time)
        
        return self.zarr_root
    
    def _resolve_dependencies(self, requested_stages: List[str]) -> List[str]:
        """Resolve stage dependencies and return ordered list of stages to run."""
        required_stages = set()
        
        # Open zarr to check what exists (if it exists)
        existing_stages = set()
        if Path(self.config.zarr_path).exists():
            try:
                root = zarr.open_group(self.config.zarr_path, mode='r')
                
                # Check which stages have outputs
                if 'raw_video' in root:
                    existing_stages.add('import')
                    if 'images_ds' in root['raw_video']:
                        existing_stages.add('downsample')
                if 'background_runs' in root and root['background_runs'].attrs.get('latest'):
                    existing_stages.add('background')
                if 'detect_runs' in root and root['detect_runs'].attrs.get('latest'):
                    existing_stages.add('detect')
            except:
                pass
        
        for stage in requested_stages:
            # Add the stage itself
            required_stages.add(stage)
            
            # Add only MISSING dependencies
            deps = self.STAGE_DEPENDENCIES.get(stage, [])
            for dep in deps:
                if dep not in existing_stages:
                    required_stages.add(dep)
        
        # Return in proper order
        return [s for s in self.STAGE_ORDER if s in required_stages]
    
    def _run_stage(self, stage: str) -> None:
        """Run a single pipeline stage."""
        self.console.rule(f"[bold]Stage: {stage.title()}[/bold]")
        
        # Only check completion for data import stages
        if stage in [
            'import',
            'downsample',
            'background',
            'detect',
            'crop',
            'keypoints',
            'assign_ids'] and self._is_stage_complete(stage):
            self.console.print(f"[green]✓ Stage '{stage}' already complete, skipping[/green]")
            return
        
        stage_start = time.perf_counter()
        
        try:
            if stage == 'import':   
                self._run_import()
            elif stage == 'background':
                self._run_background()
            elif stage == 'detect':
                self._run_detect()
            elif stage == 'crop':
                self._run_crop()
            elif stage == 'keypoints':
                self._run_keypoints()
            elif stage == 'track':
                self._run_track()
            elif stage == 'refine':
                self._run_refine()
            elif stage == 'assign_ids':
                self._run_assign_ids()
            else:
                raise ValueError(f"Unknown stage: {stage}")
        finally:
            self.stage_timings[stage] = time.perf_counter() - stage_start

    def _run_import(self) -> None:
        """Run video import stage."""
        if not self.config.video_path:
            raise ValueError("Video path required for import stage")
        
        self.zarr_root = import_video(
            video_path=self.config.video_path,
            zarr_path=self.config.zarr_path,
            config=self.pipeline_params,
            console=self.console,
            use_gpu=self.config.use_gpu,
            force_cpu=self.config.force_cpu
        )

    
    def _run_background(self) -> None:
        """Run background calculation stage."""
        if self.zarr_root is None:
            self.zarr_root = zarr.open_group(self.config.zarr_path, mode='a')
        
        # Get background parameters from pipeline config
        bg_params = self.pipeline_params.get('background', {})
        
        # Run background computation
        results = compute_background(
            zarr_path=self.config.zarr_path,
            sample_size=bg_params.get('sample_size'),
            seed=bg_params.get('seed'),
            method=bg_params.get('method'),
            compute_full=bg_params.get('compute_full'),
            compute_ds=bg_params.get('compute_ds'),
            config_path=self.config.config_path,
            console=self.console
        )
        
        self.console.print(f" Background computed using {results['frames_used']} frames")
    
    def _run_detect(self) -> None:
        """Run detection stage."""
        if self.zarr_root is None:
            self.zarr_root = zarr.open_group(self.config.zarr_path, mode='a')
        
        # Run fish detection
        results = detect_fish(
            zarr_path=self.config.zarr_path,
            config_path=self.config.config_path,
            scheduler=self.config.scheduler,
            num_workers=self.config.num_workers,
            console=self.console
        )
        
        self.console.print(f" Detected {results['total_detections']} fish in {results['frames_with_detections']} frames")
    
    def _run_crop(self) -> None:
        """Run cropping stage to extract ROIs from detections."""
        if self.zarr_root is None:
            self.zarr_root = zarr.open_group(self.config.zarr_path, mode='a')
        
        # Run cropping
        results = crop_detections(
            zarr_path=self.config.zarr_path,
            config=self.pipeline_params,
            scheduler=self.config.scheduler,
            num_workers=self.config.num_workers,
            console=self.console
        )
        
        self.console.print(f" Cropped {results['total_crops']} ROIs from {results['frames_with_crops']} frames")

    def _run_keypoints(self) -> None:
        """Run keypoints stage to detect anatomical keypoints in cropped ROIs."""
        if self.zarr_root is None:
            self.zarr_root = zarr.open_group(self.config.zarr_path, mode='a')
        
        # Run keypoint detection
        results = detect_keypoints(
            zarr_path=self.config.zarr_path,
            config=self.pipeline_params,
            scheduler=self.config.scheduler,
            num_workers=self.config.num_workers,
            console=self.console
        )
        
        self.console.print(
            f"✓ Detected keypoints in {results['successful_detections']}/{results['total_rois']} "
            f"ROIs ({results['success_rate_percent']:.1f}% success rate)"
        )
    
    def _run_refine(self) -> None:
        """Run refinement stage."""
        self.console.print("[yellow]Refinement stage not needed for multi-fish tracking[/yellow]")
    
    def _run_assign_ids(self) -> None:
        """Run ID assignment stage."""
        if self.zarr_root is None:
            self.zarr_root = zarr.open_group(self.config.zarr_path, mode='a')
        
        # Import the spatial assignment function
        from ..tracking.assign_ids import assign_ids_spatial
        
        # Run spatial ID assignment
        results = assign_ids_spatial(
            zarr_path=self.config.zarr_path,
            config=self.pipeline_params,
            console=self.console
        )
        
        self.console.print(
            f"✓ Assigned IDs to {results['assigned_detections']}/{results['total_detections']} "
            f"detections ({results['assignment_rate_percent']:.1f}% success rate)"
        )
    
    def _validate_stage(self, stage: str) -> None:
        """Validate the output of a stage."""
        if stage == 'import':
            # Validate import using the utility function
            if self.zarr_root:
                stats = get_import_stats(self.config.zarr_path)
                self.console.print(f"✓ Imported {stats['total_frames']} frames")
    
    def _display_pipeline_plan(self, stages: List[str]) -> None:
        """Display the pipeline execution plan."""
        table = Table(title="Pipeline Execution Plan", show_header=True)
        table.add_column("Stage", style="cyan")
        table.add_column("Status", style="yellow")
        table.add_column("Dependencies", style="green")
        
        for stage in stages:
            deps = self.STAGE_DEPENDENCIES.get(stage, [])
            deps_str = ", ".join(deps) if deps else "None"
            table.add_row(stage.title(), "Pending", deps_str)
        
        self.console.print(table)
    
    def _display_summary(self, stages_run: List[str], total_time: float) -> None:
        """Display pipeline execution summary."""
        table = Table(title="Pipeline Summary", show_header=True)
        table.add_column("Stage", style="cyan")
        table.add_column("Time (s)", style="yellow", justify="right")
        table.add_column("Percentage", style="green", justify="right")
        
        for stage in stages_run:
            stage_time = self.stage_timings.get(stage, 0)
            percentage = (stage_time / total_time * 100) if total_time > 0 else 0
            table.add_row(
                stage.title(),
                f"{stage_time:.2f}",
                f"{percentage:.1f}%"
            )
        
        table.add_row(
            "[bold]Total[/bold]",
            f"[bold]{total_time:.2f}[/bold]",
            "[bold]100%[/bold]"
        )
        
        self.console.print(table)
        
        # Final status
        if self.config.zarr_path and Path(self.config.zarr_path).exists():
            self.console.print(Panel(
                f"✓ Pipeline completed successfully\n"
                f"Output: {self.config.zarr_path}\n"
                f"Total time: {total_time:.1f} seconds",
                title="Success",
                style="green"
            ))

    def _is_stage_complete(self, stage: str) -> bool:
        """Check if a stage has already been completed."""
        if not Path(self.config.zarr_path).exists():
            return False

        try:
            root = zarr.open_group(self.config.zarr_path, mode='r')
            
            if stage == 'import':
                return 'raw_video' in root and 'images_full' in root['raw_video']
            elif stage == 'downsample':
                return 'raw_video' in root and 'images_ds' in root['raw_video']
            elif stage == 'background':
                # Check if background_runs exists with at least one run
                if 'background_runs' not in root:
                    return False
                return len(list(root['background_runs'].group_keys())) > 0
            elif stage == 'detect':
                # Check if detect_runs exists with at least one run
                if 'detect_runs' not in root:
                    return False
                return len(list(root['detect_runs'].group_keys())) > 0
            elif stage == 'crop':
                # Check if crop_runs exists with at least one run
                if 'crop_runs' not in root:
                    return False
                return len(list(root['crop_runs'].group_keys())) > 0
            elif stage == 'keypoints':
                # Check if keypoints_runs exists with at least one run
                if 'keypoints_runs' not in root:
                    return False
                return len(list(root['keypoints_runs'].group_keys())) > 0
            elif stage == 'assign_ids':
                if 'id_assignment_runs' not in root:
                    return False
                return len(list(root['id_assignment_runs'].group_keys())) > 0
            # Add more stage checks as needed
            
        except Exception:
            return False

        return False


def main():
    """Main entry point for the pipeline CLI."""
    
    # Check for --interactive flag early
    if "--interactive" in sys.argv or "-i" in sys.argv:
        from ..cli.interactive_launcher import run_interactive_launcher
        cmd = run_interactive_launcher()
        
        if not cmd:
            print("Pipeline launch cancelled")
            return 0
        
        # Remove the script name and replace with actual args
        sys.argv = cmd
        # Fall through to normal argument parsing
    
    parser = argparse.ArgumentParser(
        description="FishEye tracking pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python -m fisheye data.zarr --video-path video.mp4 --stages all
  
  # Run specific stages
  python -m fisheye data.zarr --stages import background detect
  
  # Tune parameters before running pipeline
  python -m fisheye data.zarr --tune mask
  python -m fisheye data.zarr --tune detect --frame 100
  python -m fisheye data.zarr --tune keypoint --frame 50
  
  # List available tuners
  python -m fisheye --list-tuners
        """
    )
    
    # Tuning arguments (mutually exclusive with normal pipeline operation)
    tuner_group = parser.add_argument_group('tuning')
    tuner_group.add_argument(
        "--tune",
        type=str,
        metavar="TUNER",
        help="Run interactive parameter tuner (mask, detect, threshold, keypoint)"
    )
    tuner_group.add_argument(
        "--list-tuners",
        action="store_true",
        help="List available tuners and exit"
    )
    tuner_group.add_argument(
        "--frame",
        type=int,
        help="Specific frame index to use in tuner"
    )
    tuner_group.add_argument(
        "--full",
        action="store_true",
        help="Use full resolution in tuner (instead of downsampled)"
    )
    
    # Required arguments (not required if using --list-tuners)
    parser.add_argument(
        "zarr_path",
        type=str,
        nargs='?',  # Make optional for --list-tuners
        help="Path to output zarr file"
    )
    
    # Optional arguments for pipeline
    parser.add_argument(
        "--video-path",
        type=str,
        help="Path to input video (required for import stage)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/fisheye/default.yaml",
        help="Path to pipeline configuration YAML"
    )
    
    parser.add_argument(
        "--stages",
        nargs='+',
        choices=['import',
                 'downsample',
                 'background',
                 'detect',
                 'crop',
                 'keypoints',
                 'track',
                 'refine',
                 'assign_ids',
                 'all'
        ],
        default=['all'],
        help="Stages to run"
    )
    
    parser.add_argument(
        "--scheduler",
        choices=['processes', 'threads', 'single-thread', 'distributed'],
        default='processes',
        help="Dask scheduler to use"
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number of workers (default: CPU count)"
    )
    
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration"
    )
    
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU processing even if GPU available"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show plan without executing"
    )
    
    args = parser.parse_args()
    
    # Create console
    console = Console()
    
    # Handle --list-tuners
    if args.list_tuners:
        from ..tune.dispatcher import list_tuners
        list_tuners(console)
        return 0
    
    # Handle --tune
    if args.tune:
        if not args.zarr_path:
            parser.error("zarr_path is required when using --tune")
        
        from ..tune.dispatcher import run_tuner
        return run_tuner(
            tuner_name=args.tune,
            zarr_path=args.zarr_path,
            config_path=args.config,
            frame_idx=args.frame,
            use_full_res=args.full,
            console=console
        )
    
    # Validate requirements for normal pipeline operation
    if not args.zarr_path:
        parser.error("zarr_path is required")
    
    if 'import' in args.stages and not args.video_path:
        parser.error("--video-path required when running import stage")
    
    # Create pipeline config
    config = PipelineConfig.from_args(args)
    
    # Create and run pipeline
    pipeline = Pipeline(config, console)
    
    try:
        pipeline.run()
    except KeyboardInterrupt:
        console.print("\n[red]Pipeline interrupted by user[/red]")
        return 1
    except Exception as e:
        console.print(f"\n[red]Pipeline failed: {e}[/red]")
        if args.verbose:
            import traceback
            console.print(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
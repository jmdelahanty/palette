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
from rich import box
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
from ..tracking.assign_ids import assign_ids_spatial
from ..refinement.refine_detect import create_refined_run
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
    crop_source: str = "detect"
    refine_max_gap: Optional[int] = None
    refine_method: Optional[str] = None
    refine_remove_jumps: Optional[bool] = None
    refine_remove_blips: Optional[bool] = None
    
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
            dry_run=getattr(args, 'dry_run', False),
            crop_source=getattr(args, 'crop_source', 'detect'),
            refine_max_gap=getattr(args, 'refine_max_gap', None),
            refine_method=getattr(args, 'refine_method', None),
            refine_remove_jumps=getattr(args, 'refine_remove_jumps', None),
            refine_remove_blips=getattr(args, 'refine_remove_blips', None)
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
        'refine',
        'crop',
        'keypoints',
        'assign_ids',
        'track',
    ]
    
    STAGE_DEPENDENCIES = {
        'import': [],
        'downsample': ['import'],
        'background': ['import'],
        'detect': ['background'],
        'refine': ['detect'],
        'crop': ['detect'],
        'keypoints': ['crop', 'background'],
        'assign_ids': ['detect'],
        'track': ['keypoints'],
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
        self.stage_results = {}
        
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
            'keypoints': {
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
            },
            'refine_detect': {
                'filters': {'remove_jumps': True, 'remove_blips': False},
                'max_gap': 20,
                'interpolation_method': 'linear'
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
        required_stages = set()  # Using set to avoid duplicates
        
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
                if 'id_assignment_runs' in root and root['id_assignment_runs'].attrs.get('latest'):
                    existing_stages.add('assign_ids')
            except:
                pass
        
        for stage in requested_stages:
            # Add the stage itself
            required_stages.add(stage)  # Set automatically handles duplicates
            
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
            'refine',
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
        
        # Validate refined source is available if requested
        if self.config.crop_source in ['filtered', 'interpolated']:
            if 'refined_runs' not in self.zarr_root or self.zarr_root['refined_runs'].attrs.get('latest') is None:
                self.console.print(
                    f"[red]Error: Crop source '{self.config.crop_source}' requires refined detections.[/red]\n"
                    "[yellow]Run the 'refine' stage first or use '--crop-source detect'.[/yellow]"
                )
                raise ValueError(f"Refined runs not found for crop source '{self.config.crop_source}'")
        
        # Run cropping with specified source
        results = crop_detections(
            zarr_path=self.config.zarr_path,
            config=self.pipeline_params,
            source_type=self.config.crop_source,
            scheduler=self.config.scheduler,
            num_workers=self.config.num_workers,
            console=self.console
        )
        
        # Display results with source info
        source_info = f"from {self.config.crop_source} detections"
        if 'detection_source_type' in results:
            source_info = f"from {results['detection_source_type']} detections"
        
        self.console.print(f"[green]✓[/green] Cropped {results['total_crops']} ROIs from {results['frames_with_crops']} frames ({source_info})")

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
        """Run detection refinement stage (filter & interpolate detections)."""
        if self.zarr_root is None:
            self.zarr_root = zarr.open_group(self.config.zarr_path, mode='a')
        
        run_name = create_refined_run(
            zarr_path=self.config.zarr_path,
            config=self.pipeline_params,
            max_gap=self.config.refine_max_gap,
            interpolation_method=self.config.refine_method,
            remove_jumps=self.config.refine_remove_jumps,
            remove_blips=self.config.refine_remove_blips,
            console=self.console
        )
        
        self.console.print(f"[green]✓[/green] Detection refinement saved as [cyan]{run_name}[/cyan]")
    
    def _run_assign_ids(self) -> None:
        """
        Run ID assignment stage.
        
        For single-dish experiments: Automatically uses dish mask as single ROI
        For multi-dish experiments: Requires sub-dish ROI definitions
        
        Checks experiment_setup metadata to determine mode automatically.
        """
        # Check if experiment setup is configured
        root = zarr.open(self.config.zarr_path, mode='r')
        
        has_setup = 'experiment_setup' in root.attrs
        if not has_setup:
            self.console.print("[yellow]⚠️  No experiment setup metadata found[/yellow]")
            self.console.print("[yellow]This helps determine single vs multi-dish mode[/yellow]")
            self.console.print()
            
            # Prompt user to configure
            response = input("Configure experiment setup now? (Y/n): ").strip().lower()
            if response != 'n':
                import subprocess
                setup_script = Path(__file__).parent.parent.parent / "setup_experiment_metadata.py"
                subprocess.run([
                    sys.executable, 
                    str(setup_script), 
                    self.config.zarr_path,
                    "--interactive"
                ])
                
                # Reload to check if configured
                root = zarr.open(self.config.zarr_path, mode='r')
                if 'experiment_setup' not in root.attrs:
                    self.console.print("[yellow]Setup not configured, continuing anyway...[/yellow]")
        
        # Get experiment setup info
        experiment_setup = root.attrs.get('experiment_setup', {})
        setup_type = experiment_setup.get('setup_type', 'unknown')
        
        # For multi-dish, check if subdish masks are defined
        if setup_type == 'multi_dish':
            has_subdish_masks = False
            
            if 'analysis_metadata' in root:
                has_subdish_masks = 'subdish_mask_tuning' in root['analysis_metadata'].attrs
            
            if not has_subdish_masks:
                # Check config as fallback
                assign_params = self.pipeline_params.get('assign_ids', {})
                has_subdish_masks = 'sub_dish_rois' in assign_params
            
            if not has_subdish_masks:
                self.console.print("[yellow]⚠️  Multi-dish mode requires sub-dish ROI definitions[/yellow]")
                self.console.print()
                response = input("Run sub-dish mask tuner now? (Y/n): ").strip().lower()
                
                if response != 'n':
                    import subprocess
                    tune_script = Path(__file__).parent.parent / "tune" / "subdish_mask_tuner.py"
                    subprocess.run([
                        sys.executable,
                        str(tune_script),
                        self.config.zarr_path
                    ])
                else:
                    self.console.print("[red]Cannot proceed without sub-dish masks[/red]")
                    return
        
        # Run assign_ids
        self.console.print(f"\n[bold cyan]Running ID Assignment ({setup_type} mode)[/bold cyan]\n")
        
        results = assign_ids_spatial(
            zarr_path=self.config.zarr_path,
            config=self.pipeline_params,
            console=self.console
        )
        
        # Store results
        self.stage_results['assign_ids'] = results
    
    def _validate_stage(self, stage: str) -> None:
        """Validate the output of a stage."""
        if stage == 'import':
            # Validate import using the utility function
            if self.zarr_root:
                stats = get_import_stats(self.config.zarr_path)
                self.console.print(f"✓ Imported {stats['total_frames']} frames")
    
    def _display_pipeline_plan(self, stages: List[str]) -> None:
        """Display the pipeline execution plan"""
        from rich.table import Table
        from rich.panel import Panel
        
        # Check for experiment setup
        setup_panel = None
        if Path(self.config.zarr_path).exists():
            try:
                root = zarr.open(self.config.zarr_path, mode='r')
                experiment_setup = root.attrs.get('experiment_setup', {})
                
                if experiment_setup:
                    # Create experiment setup info panel
                    setup_type = experiment_setup.get('setup_type', 'unknown')
                    num_dishes = experiment_setup.get('num_dishes', '?')
                    fish_per_dish = experiment_setup.get('fish_per_dish', '?')
                    total_fish = experiment_setup.get('total_expected_fish', '?')
                    source = experiment_setup.get('source', 'unknown')
                    
                    # Color code based on setup type
                    setup_color = "cyan" if setup_type == "single_dish" else "yellow"
                    
                    setup_text = (
                        f"[bold {setup_color}]Setup Type:[/bold {setup_color}] {setup_type}\n"
                        f"[bold]Dishes:[/bold] {num_dishes}\n"
                        f"[bold]Fish per dish:[/bold] {fish_per_dish}\n"
                        f"[bold]Total expected:[/bold] {total_fish} fish\n"
                        f"[dim]Source: {source}[/dim]"
                    )
                    
                    setup_panel = Panel(
                        setup_text,
                        title="🐠 Experiment Configuration",
                        border_style=setup_color,
                        padding=(1, 2)
                    )
            except Exception as e:
                # Silently fail if can't read zarr
                pass
        
        # Display experiment setup if available
        if setup_panel:
            self.console.print(setup_panel)
            self.console.print()
        else:
            # Warn if assign_ids is in stages but no setup
            if 'assign_ids' in stages:
                self.console.print(
                    "[yellow]ℹ No experiment setup configured. "
                    "Run with --setup to configure.[/yellow]\n"
                )
        
        # Display stage execution plan
        table = Table(title="Pipeline Execution Plan", show_header=True, box=box.ROUNDED)
        table.add_column("Order", style="dim", width=6)
        table.add_column("Stage", style="cyan")
        table.add_column("Status", style="yellow")
        table.add_column("Dependencies", style="green")
        
        for i, stage in enumerate(stages, 1):
            deps = self.STAGE_DEPENDENCIES.get(stage, [])
            deps_str = ", ".join(deps) if deps else "None"
            
            # Check if stage is complete
            is_complete = self._is_stage_complete(stage)
            status = "✓ Complete" if is_complete else "→ Pending"
            status_style = "green" if is_complete else "yellow"
            
            table.add_row(
                str(i),
                stage.title(),
                f"[{status_style}]{status}[/{status_style}]",
                deps_str
            )
        
        self.console.print(table)
        self.console.print()


    def _display_summary(self, stages_run: List[str], total_time: float) -> None:
        """Display pipeline execution summary with experiment results."""
        from rich.table import Table
        from rich.panel import Panel
        
        # Timing table
        timing_table = Table(
            title="Pipeline Timing", 
            show_header=True, 
            box=box.ROUNDED
        )
        timing_table.add_column("Stage", style="cyan")
        timing_table.add_column("Time (s)", style="yellow", justify="right")
        timing_table.add_column("Percentage", style="green", justify="right")
        
        for stage in stages_run:
            stage_time = self.stage_timings.get(stage, 0)
            percentage = (stage_time / total_time * 100) if total_time > 0 else 0
            timing_table.add_row(
                stage.title(),
                f"{stage_time:.2f}",
                f"{percentage:.1f}%"
            )
        
        timing_table.add_row(
            "[bold]Total[/bold]",
            f"[bold]{total_time:.2f}[/bold]",
            "[bold]100%[/bold]"
        )
        
        self.console.print(timing_table)
        self.console.print()
        
        # Results summary if zarr exists
        if Path(self.config.zarr_path).exists():
            try:
                root = zarr.open(self.config.zarr_path, mode='r')
                
                # Get experiment setup
                experiment_setup = root.attrs.get('experiment_setup', {})
                
                # Build results summary
                results_lines = []
                
                # Experiment configuration
                if experiment_setup:
                    setup_type = experiment_setup.get('setup_type', 'unknown')
                    total_expected = experiment_setup.get('total_expected_fish', '?')
                    results_lines.append(f"[bold cyan]Experiment:[/bold cyan] {setup_type}")
                    results_lines.append(f"[bold]Expected fish:[/bold] {total_expected}")
                
                # Detection results
                if 'detect_runs' in root and root['detect_runs'].attrs.get('latest'):
                    latest_detect = root['detect_runs'].attrs['latest']
                    detect_group = root[f'detect_runs/{latest_detect}']
                    if 'summary_statistics' in detect_group.attrs:
                        stats = detect_group.attrs['summary_statistics']
                        total_detections = stats.get('total_detections', 0)
                        frames_with_detections = stats.get('frames_with_detections', 0)
                        total_frames = stats.get('total_frames', 1)
                        detection_rate = (frames_with_detections / total_frames) * 100
                        
                        results_lines.append(f"[bold]Detections:[/bold] {total_detections:,} total")
                        results_lines.append(f"[bold]Detection rate:[/bold] {detection_rate:.1f}% of frames")
                
                # Crop results - show source type
                if 'crop_runs' in root and root['crop_runs'].attrs.get('latest'):
                    latest_crop = root['crop_runs'].attrs['latest']
                    crop_group = root[f'crop_runs/{latest_crop}']
                    
                    # Get crop source info
                    crop_source = crop_group.attrs.get('detection_source_type', 'detect')
                    includes_interpolated = crop_group.attrs.get('includes_interpolated', False)
                    
                    if 'summary_statistics' in crop_group.attrs:
                        stats = crop_group.attrs['summary_statistics']
                        total_crops = stats.get('total_rois_cropped', 0)
                        frames_with_crops = stats.get('frames_with_crops', 0)
                        
                        # Build crop info string
                        crop_info = f"[bold]Crops:[/bold] {total_crops:,} ROIs from {frames_with_crops:,} frames"
                        
                        # Add source info
                        source_color = "yellow" if crop_source in ['filtered', 'interpolated'] else "cyan"
                        crop_info += f" ([{source_color}]{crop_source}[/{source_color}])"
                        
                        results_lines.append(crop_info)
                        
                        # If interpolated, show breakdown
                        if includes_interpolated:
                            n_real = crop_group.attrs.get('n_real_detections', 0)
                            n_interp = crop_group.attrs.get('n_interpolated_detections', 0)
                            results_lines.append(
                                f"  [dim]└─ {n_real:,} real + {n_interp:,} interpolated[/dim]"
                            )
                
                # ID assignment results
                if 'id_assignment_runs' in root and root['id_assignment_runs'].attrs.get('latest'):
                    latest_assign = root['id_assignment_runs'].attrs['latest']
                    assign_group = root[f'id_assignment_runs/{latest_assign}']
                    if 'summary_statistics' in assign_group.attrs:
                        stats = assign_group.attrs['summary_statistics']
                        assigned = stats.get('assigned_detections', 0)
                        total = stats.get('total_detections', 1)
                        assignment_rate = stats.get('assignment_rate_percent', 0)
                        num_masks = stats.get('num_masks', 0)
                        
                        results_lines.append(f"[bold]ID Assignment:[/bold] {assigned:,}/{total:,} ({assignment_rate:.1f}%)")
                        results_lines.append(f"[bold]ROIs tracked:[/bold] {num_masks}")
                        
                        # Per-mask summary
                        per_mask_stats = stats.get('per_mask_statistics', [])
                        if per_mask_stats:
                            results_lines.append("\n[bold]Per-ROI Coverage:[/bold]")
                            for mask_stat in per_mask_stats:
                                mask_id = mask_stat['mask_id']
                                coverage = mask_stat['coverage_percent']
                                detections = mask_stat['total_detections']
                                
                                # Color code based on coverage
                                if coverage > 80:
                                    color = "green"
                                elif coverage > 50:
                                    color = "yellow"
                                else:
                                    color = "red"
                                
                                roi_label = "Dish" if experiment_setup.get('setup_type') == 'single_dish' else f"ROI {mask_id}"
                                results_lines.append(
                                    f"  [{color}]{roi_label}:[/{color}] "
                                    f"{coverage:.1f}% coverage ({detections:,} detections)"
                                )
                
                # Keypoint results
                if 'keypoint_runs' in root and root['keypoint_runs'].attrs.get('latest'):
                    latest_kp = root['keypoint_runs'].attrs['latest']
                    kp_group = root[f'keypoint_runs/{latest_kp}']
                    if 'summary_statistics' in kp_group.attrs:
                        stats = kp_group.attrs['summary_statistics']
                        successful = stats.get('successful_detections', 0)
                        total_rois = stats.get('total_rois', 1)
                        success_rate = (successful / total_rois) * 100
                        
                        results_lines.append(f"[bold]Keypoints:[/bold] {successful:,}/{total_rois:,} ({success_rate:.1f}%)")
                
                # Display results panel
                if results_lines:
                    results_text = "\n".join(results_lines)
                    results_panel = Panel(
                        results_text,
                        title="📊 Pipeline Results",
                        border_style="green",
                        padding=(1, 2)
                    )
                    self.console.print(results_panel)
                    self.console.print()
            
            except Exception as e:
                # Silently fail if can't read results
                pass
        
        # Final status panel
        status_text = (
            f"[green]✓[/green] Pipeline completed successfully\n"
            f"[bold]Output:[/bold] {self.config.zarr_path}\n"
            f"[bold]Total time:[/bold] {total_time:.1f} seconds"
        )
        
        # Add next steps based on what was run
        if 'assign_ids' in stages_run:
            status_text += "\n\n[bold cyan]Next steps:[/bold cyan]"
            if Path(self.config.zarr_path).exists():
                root = zarr.open(self.config.zarr_path, mode='r')
                experiment_setup = root.attrs.get('experiment_setup', {})
                
                # Suggest interpolation and analysis
                status_text += "\n  • Run interpolation: python batch_roi_interpolator.py"
                status_text += "\n  • Analyze behavior: python fish_behavior_metrics.py"
                status_text += "\n  • Visualize: python roi_heatmap_generator.py"
        
        self.console.print(Panel(
            status_text,
            title="Success",
            border_style="bold green",
            padding=(1, 2)
        ))

    def _is_stage_complete(self, stage: str) -> bool:
        """Check if a stage has already been completed."""
        if not Path(self.config.zarr_path).exists():
            return False
        
        try:
            root = zarr.open(self.config.zarr_path, mode='r')
            
            if stage == 'import':
                return 'raw_video' in root and 'images_ds' in root['raw_video']
            
            elif stage == 'background':
                if 'background_runs' not in root:
                    return False
                latest = root['background_runs'].attrs.get('latest')
                return latest is not None
            
            elif stage == 'detect':
                if 'detect_runs' not in root:
                    return False
                latest = root['detect_runs'].attrs.get('latest')
                return latest is not None
            
            elif stage == 'crop':
                if 'crop_runs' not in root:
                    return False
                latest = root['crop_runs'].attrs.get('latest')
                return latest is not None
            
            elif stage == 'keypoints':
                if 'keypoint_runs' not in root:
                    return False
                latest = root['keypoint_runs'].attrs.get('latest')
                return latest is not None
            
            elif stage == 'assign_ids':
                if 'id_assignment_runs' not in root:
                    return False
                latest = root['id_assignment_runs'].attrs.get('latest')
                return latest is not None
            
            elif stage == 'refine':
                if 'refined_runs' not in root:
                    return False
                latest = root['refined_runs'].attrs.get('latest')
                return latest is not None
            
            elif stage == 'track':
                if 'tracking_runs' not in root:
                    return False
                latest = root['tracking_runs'].attrs.get('latest')
                return latest is not None
            
            return False
            
        except Exception:
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
        "--refine-max-gap",
        type=int,
        help="Override maximum gap (in frames) for refinement"
    )

    parser.add_argument(
        "--refine-method",
        type=str,
        choices=['linear'],
        help="Interpolation method for refinement"
    )

    parser.set_defaults(refine_remove_jumps=None, refine_remove_blips=None)
    refine_jump_group = parser.add_mutually_exclusive_group()
    refine_jump_group.add_argument(
        "--refine-remove-jumps",
        dest="refine_remove_jumps",
        action="store_true",
        help="Force refinement to drop jump detections"
    )
    refine_jump_group.add_argument(
        "--refine-keep-jumps",
        dest="refine_remove_jumps",
        action="store_false",
        help="Keep jump detections during refinement"
    )

    refine_blip_group = parser.add_mutually_exclusive_group()
    refine_blip_group.add_argument(
        "--refine-remove-blips",
        dest="refine_remove_blips",
        action="store_true",
        help="Remove blip detections during refinement"
    )
    refine_blip_group.add_argument(
        "--refine-keep-blips",
        dest="refine_remove_blips",
        action="store_false",
        help="Keep blip detections during refinement"
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
        "--crop-source",
        type=str,
        default="detect",
        choices=["detect", "filtered", "interpolated"],
        help="Detection source for cropping (default: detect)"
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

"""
Legacy core pipeline orchestrator for FishEye tracking system.

This module predates the current cluster workflow. Current processing uses
``fisheye.utils.run_recording_analysis_pipeline`` plus the per-stage
``scripts/submit_*_bsub.sh`` LSF submitters coordinated through Zarr completion
markers; see ``docs/operator_guide/pipeline_workflow.md``. Keep this
orchestrator runnable for compatibility, but do not extend it as the live path.
"""

import sys
import yaml
import time
import argparse
import logging
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
from ..detection.detect_keypoints_traditional import detect_keypoints as detect_keypoints_traditional
from ..tracking.crop import crop_detections, infer_detection_source_type
from ..tracking.arena_assignment import assign_arenas_spatial
from ..refinement.refine_detect import create_refined_run
from ..refinement.detect_quality import analyze_detect_quality, save_quality_report
from ..refinement.refine_keypoints import create_refined_keypoint_run
from ..registry.stage_catalog import canonical_stage_id
from ..shared.experiment_setup import infer_experiment_setup
from ..shared.zarr_run_completion import resolve_latest_complete_run_name
from ..shared.zarr.schema import validate_zarr_structure

REFINED_DETECT_GROUP = "refined_detect_runs"
LEGACY_REFINED_DETECT_GROUP = "refined_runs"
REFINED_KEYPOINT_GROUP = "refined_keypoints_runs"
LEGACY_REFINED_KEYPOINT_GROUP = "keypoints_refined_runs"
_DEPRECATED_REFINE_INTERPOLATION_OVERRIDE_MESSAGE = (
    "Interpolation overrides are deprecated and unsupported for refine_detect. "
    "The current sparse-first refine_detect workflow always runs with interpolation disabled."
)
LEGACY_ORCHESTRATOR_NOTICE = (
    "Legacy orchestrator notice: fisheye.core.pipeline predates the current cluster workflow; "
    "use fisheye.utils.run_recording_analysis_pipeline and scripts/submit_*_bsub.sh "
    "(see docs/operator_guide/pipeline_workflow.md)."
)
LOGGER = logging.getLogger(__name__)


def _reject_deprecated_refine_interpolation_overrides(
    *,
    refine_max_gap: Optional[int] = None,
    refine_method: Optional[str] = None,
) -> None:
    if refine_max_gap is None and refine_method is None:
        return
    flags: list[str] = []
    if refine_max_gap is not None:
        flags.append("--refine-max-gap")
    if refine_method is not None:
        flags.append("--refine-method")
    raise ValueError(
        f"{_DEPRECATED_REFINE_INTERPOLATION_OVERRIDE_MESSAGE} "
        f"Remove {' and '.join(flags)}."
    )


def _get_group_with_fallback(root: zarr.Group, primary: str, legacy: str) -> Optional[zarr.Group]:
    if primary in root:
        return root[primary]
    if legacy in root:
        return root[legacy]
    return None


def _latest_complete(parent: Optional[zarr.Group]) -> Optional[str]:
    if parent is None:
        return None
    return resolve_latest_complete_run_name(parent)


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
    crop_source: Optional[str] = None
    crop_source_path: Optional[str] = None
    crop_selection_policy: Optional[str] = None
    crop_acceleration: str = "auto"
    refine_max_gap: Optional[int] = None
    refine_method: Optional[str] = None
    refine_remove_jumps: Optional[bool] = None
    refine_remove_blips: Optional[bool] = None
    training_data: bool = False
    frame_step: Optional[int] = None
    no_dask_progress: bool = False
    registry_path: Optional[str] = None

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
            crop_source=getattr(args, 'crop_source', None),
            crop_source_path=getattr(args, 'crop_source_path', None),
            crop_selection_policy=getattr(args, 'crop_selection_policy', None),
            crop_acceleration=getattr(args, 'crop_acceleration', 'auto'),
            refine_max_gap=getattr(args, 'refine_max_gap', None),
            refine_method=getattr(args, 'refine_method', None),
            refine_remove_jumps=getattr(args, 'refine_remove_jumps', None),
            refine_remove_blips=getattr(args, 'refine_remove_blips', None),
            training_data=getattr(args, 'training_data', False),
            frame_step=getattr(args, 'frame_step', None),
            no_dask_progress=getattr(args, 'no_dask_progress', False),
            registry_path=getattr(args, 'registry', None),
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
        'detect_quality',
        'refine',
        'crop',
        'keypoints',
        'refined_subject_masks',
        'keypoints_refine',
        'assign_ids',
        'track',
    ]
    
    STAGE_DEPENDENCIES = {
        'import': [],
        'downsample': ['import'],
        'background': ['import'],
        'detect': ['background'],
        'detect_quality': ['detect'],
        'refine': ['detect_quality'],
        'crop': ['detect'],
        'keypoints': ['crop', 'background'],
        'keypoints_refine': ['keypoints'],
        'refined_subject_masks': [],
        'assign_ids': ['detect'],
        'track': ['keypoints'],
    }

    STAGE_CANONICAL_IDS = {
        stage: canonical_stage_id(stage)
        for stage in STAGE_ORDER
    }

    ANALYSIS_STAGES = {
        'background',
        'detect',
        'detect_quality',
        'track',
        'refine',
        'keypoints_refine',
        'refined_subject_masks',
    }
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
        self._explicitly_requested_stages: set[str] = set()
        
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
            },
            'refine_keypoints': {
                'chunk_size': 4096,
                'scheduler': 'processes',
                'num_workers': None,
                'memory_limit': None
            },
            'refine_subject_masks': {
                'enabled': False,
                'subject_run': None,
                'source_run': None,
                'refined_run': None,
                'run_name': None,
                'components': None,
                'roi_indices': None,
                'chunk_size': 512,
                'scheduler': 'processes',
                'num_workers': None,
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
        self._explicitly_requested_stages = {str(stage) for stage in stages if str(stage) != 'all'}
        
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
                if 'background_runs' in root and _latest_complete(root['background_runs']):
                    existing_stages.add('background')
                if 'detect_runs' in root and _latest_complete(root['detect_runs']):
                    existing_stages.add('detect')
                if 'arena_assignment_runs' in root and _latest_complete(root['arena_assignment_runs']):
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
                    # Special case: refine/detect_quality only need detect data, not video/background
                    # If detect_runs exists, skip import/background/downsample dependencies
                    if stage in ('refine', 'detect_quality') and dep in ['import', 'background', 'downsample']:
                        if 'detect' in existing_stages:
                            continue  # Skip video pipeline deps if detections exist
                    
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
            'detect_quality',
            'refine',
            'crop',
            'keypoints',
            'refined_subject_masks',
            'keypoints_refine',
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
            elif stage == 'detect_quality':
                self._run_detect_quality()
            elif stage == 'crop':
                self._run_crop()
            elif stage == 'keypoints':
                self._run_keypoints()
            elif stage == 'refined_subject_masks':
                self._run_refined_subject_masks()
            elif stage == 'keypoints_refine':
                self._run_keypoints_refine()
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

        # Build cli_args dict for import_video
        cli_args = {
            'video_path': self.config.video_path,
            'zarr_path': self.config.zarr_path,
            'training_data': self.config.training_data,
            'frame_step': self.config.frame_step,
        }

        self.zarr_root = import_video(
            video_path=self.config.video_path,
            zarr_path=self.config.zarr_path,
            config=self.pipeline_params,
            cli_args=cli_args,
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
            console=self.console,
            show_progress=not self.config.no_dask_progress,
        )
        
        self.console.print(f" Detected {results['total_detections']} fish in {results['frames_with_detections']} frames")
    
    def _run_crop(self) -> None:
        """Run cropping stage to extract ROIs from detections."""
        if self.zarr_root is None:
            self.zarr_root = zarr.open_group(self.config.zarr_path, mode='a')

        crop_params = self.pipeline_params.get('crop', {}) or {}
        config_source_type = crop_params.get('source_type')
        config_source_path = crop_params.get('source_path')
        selection_policy = self.config.crop_selection_policy or crop_params.get('selection_policy')
        
        cli_source_type = self.config.crop_source
        cli_source_path = self.config.crop_source_path

        source_path = cli_source_path or config_source_path
        if source_path:
            source_path = str(source_path).strip().strip('/')
            if not source_path:
                source_path = None

        raw_source_type = cli_source_type or config_source_type
        source_type = infer_detection_source_type(source_path, raw_source_type)
        
        # Run cropping with specified source
        results = crop_detections(
            zarr_path=self.config.zarr_path,
            config=self.pipeline_params,
            source_type=source_type,
            source_path=source_path,
            selection_policy=selection_policy,
            scheduler=self.config.scheduler,
            num_workers=self.config.num_workers,
            console=self.console,
            acceleration=self.config.crop_acceleration,
            use_gpu_allowed=self.config.use_gpu,
            force_cpu=self.config.force_cpu,
            verbose=self.config.verbose
        )
        
        # Display results with source info
        source_label = results.get('detection_source_type', source_type)
        source_path_display = results.get('detection_source_path')
        if source_path_display:
            source_info = f"from {source_label} detections ({source_path_display})"
        else:
            source_info = f"from {source_label} detections"
        
        self.console.print(f"[green]✓[/green] Cropped {results['total_crops']} ROIs from {results['frames_with_crops']} frames ({source_info})")

    def _run_keypoints(self) -> None:
        """Run keypoints stage to detect anatomical keypoints in cropped ROIs."""
        if self.zarr_root is None:
            self.zarr_root = zarr.open_group(self.config.zarr_path, mode='a')

        params = self.pipeline_params.get('keypoints', {}) or {}
        method = str(params.get('method', 'traditional')).lower()

        if method in {'traditional', 'trad', 'traditional_pose'}:
            results = detect_keypoints_traditional(
                zarr_path=self.config.zarr_path,
                config=self.pipeline_params,
                scheduler=self.config.scheduler,
                num_workers=self.config.num_workers,
                console=self.console,
                registry=Path(str(self.config.registry_path)) if self.config.registry_path else None,
            )

            self.console.print(
                f"✓ Detected keypoints in {results['successful_detections']}/{results['total_rois']} "
                f"ROIs ({results['success_rate_percent']:.1f}% success rate)"
            )
        elif method in {'yolo', 'yolo_pose'}:
            model_path = params.get('model') or params.get('model_path')
            if not model_path:
                raise ValueError("YOLO keypoint detection requires 'model' (or 'model_path') in keypoints config.")

            from ..detection.detect_keypoints_yolo import detect_keypoints_yolo

            run_name = detect_keypoints_yolo(
                zarr_path=self.config.zarr_path,
                model_path=model_path,
                run_name=params.get('run_name'),
                crop_run=params.get('crop_run'),
                batch_size=params.get('batch_size', 256),
                device=params.get('device'),
                imgsz=params.get('imgsz'),
                conf=params.get('conf', 0.25),
                iou=params.get('iou', 0.5),
                max_det=params.get('max_det', 1),
                mask_threshold=params.get('mask_threshold', 0.5),
                roi_cache_policy=params.get('roi_cache_policy', 'auto'),
                roi_cache_dir=params.get('roi_cache_dir'),
                input_mode=params.get('input_mode', 'numpy-list'),
                verbose=params.get('verbose', False),
                registry=Path(str(self.config.registry_path)) if self.config.registry_path else None,
                console=self.console,
            )

            self.console.print(
                f"✓ YOLO pose inference saved as keypoints_runs/{run_name}"
            )
        else:
            raise ValueError(f"Unknown keypoint method '{method}'. Expected 'traditional' or 'yolo'.")

    def _run_keypoints_refine(self) -> None:
        """Run keypoint refinement stage."""
        if self.zarr_root is None:
            self.zarr_root = zarr.open_group(self.config.zarr_path, mode='a')

        run_name = create_refined_keypoint_run(
            zarr_path=self.config.zarr_path,
            config=self.pipeline_params,
            console=self.console,
            command="pipeline:keypoints_refine",
            created_at_utc=datetime.now(timezone.utc).isoformat(),
        )

        self.console.print(f"[green]✓[/green] Keypoint refinement saved as [cyan]{run_name}[/cyan]")

    def _run_refined_subject_masks(self) -> None:
        """Run refined subject-mask batch apply stage."""
        if self.zarr_root is None:
            self.zarr_root = zarr.open_group(self.config.zarr_path, mode='a')

        params = self.pipeline_params.get('refine_subject_masks', {}) or {}
        explicitly_requested = self._is_stage_explicitly_requested('refined_subject_masks')
        if not bool(params.get('enabled', False)) and not explicitly_requested:
            self.console.print(
                "[yellow]Refined subject-mask batch apply disabled via "
                "refine_subject_masks.enabled=false; skipping.[/yellow]"
            )
            return
        if not bool(params.get('enabled', False)) and explicitly_requested:
            self.console.print(
                "[cyan]Running refined subject-mask batch apply because "
                "'refined_subject_masks' was explicitly requested.[/cyan]"
            )

        def _resolve_alias(primary: str, alias: str):
            primary_value = params.get(primary)
            alias_value = params.get(alias)
            if primary_value is not None and alias_value is not None and primary_value != alias_value:
                raise ValueError(
                    f"refine_subject_masks.{primary} and refine_subject_masks.{alias} disagree: "
                    f"{primary_value!r} != {alias_value!r}"
                )
            return primary_value if primary_value is not None else alias_value

        source_run = _resolve_alias('subject_run', 'source_run')
        refined_run = _resolve_alias('refined_run', 'run_name')

        scheduler_param = params.get('scheduler')
        scheduler_source = 'refine_subject_masks.scheduler' if scheduler_param is not None else 'pipeline --scheduler'
        scheduler = str(scheduler_param or self.config.scheduler or 'processes').lower()
        if scheduler in {'single-thread', 'single_thread'}:
            scheduler = 'single-threaded'
        if scheduler not in {'threads', 'processes', 'distributed', 'single-threaded'}:
            self.console.print(
                f"[yellow]Unsupported refine_subject_masks scheduler '{scheduler}', using 'processes'[/yellow]"
            )
            scheduler = 'processes'
        chunk_size = params.get('chunk_size', 512)
        if chunk_size is None:
            chunk_size = 512
        num_workers = params.get('num_workers', self.config.num_workers)
        num_workers_source = (
            'refine_subject_masks.num_workers'
            if params.get('num_workers') is not None
            else 'pipeline --num-workers'
        )

        from ..refinement.refine_subject_masks import refine_subject_masks

        self.console.print(
            f"[cyan]Refined subject-mask execution config:[/cyan] scheduler={scheduler} "
            f"({scheduler_source}), num_workers={num_workers} ({num_workers_source})"
        )

        apply_summary = refine_subject_masks(
            zarr_path=self.config.zarr_path,
            subject_run=source_run,
            refined_run=refined_run,
            components=params.get('components'),
            roi_indices=params.get('roi_indices'),
            chunk_size=chunk_size,
            scheduler=scheduler,
            num_workers=num_workers,
            console=self.console,
        )

        self.stage_results['refined_subject_masks'] = apply_summary
        self.console.print(
            "[green]✓[/green] Refined subject-mask batch apply updated "
            f"[cyan]{apply_summary['refined_run']}[/cyan] "
            f"(changed={apply_summary['changed_roi_count']}, noop={apply_summary['noop_roi_count']})"
        )
    
    def _run_refine(self) -> None:
        """Run detection refinement stage (filter artifacts, write curated instances)."""
        _reject_deprecated_refine_interpolation_overrides(
            refine_max_gap=self.config.refine_max_gap,
            refine_method=self.config.refine_method,
        )
        if self.zarr_root is None:
            self.zarr_root = zarr.open_group(self.config.zarr_path, mode='a')
        
        run_name = create_refined_run(
            zarr_path=self.config.zarr_path,
            config=self.pipeline_params,
            remove_jumps=self.config.refine_remove_jumps,
            remove_blips=self.config.refine_remove_blips,
            console=self.console,
            command="pipeline:refine_detect",
            created_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        
        self.console.print(f"[green]✓[/green] Detection refinement saved as [cyan]{run_name}[/cyan]")

    def _run_detect_quality(self) -> None:
        """Run detection quality assessment stage."""
        quality_params = self.pipeline_params.get('detect_quality', {})

        report = analyze_detect_quality(
            zarr_path=self.config.zarr_path,
            jump_threshold=quality_params.get('jump_threshold', 100.0),
            blip_gap_threshold=quality_params.get('blip_gap_threshold', 10),
            threshold_mode=quality_params.get('threshold_mode', 'scaled'),
            threshold_reference_width=quality_params.get('threshold_reference_width', 640.0),
        )

        save_quality_report(
            zarr_path=self.config.zarr_path,
            quality_report=report,
            console=self.console,
        )

        score = report['quality_score']
        self.console.print(
            f"[green]✓[/green] Detection quality: grade [cyan]{score['grade']}[/cyan] "
            f"(score {score['overall_score']:.1f})"
        )

    def _run_assign_ids(self) -> None:
        """
        Run arena assignment stage.
        
        For single-dish experiments: Automatically uses dish mask as single ROI
        For multi-dish experiments: Requires sub-dish ROI definitions
        
        Checks experiment_setup metadata to determine mode automatically.
        """
        # Check if experiment setup is configured
        root = zarr.open(self.config.zarr_path, mode='r')
        setup_info = infer_experiment_setup(root.attrs)

        if setup_info.source == "unknown":
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
                setup_info = infer_experiment_setup(root.attrs)
                if setup_info.source == "unknown":
                    self.console.print("[yellow]Setup not configured, continuing anyway...[/yellow]")
        elif setup_info.source == "experimental_chamber":
            self.console.print("[yellow]⚠️  No experiment_setup metadata found; using experimental_chamber to infer single-dish mode[/yellow]")
            self.console.print("[yellow]Consider backfilling experiment_setup for future runs.[/yellow]")
            self.console.print()
        
        # Get experiment setup info
        setup_type = setup_info.setup_type
        if setup_type not in {'single_dish', 'multi_dish'}:
            self.console.print("[yellow]Unknown setup type; defaulting to multi-dish mode.[/yellow]")
            setup_type = 'multi_dish'
        
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
        
        results = assign_arenas_spatial(
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

    def _is_stage_explicitly_requested(self, stage: str) -> bool:
        return stage in self._explicitly_requested_stages
    
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
                latest_detect = _latest_complete(root['detect_runs']) if 'detect_runs' in root else None
                if latest_detect:
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
                latest_crop = _latest_complete(root['crop_runs']) if 'crop_runs' in root else None
                if latest_crop:
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
                        source_color = "yellow" if crop_source in ['filtered', 'interpolated', 'manual'] else "cyan"
                        crop_info += f" ([{source_color}]{crop_source}[/{source_color}])"
                        source_path = crop_group.attrs.get('detection_source_path')
                        if source_path:
                            crop_info += f" [dim]{source_path}[/dim]"
                        
                        results_lines.append(crop_info)
                        
                        # If interpolated, show breakdown
                        if includes_interpolated:
                            n_real = crop_group.attrs.get('n_real_detections', 0)
                            n_interp = crop_group.attrs.get('n_interpolated_detections', 0)
                            results_lines.append(
                                f"  [dim]└─ {n_real:,} real + {n_interp:,} interpolated[/dim]"
                            )
                
                # Arena assignment results
                latest_assign = _latest_complete(root['arena_assignment_runs']) if 'arena_assignment_runs' in root else None
                if latest_assign:
                    assign_group = root[f'arena_assignment_runs/{latest_assign}']
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
                latest_kp = _latest_complete(root['keypoint_runs']) if 'keypoint_runs' in root else None
                if latest_kp:
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
                
                # Suggest downstream analysis
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

        # Refinement stages are designed to be repeatable; always allow rerun.
        if stage in {'refine', 'keypoints_refine', 'refined_subject_masks'}:
            return False
        
        try:
            root = zarr.open(self.config.zarr_path, mode='r')
            
            if stage == 'import':
                return 'raw_video' in root and 'images_ds' in root['raw_video']
            
            elif stage == 'background':
                if 'background_runs' not in root:
                    return False
                return _latest_complete(root['background_runs']) is not None
            
            elif stage == 'detect':
                if 'detect_runs' not in root:
                    return False
                return _latest_complete(root['detect_runs']) is not None

            elif stage == 'detect_quality':
                if 'detect_runs' not in root:
                    return False
                detect_latest = _latest_complete(root['detect_runs'])
                if detect_latest is None:
                    return False
                detect_run = root['detect_runs'].get(detect_latest)
                if detect_run is None:
                    return False
                qr = detect_run.get('quality_reports')
                if qr is None:
                    return False
                return _latest_complete(qr) is not None

            elif stage == 'crop':
                if 'crop_runs' not in root:
                    return False
                return _latest_complete(root['crop_runs']) is not None
            
            elif stage == 'keypoints':
                if 'keypoints_runs' not in root:
                    return False
                return _latest_complete(root['keypoints_runs']) is not None
            elif stage == 'refined_subject_masks':
                if 'refined_subject_masks_runs' not in root:
                    return False
                return _latest_complete(root['refined_subject_masks_runs']) is not None

            elif stage == 'keypoints_refine':
                group = _get_group_with_fallback(
                    root,
                    REFINED_KEYPOINT_GROUP,
                    LEGACY_REFINED_KEYPOINT_GROUP,
                )
                return group is not None and _latest_complete(group) is not None
            
            elif stage == 'assign_ids':
                if 'arena_assignment_runs' not in root:
                    return False
                return _latest_complete(root['arena_assignment_runs']) is not None
            
            elif stage == 'refine':
                group = _get_group_with_fallback(
                    root,
                    REFINED_DETECT_GROUP,
                    LEGACY_REFINED_DETECT_GROUP,
                )
                return group is not None and _latest_complete(group) is not None
            
            elif stage == 'track':
                if 'tracking_runs' not in root:
                    return False
                return _latest_complete(root['tracking_runs']) is not None
            
            return False
            
        except Exception as exc:
            LOGGER.warning(
                "Failed to inspect stage completion for stage=%s store=%s: %r",
                stage,
                self.config.zarr_path,
                exc,
            )
            return False


def main():
    """Main entry point for the pipeline CLI."""
    print(LEGACY_ORCHESTRATOR_NOTICE)

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
  scripts/py -m fisheye data.zarr --video-path video.mp4 --stages all
  
  # Run specific stages
  scripts/py -m fisheye data.zarr --stages import background detect
  
  # Tune parameters before running pipeline
  scripts/py -m fisheye data.zarr --tune mask
  scripts/py -m fisheye data.zarr --tune detect --frame 100
  scripts/py -m fisheye data.zarr --tune subject-mask --frame 250
  scripts/py -m fisheye data.zarr --tune keypoint --frame 50
  
  # List available tuners
  scripts/py -m fisheye --list-tuners
        """
    )
    
    # Tuning arguments (mutually exclusive with normal pipeline operation)
    tuner_group = parser.add_argument_group('tuning')
    tuner_group.add_argument(
        "--tune",
        type=str,
        metavar="TUNER",
        help="Run interactive parameter tuner (mask, detect, threshold, subject-mask, keypoint)"
    )
    tuner_group.add_argument(
        "--list-tuners",
        action="store_true",
        help="List available tuners and exit"
    )
    tuner_group.add_argument(
        "--frame",
        type=int,
        help="Specific frame/ROI index to use in tuner"
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
        "--training-data",
        action="store_true",
        help="Enable sampled import mode for training data collection (requires --frame-step)"
    )

    parser.add_argument(
        "--frame-step",
        type=int,
        metavar="N",
        help="Import every Nth frame (e.g., 100 = frames 0, 100, 200, ...). Requires --training-data flag."
    )

    parser.add_argument(
        "--registry",
        type=str,
        help="Optional registry SQLite path used by stages that emit registry status.",
    )

    parser.add_argument(
        "--refine-max-gap",
        type=int,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--refine-method",
        type=str,
        choices=['linear'],
        help=argparse.SUPPRESS,
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
        choices=[
            'import',
            'downsample',
            'background',
            'detect',
            'detect_quality',
            'crop',
            'keypoints',
            'refined_subject_masks',
            'keypoints_refine',
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
        default=None,
        choices=["auto", "refined", "detect", "manual", "filtered", "interpolated"],
        help=(
            "Detection source stage for cropping (default: config value). "
            "'auto' prefers the canonical current refined surface and falls back "
            "to raw detect; 'refined' requires the canonical curated refined "
            "surface. 'manual'/'filtered'/'interpolated' are legacy sparse "
            "compatibility modes for older archives."
        )
    )

    parser.add_argument(
        "--crop-source-path",
        type=str,
        default=None,
        help=(
            "Explicit detection source path inside the zarr (e.g. "
            "detect_runs/<run> or the canonical refined path "
            "refined_detect_runs/<run>/instances)"
        )
    )

    parser.add_argument(
        "--crop-selection-policy",
        type=str,
        default=None,
        choices=["training", "full_recording"],
        help="Policy for auto crop source selection (default: config value)"
    )
    
    parser.add_argument(
        "--crop-acceleration",
        choices=['auto', 'gpu', 'cpu'],
        default='auto',
        help="Acceleration strategy for cropping external videos (default: auto)"
    )
    
    parser.add_argument(
        "--scheduler",
        choices=['processes', 'threads', 'single-thread', 'distributed'],
        default='processes',
        help="Dask scheduler to use"
    )
    parser.add_argument(
        "--no-dask-progress",
        action="store_true",
        help="Disable Dask progress bars (useful when running batch jobs).",
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

    try:
        _reject_deprecated_refine_interpolation_overrides(
            refine_max_gap=args.refine_max_gap,
            refine_method=args.refine_method,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    
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
        import traceback
        console.print_exception()
        if args.verbose:
            console.print(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

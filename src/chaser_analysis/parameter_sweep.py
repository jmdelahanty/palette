#!/usr/bin/env python3
"""
Parameter Sweep for Gap Interpolation

Helps explore different gap filling parameters and compare results.
Supports parallel execution for faster parameter exploration.
"""

import zarr
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import subprocess
import json
from multiprocessing import Pool, cpu_count
import time
from functools import partial


def list_preprocessing_runs(zarr_path):
    """List all preprocessing runs with their parameters."""
    root = zarr.open(zarr_path, mode='r')
    
    if 'preprocessing' not in root:
        print("No preprocessing runs found.")
        return []
    
    runs = []
    for run_name in root['preprocessing'].keys():
        if run_name == 'metadata':
            continue
            
        run = root['preprocessing'][run_name]
        run_info = {
            'name': run_name,
            'created_at': run.attrs.get('created_at', 'unknown'),
            'frames_with_detections': np.sum(run['n_detections'][:] > 0)
        }
        
        # Parse history for parameters
        if 'history' in run.attrs:
            history = json.loads(run.attrs['history'])
            for step in history:
                if step['step'] == 'interpolate_gaps':
                    run_info['params'] = step['params']
                    run_info['gaps_filled'] = step.get('gaps_filled', 0)
                    run_info['frames_added'] = step.get('frames_modified', 0)
        
        runs.append(run_info)
    
    return runs


def run_single_parameter_set(params_tuple):
    """
    Run gap interpolation with a single parameter set.
    This function is designed to be called in parallel.
    
    Args:
        params_tuple: Tuple of (zarr_path, source, params_dict, run_index, total_runs, script_dir)
    """
    zarr_path, source, params, run_index, total_runs, script_dir = params_tuple
    
    print(f"[Worker {run_index}/{total_runs}] Starting with params: {params}")
    start_time = time.time()
    
    # Find the gap_interpolator script
    gap_interpolator_path = Path(script_dir) / 'gap_interpolator.py'
    if not gap_interpolator_path.exists():
        # Try current directory
        gap_interpolator_path = Path('gap_interpolator.py')
        if not gap_interpolator_path.exists():
            # Try src directory
            gap_interpolator_path = Path('src') / 'gap_interpolator.py'
    
    if not gap_interpolator_path.exists():
        print(f"[Worker {run_index}/{total_runs}] ✗ Cannot find gap_interpolator.py")
        return {
            'params': params,
            'success': False,
            'error': 'gap_interpolator.py not found',
            'elapsed_time': 0,
            'run_index': run_index
        }
    
    # Build command
    cmd = [
        'python', str(gap_interpolator_path),
        str(zarr_path),
        '--source', source,
        '--save'  # Auto-save without visualization
    ]
    
    # Add parameters
    if 'max_gap' in params:
        cmd.extend(['--max-gap', str(params['max_gap'])])
    if 'method' in params:
        cmd.extend(['--method', params['method']])
    if 'confidence_decay' in params:
        cmd.extend(['--confidence-decay', str(params['confidence_decay'])])
    if 'min_confidence' in params:
        cmd.extend(['--min-confidence', str(params['min_confidence'])])
    
    # Run command
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"[Worker {run_index}/{total_runs}] ✓ Success in {elapsed:.1f}s - {params}")
        return {
            'params': params,
            'success': True,
            'elapsed_time': elapsed,
            'run_index': run_index
        }
    else:
        # Extract meaningful error message
        error_msg = result.stderr.strip()
        if not error_msg:
            error_msg = result.stdout.strip()
        if len(error_msg) > 200:
            error_msg = error_msg[:200] + "..."
        
        print(f"[Worker {run_index}/{total_runs}] ✗ Failed after {elapsed:.1f}s - {params}")
        print(f"  Error: {error_msg}")
        return {
            'params': params,
            'success': False,
            'error': error_msg,
            'elapsed_time': elapsed,
            'run_index': run_index
        }


def run_parameter_sweep(zarr_path, parameter_sets, source='latest', dry_run=False, 
                        parallel=True, n_workers=None, force_base_source=True):
    """
    Run gap interpolation with multiple parameter sets.
    
    Args:
        zarr_path: Path to zarr file
        parameter_sets: List of parameter dictionaries
        source: Source data to use
        dry_run: If True, just print what would be run
        parallel: If True, run in parallel
        n_workers: Number of parallel workers (None = number of CPUs)
        force_base_source: If True, all runs use the same source (not cumulative)
    """
    print(f"\n{'='*60}")
    print("PARAMETER SWEEP")
    print(f"{'='*60}")
    print(f"Zarr file: {zarr_path}")
    
    # Determine the actual source to use for all runs
    if force_base_source and source == 'latest':
        # Check what the best non-interpolated source is
        import zarr
        root = zarr.open(zarr_path, mode='r')
        if 'filtered_runs' in root and 'latest' in root['filtered_runs'].attrs:
            latest_filtered = root['filtered_runs'].attrs['latest']
            actual_source = latest_filtered  # Use the actual run name
            print(f"Source (fixed for all runs): filtered_runs/{actual_source}")
            print(f"  This ensures all parameter sets start from the same filtered data")
        else:
            actual_source = 'original'
            print(f"Source (fixed for all runs): original data")
            print(f"  No filtered data found, using original")
    else:
        actual_source = source
        print(f"Source: {actual_source}")
        if not force_base_source:
            print(f"  WARNING: Without force_base_source, runs may chain together!")
    
    print(f"Parameter sets to test: {len(parameter_sets)}")
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    if parallel and not dry_run:
        if n_workers is None:
            n_workers = min(cpu_count(), len(parameter_sets))
        print(f"Parallel execution: {n_workers} workers")
    else:
        print(f"Execution mode: {'Dry run' if dry_run else 'Sequential'}")
    
    if force_base_source:
        print(f"Mode: Independent runs (all from same source)")
    else:
        print(f"Mode: Cumulative (each builds on previous)")
    print()
    
    if dry_run:
        print("DRY RUN - Commands that would be executed:")
        for i, params in enumerate(parameter_sets, 1):
            print(f"\nRun {i}:")
            print(f"  Parameters: {params}")
            print(f"  Source: {actual_source}")
            print(f"  Command equivalent:")
            cmd_parts = ['python', 'gap_interpolator.py', str(zarr_path)]
            cmd_parts.extend(['--source', actual_source])
            if 'max_gap' in params:
                cmd_parts.extend(['--max-gap', str(params['max_gap'])])
            if 'method' in params:
                cmd_parts.extend(['--method', params['method']])
            print(f"    {' '.join(cmd_parts)} --save")
        return []
    
    start_time = time.time()
    
    if parallel and len(parameter_sets) > 1:
        # Prepare arguments for parallel execution
        # Use actual_source for all runs if force_base_source is True
        worker_args = [
            (zarr_path, actual_source, params, i, len(parameter_sets), script_dir)
            for i, params in enumerate(parameter_sets, 1)
        ]
        
        # Run in parallel
        print(f"Starting {len(parameter_sets)} parallel jobs on {n_workers} workers...")
        print("="*60)
        
        with Pool(processes=n_workers) as pool:
            results = pool.map(run_single_parameter_set, worker_args)
        
    else:
        # Run sequentially
        print("Running sequentially...")
        results = []
        for i, params in enumerate(parameter_sets, 1):
            # Use actual_source for all runs if force_base_source is True
            result = run_single_parameter_set((zarr_path, actual_source, params, i, len(parameter_sets), script_dir))
            results.append(result)
    
    total_time = time.time() - start_time
    
    # Summary
    print(f"\n{'='*60}")
    print("SWEEP COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f}s")
    
    if parallel and len(parameter_sets) > 1:
        sequential_estimate = sum(r['elapsed_time'] for r in results)
        speedup = sequential_estimate / total_time
        print(f"Estimated sequential time: {sequential_estimate:.1f}s")
        print(f"Speedup: {speedup:.1f}x")
    
    successful = sum(1 for r in results if r['success'])
    print(f"Successful runs: {successful}/{len(results)}")
    
    # Show individual timings
    if len(results) > 1:
        print("\nIndividual run times:")
        sorted_results = sorted(results, key=lambda x: x['elapsed_time'])
        for r in sorted_results:
            status = "✓" if r['success'] else "✗"
            params_str = f"max_gap={r['params'].get('max_gap', 'N/A')}"
            print(f"  {status} {params_str}: {r['elapsed_time']:.1f}s")
    
    return results


def compare_runs(zarr_path, run_names=None, metric='coverage'):
    """
    Compare different preprocessing runs.
    
    Args:
        zarr_path: Path to zarr file
        run_names: List of run names to compare (None = all)
        metric: What to compare ('coverage', 'gaps', 'movement')
    """
    root = zarr.open(zarr_path, mode='r')
    
    if 'preprocessing' not in root:
        print("No preprocessing runs found.")
        return
    
    # Get runs to compare
    if run_names is None:
        run_names = [k for k in root['preprocessing'].keys() if k != 'metadata']
    
    # Collect metrics
    comparison = {}
    
    for run_name in run_names:
        if run_name not in root['preprocessing']:
            print(f"Warning: Run '{run_name}' not found")
            continue
        
        run = root['preprocessing'][run_name]
        n_detections = run['n_detections'][:]
        
        # Calculate metrics
        metrics = {
            'coverage': np.sum(n_detections > 0) / len(n_detections) * 100,
            'total_frames': len(n_detections),
            'frames_with_detections': np.sum(n_detections > 0)
        }
        
        # Get parameters from history
        if 'history' in run.attrs:
            history = json.loads(run.attrs['history'])
            for step in history:
                if step['step'] == 'interpolate_gaps':
                    metrics['params'] = step['params']
                    metrics['gaps_filled'] = step.get('gaps_filled', 0)
        
        comparison[run_name] = metrics
    
    # Display comparison
    print(f"\n{'='*60}")
    print("RUN COMPARISON")
    print(f"{'='*60}")
    
    # Sort by coverage
    sorted_runs = sorted(comparison.items(), 
                        key=lambda x: x[1]['coverage'], 
                        reverse=True)
    
    for run_name, metrics in sorted_runs:
        print(f"\n{run_name}:")
        print(f"  Coverage: {metrics['coverage']:.2f}%")
        print(f"  Frames with detections: {metrics['frames_with_detections']}")
        
        if 'params' in metrics:
            print(f"  Parameters:")
            for key, value in metrics['params'].items():
                print(f"    - {key}: {value}")
        
        if 'gaps_filled' in metrics:
            print(f"  Gaps filled: {metrics['gaps_filled']}")
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Coverage comparison
    run_labels = [r.split('_')[0] for r, _ in sorted_runs]  # Shorten names
    coverages = [m['coverage'] for _, m in sorted_runs]
    
    ax1.bar(range(len(coverages)), coverages)
    ax1.set_xticks(range(len(coverages)))
    ax1.set_xticklabels(run_labels, rotation=45)
    ax1.set_ylabel('Coverage (%)')
    ax1.set_title('Coverage Comparison')
    ax1.grid(True, alpha=0.3)
    
    # Parameter comparison (if all have same param keys)
    if all('params' in m for _, m in sorted_runs):
        param_data = {}
        for run_name, metrics in sorted_runs:
            if 'params' in metrics:
                for key, value in metrics['params'].items():
                    if key not in param_data:
                        param_data[key] = []
                    param_data[key].append(value if isinstance(value, (int, float)) else 0)
        
        # Plot max_gap vs coverage
        if 'max_gap' in param_data:
            ax2.scatter(param_data['max_gap'], coverages)
            ax2.set_xlabel('Max Gap Parameter')
            ax2.set_ylabel('Coverage (%)')
            ax2.set_title('Coverage vs Max Gap Size')
            ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return comparison


def main():
    parser = argparse.ArgumentParser(
        description='Parameter sweep and comparison for gap interpolation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all existing runs
  %(prog)s data.zarr --list
  
  # Compare existing runs
  %(prog)s data.zarr --compare
  
  # Run parameter sweep
  %(prog)s data.zarr --sweep
  
  # Custom parameter sweep
  %(prog)s data.zarr --sweep --max-gaps 5 10 15 20
        """
    )
    
    parser.add_argument('zarr_path', help='Path to zarr file')
    
    parser.add_argument('--list', action='store_true',
                       help='List all preprocessing runs')
    
    parser.add_argument('--compare', action='store_true',
                       help='Compare all preprocessing runs')
    
    parser.add_argument('--sweep', action='store_true',
                       help='Run parameter sweep')
    
    parser.add_argument('--max-gaps', nargs='+', type=int,
                       default=[5, 10, 15, 20, 30, 50],
                       help='Max gap sizes to test in sweep')
    
    parser.add_argument('--source', default='latest',
                       help='Source data for sweep')
    
    parser.add_argument('--parallel', action='store_true', default=True,
                       help='Run sweep in parallel (default: True)')
    
    parser.add_argument('--sequential', action='store_true',
                       help='Force sequential execution')
    
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: number of CPUs)')
    
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be run without executing')
    
    args = parser.parse_args()
    
    if args.list:
        runs = list_preprocessing_runs(args.zarr_path)
        print(f"\nFound {len(runs)} preprocessing runs:")
        for run in runs:
            print(f"\n{run['name']}:")
            print(f"  Created: {run['created_at']}")
            print(f"  Coverage: {run['frames_with_detections']} frames")
            if 'params' in run:
                print(f"  Max gap: {run['params'].get('max_gap', 'N/A')}")
                print(f"  Method: {run['params'].get('method', 'N/A')}")
    
    elif args.compare:
        compare_runs(args.zarr_path)
    
    elif args.sweep:
        # Create parameter sets
        parameter_sets = []
        for max_gap in args.max_gaps:
            parameter_sets.append({
                'max_gap': max_gap,
                'method': 'linear',
                'confidence_decay': 0.95,
                'min_confidence': 0.1
            })
        
        # Determine if running in parallel
        use_parallel = not args.sequential and len(parameter_sets) > 1
        
        results = run_parameter_sweep(
            args.zarr_path, 
            parameter_sets,
            source=args.source,
            dry_run=args.dry_run,
            parallel=use_parallel,
            n_workers=args.workers
        )
        
        if not args.dry_run:
            print(f"\n✓ Completed {len(results)} runs")
            # Compare the new runs
            compare_runs(args.zarr_path)
    
    else:
        parser.print_help()
    
    return 0


if __name__ == '__main__':
    exit(main())